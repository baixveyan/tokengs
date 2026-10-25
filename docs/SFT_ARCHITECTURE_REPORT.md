# TokenGS 激光雷达后训练微调（`[Exp-SFT-3: Fused + Unfrozen deconv, 100 Epochs]`）逐层解剖报告

本报告详细说明 <ref_file file="/home/jms/xveyan/4DGS/TokenGS/tokengs/models/lidar_sft_tokengs.py" />（预设名 `sft3_scannetpp_lidar_fused_100ep`，启动脚本 `scripts/run_sft3_lidar_fused_100ep.sh`）中：**全网络每一层到底是冻结（❄️）还是参与微调（🔥）、哪些地方新增了 SFT 适配器、各自的峰值学习率是多少、以及新增的 6 项激光雷达几何损失具体作用在哪一层**。

---

## 一、全网络逐层解剖总表（哪里冻结？哪里微调？哪里是新增 SFT 模块？）

全模型总参数量为 **`365.88 M`**（原版 `362.70 M` + 新增零初始化 SFT 适配器 `3.18 M`），其中 **❄️ 冻结 `281.99 M`（77.1%）**，**🔥 参与 SFT 微调 `83.89 M`（22.9%）**：

| 序号 | 网络阶段与具体模块名 | 参数量 | 状态（❄️冻结 / 🔥微调） | 是否新增 SFT 层 | 峰值学习率 `lr` | 为什么要这样设置？ |
|---:|---|---:|---|---|---:|---|
| **1** | **[部位 1] 图像与光线切块**<br>`patch_embed` ($8\times 8$ Conv) | `0.20 M` | ❄️ **完全冻结** | 原版预训练层 | `0` | 保持底层 RGB + Plücker 射线切块特征不变 |
| **2** | **[部位 1] 12 层 ViT 图像编码器**<br>`enc_dec_backbone.encoder.0 ~ 11` + `multiscale_norms` | `151.19 M` | ❄️ **完全冻结** | 原版预训练层 | `0` | 完整保留官方在 `DL3DV-10K`（1 万场景）上训好的通用多视角 2D 视觉特征提取能力，防过拟合且省显存 |
| **3** | **[部位 2 前段] Latent Bottleneck 前 8 层**<br>`enc_dec_backbone.latent_blocks.0 ~ 7` | `126.15 M` | ❄️ **完全冻结** | 原版预训练层 | `0` | 前 8 层负责基础的多视角图像块跨视图聚合，保持稳定不乱动 |
| **4** | **[部位 2 尾段] 场景潜变量查询与后 4 层**<br>`enc_dec_backbone.latents` (`[4096, 1024]`) + `latent_blocks.8 ~ 11` | `67.29 M` | 🔥 **参与微调** | 原版预训练层 | `1.0e-5`<br>*(0.25× 温和 LR)* | 让 `scene_latents` 形成尾段适应室内几何并与下方激光雷达探针协同演化 |
| **5** | **★ [新增 SFT 模块 1] 激光雷达 3D 表面内省与零初始化残差回注器**<br>`lidar_adapter` (`probe_head` + `geo_encoder` + `zero_inject_proj`) | **`2.13 M`** | 🔥 **参与微调** | **★ 新增 SFT 适配器**<br>*(Zero-Init 残差)* | **`4.0e-5`**<br>*(1.0× 全速 LR)* | 从 `scene_latents` 直接内省解码 4,096 个 3D 表面点坐标 $\hat{\mathbf{p}}_s$ 与法线 $\hat{\mathbf{n}}_s$（受激光雷达真值点云监督），编码后通过**零初始化线性层**残差加回 `scene_latents`（保证第 0 步与官方模型 `30.56 dB` 逐位一致！） |
| **6** | **[部位 3] 潜变量 KV 桥接投影层**<br>`latent_proj_k/v` + `k_proj_norm / v_proj_norm` | `2.10 M` | 🔥 **参与微调** | 原版预训练层 | `1.0e-5`<br>*(0.25× 温和 LR)* | 将注入了 3D 激光雷达表面知识的 `scene_latents` 映射为供下游解码器读取的 Keys / Values |
| **7** | **★ [新增 SFT 模块 2] 高斯 Token 3D 空间位置注入层**<br>`token_3d_grounding` (`Linear(1024 -> 1024)`) | **`1.05 M`** | 🔥 **参与微调** | **★ 新增 SFT 适配器**<br>*(Zero-Init 残差)* | **`4.0e-5`**<br>*(1.0× 全速 LR)* | 将 `lidar_adapter` 提炼出的 3D 表面几何残差零初始化注入 `gs_tokens`，赋予 4,096 个 `gs_tokens` 显式三维空间位置感 |
| **8** | **[部位 4A] 解码器跨注意力层（纹理读取）**<br>`decoder_blocks.0.gs_cross_attn` + `gs_cross_attn_scale` | `2.10 M` | ❄️ **完全冻结** | 原版预训练层 | `0` | **关键保护设计**：在 `SFT-2` 中解冻该层曾导致第 0 轮掉 `-1.38 dB`；冻结它可 100% 保护原版从 `scene_latents` 读取高频纹理的能力！ |
| **9** | **[部位 4B] 高斯 Token 与解码器横向自注意力 + MLP**<br>`gs_tokens` (`[4096, 1024]`) + `decoder_blocks.0.(gs_self_attn + mlp)` | `10.50 M` | 🔥 **参与微调** | 原版预训练层 | `1.0e-5`<br>*(0.25× 温和 LR)* | 唤醒原本沉睡在 `gamma = 0.00036` 的 Token 间横向自注意力，让落在同一面墙/桌面上的相邻 Token 互相拉紧，消除表面断裂 |
| **10** | **[部位 5] 高斯属性输出头（反卷积层）**<br>**`activation_head.deconv` (`Linear(1024 -> 896)`)** | **`0.92 M`** | 🔥 **参与微调（已解冻！）** | 原版预训练层 | **`1.0e-5`**<br>*(0.25× 温和 LR)* | **核心解冻项**：打破官方预训练模型“顺着视线吐 42.5% 细针与 0.846 刀片”的死模具，在下方软几何损失驱动下重塑为输出贴面扁平圆盘！ |

---

## 二、SFT 训练中作用的 6 项损失函数（分别监督哪里？）

在 `LiDARSFTTokenGS.compute_lidar_grounding_loss`（<ref_snippet file="/home/jms/xveyan/4DGS/TokenGS/tokengs/models/lidar_sft_tokengs.py" lines="174-279" />）中，除了原有的 2D 图像渲染损失（`0.8 * L1 + 0.2 * SSIM + 1.0 * Visibility`）外，激光雷达三维点云与形态损失分层作用在两个位置：

### A. 作用在中间层 `scene_latents`（编号 4, 5）的 2 项激光雷达内省损失：
1. **双向 3D Chamfer 表面贴合损失（`0.20 * loss_chamfer`）**：
   强迫 4,096 个 `scene_latents` 内省预测出的 3D 点 $\{\hat{\mathbf{p}}_s\}$ 均匀铺满并贴合在真实激光雷达表面点云 $\mathcal{P}_{\text{lidar}}$ 上。
2. **激光雷达局部切平面法线正交损失（`0.05 * loss_normal`）**：
   强迫每个 `scene_latent` 预测的法线 $\hat{\mathbf{n}}_s$ 垂直于真实激光雷达表面局部切向量。

### B. 直接作用在末端解冻后的 `deconv` 与 `decoder`（编号 7, 9, 10）上的 4 项高斯微粒整形损失：
3. **子球个体到激光雷达真值表面的 3D 距离损失（`0.15 * loss_child_surf`）**：
   不再只拉 64 个子球的平均重心，而是直接对前 8,192 颗高活性子球个体计算到最近激光雷达表面点的 3D 距离——逼迫每一颗子球个体老老实实贴在真实墙面/桌面上！
4. **单 Token 内部视线方向深度厚度压缩损失（`0.05 * loss_radial_thick`）**：
   直接惩罚同一个 Token 内 64 个子球沿相机视线方向的前后散开偏移 $\big|(\mathbf{x}_{s,c} - \bar{\mathbf{x}}_s) \cdot \mathbf{d}_{\text{ray}}\big|$——彻底堵死“32 个子球在墙前 20cm、32 个子球在墙后 20cm 凑平均值”的漏洞，直接压薄 `RayThick`！
5. **去细针软损失（`0.02 * loss_aniso`）**：
   对所有存活高斯惩罚长宽比超标项 $\text{ReLU}(s_1 / s_2 - 1.8) + 0.5\,\text{ReLU}(s_3 / s_2 - 0.35)$——训练解冻后的 `deconv` 权重自己学会输出 $s_1 \approx s_2$、薄轴 $s_3$ 扁平的圆盘，从根源上降低 `42.5%` 的细针占比！
6. **消除“顺视线刀片”的旋转对齐软损失（`0.02 * loss_blade_orient`）**：
   惩罚高斯椭球长轴在相机视线方向上的投影平方 $\frac{\sum_a s_a (\mathbf{e}_a \cdot \mathbf{d}_{\text{ray}})^2}{s_1}$——训练解冻后的 `deconv` 旋转行权重把顺着视线竖着插的刀片转 $90^\circ$，平贴在物体表面切平面上！
