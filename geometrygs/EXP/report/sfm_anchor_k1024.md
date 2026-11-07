# sfm_anchor_k1024

把 TokenGS 4096 个 GS token 中的前 1024 个槽位，替换为固定在“输入视角自算 SfM 点”上的锚点 token。scene latent、其余 3072 个自由 GS token、解码器和读出头都保留。状态：训练和评测已完成；2026-11-06 12:34（UTC+8）完成 50 场景、400 帧 FP32 评测，退出码 0。

## 方法

| 部分 | 做法 | 代码 |
|---|---|---|
| SfM | 只用本场景被选中的输入视角。GPU 上 kornia SIFT（8192 特征，1024×1792，与模型输入相同的中心裁剪与翻转）→ 互近邻 + 比值 0.9 → 已知位姿极线过滤（Sampson < 2 px）→ 轨迹合并（同视角重复的轨迹丢弃）→ DLT 三角化，去粗差后重三角化 → 相机前方、重投影 < 2 px、夹角 > 1.5° | `geometrygs/sfm.py` |
| 锚点筛选 | 可靠点：轨迹 ≥ 3 视角或夹角 ≥ 3°，且 5 近邻平均距离 < 10 cm（孤立点基本是错误三角化）；再用最远点采样取至多 1024 个，保证空间覆盖 | `select_anchors` |
| 锚点 token | 共享可学习基向量 + LayerNorm(位置傅里叶特征 MLP + 编码器 patch 特征在其观测视角投影处的平均) × 0.02 | `SfmTokenGS._anchor_tokens` |
| 解码 | 锚点 token 与自由 token 一起过原解码器（读 scene latent、互相自注意力）和原读出头；锚点 token 的 64 个子高斯位置改为“锚点 + 0.045·tanh(偏移)”（约 15 cm 有界），颜色、尺度、旋转、透明度仍由原读出头给出 | `forward_reconstruction` |
| 不足 1024 个可靠点 | 多出的槽位保持原 gs token 与原读出，等价于原 TokenGS | 单元测试已验证 0 锚点时与 TokenGS 输出逐位一致 |

推理只需要 RGB + 相机（SfM 由输入图像现算，不使用激光或官方 COLMAP）。训练期激光只用于损失。

## 训练配置

- 起点：官方 `dl3dv_latent_6v_ssim.safetensors`；新模块随机初始化，第零步与原模型不同（初始验证 PSNR 18.84，冒烟 10 个优化步后恢复到 25.4）。
- 配方与 `sft_latenttail_readout_rgb_depth_surface3d_posmlp`（10 epoch）一致：解冻 scene latent 尾段 + 读出链路，位置 MLP，RGB(L1 0.8 + SSIM 0.2) + 渲染对数深度 0.1 + token 三维贴面 0.05（1 epoch 升温），输入视角也监督，FP32，每卡 batch 1 × 累积 2，lr 4e-5，锚点模块 lr × 5。
- 与该对照的差异：① 锚点 token；② **每个训练场景随机取 4–8 个输入视角**（窗口间隔 8–20 帧，4 个监督目标视角），对照固定 6 个。两项同时改变，比较时不能把差异全部归因于锚点。
- 7 × RTX PRO 5000（GPU 1–7，严格空闲），每 epoch 487 步/卡，约 0.67 it/s。
- workspace：`/mnt/disk/xveyan/workspace_scannetpp/geometrygs_sfm_anchor_k1024`；完整日志 `EXP/sfm_anchor_k1024/run.log`。

## 评测

沿用 TokenGS ScanNet++ 协议（50 个验证场景、6 输入、8 新视角、FP32、G1–G4 几何），`python -m geometrygs.evaluate`。demo：`EXP/demo/scene_7b6477cb95/sfm_anchor_k1024/`，额外包含 `sfm_points.ply`（全部三角化点）、`sfm_anchors.ply`（被选为 token 位置的点）、`anchored_vs_free_gaussians.ply`（α ≥ 0.05 的高斯中心，锚点 token 红、自由 token 灰）。

对照（旧协议同口径，数字来自 TokenGS `EXP_ScanNetPP/*/summary.json`）：

| 模型 | PSNR | LPIPS | AbsRel | 厚度 p90 (cm) | Acc (cm) | Comp (cm) | F@5 | 法线中位 (°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 官方 release | 30.50 | 0.180 | 0.174 | 25.7 | 22.4 | 10.2 | 0.445 | 37.3 |
| 同配方 10 epoch（主对照） | 30.63 | 0.185 | 0.071 | 13.9 | 7.4 | 7.8 | 0.626 | 22.2 |
| 同配方 20 epoch（历史最强） | 30.69 | 0.184 | 0.066 | 13.6 | 6.9 | 7.4 | 0.641 | 21.0 |
| **sfm_anchor_k1024** | **30.624** | **0.1863** | **0.0702** | **13.42** | **7.315** | **7.686** | **0.6271** | **21.89** |

相对 10 epoch 历史对照：PSNR -0.008 dB，AbsRel -0.00085，厚度 p90 -0.44 cm（约 3.2%），表面准确度误差 -0.051 cm，完整度误差 -0.066 cm，F@5 +0.00091。整体接近对照，厚度有小幅下降；单次运行且训练视角设置不同，不能认定为稳定的 SfM 增益。尚未超过历史 20 epoch 最强模型的准确度与完整度。

本次仍使用 RGB + 激光渲染深度 + 贴面损失；不是讨论中的去贴面版本，也未加入新的几何交互层或专门的局部 MLP 输出头。尚未拆分测量锚定高斯的离面距离/法向宽度、自由高斯的补充覆盖及两者重叠，因此不能仅凭总体结果判断两类 token 是否完成预期分工。

除总体指标外，需要按“离最近锚点的距离”分层看几何改善，判断 SfM 几何是否传播到没有点支撑的区域（尚未实现该分层统计）。

## 已知局限

- 可靠 SfM 点集中在纹理区域：50 个验证窗口上只有约 29% 的输入可见表面离锚点 < 10 cm（见 `sfm_quality_val6v.md`）。
- 自由 token 只能通过 1 层解码器的自注意力读取锚点 token，而预训练模型的自注意力 LayerScale 接近 0。
- 锚点位置固定；逐坐标 tanh 将各轴偏移限制在 ±15 cm，欧氏距离上界约 26 cm，并非半径 15 cm 的球约束。该约束不限制高斯协方差的范围；SfM 点错误会直接带偏该 token。
