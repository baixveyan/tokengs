# 2026 顶会前馈式 3DGS / 4D 重建文献阅读目录与方法对比手册

> **文档定位**：面向本项目（TokenGS 几何机制剖析与测试时/前馈改进）整理的 CVPR 2026 / ICLR 2026 / ECCV 2026 / TPAMI 2026 核心文献手册。不仅列出论文元数据与链接，还从**数学公式与模块设计**层面提炼每篇论文如何解决 **“空间锚定、局部连续性、冗余/死点控制、迭代/测试时精炼、动态扩展”** 五大核心问题，并直接映射到 TokenGS 的下一步改进方案。

---

## 一、19 篇核心文献总览表（按研究主线分类）

| 类别 | 论文简称 | 会议 / 期刊 | arXiv / 链接 | 核心表征范式 | 与 TokenGS 的核心关系 | 推荐度 |
|---|---|---|---|---|---|---|
| **I. 基元表征解耦**<br>*(Pixel → Token / Anchor / Voxel / Keypoint)* | **TokenGS** | CVPR 2026 Highlight | [2604.15239](https://arxiv.org/abs/2604.15239) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Ren_TokenGS_Decoupling_3D_Gaussian_Prediction_from_Pixels_with_Learnable_Tokens_CVPR_2026_paper.html) | Learnable Tokens (4096×64) | **本项目基座**：自由空间直接预测 XYZ + 测试时潜变量/Token 微调 | ★★★★★ |
| | **AnchorSplat** | CVPR 2026 | [2604.07053](https://arxiv.org/abs/2604.07053) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_AnchorSplat_Feed-Forward_3D_Gaussian_Splatting_With_3D_Geometric_Priors_CVPR_2026_paper.html) | 3D Anchor + 有界偏移 (×4 GS) | **最直接对照**：用 MVS 3D Anchor 锚定中心 + $\mu_j = A_j + \delta\mu_j$（限幅 $10/128$）+ 渲染误差反投影 Refiner | ★★★★★ |
| | **VolSplat** | ECCV 2026 | [2509.19297](https://arxiv.org/abs/2509.19297) / [Springer](https://link.springer.com/chapter/10.1007/978-3-032-37232-1_1) | 3D Sparse Voxel Grid | 与 TokenGS / AnchorSplat 构成“Token vs Anchor vs Voxel”三大非像素对齐路线 | ★★★★★ |
| | **Off The Grid** | CVPR 2026 | [2512.15508](https://arxiv.org/abs/2512.15508) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Moreau_Off_The_Grid_Detection_of_Primitives_for_Feed-Forward_3D_Gaussian_CVPR_2026_paper.html) | 2D 亚像素关键点 + 深度反投影 | 提供了**尺度乘深度（$s \propto d$）**、**置信度乘不透明度（$\alpha \cdot c$ 剪枝）**与**深度-法线自洽损失**三大即插即用工具 | ★★★★★ |
| | **SparseSplat** | CVPR 2026 | [2604.03069](https://arxiv.org/abs/2604.03069) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_SparseSplat_Towards_Applicable_Feed-Forward_3D_Gaussian_Splatting_with_Pixel-Unaligned_Prediction_CVPR_2026_paper.html) | 熵采样 3D 点 + 3D KNN 局部预测 | 深刻指出**“3DGS 优化本质是局部的”**，提出基于 3D KNN 的局部属性预测器，解决全局感知与局部预测的感受野错配 | ★★★★★ |
| | **C3G** | CVPR 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/An_Learning_Compact_3D_Representations_from_Feed-Forward_Novel_View_Synthesis_CVPR_2026_paper.html) | ~2K Learnable Query Tokens | 极简版 Token 解耦（每个 Token 解码 1 个 GS，共 ~2K 个），并复用 Cross-Attention 图将 2D 语义特征无损提升至 3D | ★★★★☆ |
| **II. 容量控制与空间排序**<br>*(Compaction & Floater Removal)* | **EcoSplat** | CVPR 2026 | [2512.18692](https://arxiv.org/abs/2512.18692) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Bui_EcoSplat_Efficiency-controllable_Feed-forward_3D_Gaussian_Splatting_from_Multi-view_Images_CVPR_2026_paper.html) | 重要性排序可控数量 GS | 提出基于**光度梯度 + 法线梯度变异图**的伪真值重要性掩码与重要性感知 Opacity 抑制损失 $\mathcal{L}_{\text{io}}$ | ★★★★☆ |
| | **Z-Order Transformer** | CVPR 2026 Oral | [2605.13465](https://arxiv.org/abs/2605.13465) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Z-Order_Transformer_for_Feed-Forward_Gaussian_Splatting_CVPR_2026_paper.html) | 3D Morton 曲线序列 + Z-Pooling | 用 Z-order 曲线将无序 3D 点排列为保局部性的一维序列，通过右移位 `Z >> h` 聚类池化自然合并重叠冗余 GS | ★★★★☆ |
| | **FreeSplat++** | TPAMI 2026 | [2503.22986](https://arxiv.org/abs/2503.22986) / [PubMed](https://pubmed.ncbi.nlm.nih.gov/41701597/) | 像素三元组 GRU 融合 + 深度正则微调 | **直接命中我们的痛点**：提出 **加权漂浮物剔除（WFR，按深度一致性衰减前景漂浮物 opacity）** 与 **深度正则化单场景微调 $\mathcal{L}_{\text{depth}}^{\text{ft}}$** | ★★★★☆ |
| **III. 迭代前馈与单场景微调**<br>*(Iterative Refinement vs Scene / Token Tuning)* | **iLRM** | CVPR 2026 | [2507.23277](https://arxiv.org/abs/2507.23277) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_iLRM_An_Iterative_Large_3D_Reconstruction_Model_CVPR_2026_paper.html) | 视角 Token 逐层反馈更新 | 将每一层 Decoder 视为一步梯度下降 $V^{(l)} \approx V^{(l-1)} - \eta\nabla\mathcal{E}$，并用 **Token Uplifting（$1\to k$ 细粒度子查询）** 解决一对多解码粗糙问题 | ★★★★★ |
| | **iSplat** | CVPR 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_iSplat_Iterative_Learning_for_Fine-Grained_Gaussian_Splatting_CVPR_2026_paper.html) | 3 次前馈迭代 + GRU 优化器 | 提出 **不确定度驱动的深度搜索空间收窄（Uncertainty-driven Depth Refinement）**，跨域泛化（Re10K→DTU）提升 +2.88 dB | ★★★★★ |
| | **GIFSplat** | CVPR 2026 | [2602.22571](https://arxiv.org/abs/2602.22571) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_GIFSplat_Generative_Prior-Guided_Iterative_Feed-Forward_3D_Gaussian_Splatting_from_Sparse_CVPR_2026_paper.html) | 前馈残差更新 + 扩散先验蒸馏 | 将 2D 渲染残差 $\psi(I_m)-\psi(R_m)$ 按高斯渲染权重 $w_i(u)$ 池化回每个 3DGS，并在稀疏视角下用单步扩散模型（DIFIX）约束新视角 | ★★★★☆ |
| **IV. 无位姿与非受控解耦**<br>*(Pose-Free & Disentanglement)* | **TokenSplat** | CVPR 2026 | [2603.00697](https://arxiv.org/abs/2603.00697) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TokenSplat_Token-aligned_3D_Gaussian_Splatting_for_Feed-forward_Pose-free_Reconstruction_CVPR_2026_paper.html) | 粗位置引导 Token 融合 + 双流解耦 | **解决跨 Token 混叠的关键参考**：先为每个 Token 预测粗 3D 位置与融合置信度，在 $\epsilon$ 邻域内按置信度 Softmax 融合重叠 Token，再做 $1\to M$ 解码 | ★★★★★ |
| | **WildSplat** | ECCV 2026 | [2607.05347](https://arxiv.org/abs/2607.05347) / [Project](https://zju3dv.github.io/wildsplat/) | 几何分支 + 外观分支显式解耦 | 几何分支只预测光照不变的 3DGS 几何结构与相机位姿，外观分支通过全局预调制 Cross-Attention 注入目标视角外观 | ★★★★☆ |
| **V. 在线流式与 4D 动态重建**<br>*(Online Stream & 4D)* | **StreamSplat** | ICLR 2026 | [2506.08862](https://arxiv.org/abs/2506.08862) / [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c822d05dcc00695ed6b63c9e97bb3449-Abstract-Conference.html) | 概率采样 + 双向形变场 + 自适应融合 | 代表从静态前馈走向未标定视频流在线动态 3DGS（1200× 加速），用自适应融合传播持久 GS、处理新增与消失 GS | ★★★★★ |
| | **DynamicVGGT** | CVPR 2026 | [2603.08254](https://arxiv.org/abs/2603.08254) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/He_DynamicVGGT_Learning_Dynamic_Point_Maps_for_4D_Scene_Reconstruction_in_CVPR_2026_paper.html) | 动态点图 + Learnable Motion Tokens | 在 VGGT 基础上引入可学习 Motion Tokens 预测每个 Gaussian 的三维速度，并用连续 3DGS 优化精炼动态几何 | ★★★★★ |
| | **MoRe** | CVPR 2026 | [2603.05078](https://arxiv.org/abs/2603.05078) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Fang_MoRe_Motion-aware_Feed-forward_4D_Reconstruction_Transformer_CVPR_2026_paper.html) | Attention Forcing 动静分离 + 因果注意力 | 通过 Attention-Forcing 策略将动态运动从静态结构中解耦，避免运动物体干扰相机位姿与静态背景几何 | ★★★★★ |
| | **Any4D** | CVPR 2026 | [2512.10935](https://arxiv.org/abs/2512.10935) / [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Karhade_Any4D_Unified_Feed-Forward_Metric_4D_Reconstruction_CVPR_2026_paper.html) | 自中心（相机系深度）+ 他中心（世界系位姿/流） | 将 4D 重建分解为相机局部坐标系下的深度/内参（Egocentric）与世界坐标系下的外参/场景流（Allocentric），避免全局坐标直接回归的尺度漂移 | ★★★★★ |
| | **OmniX** | ECCV 2026 | [2607.10840](https://arxiv.org/abs/2607.10840) / [GitHub](https://github.com/yanqinJiang/OmniX) | 紧凑 Dynamic Tokens 轨迹场 | 利用 3D 运动的稀疏与低秩特性，用少量紧凑的 Dynamic Tokens 生成全像素 3D 轨迹场，将动态运动建模与静态几何预测彻底解耦 | ★★★★★ |

---

## 二、重点论文方法精读与核心公式（按对 TokenGS 的启发排序）

### 1. AnchorSplat（CVPR 2026）—— 用“3D 几何锚点 + 有界偏移”替代自由空间盲猜
- **核心诊断**：Pixel-aligned 冗余且缺乏 3D 邻域交互；而完全无锚定的预测容易在模糊区域产生漂浮物（Floaters）和重影。
- **三大核心机制**：
  1. **MVS 3D Anchor 生成与 FPS 下采样**：先用多视角几何先验（如 MapAnything / MVS 深度反投影）得到 3D 点云，再通过体素化 + 最远点采样（FPS）下采样出 $N \ll H\times W\times V$ 个紧贴物体表面的 3D 锚点 $\{A_j \in \mathbb{R}^3\}_{j=1}^N$。
  2. **锚点有界局部偏移解码（解决 TokenGS 子球乱飞的关键！）**：
     每个 3D 锚点 $A_j$ 聚合多视角图像/深度/射线特征后，通过 Transformer 交互并由 MLP 解码出 **4 个子 Gaussian**，其中心坐标定义为：
     $$\mu_j = A_j + \delta\mu_j, \qquad \|\delta\mu_j\|_\infty \le \frac{10}{128} \approx 0.078$$
     **对比 TokenGS**：TokenGS 的子球用的是无锚定的自由空间指数函数 $\text{sign}(u)(e^{|u|}-1)$，一旦 $|u|>1$ 就飞出几米外；AnchorSplat 则把每个 Token/Anchor 严格锁在真实表面锚点 $A_j$ 周围 $\pm 0.078$ 的局部邻域内，从根源上杜绝了子球跨场景乱飞！
  3. **基于渲染误差反投影的轻量级 Gaussian Refiner**：
     将当前高斯渲染图 $\hat{I}_i$ 与输入真值 $I_i$ 的多尺度 ResNet 特征差 $e_i = F_i - \hat{F}_i$ 通过深度与相机参数反投影回每个 3D Gaussian（得到 $E_j$），再用几步前馈 Transformer 修正 $(\delta\mu_j, \alpha_j, s_j, r_j, sh_j)$。

---

### 2. FreeSplat++（TPAMI 2026）—— 专治“漂浮物”与“单场景微调深度崩塌”
- **核心诊断**：前馈式 3DGS 在多视角拼接或微调时，预测深度的噪声会产生严重偏离表面的前景漂浮物（Floaters）；且纯靠 RGB/SSIM 做单场景微调（Fine-tuning）会导致几何退化。
- **三大核心机制（对我们的 scene 微调 / token 微调极具实用价值！）**：
  1. **像素级三元组融合（PTF）**：把每个视角反投影的局部高斯 $(\mu_l^t, \omega_l^t, f_l^t)$ 与全局高斯投影做像素对齐，若深度差在阈值 $\delta$ 内则按权重加权平均中心位置，并用 GRU 融合特征，消除跨视角重叠冗余。
  2. **加权漂浮物剔除（Weighted Floater Removal, WFR）**：
     遍历各输入视角，若某个投影到像素 $i$ 的全局高斯深度 $d_g^t(j)$ 显著小于该视角的一致性深度 $d_l^t(i) - \delta$（说明它是挡在真实表面前面的半空漂浮物），则根据真实表面邻域累积权重 $\tilde{\omega}_l(i)$ 与漂浮物邻域累积权重 $\tilde{\omega}_g(m'_i)$ **直接衰减该漂浮物的 Opacity**：
     $$\alpha_g(m'_i) \leftarrow \frac{\tilde{\omega}_g(m'_i)}{\tilde{\omega}_g(m'_i) + \tilde{\omega}_l(i)} \alpha_g(m'_i)$$
  3. **深度正则化单场景微调（Depth-Regularized Per-Scene Fine-Tuning）**：
     在单场景微调（对应我们的 scene 微调 / token 微调）时，不能只用 $\mathcal{L}_{\text{color}} + \mathcal{L}_{\text{ssim}}$！FreeSplat++ 用前馈阶段得到的稠密多视角一致深度图 $\mathcal{D}_r^n$ 作为锚定约束，在微调内循环加入渲染深度 L1 正则：
     $$\mathcal{L}_{\text{rft}} = (1-\lambda_1)\mathcal{L}_{\text{color}}^{\text{ft}} + \lambda_1\mathcal{L}_{\text{ssim}}^{\text{ft}} + \lambda_2 \frac{1}{N}\sum_{n=1}^N \left\|\hat{\mathcal{D}}_r^n - \mathcal{D}_r^n\right\|_1 \quad (\lambda_1=0.2,\ \lambda_2=0.1)$$
     **直接启发**：我们在 TokenGS 中发现，纯靠 6 个输入视角 RGB+SSIM 微调会在多层半透明深度上过拟合（甚至把离群点的 opacity 从 `0.117` 推高到 `0.147`）；如果在微调内循环加入多视角一致深度锚定（或深度方差压缩损失），就能直接把“多层半透明雾”压回真实表面！

---

### 3. Off The Grid（CVPR 2026）—— 尺度-深度解耦、置信度剪枝与深度-法线自洽
- **四大亮点设计**：
  1. **Patch 内软 argmax（DSNT）连续定位**：不在网格中心放点，而在每个 $14\times 14$ patch 内用 $P$ 个通道的 Softmax 热力图期望 $(x,y) = \sum c(i,j)h(i,j)$ 预测亚像素连续 2D 位置，再借由插值深度反投影到 3D。
  2. **按图像块香农熵（Shannon Entropy）分三档分配基元密度**：最平滑的 55% patch 分配 16 个 GS，中间 35% 分配 32 个，最复杂的 15% 分配 64 个（完美印证我们在 TokenGS 中发现的“近景/复杂块开 50 球、远景/平滑块开 35 球”的物理合理性）。
  3. **尺度相对深度参数化（Scale $\propto$ Depth）**：
     $$\mathbf{s}_{\text{world}} = d \cdot \left(s_{\min} + (s_{\max}-s_{\min})\sigma(\text{raw}_s)\right)$$
     让网络预测的是**投影像素尺度**，再乘以该高斯到相机的深度 $d$，避免近景过大或远景触及硬上限 `0.075` 饱和（在 TokenGS 中有 31.7% 的点卡死在 `0.075` 上限）。
  4. **几何自洽损失（Depth-Normal Consistency）**：
     渲染深度图 $\hat{D}$ 与骨干深度 $D$ 做 L1 约束，并要求**高斯最短轴方向（即我们的 2D 薄瓦片法线！）渲染出的法线图**与深度图导出的表面法线对齐（$\mathcal{L}_{\text{normal}}$），强迫薄瓦片真正贴平在物体表面上！

---

### 4. SparseSplat（CVPR 2026）—— 揭示“全局感知 vs 局部预测”的感受野错配
- **核心洞见**：经典 3DGS 的梯度更新是**高度局部化**的——一个 Gaussian 的最佳尺寸、朝向、不透明度取决于它周围紧邻的其他 Gaussian（因为它们在同一个像素上做 Alpha 连乘 $\prod (1-\alpha_j G_j)$）。但现有前馈模型用大感受野 Transformer 提取全局特征后，却让每个点孤立地预测自己的属性，导致相邻点互相打架、重叠冗余。
- **方法**：先按局部窗口香农熵做概率采样 $P_\tau(u,v) = \text{clip}(\tau E(u,v)/\log L, 0, 1)$ 得到稀疏 3D 锚点，再对每个 3D 点查找其 **3D K 近邻（KNN）**，把中心点与 $K$ 个邻居的几何特征（xyz、法线、视线）和视觉特征一起送入局部位置感知注意力头（Geo-Aware Attention Head）预测高斯属性。仅用 22% 的 Gaussian 数量即超越 DepthSplat。

---

### 5. TokenSplat（CVPR 2026）—— 粗位置引导的跨视角 Token 空间聚类融合
- **核心问题**：多个视角的 Token 直接解码会产生大量空间重叠的 Gaussian，导致几何模糊和颜色不一致（对应我们在 TokenGS 中发现的“98% Token 半径大于最近邻间距、每个像素块混叠 ~20 个 Token”）。
- **核心方法（Token Fusion Module）**：
  1. 在解码 64 个子高斯之前，先让每个 Token 预测一个**粗略 3D 位置（Coarse Position）**和一个**融合置信度（Fusion Confidence）**；
  2. 在 3D 空间中按距离阈值 $\epsilon$ 将空间位置相近的 Token 归入同一局部簇，用置信度的 Softmax 加权融合它们的特征与粗位置；
  3. 再由融合后的非重叠 Token 通过一对多（One-to-Many）预测头生成最终的局部高斯簇。

---

### 6. iLRM & iSplat & GIFSplat（CVPR 2026 三篇迭代精炼代表作）
这三篇论文从不同角度回答了：**“如何在不跑上千步慢速优化的前提下，修正单次前馈预测的几何与外观误差？”**
- **iLRM**：
  - 将每层 Decoder 建模为显式反馈步骤 $V^{(l)} = V^{(l-1)} + F(V^{(l-1)}, S)$（保持图像特征 $S$ 固定作“梯度信号”）；
  - 提出 **Token Uplifting**：一个低分辨率 Token 先线性展开为 $k$ 个细粒度子查询 Token（$d \to k\cdot d \to k \times d$），分别去 Cross-Attend 高分辨率图像特征，再合回原 Token 解码多个 Gaussian，解决了“单个粗 Token 无法同时为内部多个子 Gaussian 对齐局部高频细节”的瓶颈。
- **iSplat**：
  - 采用 3 次前馈迭代 + GRU 优化器；
  - 提出 **不确定度驱动的深度搜索收窄（Uncertainty-driven Depth Refinement）**：根据上一步预测的不确定度动态缩小当前步每个 Gaussian 的深度调整范围，使跨域零样本泛化（Re10K→DTU）暴涨 **+2.88 dB**。
- **GIFSplat**：
  - 在每次迭代步 $t$，渲染输入视角并计算特征差 $\mathcal{O}_m^{(t)} = \psi(I_m) - \psi(R_m^{(t)})$，按每个高斯对像素的渲染贡献权重 $w_i(u)$ 将 2D 误差池化回 3D 高斯：
    $$\mathbf{o}_i^{(t)} = \frac{\sum_{m}\sum_u w_i(u)\mathcal{O}_m^{(t)}(u)}{\sum_m\sum_u w_i(u) + \varepsilon}$$
  - 对稀疏视角下的新视角（Novel Views），用冻结的单步扩散增强器（DIFIX）增强渲染图，提取特征差反投影回 3D 高斯，从而**为输入视角看不到的远视角（对应我们的 `d4+`）提供额外监督**！

---

### 7. EcoSplat & Z-Order Transformer & C3G（CVPR 2026 紧凑表征三篇）
- **EcoSplat**：构造结合图像梯度 $\|\nabla I\|_2$ 与几何法线梯度 $\|\nabla \mathbf{n}\|_2$ 的变异度图（Variation Map），对低变异度区域的冗余点施加 **重要性感知 Opacity 抑制损失 $\mathcal{L}_{\text{io}}$**，直接将非关键 Gaussian 的 opacity 压向 0，测试时按 Top-$K$ opacity 直接截断即可无损精简。
- **Z-Order Transformer（Oral）**：将 3D 高斯坐标量化后按位交错编码为一维 Morton 码（Z-order 曲线），使 3D 空间相邻的高斯在一维序列中也相邻；通过位右移 `Z >> h` 即可极速完成 3D 局部块平均池化（Z-order Pooling），消除空间重叠点。
- **C3G**：用约 2000 个 Learnable Query Tokens 与 VGGT 特征拼接做全局 Self-Attention（让 Token 之间互相通信避免抢占同一空间区域），每个 Token 只解码 1 个高信息量 Gaussian，复用注意力权重即可无损将 2D 特征提升到 3D。

---

### 8. VolSplat（ECCV 2026）& WildSplat（ECCV 2026）
- **VolSplat**：先由多视角平面扫描代价体预测深度图，将 2D 特征反投影到 3D 稀疏体素网格（Sparse Voxel Grid），再用稀疏 3D 卷积解码器在每个非空体素内预测 Gaussian，彻底保证每个 3D 局部体素至多只有一组局部高斯，无跨视角混叠。
- **WildSplat**：针对光照不一致的真实场景，设计双分支解耦架构——几何分支提取光照不变的 3DGS 结构与位姿，外观分支通过全局预调制交叉注意力注入参考图色彩风格（与本机 `FFRGS` 的几何/外观解耦思想高度一致）。

---

### 9. StreamSplat（ICLR 2026）& DynamicVGGT / MoRe / Any4D / OmniX（2026 4D 动态五篇）
- **StreamSplat**：面向未标定视频流在线重建，结合 ① 概率采样生成紧凑 3DGS、② 双向形变场对齐跨帧关联、③ 自适应高斯融合（传播持久高斯、动态增删新出现和消失的高斯）。
- **DynamicVGGT**：在静态点图基础上用可学习 Motion Tokens 预测每个 Gaussian 的三维速度向量，结合场景流监督与连续 3DGS 优化。
- **MoRe**：用 Attention-Forcing 强制将动态前景 Token 与静态背景 Token 分离，并用分组因果注意力适应可变长度的帧级 Token。
- **Any4D**：提出 **自中心（Egocentric：相机系深度与内参）+ 他中心（Allocentric：世界系外参与场景流）** 的模块化分解，避免直接在全局世界坐标系下回归所有三维坐标带来的尺度退化。
- **OmniX**：利用 3D 运动的低秩性，用少量紧凑的 Dynamic Tokens 生成全图稠密 3D 轨迹场，将动态运动与静态几何彻底解耦。

---

## 三、对照文献：TokenGS 现有问题与 2026 顶会解法的逐一映射

| 我们在 TokenGS 中实证发现的客观问题 | 之前失败的旧做法（`SW + Packing + Inter`） | 2026 顶会文献给出的正确解法参考 |
|---|---|---|
| **问题 1：无显式空间锚定，沿视线形成 12%~38% 厚度的“多层半透明雾”，导致输入视角 `d0` 极高（31.47 dB）但远视角 `d4+` 掉到 25.33 dB** | 在无表面监督的自由空间里盲目把点往悬空的 Token 均值拉，反而破坏多层 Alpha 混合 | **FreeSplat++**（渲染深度自洽正则 $\mathcal{L}_{\text{depth}}^{\text{ft}}$ + 漂浮物 Opacity 衰减 WFR）、**Off The Grid**（深度-法线自洽 $\mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{normal}}$）、**AnchorSplat**（3D 几何先验锚定） |
| **问题 2：单 Token 内 2.65 万个“半逃逸点（离核 `0.25~1.0m`，`opacity >= 1/255`）”没关干净，微调时反而被拿去作弊（opacity 从 `0.117` 升至 `0.147`），破坏单 Token 连续性** | 把 `opacity` 设为 `detach()`，并把 `<1/255` 死点也算进方差，强行通过冻结的 `deconv` 拉 `xyz`，把 34 个核心表面点拽离表面 | **EcoSplat / FreeSplat++ / Off The Grid**：**放开或直接调控 Opacity**（对偏离局部核心或违背深度一致性的离群点施加 Opacity 抑制/衰减，让其彻底归零，而不是硬拉锁死线性头下的 `xyz`）；或如 **AnchorSplat** 将子球偏移限制在有界局部范围内 |
| **问题 3：跨 Token 空间严重混叠（98% Token 半径 > 最近邻间距，每小块由 ~20 个 Token 叠加）** | 在 3D 空间对 4096 个已高度重叠的父高斯强加 `Inter-slot` 马氏排斥，把 3,879 个高不透明点推到视锥外 | **TokenSplat**（按粗 3D 位置与置信度在局部 $\epsilon$ 邻域内做 Softmax 融合）、**Z-Order Transformer**（按 3D Morton 码局部聚类池化）、**SparseSplat**（3D KNN 局部协同预测） |
| **问题 4：scene 微调 / token 微调内循环视锥外 `gsplat` 零梯度漏洞 + 远视角欠约束** | 沿用官方依赖 `gsplat means2d` 的 `visibility_loss`（对剔除点梯度为 0） | 用**解析透视投影**替代 `gsplat means2d` 计算视锥约束；借鉴 **iSplat / FreeSplat++** 在微调内循环锁定初始多视角一致深度/核心锚点防止过拟合漂移 |

---

## 四、基于文献与实证剖析的下一步研究与实验路线图（2026-10-20 记录）

结合上述 19 篇 2026 顶会论文的共识（**“用 Opacity 门控/抑制处理离群点而非硬拽 3D 坐标”**、**“在视角深度/法线空间做表面锚定而非在无参照自由空间推拉”**、**“父锚点 + 有界局部偏移”**），我们将后续实验方向按改动范围分为三条递进路线：

### 路线 A（零重训成本，直接改进 `scene + token 联合微调` 内循环）
以当前 140 场景最佳画质方案 `scene_lr1e-2_token_lr1e-4_tuning`（**27.481 dB**）为基底，抛弃旧的全局 `SW + Packing + Inter` 坐标硬拉，针对三个真实病根分别设计：
1. **A1：过滤 `< 1/255` 死点后的干净几何口径验证**
   - 在 `tokengs/losses/geo_losses.py` 中将 `opacity < 1/255`（被 `gsplat` 硬件剔除的死点）的质量权重严格置为 `0`，消除 8 个死点槽位占去 59.5% 中心方差并拖拽 34 个核心表面点的 bug。
2. **A2：离核“半逃逸点”的距离门控 Opacity 清理（$\mathcal{L}_{\text{contig-opac}}$，借鉴 EcoSplat / FreeSplat++）**
   - 针对每个 Token 内偏离核心 `[0.25, 1.00m)` 却未死透的 26,534 个半逃逸点（在无约束联合微调中平均 opacity 反而从 `0.117` 被推高到 `0.147`）：
   - 用每个 Token 前 32 个高不透明度点计算稳健核心位置 $\boldsymbol{\mu}_{s,\text{core}}$（完全 `detach`，不通过冻结的 `deconv` 拽任何点的 `xyz`），仅对超出局部半径阈值的子球施加 **Opacity 抑制损失**：
     $$\mathcal{L}_{\text{contig-opac}} = \frac{1}{S\cdot C}\sum_{s,c} \alpha_{s,c} \cdot \text{ReLU}\left(\frac{\|\mathbf{x}_{s,c}.\text{detach}() - \boldsymbol{\mu}_{s,\text{core}}\|}{\sigma_s} - \tau\right)^2$$
   - 目标：在不损伤（乃至提升）PSNR 的前提下，把偏离 Token 局部区域的半逃逸点压到 `< 1/255` 剔除，消除单 Token 内部孤岛（25.8%）并提升局部连续性。
3. **A3：视线深度自洽 / 深度厚度压缩 + 解析视锥投影约束（借鉴 FreeSplat++ / Off The Grid）**
   - 针对沿视线 12%~38% 厚度的“多层半透明雾”以及第 32 步后输入视角 `d0` 过拟合、远视角 `d4+` 停滞的问题：
   - 在微调内循环利用 `gsplat` 的期望深度渲染（`RGB+ED`）施加深度锚定 $\|\hat{D}^{(t)} - \hat{D}^{(0)}\|_1$ 或视线深度方差/不透明度二值化约束；
   - 同时用解析相机矩阵投影替代 `gsplat` 的 `means2d`（修复剔除点读 `at::empty` 未初始化内存且梯度为 0 的漏洞）。

### 路线 B（中等改动，打破 `deconv` 线性层刚性耦合）
- 在 `scene + token 联合微调` 的基础上，允许对 `ClipActivationHead.deconv` 的位置/不透明度行做极低学习率微调，或对存活子球引入有界残差偏移 $\delta\mu$（借鉴 **AnchorSplat** 的 $\|\delta\mu\|_\infty \le r_{\max}$），打破 64 个子球在冻结线性头下余弦相似度高达 `+0.97~+0.99` 的刚性锁死。

### 路线 C（架构级创新，对标 CVPR/ECCV 2026 新范式）
1. **父锚点 + 有界局部子球参数化（借鉴 AnchorSplat）**：每个 Token 先预测 3D 父锚点 $\mathbf{p}_s$，64 个子球定义为 $\mathbf{x}_{s,c} = \mathbf{p}_s + \tau_{\max}\tanh(\Delta \mathbf{u}_{s,c})$，从数学上保证每个 Token 100% 对应单片连续局部区域，彻底根除远端死点与半逃逸点。
2. **粗位置引导的 Token 局部去重与 3D KNN 交互（借鉴 TokenSplat + SparseSplat + Z-Order Transformer）**：在 Decoder 中利用粗位置 $\mathbf{p}_s$ 做 3D KNN 局部注意力或 Morton 码聚类融合，解决“98% Token 半径大于最近邻间距、每像素块 ~20 个 Token 混叠”的问题。
3. **基于渲染残差反投影的前馈 Refiner（借鉴 GIFSplat + iSplat）**：将 50 步梯度微调（scene 微调 / token 微调）蒸馏为 2~3 步前馈残差网络（把输入视角渲染误差 $I - \hat{I}$ 按高斯权重池化回 Token 直接预测 $\Delta\text{scene\_latents}$ 和 $\Delta\text{gs\_tokens}$）。

---

## 五、3D World Models（单图/少图/多模态先验生成完整 3D 场景）专题阅读目录与核心借鉴

> **核心问题**：3D World Models 是如何从 1 张图（或少量图 + 几何先验）生成整个可漫游大场景的？它们的哪些架构设计可以直接指导我们的“扫描仪点云 + 多视角大图前馈 3DGS”？

### 1. 代表性 3D World Model 论文与技术报告总表（2025–2026）

| 流派分类 | 论文 / 报告简称 | 机构 & 发表年份 | 链接 | 输入 → 输出 | 核心技术路线 |
|---|---|---|---|---|---|
| **流派 I：任意几何先验提示的统一世界重建**<br>*(Any-Prior World Reconstruction)* | **WorldMirror 1.0 / 2.0**<br>*(HunyuanWorld-Mirror)* | Tencent Hunyuan & CUHK & ZJU (2025–2026) | [arXiv:2510.10726](https://arxiv.org/abs/2510.10726) / [GitHub](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) | 多图 + **可选先验（点云/深度/位姿/内参）** → 3DGS + 点图 + 法线 + 深度 | **与我们新方向最贴合**：多模态 Token 化（深度 Token 直接加到图像 Token 上 $\mathbf{T}^{\text{img}}+\mathbf{T}^{\text{depth}}$）+ 几何与 3DGS 外观两阶段解耦训练，支持 200 视图大场景前馈 |
| **流派 II：视频扩散世界模型 + 前馈 3DGS 提升**<br>*(Video World Model → 3DGS Lifting)* | **Lyra 1.0 & Lyra 2.0**<br>*(Explorable Generative 3D Worlds)* | **NVIDIA Toronto AI Lab** *(与 TokenGS 同作者团队)* (ICLR 2026 / 2026.04) | [arXiv:2604.13036](https://arxiv.org/abs/2604.13036) / [Project](https://nv-tlabs.github.io/Project-Lyra/) | 单图 + 相机轨迹 → 长视频 → 完整大场景 3DGS / Mesh | 维护每帧 3D 点云缓存（3D Cache）做**基于可见性得分的贪心最大覆盖选帧**，将 3D 对应关系仅注入 Q/K（不污染 V），最后用**重叠分块（Chunked）前馈 3DGS** 重建长轨迹大场景 |
| | **Wonderland** | Snap & UToronto & UCLA (CVPR 2025) | [arXiv:2412.12091](https://arxiv.org/abs/2412.12091) / [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Liang_Wonderland_Navigating_3D_Scenes_from_a_Single_Image_CVPR_2025_paper.html) | 单图 → 相机控制视频 Latent → 前馈 3DGS（LaLRM） | 不把视频解码成像素图，而是训练前馈 3D 重建模型（LaLRM）直接从视频扩散模型的压缩潜空间（Video Latents）解码大范围 3DGS |
| | **Aether** | Shanghai AI Lab & USTC & ZJU (ICCV 2025) | [arXiv:2503.18945](https://arxiv.org/abs/2503.18945) / [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Aether_Geometric-Aware_Unified_World_Modeling_ICCV_2025_paper.html) | 初始观测图 + 相机轨迹动作 → 4D 动态重建 + 视频预测 + 视觉规划 | 将 4D 几何重建、相机轨迹动作控制视频预测与目标规划三任务交错统一训练，实现零样本合成到真实泛化 |
| **流派 III：点云缓存引导的迭代补全扩展**<br>*(Point-Conditioned Iterative World Expansion)* | **ViewCrafter & ReconX** | CUHK / Tencent / HKUST (2024–2025) | [arXiv:2409.02048](https://arxiv.org/abs/2409.02048) / [arXiv:2408.16767](https://arxiv.org/abs/2408.16767) | 单图 / 极少图 → 初始点云投影渲染 → 视频扩散补全 → 稠密 3DGS | 先构建粗糙全局点云并投影到新视角（含遮挡空洞），以此作为强几何条件引导扩散模型补全缺失区域并反投影更新点云，最后初始化 3DGS |
| | **WonderWorld** | Stanford & MIT & Google (CVPR 2025) | [arXiv:2406.09394](https://arxiv.org/abs/2406.09394) / [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_WonderWorld_Interactive_3D_Scene_Generation_from_a_Single_Image_CVPR_2025_paper.html) | 单图 + 交互式文本/移动 → 实时扩展的相连 3D 场景（<10s） | 提出 **快速分层高斯薄片（FLAGS, Fast Layered Gaussian Surfels）**，直接用深度、法线、焦距做几何解析初始化（把耗时几何优化降维成轻量外观微调）+ 引导深度扩散实现场景无缝拼接 |
| | **HunyuanWorld 1.0 & Voyager** | Tencent Hunyuan (2025) | [arXiv:2507.21809](https://arxiv.org/abs/2507.21809) / [Report](https://3d-models.hunyuan.tencent.com/world/HY_World_1_technical_report.pdf) | 单图 / 文本 → 360° 全景代理 → 语义分层 3D 世界（3DGS / Mesh） | 以 360° 全景图作为世界代理，按语义拆分为前景可交互物体层与背景环境层，做层对齐深度估计与分层 3D 重建 |
| **流派 IV：单图多层深度偏移前馈 3DGS**<br>*(Single-Image Layered Feed-Forward 3DGS)* | **Flash3D** | Oxford VGG (3DV 2025) | [arXiv:2406.04343](https://arxiv.org/abs/2406.04343) / [Project](https://www.robots.ox.ac.uk/~vgg/research/flash3d/) | 单张 RGB 图 → 毫秒级前馈完整 3DGS 场景 | 用冻结单目度量深度模型预测第 1 层可见表面深度 $D_1$，再让前馈网络预测正深度偏移 $\Delta D_2,\dots,\Delta D_K > 0$ 在射线后方生成多层高斯以填补被遮挡背景 |

---

### 2. 核心解惑：World Model 是怎么“从一张图输出整个场景”的？
单张图只包含正前方可见表面的信息，**没有任何模型能靠纯几何计算算出一堵墙背后的样子**。World Model 之所以能从 1 张图输出完整大场景，本质上靠的是 **“显式 3D 几何骨架（点云 / 深度 / Surfel）+ 生成式补全 / 分块扩展”的双引擎配合**：
1. **对看得见的区域（Visible Surface）**：绝不让网络在自由空间里盲猜 3D 坐标，而是用深度估计或点云直接建立**显式 3D 表面骨架**（如 WonderWorld 的 FLAGS 薄片初始化、Flash3D 的首层深度锚定、ViewCrafter/Lyra 2.0 的 3D 点云缓存）；
2. **对看不见的区域 / 大范围扩展（Occluded / Unbounded Regions）**：要么在单条射线上预测多层后景深度偏移 $D + \Delta D$（Flash3D，适合小范围视差），要么将当前已知的 3D 点云投影到新相机视角，用视频扩散模型根据 3D 投影提示“脑补”出新视角画面，再用**分块前馈 3DGS（Chunked Feed-Forward 3DGS）**把生成的视频段提升并拼接成完整大场景（Lyra 2.0 / Wonderland / ViewCrafter）。

---

### 3. 从 World Model 提炼出的 5 条可直接用于我们“点云 + 大场景前馈 3DGS”的设计准则

1. **准则 1（来自 WorldMirror Eq. 1 & Appendix E.1）：点云/深度先验注入 Encoder 时，用“投影深度 Patch 相加（$\mathbf{T}^{\text{img}} + \mathbf{T}^{\text{depth}}$）”而非拼接 Token**
   - 如果把点云单独做成一串 Token 拼在图像 Token 后面，Transformer 序列长度翻倍、注意力计算量暴增 4 倍。
   - **最佳做法**：将已知点云投影到输入视角得到归一化深度图 $D_i$（及有效掩码），过一层 Patch Embedding 得到与图像块一一对齐的 $\mathbf{T}_i^{\text{depth}}$，直接**逐元素加到图像 Token 上**（在 TokenGS 中即直接并入 `_embed_encoder_input` 的通道中）——**零额外序列长度、零额外注意力开销**，却能让 Encoder 从第 1 层起获得精确的 3D 深度先验。
2. **准则 2（来自 Lyra 2.0 Appendix A.1）：几何/位置引导信号只加进 Attention 的 Query 和 Key，保持 Value 纯净！**
   - Lyra 2.0 发现：把 3D 几何对应坐标或射线编码加进 Value（$V$）会污染预训练模型的高频纹理与色彩特征；**只加进 Query 和 Key（$\mathbf{q}=W_Q(\mathbf{x}+\mathbf{p}),\ \mathbf{k}=W_K(\mathbf{x}+\mathbf{p})$，用于引导跨视角正确配对路由），而让 Value 专心传递外观特征**，重建画质最清晰。
3. **准则 3（来自 WorldMirror Sec. 3.2 & A.2）：几何定位与 3DGS 外观必须“解耦顺序训练” + 渲染深度锁定（$\lambda_{\text{gsdepth}}=0.1$）**
   - 联合硬训几何与渲染会导致 3DGS 为了拟合 2D 颜色而破坏 3D 表面结构（产生多层半透明雾）。应当先由点云/深度锁死 3D 几何特征，再训练/微调 3DGS 头部的颜色、尺度与不透明度，并始终保持 $\lambda_{\text{gsdepth}} \|\hat{D}_{\text{render}} - D_{\text{prior}}\|_1$ 约束。
4. **准则 4（来自 WonderWorld 的 FLAGS）：用点云的坐标、法线和焦距直接做 3DGS 的物理几何初始化**
   - 已知点云 $\mathbf{p}$、表面法线 $\mathbf{n}$、相机距离 $d$ 和焦距 $f$ 时，高斯薄片的初始中心就是 $\mathbf{p}$、最短轴旋转直接对齐法线 $\mathbf{n}$、初始切向尺度直接设为 $s_x, s_y \propto d / f$，网络只需预测微小的残差修正量（把困难的“从零回归几何”变成简单的“表面残差精修”）。
5. **准则 5（来自 Lyra 2.0 Sec. 4.2 & README）：几百张图的大场景必须采用“基于点云可见性最大覆盖的选帧 + 重叠分块（Chunked）前馈重建”**
   - Lyra 2.0（与 TokenGS 同作者团队）在官方实现中明确验证：面对几百帧的长序列大场景，单次前馈全序列会导致严重模糊；正确做法是利用点云投影可见性得分（Visibility Score）挑出覆盖率最高且不冗余的局部视图子集，切成多个**重叠子块（Overlapping Chunks）**分别前馈生成 3DGS，再在统一几何骨架下合并。

---

## 六、专题调研：点云 / LiDAR / 几何先验在重建中的使用方式（2025–2026，面向“训练用点云、推理只用 RGB”的后训练）

> 调研时间：2026-10-26。目的：回答“别人重建时点云到底怎么用”，并对照我们的硬门槛（只能在 10T 级预训练模型上做小数据后训练、推理只用 RGB + 相机）。只收录已核实出处的论文（CVPR/ICLR/ICCV/NeurIPS/3DV/WACV 及 NVIDIA、Meta、ByteDance、Tencent、ETH 等实验室的 arXiv）。按“点云在流程里扮演的角色”分四类。

### 1. 分类总览表

| 角色 | 论文（出处） | 点云/几何先验具体怎么进入 | 训练规模与前提 | 对我们的直接意义 |
|---|---|---|---|---|
| **A. 推理期可选输入（提示/Prompt）** | **WorldMirror**（Tencent Hunyuan，arXiv 2510.10726，2025-10） | “多模态先验提示”：位姿、内参、深度图各自经轻量编码层变成结构化 token 注入前馈模型，任意子集可缺省；同时输出点云/深度/法线/3DGS。论文明确发现“先验注入对所有任务普遍增益” | 大规模多数据集训练，训练时随机丢弃先验 | 证实“输入提示 + 随机遮盖”在**大规模预训练**阶段成立；我们无法复现其训练规模 |
| | **MapAnything**（Meta，arXiv 2509.13414，3DV 2026） | 把几何拆成“深度图 + 局部射线图 + 位姿 + 米制尺度因子”的因子化表示，**输入和输出用同一套表示**，可选输入深度/位姿/内参/部分重建，共 64 种输入组合，靠“输入增强”统一训练 | 大规模多数据集；单一前馈模型覆盖 12+ 任务 | 因子化表示值得借鉴：几何先验用“深度图 + 射线”注入最自然；但同样依赖大规模训练 |
| | **Prompt Depth Anything**（ByteDance/ZJU，CVPR 2025） | 低成本 LiDAR 作为“提示”，在深度解码器多尺度处融合；为解决“LiDAR + 精确 GT 成对数据少”，构建**合成 LiDAR 噪声模拟 + 真实数据伪 GT** 的数据管线；ScanNet++ 上 SOTA | 基于 Depth Anything 预训练模型做下游适配 | 两点可借：① 先验噪声要在训练里主动模拟；② 它是在**预训练深度模型上做适配**，与我们的后训练处境最接近，但它推理时仍需要 LiDAR |
| | **Prior Depth Anything**（ZJU/HKU，arXiv 2505.10565） | 粗到细：先用相对深度预测“预填充”不完整的米制先验（像素级对齐 + 距离加权），再用以先验为条件的单目深度模型精修先验噪声 | 零样本泛化 7 个数据集 | 对**不可靠/不完整先验**的处理范式：先用预测补洞、再条件精修——对应我们“网格有洞、细结构缺失”的处理 |
| | **AnchorSplat**（Huawei，CVPR 2026，arXiv 2604.07053） | 3D 几何先验（稀疏点云/体素/RGB-D 点云）直接作为**锚点**，预测锚点对齐的高斯 + 有界偏移；推理时锚点由预训练 MVS（MapAnything/MVSAnywhere）从图像预测；ScanNet++ v2 NVS SOTA，高斯数少 20×。注意其协议是 **32–64 张输入图、4–8 新视角、1168×1752 输入** | 在 ScanNet++ v2 上训练 | 锚点思想即我们的 S4；但它推理时的“点云”来自 RGB 预测模型，不是传感器——这正是“推理只用 RGB”的一种合法实现 |
| | **GGPT**（ETH，CVPR 2026） | 改进 SfM 得到稀疏但可靠的点云与位姿，作为“引导编码”送入几何引导点 Transformer，精修 VGGT 的稠密点图；**只在 ScanNet++ 上训练**即可跨数据集泛化 | 中等规模（ScanNet++） | 证明“少量可靠稀疏几何引导”足以纠正前馈模型的多视角不一致——与我们 DL3DV 上“1.35% COLMAP 像素锚定胜过 80% 稠密预测深度”的结论一致 |
| | **YoNoSplat**（ETH，ICLR 2026） | 可用 GT 位姿/内参也可不用；核心是“混合训练策略”：先用 GT 位姿聚合局部高斯，**逐步过渡到预测位姿与 GT 位姿的混合**，避免训练不稳与曝光偏差 | 大规模 | 退火式“拐杖抽离”在大规模训练下可行（对应我们搁置的方案三）；小数据后训练无此条件 |
| | **Points-to-3D**（Adelaide，CVPR 2026） | 用可见区域点云（LiDAR 或 VGGT 预测）替换 TRELLIS 稀疏结构潜变量的纯噪声初始化 + 结构补全网络 | 生成任务 | 同一思想在生成领域的版本；说明“点云作为潜变量初始化/锚定”是通用做法 |
| **B. 只在训练期使用（监督 / 老师 / 蒸馏）** | **DistillNeRF**（NVIDIA，NeurIPS 2024，arXiv 2406.12095） | **两阶段**：① 对每个驾驶片段离线训练逐场景 NeRF（EmerNeRF，用全部多帧多视角 + LiDAR），生成**稠密深度与虚拟视角目标**；② 把这些目标蒸馏进单帧多相机的前馈模型。论文指出 LiDAR 点云稀疏、只覆盖有限视场，所以用逐场景优化模型做“深度自动标注” | 两阶段，前馈模型只看单帧图像 | **我们第四条路的直接先例**：老师 = 逐样本优化 + 特权几何；学生 = 前馈 RGB。差别在于我们的网格已经稠密，老师的任务是“把真值翻译成解码器听得懂的记忆格” |
| | **UniPAD**（CVPR 2024）/ **GaussianPretrain**（arXiv 2411.12452） | LiDAR 点云作为**预训练监督目标**：图像特征体渲染/高斯渲染去拟合 LiDAR 几何；下游只用相机 | 驾驶数据预训练 | “LiDAR 只当训练目标”范式；与我们 S1 同类（输出端监督），论文结论是能显著提升下游相机感知 |
| | **MemDistill**（POSTECH/Samsung，ICCV 2025） | LiDAR 老师 → 相机学生，中间加一个**可学习记忆库 + 场景相关检索器**：先冻住学生、只训记忆与检索器去复现老师的中间特征与预测，再微调学生；推理时检索器从图像特征取出“LiDAR 知识” | 检测任务 | 极有启发：TokenGS 的 4,096 个可学习 `latents` 本身就是“记忆库”，我们解冻 `latents` 去逼近老师 Z\*，在形式上正是“记忆蒸馏” |
| | **L2COcc**（arXiv 2503.12369，2025） | LiDAR 模型 → 相机模型的三层蒸馏：特征相似度蒸馏、TPV 中间表征蒸馏、预测对齐蒸馏 | 占据预测 | 验证“中间表征 + 输出”双层蒸馏优于单层——对应我们学生损失里的“记忆格贴近 Z\* + 渲染深度贴近老师” |
| | **Lyra / Lyra 2.0**（NVIDIA Toronto，arXiv 2509.19296，2025） | 自蒸馏：3DGS 解码器只用视频扩散模型 RGB 解码器的输出做监督；Lyra 2.0 用 VIPE 位姿 + DA3 深度把生成视频提升为 3DGS，长序列必须分块 | 冻结扩散模型，只训 3DGS 解码器 | 与 TokenGS 同团队；“老师是同一模型的另一条解码路径、学生只训一小块”的组织方式与我们一致 |
| | **Splat and Distill**（ICLR 2026） | 用前馈 3DGS 把老师 2D 特征提升到 3D 再投影到新视角，作为“3D 一致的老师信号”蒸馏给 2D 学生；替代慢速逐场景优化 | 2D 基础模型 | “老师端先做 3D 一致化再蒸馏”的思想：我们的老师 Z\* 经几何修正后再蒸，正是让老师先“3D 一致” |
| | **Depth Anything 3**（ByteDance，arXiv 2511.10647，2025-11） | 单一 Transformer + “深度-射线”单一预测目标；**老师-学生范式**（老师生成伪标签）达到 DA2 级细节 | 公开数据集大规模 | 基础模型本身也靠老师-学生；其“深度-射线”表示与 TokenGS 的 Plücker 射线输入天然兼容，可作后续 RGB 侧几何先验候选 |
| | **MeshSplat**（arXiv 2508.17811，2025） | 前馈 2DGS：深度图上的加权 Chamfer 损失（重叠区加权）+ 单目法线估计器对齐 2DGS 朝向；无需 3D 真值 | ScanNet/Replica 级 | 法线来自单目估计器而非真值网格——当网格法线不可信（细结构）时的替代来源 |
| | **DepthSplat**（ETH，CVPR 2025） | 预训练单目深度特征（Depth Anything）**在特征层**与多视角代价体融合；证明深度与 GS 互相促进、深度预训练有益 GS | 大规模 | 另一条低风险路线：RGB 可得的深度基础模型特征作几何先验（不是点云，但推理合法），可作对照 |
| | **G4Splat**（BIGAI，ICLR 2026） | 利用**平面结构**从稀疏观测推导米制深度图作为可靠监督；几何引导可见性掩码、选视角、约束生成一致性；ScanNet++/Replica 验证 | 逐场景 + 生成先验 | 支持我们“网格可信度标签”里“平面区域最可信”的判断；平面是室内最稳的几何先验 |
| | **DN-Splatter**（WACV 2025） | 深度 + 法线先验正则（含按颜色梯度自适应加权的深度损失、局部平滑、单目法线），室内 3DGS 与网格提取 | 逐场景 | 深度损失按图像梯度自适应加权（边缘处放松）——可直接用于我们老师/S1 的深度损失设计 |
| **C. 表征层面：几何住在哪里** | **Omni-Scene**（Westlake，CVPR 2025） | 像素对齐高斯 + 体素（三平面）高斯互补，专治少重叠/遮挡/截断 | 驾驶 | TokenGS 的 token 本来就不是像素对齐的“3D 槽位”；文献共识是“3D 对齐表征 + 几何锚定”才能稳住几何 |
| | **SparseSplat / VolSplat / CSG-Fusion**（CVPR 2026 / ECCV 2026 / ICCVW 2025） | 像素非对齐、体素对齐、匹配先验剪枝合并，减少冗余并提升一致性 | — | 已在第一～三节精读，不重复 |
| **D. 把 LiDAR 当可渲染传感器** | **SplatAD**（Zenseact/Chalmers，CVPR 2025）/ **LiHi-GS**（arXiv 2412.15447） | 从高斯**可微渲染 LiDAR 距离图像**（球坐标光栅化、射线丢失、强度、卷帘效应），用真实 LiDAR 直接监督 | 驾驶，逐场景 | 提示我们：若未来用原始激光点云（而非网格）监督，应渲染“传感器域”的距离图而非只比较深度图；当前有稠密网格，深度光线投射已足够 |

### 2. 四条共识与对我们路线的核对

1. **“点云作为推理期输入”的方法无一例外依赖大规模训练 + 随机先验丢弃/输入增强**（WorldMirror、MapAnything、YoNoSplat）。这与 15.4 节的后训练门槛分析一致：在 853 个房间上让预训练 Transformer 学会读一种新模态，没有文献先例支持。AnchorSplat 给出的合法变通是：推理时“点云”由 RGB 预测模型（MapAnything）提供——这不违反“推理只用 RGB”，但相当于把几何能力外包给另一个大模型，不是我们要的“内化”。
2. **“只在训练期用点云”的成熟范式是“逐样本优化老师 → 前馈学生”（DistillNeRF）或“LiDAR 老师 → 相机学生的中间表征蒸馏”（MemDistill、L2COcc）**。两者合起来就是我们的第四条路：老师 = 原模型 + 真值几何做记忆格优化；学生 = 解冻 `latents` + 尾部 4 层去逼近老师的记忆格与渲染深度。MemDistill 还提示：**可以先只训 `latents`（记忆库）去逼近老师，再放开其余层**，分两步降低风险。
3. **先验不可靠是常态，顶会方法都显式处理**：Prompt Depth Anything 主动模拟 LiDAR 噪声；Prior Depth Anything 先补洞再条件精修；G4Splat 只信平面；DN-Splatter 在图像边缘放松深度损失；MeshSplat 用单目法线替代不可靠真值法线。这些共同支持 15.4 节的“网格可信度标签”作为正式中间产物：平面可信、细结构/透明反射不可信、洞用预测低权重补。
4. **双层蒸馏（中间表征 + 输出）优于单层**（L2COcc、Splat and Distill）。学生损失保留“记忆格贴近 Z\*”与“渲染深度贴近老师渲染深度”两项，而不是只选其一。

### 3. 由调研新增的候选与对照

- **对照 D1（RGB 侧几何先验，推理合法）**：仿 DepthSplat，把冻结的 Depth Anything 3 / DA2 特征在阶段 ② 零初始化相加注入。它不用点云，但推理只用 RGB，能回答“几何提升是否必须来自点云”；建议在第四条路跑通后作为对照，而非替代。
- **候选 T1（老师增强）**：若“天花板诊断”显示老师在冻结解码器下提升有限，参考 AnchorSplat/WonderWorld 在老师中放开 `deconv` 或引入锚点 + 有界偏移（S4），但只在老师阶段放开，学生仍保持基线结构。
- **候选 T2（两步学生）**：仿 MemDistill，学生第一步只训 `latents` 逼近 Z\*，第二步再放开 `latent_blocks.8..11`，观察哪一步贡献几何提升。
- **评测可比性提醒**：AnchorSplat 等 ScanNet++ v2 NVS 结果基于 32–64 张输入图与官方 benchmark 协议，与我们的 6 输入/8 新视角/18 帧窗口协议不可直接对比；总表中不得混列。


