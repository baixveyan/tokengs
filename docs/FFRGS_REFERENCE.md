# FFRGS 参考笔记（供 TokenGS 研究线借鉴）

> 来源：`4DGS/FFRGS/`（代码仓库，commit 80e6ed2，2026-10-12）与 `4DGS/ffrgs_workspace/`（68 个实验目录，只有 run.sh / args / log / train_log.jsonl，无 checkpoint）。2026-10-19 由会话 b1a5cb35 通读代码与文档后整理。
>
> 定位：FFRGS 是**另一条研究线**（在冻结 VGGT-Ω 上做 pose-free 前馈 3DGS），本仓库只把它当参考。本文只记录对 TokenGS baseline / TTT / 几何约束研究有用的事实。数字均注明出处（`AGENTS.md` 指 `4DGS/FFRGS/AGENTS.md`），与原文冲突时以原文为准。
>
> 第 7 节是用这些知识反查我们自己实验的结论，是本文最需要看的部分。

## 1. 项目一句话与主线结果

- 目标：只输入图像（无位姿），由冻结的 VGGT-Ω 预测相机和深度，再用 parent / child 两级 slot decoder 回归 3DGS。
- 最佳：v18@15k，held-out 8 场景 PSNR 23.74，HF 0.601（`REPRODUCE_v18.md`）。
- 同口径 TokenGS（官方 latent-6v）：喂 GT pose 25.84；喂 Ω pose、Sim3 对齐后 24.60；直接在 Ω 坐标系 23.94。TokenGS + Ω 相机：目标视图 −2.5 dB，HF 0.89→0.88，说明位姿误差只造成错位、不造成模糊（`design_notes_appearance_encoder.md` §1）。
- 关键文件：`models/ffrgs.py`（总装）、`models/omega_content_encoder.py`、`models/parent_decoder.py`、`models/patch_routing.py`、`models/child_decoder.py`、`models/direct_child_decoder.py`、`models/losses.py`、`models/render.py`、`data/dl3dv_dataset.py`、`scripts/train.py`、`scripts/export_ply.py`。

## 2. 架构（代码级）

### 2.1 OmegaContentEncoder
- VGGT-Ω 的 aggregator / camera_head / depth_head 全部冻结；aggregator 以 bf16 常驻（省 2.3 GB），只物化第 4/11/17/23 层 token（patch 16）。
- 每层 LN + Linear(2048→256)，拼接成 1024 维；加 patch-16 Plücker 卷积 embedding 和 2 个 type embedding（patch / camera-register）。
- 12 层 `LatentBottleneck`（4096 latents × 1024，QK-norm，LayerScale 0.01），**权重直接从 TokenGS latent checkpoint 的 `enc_dec_backbone.latents / latent_blocks` 拷贝**。
- 坐标尺度（gauge）：`pointmap_mean_distance` 用 Ω 深度反投影点到平均相机中心的平均距离，把场景缩放到 1.0。发布的 TokenGS 6v checkpoint 在 constant-0.15 gauge 下反推出的点距 p10/p50/p90 为 0.53/0.92/2.73，所以目标值取 1.0 而不是 0.15（`AGENTS.md` L517–525）。
- Pose-free：Ω camera head 预测输入外参；训练时用 GT 输入相机与 Ω 相机解 Sim3（`sim3_from_cameras`：旋转 Kabsch，尺度和平移由相机中心最小二乘），把 GT 目标相机映射到 Ω 坐标系后再渲染监督。Sim3 必须在 fp32、autocast 关闭时计算。
- 曾经的 GT pose 泄漏：source K 一度使用 dataset GT Plücker，修正后同一 checkpoint 从 22.40 掉到 19.58（`AGENTS.md` L568–575）。**教训：pose-free 的声明要逐一检查每个输入通道**。

### 2.2 ParentDecoder（1 个 block）
- memory cross-attn → self-attn → FFN，LayerScale 0.01，QK-LN。
- `X0_init` 从 TokenGS gs_tokens 初始化；ParentHead 由 TokenGS deconv head 的 64 个 child readout 取平均初始化，LN gain：几何 0.11，颜色 0.42。
- 每个 parent 是一个 3D 高斯（center / scale / quat / opacity / color），既参与渲染（parent loss），也用来路由。

### 2.3 足迹路由 `parent_footprint_mask`（无参数，no_grad）
- 把 parent 高斯用 EWA 投影到每个输入视图，按 patch 网格算覆盖率（最小半径 0.5 patch），再按 splat 透射率做遮挡；任何 patch 都覆盖不到时退回全 1。
- child decoder 的 patch cross-attn 用 `log(mask.clamp_min(1e-30))` 作 attention bias（不要直接 `log(mask)`，`log(0)` 反传会出 NaN）。

### 2.4 Child decoder（每 slot 64 个 child）
- `SlotChildDecoder`（相对坐标）：`p = c + R(σδ)`，`scale = 0.25σ·exp(clamp(raw, ±2))`；patch 的 xyz 以 hi/lo 两路 bf16（`split_hi_lo`）挂在 V 上做“注意力加权平均位置”。bf16 直接放绝对坐标误差 1.5e-2（比 child scale 还大），拆 hi/lo 后为 7e-7。
- `DirectChildDecoder`（绝对坐标）：`xyz_blend` 从 1 退火到 0，让 head 最终不依赖位置旁路；`0·x` 让未用参数留在图里以满足 DDP。
- 激活：`opacity = sigmoid(raw − 2)`，`color = 0.5·tanh + 0.5`，与 TokenGS 一致。

## 3. 训练与评估设置

| 项目 | FFRGS 设置 | 备注 |
|---|---|---|
| 重建损失 | 0.8·L1 + 0.2·(1−SSIM)/2 | 与 TokenGS SSIM 变体相同 |
| 监督视图 | 目标视图 + 6 个输入视图（children 与 parents 都监督，parent λ 1.0） | TokenGS latent-6v finetune preset **没有** input supervision；从零训 preset 有 |
| visibility | 解析投影 `project_means2d` + 视图维 min，NaN→1e6 | 见 §5.1，与 TokenGS 写法不同 |
| 优化器 | AdamW β(0.9, 0.95)，wd 0.05（1-D 参数与 slot query 不衰减），clip 1.0 | 梯度非有限时跳过该步并打印参数名 |
| 调度 | 线性 warmup + cosine 到峰值 5% | v5 的“19.84 上限”被证实是 cosine 把 lr 压太低造成的假饱和（`AGENTS.md` L187–191） |
| 精度 | bf16 autocast，TF32 开 | 位姿相关计算 fp32 / fp64 |
| 渲染 | 256×448，背景 0.5 灰 | 与我们的 preset 相同（`bg_color=grey`，`img_size=[256,448]`） |
| held-out | `HOLDOUT_SEED=987654321`，30 个 val 取前 8 个，val 数据集 seed+991 | 不同 run 的数字可直接比 |
| 指标 | PSNR、`edge_l1`、`hf = |∂pred| / |∂gt|` | `scripts/train.py::edge_hf`；hf 是锐度而非精度，高于 1 也不好 |

**视角采样**（`data/dl3dv_dataset.py::_sample_frame_indices`，沿用 TokenGS `_get_indices_static`）：gap ∈ [10, 40]；6 个输入 = 首尾端点 + 4 个中间帧；4 个目标全部在 `[start, start+gap]` 内，**训练中从不出现外推视角**。

**TokenGS 原版训练配方**（FFRGS 对照时整理，`AGENTS.md` L187–191、L517–520）：从零 latent preset 约 165k 步 × global batch 64（≈1100 万样本），lr 4e-4，然后 10k 步 finetune lr 4e-5；从零 preset 使用 `camera_scale_method=pointmap`、全部输入视图监督、`random_reflect`、`dec_depth=1`。

## 4. 对 TokenGS 本身的测量（可直接引用为背景）

1. **针状高斯**：TokenGS aniso p50 约 77–89（FFRGS v18 约 18–25）；长轴与视线 |cos|>0.9 的比例在小视差区最高。一根沿相机射线的针在该相机里投影成点，无论深度对错都能画对像素，所以**输入视图监督对针是零代价**；内插目标视角只沿极线轻微拉开，也几乎不惩罚（`design_notes_needles.md`）。
2. **外推掉点**：输入不变，把目标放到输入区间外 ±5/10/20/40 帧，PSNR 掉约 5.5 dB、HF 0.63→0.34。用法线对齐 loss 压针（v20/v21）能把 >0.9 比例从 27–39% 压到 6–20%，**但外推 PSNR 完全不变**，说明只改了形状、深度不确定没解决。
3. **测试时优化（只用输入视图）**：FFRGS 只调颜色 HF 0.64→0.76（输入视图 +3.2 dB），颜色+opacity 0.85；**TokenGS 只调颜色几乎不变**，说明颜色不是 TokenGS 的瓶颈；由此推断 scene-latent TTT 的收益主要来自几何 / opacity 的改变（推断，待用几何统计验证）。
4. **TokenGS 每 token 的 64 个 Gaussian 是刚性模板**：child 构型跨 slot 相关 0.90–0.94；latent decoder 每个 head 实际只读约 1.1 个 latent（硬选择）；回传到 token 的梯度约为 FFRGS 的 100 倍。FFRGS 从零训的 head 也会学成“4096 个 slot 共享的 child 模板”（bias 路径梯度累加 4096 倍），几何 loss 能把 slot 推开但不能创造 child 分工（`AGENTS.md` L505–513）。
5. **官方 TokenGS 的几何统计**（12 场景，opacity 加权）：SW 0.646；packing 18.94 vs 目标 8.87；inter K4 马氏距离 0.252（目标 2），too-close 87.8%；aniso 36.71；opacity mean 0.0573（`AGENTS.md` L455–459）。与我们 AGENT_MEMORY 1.4 的实测（packing 35–161、inter ≈0.5）同向。
6. **TokenGS 上的几何微调对照**（全模型 10k，3998 场景，lr 1e-5）：在 30 个未参与微调的 5K 场景上，官方 17.052 / 纯重建 control 17.403 / 几何（SW 0.1 + packing 0.001 + inter 0.01）17.316，**几何相对同预算 control 低 0.087 dB**（中位 −0.065，7/30 提升）；SW / inter / aniso 大幅改善；10 视图平均三者 19.965 / 19.774 / 19.453，主要牺牲输入视图复现（`AGENTS.md` L460–484）。这就是我们 `EXP_TokenGS.zip` 方案里“Geo 比 Recon 低 0.087”的出处。
7. **数据泄漏教训**：FFRGS 的 12 个 held-out 场景全部在 TokenGS loader 的训练集里（各重复 6 次）；TokenGS Provider 在 `sample_list *= repeat` 之后才切最后 `batch_size` 条作 test，内部唯一 test 场景也在 train 中出现 5 次。文中“单场景 eval PSNR 37.80”就是这个泄漏场景的数字，**不是官方 benchmark 精度**。
8. **opacity 二值化**：在 TokenGS 上加 `p(1−p)` binary loss 会把 opacity 推向 0（全零吸引域），网络靠增大 scale 补偿；加全局 mass 约束后强弱 slot 垄断反而加剧（Gini 0.768）。ADC（复制强 slot 替换弱 slot）只是复制贡献、不产生新表达（`AGENTS.md` L405–447）。
9. **密度分配**：GS 密度 ∝ 像素数，根因是每像素等权的渲染 loss；**几何正则的形式必须和参数化匹配**，结构上的扰动会被训练拉回（`design_notes_density_allocation.md`）。
10. **位姿**：render loss 对相机的梯度 |g|≈1.7e-4，means3D 的约 2.3e5，相差约 9 个数量级；三种方式实测 render loss 都不指向 GT pose，用 render loss 修位姿不可行（`AGENTS.md` L1055–1060）。

## 5. Bug 与陷阱清单（多数与 TokenGS / gsplat 共通）

### 5.1 gsplat `means2d` 对被剔除的 Gaussian 无效 ⚠️ 直接影响本仓库
- gsplat 1.5.3 `Projection.cpp` 用 `at::empty` 分配 `means2d`；`ProjectionEWA3DGSFused.cu` 在以下情况直接 `radii=0; return`，**不写 means2d**：深度在 `[near, far]` 外；2D 协方差行列式 ≤0；**opacity < 1/255**（`rasterization()` 把 opacities 传进投影做紧致 bbox）；半径 ≤ radius_clip；**bbox 完全在画面外**。反传 kernel 对 `radii<=0` 的点也直接跳过，梯度为 0。
- 后果：读 `info["means2d"]` 的 visibility loss 对“在所有监督视图都出视锥”的点——也就是论文 Eq. 1–2 真正要惩罚的点——取值是未初始化内存，梯度恒为 0。只有“bbox 与画面相交但中心在外”的窄带点能收到梯度。
- FFRGS 的修复：`models/render.py::project_means2d` 用相机内外参解析投影（z clamp 到 1e-4），NaN→1e6。
- 本仓库 `tokengs/models/losses.py::compute_visibility_loss_from_means2d` 读的正是 gsplat `means2d`（`rendering/gs.py`），并且 NaN→0（0 归一化后正好落在边界 −1，惩罚为 0）。视图维 min、clip 1.0 与论文一致。**官方训练与官方 TTT 都受此影响**，所以这不是我们复现的偏差，而是官方实现的特性。

### 5.2 其他
1. **autocast 权重缓存**：同一 autocast 区域内对 leaf 张量原地更新（optimizer step、手工改权重）后，bf16 cast 仍用缓存的旧值，消融必须用 forward hook。我们在单场景复现中发现的“BF16 下 TTT 崩到 12.64，关闭 autocast cache 后恢复”与这一机制吻合：autocast 只缓存 requires_grad 的 leaf 张量的 cast，scene latents 正是这样的 leaf，Adam 原地更新后，同一 autocast 区域内仍用旧的 bf16 副本。
2. bf16 下 AdamW 更新 bf16 主权重：权重 0.015 时量化步长约 1e-4，lr 1e-6 级的更新全被吞掉（`AGENTS.md` L534–539）。需要 fp32 master weight。
3. bf16 下 covariance Cholesky 非正定（负特征值 −3.18e-5），先转 fp32 也救不回已丢的精度；应在 autocast 关闭时用 fp32 构造 `factor = R·S`，`cov = factor·factorᵀ`。
4. SVD / det 没有 bf16 kernel，Sim3 必须在 autocast 外。
5. TF32 带来约 0.4° 位姿误差下限，位姿评估用 fp64。
6. `torch.nn.functional.huber_loss` 线性区斜率是 δ（不是 1），想要“L1 斜率为 1”的鲁棒损失应用 `smooth_l1_loss(beta=δ)`。
7. `log(0)` 的反传是 NaN（即使前向结果被 mask 掉）：bias 一律 `clamp_min(1e-30)` 后再取 log。
8. DDP unused parameter：条件分支里不用的参数要以 `0·x` 留在图里，或不创建该模块。
9. 8 卡同时首次 JIT 编译 gsplat 会触发 pybind11 竞态（SIGSEGV），先单卡预热 `TORCH_EXTENSIONS_DIR` 再多卡复用；`MAX_JOBS=1`。
10. 随机初始化的 GS head（readout std 0.002、bias 0）会把 262k 个 Gaussian 全放在一个点上；从零训 decoder 必须保留 TokenGS 预训练的 X0 / heads。
11. fused-ssim 必须 `pip install --no-build-isolation`，否则 build isolation 拉新 torch 编出不兼容的 .so。

## 6. 诊断探针与可视化工具（可移植）

- **锐度 / 高频**：`edge_l1`、`hf ratio`（`scripts/train.py::edge_hf`），按“到最近输入相机的距离 / 输入相机最大间距”分桶报告（`design_notes_needles.md` 检查 2）。
- **针统计**：opacity 加权的 aniso（最大 / 最小 scale）p50；长轴与视线的 |cos|，报 p50 与 >0.9 比例；按局部视差（6 个输入相机射线的最大夹角）分桶。
- **slot 统计**：`slot_statistics`（alive 阈值 1/255）：effective children、packing ratio vs target、贡献 Gini、跨场景 bottom-10% Jaccard。
- **注意力**：Q/K effective rank、每 slot effective patches、top-k mass；消融 `scale_patch=0`、shuffle V（用 forward hook）。
- **可视化**：`scripts/export_ply.py`（按 slot 着色、parent、Ω 点云）、`standardize_ply.py`（补成 62 属性）、`prune_ply.py`、`cors_server.py`（8899 端口 + SuperSplat `?load=`），见 AGENT_MEMORY 3.1。

## 7. 反查我们的不微调与场景潜变量微调结果（2026-10-19，只读核查 ）

> 术语：本节的 Baseline = 不微调（`no_tuning`），TTT = 场景潜变量微调（scene latent tuning，官方称 TTT，只优化 scene latents），TTT+Geo = 微调内循环加几何损失。它们与论文 / FFRGS 里调 Gaussian token 的 TTT 不同。结果目录名见 `EXP_TokenGS/renamed_directories_old_to_new.md`。

### 7.1 核查确认没问题的部分
- Baseline 是官方权重原样推理（strict load、`model.eval()`、`no_grad`、FP32、无 TTT / 无几何项），口径与官方一致，见 AGENT_MEMORY 6.1。
- 渲染 256×448、背景 0.5 灰、6 输入、L1 0.8 + SSIM 0.2、visibility 1.0 / clip 1.0、`matmul_allow_tf32=False`，均与 preset 和 FFRGS 对照口径一致（`EXP_TokenGS/no_tuning/launch_contract.json`）。
- TTT 只用 context 图像监督，target 图像不参与，无泄漏。

### 7.2 评估集只有内插：TTT 增益随“离 context 帧的距离”迅速衰减
- 评估索引 `evaluation_idx_dl3dv_depthsplat_6v.json`：每个场景的 target 是连续 50 帧（如 0–49），6 个 context 跨度 37–49 帧，target 仅 2.1% 落在 context 跨度外，且只出 1–几帧。每场景实际有 304–439 帧，benchmark 只用了前 50 帧附近。
- 用现有 `frame_metrics.jsonl` 按“到最近 context 帧的帧距 d”分桶（140 场景，7000 帧，配对 baseline vs TTT）：

| 分桶 | 帧数 | Baseline PSNR | TTT PSNR | 增益 | TTT 变差比例 |
|---|---|---|---|---|---|
| d=0（context 本身） | 840 | 26.561 | 31.475 | **+4.914** | 0.0% |
| d=1 | 1400 | 26.028 | 28.755 | +2.727 | 1.7% |
| d=2–3 | 2527 | 25.245 | 27.063 | +1.818 | 5.9% |
| d=4–6 | 1896 | 24.178 | 25.394 | **+1.216** | 12.9% |
| d≥7 | 190 | 23.512 | 24.755 | +1.244 | 8.4% |
| context 跨度外 | 147 | 23.805 | 26.259 | +2.454 | 4.1% |

  （“跨度外”的帧都紧挨端点 context，帧距小，所以增益大，不代表外推能力。）
- 含义：all_frames 主指标里有 12% 帧就是 context 本身（增益 +4.9 dB），novel_only 的 6160 帧里约 2/3 离 context ≤3 帧（含紧挨端点的跨度外帧），d≥4 只有 2086 帧。“+2.2 dB”的 TTT 增益有相当部分是在拟合输入视图；离 context 4 帧以上只有约 +1.2 dB。**这个基准对几何质量几乎不敏感**，而几何约束的收益预计恰恰体现在远视角，必须另设外推口径，否则 TTT+Geo 很可能“看不出差别或略降”。

### 7.3 visibility 项基本失效（官方行为，但会影响我们的几何研究）
- 见 §5.1。在 TTT 内循环中，出视锥或 opacity < 1/255 的 Gaussian 收不到 visibility 梯度；损失值读未初始化内存，只影响日志、不影响梯度。
- 另外：多视图取 min 时，被剔除视图的垃圾值可能被选中，从而决定梯度是否流向“bbox 相交”那个视图。这理论上也是一个 run-to-run 不确定性来源（我们之前只归因于 gsplat atomicAdd），未验证。
- 正式对照（TTT vs TTT+Geo）**两组都保持官方写法**，否则基线不再是官方 TTT；若要研究 floater，可另加“解析投影 visibility”作为第三组或诊断项。

### 7.4 几何正则形式与 FFRGS 的差异（TTT+Geo 设计时要决定）
| 项 | 我们 `tokengs/losses/geo_losses.py` | FFRGS `models/losses.py` |
|---|---|---|
| 质量权重 | detached opacity | detached opacity × 体积^(2/3)（`_mass_weights`） |
| packing | log 比值平方，目标 eff^(2/3)（与 FFRGS 等价，轴向取 mean / sum 一致） | 同 |
| inter | `(d − 2)²`，全部 active slot 的 kNN | `smooth_l1(log(d/2), beta=1)`，随机 256 个 query 的 kNN |
| 视为“针”的约束 | 无 | v20/v21 法线对齐（能压针，外推无改善） |

- 预训练解 d≈0.25：`(d−2)²≈3.06`，梯度 2(d−2)≈−3.5。按比值看不对称：过近时损失有上界 4（d→0），过远时无上界；log 比值形式对 ×8 过近与 ×8 过远同等对待。这可能部分解释我们审计中“packing / inter 梯度是 sw 的 3–8 倍”（未验证）。若把 inter 换成 log 形式，λ 的有效量级会变，**必须在预注册前定下来**。
- FFRGS 在 TokenGS 上的全模型对照结论（§4.6）：几何正则相对同预算纯重建约 −0.09 dB、几何指标大幅改善。TTT 阶段的预期也应如此：**PSNR 持平或略降是正常结果，论文主张要靠几何指标与远视角指标支撑**。

### 7.5 建议（按优先级，均未执行）
1. **（用户 2026-10-19 决定：当前实验暂不做，留作后续单独实验）加外推 / 远视角诊断口径**（不替换官方 50 帧口径）：每场景 context 不变，额外渲染 `max(context) + {5, 10, 20, 40}` 帧（所有场景都有 ≥304 帧），按 FFRGS 的“到最近 context 相机中心距离 / context 最大间距”分桶报 PSNR / SSIM / LPIPS / HF。也可先用已有数据按帧距分桶（7.2 表，零成本）。
2. **评估加 `edge_l1` 与 `hf ratio`**，与 PSNR 同时报 all / novel / 分桶。
3. **几何统计进入正式汇总**：Baseline、TTT、TTT+Geo 都报 opacity 加权 aniso p50、长轴∥视线 >0.9 比例、packing ratio、inter 距离、effective children、全视图出视锥比例（解析投影）。FFRGS 的结论“TokenGS 只调颜色几乎不变”提示 TTT 主要在改几何；TTT 是否在加重针（输入视图零代价拟合），是我们故事的关键证据。
4. **TTT 过程曲线**：在 smoke 场景记录每步 context loss、远视角 PSNR、几何统计，确认 50 步、lr 1e-2 在远视角上是否已饱和或过拟合（论文 Fig. 12 只看了含 context 的口径）。只作诊断，不改正式配置。
5. **决定 inter 损失形式**（`(d−2)²` 还是 log 比值）和质量权重，写进预注册，再定 λ。
6. 可选的第三组：在 TTT+Geo 上加解析投影 visibility（或 anti-needle 项），专门针对 floater / 针。
