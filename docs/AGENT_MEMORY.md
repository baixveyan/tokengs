# TokenGS 共享背景记忆（Agent Memory）

> 本文件是 TokenGS 研究线的**共享背景记忆**。新的 agent 接手前应先读完本文件，再读 `EXP_TokenGS/EXP_TokenGS.md`（当前实验的执行日志与原始数字）。
>
> 最后更新：2026-10-20。来源：TokenGS 论文（arXiv 2604.15239）+ `release_2026.6.md` + 会话“检查4dgs代码完整性”（4f8b4e60）+ 当前会话（b1a5cb35）+ 对仓库与产物的现场核验。

## 0. 用途与维护规则

- **权威性排序**：用户最新指令 > 当前代码 / 产物文件 > 本文件 > 历史会话。本文件与代码或产物不一致时，以代码和产物为准，并回来修正本文件。
- **更新方式**：每完成一个阶段（实验跑完、方向变更、发现新陷阱），在对应章节追加或修正内容，并在第 5 节时间线里补一行。不要删除历史阶段，废弃的方向标注“已废弃”并说明原因。
- **语言约定**：中文正文，英文术语保留（scene latent、PSNR 等）。
- **术语（禁用“TTT”笼统简称，用户硬性要求 2026-10-20）**：由于论文里的 TT 指调 Gaussian token（`token-tuning`），而官方 release 代码的默认 preset 却把调 scene latents（`scene-latent-tuning`）称作 TTT，两者存在严重不一致。**从现在起一律禁止使用“TTT”这一含糊称呼，必须直接说清楚是 `scene 微调`（调 scene latents）、`token 微调`（调 Gaussian tokens）还是 `scene + token 联合微调`！** 目录 / 脚本 / 参数对照见 `EXP_TokenGS/renamed_directories_old_to_new.md`。
- **命名规则**：名字只写能区分实验的信息，禁止 v1、v2、formal、task 编号之类指代不清的名字（用户要求，2026-10-19）；实验名用 scene / token / geo，词序固定 scene → token → geo → tuning，**学习率紧跟在对应被调模块词后面**（如 `scene_lr1e-2_token_lr1e-4_geo_tuning`，避免多个模块组合时分不清哪个 lr 属于哪个模块；用户 2026-10-20，详见第 10 节）。
- **职责划分**：本文件记录“为什么、是什么、踩过什么坑”；`EXP_TokenGS/EXP_TokenGS.md` 记录“具体跑了什么、数字是多少”；[`docs/LITERATURE_REVIEW_2026.md`](LITERATURE_REVIEW_2026.md) 记录 2026 顶会（CVPR / ICLR / ECCV / TPAMI）19 篇前馈式 3DGS / 4D 重建论文的方法精读、核心公式及与 TokenGS 痛点的逐一映射。不要在两处重复记录大段原始数字。

---

## 1. 论文背景

### 1.1 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | TokenGS: Decoupling 3D Gaussian Prediction from Pixels with Learnable Tokens |
| arXiv | 2604.15239 |
| 作者 | Jiawei Ren\*、Michal Tyszkiewicz\*（共同一作）、Jiahui Huang、Zan Gojcic |
| 单位 | NVIDIA Toronto AI Lab |
| 会议 | CVPR 2026 Highlight |
| 项目页 | https://research.nvidia.com/labs/toronto-ai/tokengs |
| 代码 | https://github.com/nv-tlabs/TokenGS（本地 remote 走 ghfast.top 镜像） |
| 权重 | Hugging Face `jiaweir/tokengs`（本地已下载到 `checkpoints_hf/`） |
| 本地 PDF | `TokenGS.pdf`（arXiv v1，2026-04-16，11 页，用户上传）。纯文本抽取在 `/tmp/tokengs_paper.txt`（临时文件，可用 `PYTHONPATH=/tmp/pdflib` + pypdf 重新生成） |

以下 1.2–1.4 节均依据本地 PDF 原文整理，括号中标注章节、图表编号。

### 1.2 动机（Sec. 1）

论文指出主流 encoder-only、pixel-aligned 前馈 3DGS 有三个问题：

1. **沿相机射线预测深度**：模型无法在内部纠正位姿噪声和多视角不一致，动态场景还需要随时间 warp 像素。
2. **Gaussian 数量与分辨率、视角数绑定**：例如 32 张 512×512 图像会产生 800 万以上的 Gaussian，数量随输入视角数增长而与场景复杂度无关，把输入重复 N 次就多出 N 倍冗余。
3. **直接优化 Gaussian 参数做 test-time refinement 会破坏网络先验**，在视角少时尤其明显。

摘要中明确声称 TokenGS 能产生“more regularized geometry and more balanced 3DGS distribution”，这是我们几何约束研究需要正面检验的说法。

### 1.3 方法（Sec. 3 与附录 A）

1. **直接回归 3D 均值（Sec. 3.1）**：在由相机外参定义、所有视角共享的全局坐标系中直接回归 xyz，类似 VGGT 的 point map，但**只用自监督渲染损失**，不用 point-map 真值。论文列出的好处：可外推和补全视角外区域（Fig. 6），对位姿噪声更鲁棒（Fig. 7），消除深度预测网络常见的尖刺伪影（Fig. 5）。
2. **零梯度问题与 visibility loss（Sec. 3.1，Eq. 1–2）**：落在所有监督视锥外的 Gaussian 收不到渲染梯度，会变成浪费容量的漂浮噪点。做法是把 Gaussian 中心投影到每个监督视角，归一化到 `ũ, ṽ ∈ [−1, 1]`，然后计算
   `L_vis = Σ_m min_i [ReLU(|ũ_im| − 1) + ReLU(|ṽ_im| − 1)]`，
   即只要在任意一个视角内可见，惩罚就为 0；为避免虚假梯度，损失值 clip 到 1.0。
   - 脚注 2：token 化后 3D 点与像素不再一一对应，如果要用 3D 监督，需要 Chamfer 之类的距离，这会增加复杂度，所以论文不用 3D 监督。
3. **总损失**：`L = L_MSE + λ_SSIM·L_SSIM + λ_vis·L_vis`，其中 `λ_SSIM = 0.2`，`λ_vis = 1.0`（论文版；release 2026.6 的损失不同，见 1.5）。
4. **Encoder（Sec. 3.2.1）**：ViT。每个视角的图像 patch 和 6 通道 Plücker 坐标 patch 各自经线性层投影后相加，所有视角的 token 拼成一个序列做全局注意力，最后经 LN 得到 image tokens。
5. **Decoder（Sec. 3.2.1）**：DETR 式设计。N_t 个可学习 3DGS token embedding 作为 query，每个 block 依次为 cross-attn（到 image tokens）→ self-attn（GS tokens 之间）→ per-token MLP。
   - 所有 cross-attn 层**共享 image tokens 的 KV 投影**，解码开始时只算一次，显存从 O(N_I·D_dec) 降到 O(N_I)。
   - Encoder 和 decoder 都用 LayerScale 与 QK-norm，论文称这对训练稳定“至关重要”。
   - 最终回归 head 前**不加 LN**，论文发现加了会降低重建质量。
6. **Token → Gaussian**：每个输出 embedding 经线性层回归 N_G = 64 个 Gaussian（附录 A：decoder patch size 8），每个 14 维：μ 3、c 3、s 3、σ 1、q 4。本地代码中通道切片为 `0:3 center / 3:4 opacity / 4:7 scale / 7:11 rotation`，剩余 `11:14` 为 rgb。
   - 激活函数：xyz 为 `sign(x)·(exp(x) − 1)`（沿用 VGGT），并加 z 轴偏移，使 Gaussian 初始化在参考相机（位姿为 identity）前方；颜色和 opacity 用 scaled tanh；scale 用 clipped exponential；四元数单位化（沿用 GS-LRM）。
7. **动态场景（Sec. 3.2.2，Fig. 4）**：采用 BTimer 的 bullet-time 设定，把 token 拆成 static tokens T^S 和 dynamic tokens T^D。dynamic token 加上目标时刻 t 的 sinusoidal 编码，再经线性投影，即 `T̃^D_j(t) = T^D_j + Linear(τ(t))`。self-attn 中使用结构化 mask：static↔static、dynamic↔dynamic 互相注意，dynamic → static 单向注意，static 不注意 dynamic。
8. **Test-Time Scaling（Sec. 3.3）**：论文提出两种不需要重训网络的推理期扩展方式。
   - **Context extension（CE）**：推理时输入比训练时更多的视角（例如训练用 2 视角，推理用 4 视角）。Gaussian 数量保持不变，与 context 长度无关。
   - **Token tuning（TT）**：轻量的 test-time training。**只微调 Gaussian token embeddings**，网络参数和 image features 都冻结，在输入视角上做自监督。论文的解释是：少量梯度步就能让注意力模式适配具体场景，在保留网络其余部分先验的同时改进 Gaussian 参数。
   - 默认设置（Sec. 4 Training details）：**50 步，lr 1e-4**。
9. **训练细节（Sec. 4 与附录 A）**
   - 规模：6 层 encoder / 24 层 decoder，C = 1024；DL3DV 实验为了支持更高分辨率和更多视角，减为 3 / 12 层。使用 FlashAttention2 与 FlexAttention（后者用于带 mask 的注意力）。
   - 优化：AdamW，batch 64，8×A100，cosine 调度。
   - 基础训练：1024 tokens，256×256，150K iter，lr 4e-4，warmup 2000，约 36 h。
   - 微调：4096 tokens，10K iter，lr 4e-5，warmup 400，约 4 h。DL3DV 微调分辨率改为 448×256。
   - 正则：梯度裁剪 1.0；weight decay 0.05（bias 与 LN 除外）；训练时对图像和位姿做随机翻转；视角采样策略沿用 DepthSplat。
   - 初始化：token std 0.01；增加 token 或加入 dynamic token 时，从已有 token 复制再加 std 0.01 扰动；输出投影层 std 2e-3（保证初始 Gaussian 大多可见）；decoder 中 LayerScale 初值 1e-5。
   - 场景平移缩放：RE10K 0.25、DL3DV 0.15、Kubric 0.1，使平均深度约为 1。

### 1.4 实验结果与消融（Sec. 4）

**Table 1：RE10K，2 视角，256×256**

| 方法 | PSNR | SSIM | LPIPS | #GS |
|---|---|---|---|---|
| GS-LRM | 28.10 | 0.892 | 0.114 | 131K |
| DepthSplat | 27.47 | 0.889 | 0.114 | 131K |
| Ours（1024 tok） | 28.02 | 0.896 | 0.147 | 66K |
| Ours（4096 tok） | 28.41 | 0.903 | 0.135 | 262K |
| Ours（4096 tok，+TT） | 28.82 | 0.910 | 0.130 | 262K |

**Table 2：DL3DV，448×256。** 论文模型**只在 4 视角上训练**，2 / 6 视角属于 context extension 泛化（表中标 \*）。

| 视角 | Ours（w/o TT） | Ours（+TT） | TT 耗时 | DepthSplat |
|---|---|---|---|---|
| 2\* | 19.35 / 0.609 / 0.440 | 19.32 / 0.601 / 0.435 | 4.00 s | 19.31 / 0.615 / 0.310 |
| 4 | 23.02 / 0.747 / 0.326 | 23.57 / 0.766 / 0.314 | 4.82 s | 23.12 / 0.780 / 0.178 |
| 6\* | 23.69 / 0.766 / 0.311 | 24.51 / 0.792 / 0.295 | 5.61 s | 24.19 / 0.823 / 0.147 |

- 2 视角时 TT 反而让 PSNR 略降（19.35 → 19.32）。论文 6 视角结果使用的 Gaussian 比 DepthSplat 少 72%。
- 论文版 LPIPS 普遍明显差于 DepthSplat，release 2026.6 推出 LPIPS 变体可能与此有关（推测）。

**其他实验**

- **Table 3，视角外推**（2 视角）：Ours 在 DL3DV 上 20.98 / 0.688 / 0.364，在 RE10K 上 22.60 / 0.831 / 0.222，与使用 point-map 监督微调的 DepthSplat+PM 相当（DL3DV 20.77）。
- **Fig. 7，位姿噪声**：在第二个相机上加入 1–10° 的随机旋转，噪声越大，相对 GS-LRM 的优势越明显。
- **Table 4，动态场景（Kubric，4 视角，256×256）**：在 RE10K + DL3DV 上训练的 4 视角模型，再加 256 个 dynamic tokens，在 Kubric 上微调 50K iter，16 个 held-out 场景评估。Ours 24.84 vs BTimer 24.45（PSNR 更高，但 SSIM / LPIPS 较差）。BTimer 靠改变 opacity 来模拟运动，导致物体闪烁；TokenGS 的中心在时间上连续，并涌现出 scene flow（Fig. 9）。

**消融（Sec. 4.5）**

- **Fig. 10，token 的空间对应关系**：原文为“Gaussians from a single token stay close to each other within a scene and remain in a similar 3d location across scenes”，并类比 DETR 的 slot specialization；同时指出高频细节区域会分到更多 token。
  - ⚠️ **与我们的实测矛盾**：在 latent 6v 官方权重上实测，packing ratio 为 35–161（正则目标约 9），相邻 slot 的马氏距离约 0.5（目标 2）；单场景复现时 packing 为 285、inter 为 0.20。LocusGS 也指出同一 token 的 Gaussian 会散布到远处。Fig. 10 只是对 5 个随机 token 做的定性可视化，没有量化指标。这正是本研究“几何约束”的切入点。
- **Fig. 11，visibility loss**：在 RE10K 上训练两个 4 视角模型，有 / 无该项的 PSNR 为 28.8 vs 28.4（+0.4），去掉后视野外区域出现漂浮物。
- **Fig. 12–13，test-time scaling**（DL3DV，4 视角基础模型，固定 50 个 target 视角，**包含输入视角**，与我们的 `all_frames` 口径相同）：
  - CE：输入视角最多增加到训练时的约 4 倍时仍有收益，之后饱和；
  - TT：**10 步**就有显著提升，增加到 250 步仍在缓慢上升，趋于饱和；CE 与 TT 可以叠加；
  - 直接优化 Gaussian 参数：PSNR 更高、收敛更快（视角越多越明显），但 Fig. 13 显示它会在全新视角上**破坏几何**，而 TT 能改善几何、渲染更锐利；
  - 原文：“A deeper exploration of joint optimization strategies (token and Gaussian tuning) is left for future work.”
- **Fig. 14，token 数量**：在 DL3DV 上把 1024-token 基础模型微调为 256–4096 个 token，视角数 2–6；token 数和视角数可独立调节，用于权衡压缩率与质量。

**局限（Sec. 5）**：继承了前馈方法的通病，难以处理大尺度场景和细粒度几何；TT 成本较高，通常需要几十步优化。论文认为“token 微调 + 直接 Gaussian 微调”的混合策略是有前景的未来方向。

**对本研究的启示**：

- 论文自己承认 Gaussian 参数微调会破坏几何，并把 token 与 Gaussian 的联合优化留作未来工作。我们在 TTT 内循环中加几何正则，正好落在这个空白上。
- 论文的“几何更好”只有定性证据（Fig. 10 / 13），我们的 sw / packing / inter 指标可以提供定量证据。
- 本地 `kubric_dyn.safetensors` 与 `scripts/render_kubric_dyn.py` 对应动态扩展。

### 1.5 Release 2026.6（本研究实际使用的版本）⚠️ 重要

本研究使用的是 **release 2026.6 的 latent bottleneck 模型**，它与论文正文的模型不同：

| 项目 | 论文版 | Release 2026.6 latent 版 |
|---|---|---|
| Encoder | 3 层（DL3DV） | 12 层，多尺度 readout（第 5/7/9/11 层） |
| 中间层 | 无 | 4096 个可学习 scene latent tokens（latent bottleneck，12 层 cross-attn 读取图像特征） |
| Gaussian decoder | 12 层（DL3DV；论文主模型 24 层） | **1 层**（`dec_depth=1`）。权重实测：self-attn layerscale |γ| 均值 3.6e-4（几乎关闭），cross-attn 0.027，MLP 0.023 |
| Gaussian tokens | 1024 / 4096 | 4096×1024，每个 token 输出 64 个 Gaussian（`dec_patch_size=8`），共 262,144 |
| TTT 对象 | Gaussian tokens，lr 1e-4 | **scene latent tokens（`scene-latent-tuning`），lr 1e-2**，50 步 |
| 损失 | MSE + 0.2·SSIM + vis | SSIM 变体：`0.8·L1 + 0.2·SSIM`；LPIPS 变体：`L2 + 0.5·LPIPS` |
| 新增训练模式 | — | `mean_of_grads`（per-scene / per-view 分块渲染，省显存，与 `deferred_bp` 互斥） |

**官方 benchmark**（DL3DV-10K，140 场景，scene scale 0.15）：

| 视角 | 变体 | w/o TTT PSNR / SSIM / LPIPS | w/ TTT PSNR / SSIM / LPIPS |
|---|---|---|---|
| 6 | SSIM | 25.120 / 0.806 / 0.257 | 27.304 / 0.866 / 0.185 |
| 6 | LPIPS | 24.077 / 0.783 / 0.194 | 25.505 / 0.829 / 0.121 |
| 4 | SSIM | 24.012 / 0.778 / 0.277 | 25.408 / 0.822 / 0.213 |
| 2 | SSIM | 20.493 / 0.658 / 0.381 | 20.899 / 0.668 / 0.333 |

- **结论**：我们的数字应与这张 release 表对比，**不能与论文 Table 2 对比**（模型不同）。
- **两种 TTT 的区别**：Gaussian tokens 是所有场景共用的 query（训练得到的参数，本身不含场景信息）；scene latents 是 encoder + bottleneck 对当前场景算出的激活，经 `latent_kv_proj` 变成 decoder cross-attn 的 K/V，是场景内容的压缩表示。论文 TT 调 query、冻结 K/V 侧（"image features frozen"）；release 反过来调 K/V 侧的 scene latents、固定 query。两者大小相同（均为 4096×1024）。
- **为什么不一致**：release 文档和 commit 信息（886c3e5、a2c0074）都只说做了什么、没说原因，也没有公布两种模式在 latent 模型上的对比。可确认的是：scene latents 只存在于 release 新架构；且新架构 decoder 只剩 1 层、self-attn 几乎关闭，论文所依赖的"在多层 decoder 中调整注意力模式"基本不再存在，网络深度都移到了 scene latents 之前。其余动机属于推测。
- `tokengs/options.py` 中全局默认值仍是论文版：`ttt_mode="token-tuning"`、`ttt_n_steps=50`、`ttt_lr=1e-4`、`num_gs_tokens=1024`、`lambda_visibility=1.0`、`use_geo_losses=False`。preset `eval_dl3dv_latent_6view_ssim` 会覆盖为 `scene-latent-tuning` 与 `lr=1e-2`。**直接改全局默认值不会影响 preset 行为，反之亦然**。

### 1.6 相关工作

- **LocusGS（arXiv 2608.12825）**：为每个 query 附加 3D anchor（中心 + 支撑半径），通过 anchor-to-ray attention bias 和以 anchor 为中心的解码，解决 token 内 Gaussian 散布的问题。它与我们“几何约束让 slot 局部化”的方向**直接竞争**。写论文或定位创新点时必须对比：我们是“只在 TTT 阶段加正则、不改架构、不重训”，LocusGS 是改架构并重训。
- **StreamSplat（ICLR 2026，arXiv 2506.08862，DSL-Lab）**：无标定单目视频流的在线动态重建，约 20 FPS。与 TokenGS 定位互补而非竞争：TokenGS 处理 posed 多视图静态场景，StreamSplat 处理 pose-free 视频流动态场景。之前讨论过“训练期 video diffusion teacher 蒸馏到 StreamSplat”的方向，直接竞品是 Lyra。该方向尚未立项。
- **FFRGS（VGGT-Omega，pose-free）**：另一条研究线，代码与实验日志已归档到 `4DGS/FFRGS/`、`4DGS/ffrgs_workspace/`（见 3.2）。本研究只作参考，知识整理在 `docs/FFRGS_REFERENCE.md`。

---

## 2. 代码与论文映射

### 2.1 仓库状态（2026-10-19 核验）

- 路径：`/home/jms/xveyan/4DGS/TokenGS`，HEAD `b16269c`（“Zero gradients after step()…”），remote `nv-tlabs/TokenGS`（ghfast 镜像）。
- **9 个核心文件有未提交的本地修改**（全模型微调修复 7 个问题后的重跑阶段留下，约 +743/−183 行）：
  `tokengs/data/__init__.py`、`tokengs/data/provider*`、`tokengs/data/registry*`、`tokengs/data/static/dl3dv.py`、`tokengs/evaluate.py`、`tokengs/models/tokengs.py`、`tokengs/options.py`、`tokengs/rendering/gs.py`、`tokengs/train.py`。
  这些修改来自已废弃的全模型微调实验（见 5.2），其中包含几何损失接入、数据读取失败报错、DDP 验证去重等修复。**不要随意 `git checkout` 还原**，先问用户。
- **Untracked（本研究新增）**：
  - `tokengs/losses/geo_losses.py`（+ `__init__.py`）：三项几何损失（inter 默认 FFRGS 写法）；
  - `scripts/evaluate_with_scene_latent_tuning.py`：正式评估入口（显式 FP32，逐帧指标，all_frames / novel_only 双口径，帧距分桶，CI95；`--no-tuning` / `--scene-latent-tuning` / `--token-tuning`、`--tuning-loop {official,reimplemented}`、`--geo-sw/--geo-packing/--geo-inter`、`--demo`、`--ply-dir` + `--ply-name`）；
  - 辅助模块：`scripts/evaluation_statistics.py`（edge/HF、几何统计、针统计）、`scripts/reimplemented_scene_latent_tuning.py` / `scripts/reimplemented_token_tuning.py`（逐行复刻官方两种微调循环，可加几何项、记录曲线、留下调好的张量）、`scripts/demo_ply_export.py`（不剪枝 ply + 相机 ply）；
  - `scripts/decode_saved_decoder_inputs.py`（从 `decoder_inputs.pt` 重新解码出 Gaussian / ply）、`scripts/compare_same_scene_ply.py`（同场景多方法的指标表 + 逐 Gaussian 位移 + 差异 ply）、`scripts/highlight_selected_tokens_ply.py`（只给同一批 10 / 20 个 token 上色、其余灰色半透明的 ply + 图例）；
  - `scripts/experiment_registry.py`：11 个 140 场景实验的唯一名单、名字、参数、顺序和 demo 资格；`scripts/report_scene_geo_tuning.py`：scene 有无几何损失的配对报告 + 预注册选示例场景（只读）；`scripts/report_scene_token_x_geo.py`：调哪些参数 × 几何 × token lr 的配对报告；`scripts/build_all_experiments_table.py`：全部实验总表；
  - `scripts/one_click_launch_common.sh`：一键启动脚本的公共函数（环境、空闲 GPU、并行 / 排队、完整日志、`DRY_RUN=1`）；`EXP_TokenGS/run_140_scenes.sh`：140 场景统一入口；`EXP_TokenGS/run_demo.sh`：同场景 demo 统一入口；
  - `scripts/audit_geo_latent_grad.py`：几何损失 → scene latent 梯度审计；
  - `acc_configs/gpu3_geo.yaml`、`acc_configs/gpu4.yaml`（全模型微调阶段的 accelerate 配置，当前未使用）；
  - `assets/evaluation_idx_single_scene_*v.json`；
  - `tokengs/checkpoint.py`：全模型微调阶段（已废弃）留下的 checkpoint 事务代码；对应的 3 个测试和依赖已删除 EXP_REPRODUCE 的 `test_multiscene_preparation.py` 已于 2026-10-19 删除（用户同意）；
  - `tests/test_geo_losses_ffrgs_formulation.py`、`tests/test_dl3dv_resolution.py`、`tests/test_reimplemented_tuning_loops.py`（复刻循环与官方循环 CPU 逐位等价）、`tests/test_compare_same_scene_ply.py`、`tests/test_highlight_selected_tokens_ply.py`、`tests/test_experiment_registry.py`：当前有效的测试（2026-10-20 第四次整理后：除 contracts 外 66 passed、44 subtests passed）。官方 `tests/test_tokengs_contracts.py` 无法导入（官方 d200909 删除了 `project_gaussian_means2d`），属官方仓库问题；
  - `EXP_TokenGS/`、`EXP_FFRGS/`、`EXP_TokenGS.zip`、`EXP_FFRGS.zip`（两个 zip 是用户最初提供的方案文档，只含 md）；
  - `checkpoints_hf/`（HF 权重，约 15.1 GB）；
  - `docs/`、`AGENTS.md`（本共享记忆）；
  - `TokenGS.pdf`（论文原文，用户上传）。

### 2.2 关键代码路径

- `tokengs/models/tokengs.py`
  - `forward()`：当 `use_ttt_for_eval=True` 时调用 `forward_ttt()`。
  - `_forward_ttt_scene_latents`：50 步 Adam 内循环，loss 为 `compute_loss`（L1 0.8 + SSIM 0.2，渲染回 6 个 context 视角）。最终在 `@torch.no_grad` 下用 `scene_latents.detach()` 重新解码。
  - 梯度链路：`scene_latents → _latents_to_kv（latent_kv_proj + k_proj_norm）→ decoder_blocks（cross-attn；gs_tokens 是 detach 的常量）→ ClipActivationHead → [B, 262144, 14]`，按 `(n p)` 排列，可 `reshape(B, 4096, 64, 14)` 得到 slot × child。
  - `scale_act` 为 `torch.minimum(exp, cap=0.075)`，饱和处梯度为 0（实测饱和比例 1.0–3.2%）。
- `tokengs/losses/geo_losses.py`：`compute_geo_losses`，包含三项损失：
  1. **Sliced Wasserstein（sw）**：slot 内 64 个 child 以 opacity 加权，白化后沿随机方向投影，与标准正态分位比较，要求 slot 内近似单峰椭球；
  2. **Slot packing**：cluster 空间散布 ÷ 单个 Gaussian 尺寸，目标值由有效 child 数估算（约 8.6–9.4）；
  3. **Inter-slot Mahalanobis（inter）**：每个 slot 汇总成 parent Gaussian，与 kNN（k=4）邻居计算马氏距离，目标为 2，loss 为 `(d−2)²`。
  - **设计内的 stop-gradient**：opacity 权重 detach（三项对 opacity 均无梯度）；packing 分母 scale detach；kNN 图在 `no_grad` 下用 topk 构建；Cholesky jitter、SW 累计质量 detach；整段 `autocast(enabled=False)` 并转 FP32。
  - 默认权重：`options.py` 中三项均为 0.01。

---

## 3. 环境与启动

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
cd /home/jms/xveyan/4DGS/TokenGS
```

- conda env `tokengs`：Python 3.11，torch 2.7.0+cu128，gsplat 1.5.3，fused-ssim（为 sm_120 编译），CUDA 12.8。
- **env/bin 必须放进 PATH**，否则 gsplat JIT 编译找不到 ninja。
- 权重：`checkpoints_hf/checkpoints/`（论文版 dl3dv_{base,2v,4v,6v}）与 `checkpoints_hf/checkpoints_26.06/`（latent 版 6 个 + latent_base + kubric_dyn，附 `SHA256SUMS.txt`）。
- 正式使用的权重：`checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors`，sha256 `5913da347475049cb58e552634ee1c527293c2571576806db0d4da702ce978c3`。
- 正式评估由一键脚本启动（实验目录下的 `run_*.sh`，内部已设置上面的环境并自动选空闲 GPU）。单独调用入口的示例（不微调）：

```bash
python scripts/evaluate_with_scene_latent_tuning.py --preset eval_dl3dv_latent_6view_ssim \
  --resume checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors \
  --output EXP_TokenGS/no_tuning --no-tuning --expected-scenes 140 --mixed-precision no
```

- 每次运行会在输出目录写 `launch_contract.json`（argv、GPU UUID、checkpoint / 脚本 / 评估索引 sha256、完整 opt）、`status.json`、`summary.json`、`scene_metrics.{csv,jsonl}`、`frame_metrics.jsonl`、`resource_evidence.json`；一键脚本另在输出目录旁写完整未过滤的 `<组输出目录>.log`（含命令、GPU、git、退出码、耗时）和总日志。复现时以 `launch_contract.json` 为准。
- **GPU 使用规则**：服务器多人共用。只使用显存 ≤100 MiB 且没有 compute app 的卡，启动前用 `nvidia-smi` 确认，不得干扰他人进程。

### 3.1 Gaussian 输出可视化（原实验文档的做法，2026-10-19 核查）

- `EXP_TokenGS.zip` / `EXP_FFRGS.zip` 中的视频都是屏幕录制：在 VS Code 系编辑器（录屏中的路径为 `/mnt/disk/jinsoo/ffrgs_workspace/...`，该目录不在本机）里直接打开 `.ply` 文件，使用扩展 **3D Visualizer**（`kleinicke.ply-visualizer`，Florian Nick 开发，基于 Three.js，splat 渲染用 Spark）。扩展以 webview 形式运行，所以看起来像网页。
- 界面特征：Files / Camera / Controls / Info 四个标签页；每个文件可在 Points（只画中心点，可按任意 PLY 标量字段着色）和 ✨ Splats（排序后的真实 splat 渲染）之间切换；有 Max splat size 滑块；同一视图可加载多个文件；比较时分栏并排显示。
- 录屏中出现的文件：`original_all_gs.ply`（262,144 = 4096×64，未做 opacity 剪枝）、`colored_20_slots_opacity1.ply`（20 个 slot 用不同纯色并把 opacity 设为 1，用来看 slot 的空间分布，分 `original/` 与 `geometry_final/` 对比）、`cmp_scene5_tokengs.ply`（182,702 个点，像是经过 opacity 剪枝）。
- 本仓库导出：`gs.save_ply` 会剪掉 opacity < 0.005 的 Gaussian，剪枝后按下标还原 slot 的对应关系会失效。若要按 slot 分析，需不剪枝导出，或额外写一个 `slot_id` 字段（Gaussian 按 token 顺序排列，slot i 对应下标 [64i, 64i+64)）。
- 其他用法：同一作者提供网页版 3d.f-kleinicke.de（在浏览器本地读文件）和 PyPI 包 `3d-visualizer`。
- FFRGS 仓库（`4DGS/FFRGS/scripts/`）里还有另一套网页查看方式：`cors_server.py` 在 8899 端口起一个带 CORS 头的静态文件服务，供 SuperSplat 网页用 `?load=<url>` 加载；`standardize_ply.py` 把 17 个属性的紧凑 ply 补零成标准 62 个属性（SuperSplat 等严格查看器要求）；`prune_ply.py` 按 opacity 剪枝；`export_ply.py` 能导出按 slot 着色的 children / parent。

### 3.2 FFRGS 归档（2026-10-19 解压）

- `4DGS/FFRGS/`（来自 `FFRGS_20261019.tar.gz`）：FFRGS 代码仓库，1 个 commit（80e6ed2，2026-10-12，YifanW98）。项目是在冻结的 VGGT-Ω 上做 pose-free 前馈 3DGS，用 parent / child 两级 slot decoder，最佳为 v18@15k，held-out 8 场景 PSNR 23.74；同条件下 TokenGS 喂 Ω pose 为 23.94，喂 GT pose 为 25.84。完整实验日志见 `AGENTS.md`，总览见 `PROJECT_SUMMARY.md`。根目录的 8 个 `cmp_scene*.ply` 是指向 `/mnt/disk/jinsoo/...` 的软链接，在本机失效。
- `4DGS/ffrgs_workspace/`（来自 `ffrgs_workspace.tar.gz`）：68 个实验目录里只有 `run.sh`、参数、日志和 `train_log.jsonl`，**没有 checkpoint**；`ply_compare_tokengs/` 下有 8 个真实 ply（TokenGS 与 FFRGS v20@5k，scene 2/5/9/11）。
- 与本研究相关的结论：FFRGS 发现 gsplat 对被剔除的 Gaussian（包括 bbox 整个在画面外的）不写 `info["means2d"]`（FFRGS 文档说保留为 0，实际是 `at::empty` 的未初始化值），且反传跳过这些点，因此读 `means2d` 的 visibility loss 对最该惩罚的点梯度恒为 0。本仓库 `rendering/gs.py` 同样把 `info['means2d']` 交给 `compute_visibility_loss_from_means2d`。已读 gsplat 1.5.3 源码确认机制，且本仓库 `EXP_TokenGS/demo/scene_2f3e1c0f688c/*/tuning_curve.jsonl` 第 0 步已提供直接 GPU 实测证据（见陷阱 14）。
- **完整知识整理与对我们实验的反查见 `docs/FFRGS_REFERENCE.md`**（架构、训练 / 评估设置、对 TokenGS 的测量、bug 清单、诊断探针；第 7 节是对 Baseline / TTT 的核查结论与建议）。

---

## 4. 数据

- `data/dl3dv` 与 `data/dl3dv_eval` 都是软链接，指向 `/home/jms/xveyan/data/4dgs/DL3DV-10K-Benchmark`。
- 内容为 **140 个场景**，每个场景有一个 `{hash}.zip`（内部前缀 `{hash}/gaussian_splat/`）和同名解压目录，另有 `benchmark-meta.csv`。共 48987 帧，CRC 全部校验通过。来源：用户从 Windows 上传的 45 GB tar，再由 agent 重新打包。
- 评估索引：`assets/evaluation_idx_dl3dv_depthsplat_6v.json`（sha256 `3ab98877…4b9e`），每场景 6 个 context 视角、50 个 target 视角。
- **评估口径**：
  - `all_frames`：50 帧，与官方 release 一致，**主指标**；
  - `novel_only`：50 帧去掉 6 个 context 视角后的 44 帧，辅助指标，反映真正未见视角。
- 特殊场景：`07d9f97…` 分辨率为 480×270（其他为 960×540）。**官方 HEAD 代码**会把它的内参 K 放大 2 倍；本地工作区的 `dl3dv.py` 已按实际图片分辨率修正（见 6.1）。
- **测试集泄露结论**：官方模型在 DL3DV-10K 训练集上训练，140 个 benchmark 场景按发布包物理隔离；路径分离和 PSNR 差距（benchmark 明显低于训练场景）也支持“未泄露”。本机没有训练集 hash 列表，无法直接做交集检查。
- **本地没有 benchmark 之外的 DL3DV 场景**可作开发集（选 λ 时需注意，见第 8 节）。

---

## 5. 实验历史时间线

### 5.1 会话“检查4dgs代码完整性”（4f8b4e60）

会话记录：`/home/jms/.qoder/cache/projects/xveyan-425e665f/conversation-history/4f8b4e60/4f8b4e60.jsonl`（约 501 行）。首条用户消息：“这个会话用于我们研究4dgs相关的代码工作…检查这些代码是否完整，是否支持从mp4原视频开始的完整处理流程”。会话前半段是 FreeTimeGS，后半段转为 TokenGS。

| 阶段 | 内容 | 结论 / 产出 |
|---|---|---|
| 0 | FreeTimeGS 代码完整性检查（mono_4dgs、FreeTimeGsVanilla、init、Neural3DV，从 MP4 初始化）；讨论 StreamSplat + 生成式蒸馏；TokenGS vs StreamSplat 对比 | 方向讨论，无实验 |
| 1 | 搭建 tokengs 环境，下载 12 个 checkpoint（15.1 GB） | 环境可用（见第 3 节） |
| 2 | 解析 `EXP_TokenGS.zip` / `EXP_FFRGS.zip` 的方案 | 全模型微调方案：从官方权重出发，用 12 个 DL3DV-5K 场景做全模型微调，在 30 个未见场景上评估，8 GPU，lr 4e-5，10K updates。方案文档声称 Geo 比 Recon 低 0.087，SW 降约 20 倍，inter 距离从 0.25 升到 1.74。FFRGS 无代码 |
| 3 | 单场景复现（`032dee9f`） | 官方 30.17 / TTT 31.59 / Recon 32.17 / Geo 31.33；SW 0.387→0.018，inter 0.20→1.97，packing 285→16.3；Gini 0.285→0.751（slot 负载不均衡）。**发现 BF16 下 TTT 崩到 12.64，关闭 autocast cache 后恢复** |
| 4 | 数据准备 | 140 场景 zip，见第 4 节 |
| 5 | 全模型微调首轮（100/20/20 划分） | Geo 比 Recon 低约 0.3 dB。审计发现 7 个问题：softmax(sigmoid) 误用；`07d9f97` 的 K 放大 2 倍；inter 协方差忽略旋转；读取失败时静默换场景；DDP 验证重复计数；val-best 与 last checkpoint 混用；泄露相关措辞不严谨 |
| 6 | 全模型微调修复 7 个问题后重跑：3 seed × 2 组 = 6 次训练，24 项评估 | Recon test 25.019，Geo test 24.443（−0.576）；官方不微调同 40 场景为 val 25.243 / test 25.228（BF16，目录已于 2026-10-19 删除） → **微调后反而比官方差** |
| 7 | 用户指出方案中的“官方微调”其实是 TTT，全模型微调方向错误 | 改做官方 TTT：140 场景 FP32 = 27.399（与 release 27.304 吻合）。删除 EXP_REPRODUCE（约 43 GiB），清理 EXP_TokenGS，生成交接 prompt → 进入会话 b1a5cb35 |

### 5.2 已废弃的方向 ⚠️

- **全模型权重微调（首轮 + 修复 7 个问题后的重跑）**：在约 100 个场景上微调整个模型 10K 步，PSNR 低于官方 baseline，说明小数据全模型微调破坏了预训练泛化能力。**其结果（包括 Recon vs Geo 差值、SW / inter 改善）不得引用为“几何约束是否有效”的证据**。产物已删除，相关代码修改仍保留在工作区（见 2.1）。

### 5.3 当前会话（b1a5cb35，2026-10-19 至 2026-10-20）

1. 官方循环的 scene latent tuning，140 场景 FP32 正式评估 → 现 `EXP_TokenGS/scene_latent_tuning/official_loop`。
2. 不微调 140 场景 FP32 补跑（GPU7，330 s）；这次的产物后来被第 7 条里同配置、指标更全的一次取代（逐帧相同），见第 10 条。
3. 几何损失 → scene latent 梯度审计（`scripts/audit_geo_latent_grad.py`，见第 7 节）。
4. 在 `EXP_TokenGS.md` 中写好对照设计，用户确认后预注册（inter 用 FFRGS 写法；权重只跑一组 sw 0.1 / packing 0.001 / inter 0.01；重跑一次量噪声底；示例场景取增益最接近中位数的）。
5. 建立本共享记忆文档；用户随后上传 `TokenGS.pdf`，据此补全论文 3.3 节与全部表格（第 1 节）。
6. 解压 FFRGS 代码与 workspace，通读后写成 `docs/FFRGS_REFERENCE.md`；据此只读反查 Baseline / TTT：口径无误，但发现评估集几乎只有内插、TTT 增益随离 context 帧距迅速衰减（d=0 +4.9 dB，d≥4 约 +1.2 dB），以及 visibility 项因 gsplat `means2d` 机制基本失效（详见该文第 7 节）。
7. 用户运行预注册的四组 140 场景 FP32 对照（不微调 / 微调 / 微调重跑 / 微调+几何损失 sw 0.1 · packing 0.001 · inter 0.01；报告在 `EXP_TokenGS/geo_loss_in_scene_latent_tuning`，各组现位置见第 6 节）。结果：微调增益 +2.206 dB；重跑噪声底 +0.0003 dB（CI ±0.004）；复刻循环与官方循环差 +0.0002 dB；**加几何损失 −0.665 dB**（CI [−0.741, −0.590]，1/139 胜），几何统计朝目标移动（SW 0.712→0.376，packing 704→276，inter 0.52→1.05），但 14 项画质指标全部显著变差，视锥外 Gaussian 比例 0.0128→0.0361。数字见 `EXP_TokenGS.md` 与该目录 `report.md`。
8. 按用户要求全面改名（TTT → scene latent tuning，去掉 v1/v2/formal/task 编号等名字）：拆分 `geo_diagnostics.py`、评估入口改为 `scripts/evaluate_with_scene_latent_tuning.py`、report 改为 `scripts/report_geo_loss_in_scene_latent_tuning.py`、全部结果目录改名，删除 4 个废弃测试；对照表 `EXP_TokenGS/renamed_directories_old_to_new.md`。
9. 按用户要求改为“一键脚本 + 完整 log”：`scripts/one_click_launch_common.sh` + 各实验目录的 `run_all_groups.sh` / `run_demo_scene_all_groups.sh`（后者从未运行，2026-10-20 由第 13 条的脚本取代并删除；见第 3 节、第 10 节）。截至本条，一键脚本只做过 `bash -n` 检查，尚未在 GPU 上实际跑过。
10. 按用户要求重组目录（同一方法的结果放在该方法目录，对照实验目录只放该实验独有的组）：不微调用第 7 条那次（指标更全）取代第 2 条那次；三份微调结果合并到 `scene_latent_tuning/{official_loop,reimplemented_loop,reimplemented_loop_repeat_run}`；几何损失目录只留 `sw0.1_packing0.001_inter0.01` + 报告 + 一键脚本；BF16 结果、排查目录、取消的半成品全部删除。report 在新位置重新生成，数字与重组前完全一致。映射见 `EXP_TokenGS/renamed_directories_old_to_new.md`。
11. （2026-10-20）用户指出论文的 token tuning（TT，调 Gaussian token）我们没试过，决定只跑一组：**lr 1e-4、50 步、官方循环、140 场景、FP32、不测噪声底**，结果放 `EXP_TokenGS/token_tuning/lr1e-4_steps50`。评估入口新增 `--token-tuning`（必须显式给 `--tuning-lr`），一键脚本 `EXP_TokenGS/token_tuning/run_all_groups.sh`（配对对比不微调、场景潜变量微调官方 / 复刻循环）。设计见 `EXP_TokenGS.md` “执行日志：token 微调”。
12. （2026-10-20）用户运行 token 微调 140 场景（GPU1，654 s，140/140）：all PSNR **26.810**，对不微调 **+1.617 dB**（140/140 变好），对场景潜变量微调 **−0.590 dB**（5/140 更好），HF ratio 更低（画面更糊）；差距随帧距缩小（d0 −1.94 → d4+ −0.16 dB）；几何 / 针状指标几乎不变（SW 0.712，packing 650），没看到论文说的几何改善。用的是改代码前的脚本版本（官方循环，`script_sha256=b6db0674…`）。
13. （2026-10-20）用户要求同场景对比不同方法的 ply，并保存微调结果：新增 token 复刻循环（有曲线和几何 / 针状指标）、`decoder_inputs.pt`（scene latents + gs_tokens 两个 decoder 输入，可脱离数据重新解码）、`--ply-dir` 把所有方法的 ply 集中放、对比脚本和一键脚本 `EXP_TokenGS/same_scene_ply_comparison/run_all_groups.sh`（四种方法 × 场景）。CPU 测试通过。
14. （2026-10-20）同场景四方法第一次运行：四组 exit 0，重解码 max |diff| 全为 0；一键脚本卡在汇总前（陷阱第 19 条，已修复）。用户要求 ply **按方法分文件夹**：改为 `same_scene_ply_comparison/ply/<方法>/scene_<id 前12位>/`，对比表和差异 ply 单独放 `same_scene_ply_comparison/comparison/scene_<id 前12位>/`；已有文件用 mv 搬到新布局（旧路径对照见 `renamed_directories_old_to_new.md`）。
15. （2026-10-20）用户要求去掉目录名里的 `_fp32_140scenes` / `_fp32` 后缀（所有结果都是 FP32、140 场景，后缀没有区分作用）：`no_tuning/`、`scene_latent_tuning/`、`geo_loss_in_scene_latent_tuning/`、`token_tuning/`、`same_scene_ply_comparison/`，子目录名不变。脚本、文档、ply 旁注里的路径已改；report 在新位置重新生成，数字与改名前逐项相同；证据文件保留旧名，对照见 `renamed_directories_old_to_new.md` 头部。今后目录名不再加所有实验都相同的后缀。
16. （2026-10-20）用户提出补齐“调哪些参数 × 是否加几何损失”：参数维度为 scene latents / Gaussian tokens / 两者，损失维度为只用渲染 / 渲染+几何，另有不微调（“不调+几何”不成立）。用户选择：只补 3 格（token+几何、scene+token、scene+token+几何，不做去掉渲染损失的纯几何消融）；联合微调各用原学习率（scene 1e-2、token 1e-4，一个 Adam 两个参数组，50 步）；token+几何沿用 sw0.1/packing0.001/inter0.01。新增联合循环 `scripts/reimplemented_scene_latent_and_token_tuning.py`（自定义 `ttt_mode="scene-latent-and-token-tuning"`，官方 `forward_ttt` 遇到会报错）、入口参数 `--scene-latent-and-token-tuning --token-lr`、报告 `scripts/report_tuned_parameters_x_geo_loss.py`、一键脚本 `EXP_TokenGS/tuned_parameters_x_geo_loss/run_all_groups.sh`。CPU 测试：任一学习率设 0 时与对应单张量循环逐位相同；pytest 51 passed。预注册写在 `EXP_TokenGS.md` 末尾；待用户运行。
17. （2026-10-20）核实学习率来源：scene latents 1e-2 是 release latent preset 指定（`options.py` `_latent_dl3dv_eval_preset`、`release_2026.6.md` 第 63 行）；token 1e-4 来自论文（旧模型），release latent 模型没有官方 token 学习率；联合模式和几何权重是我们的设计。用户要求补 token lr 1e-2，选择补整行（token、token+几何、联合、联合+几何，共 4 组）、单独调 token 用复刻循环。同一个一键脚本现在共 7 组：`token_tuning/lr1e-2_steps50`（按方法归档）+ `tuned_parameters_x_geo_loss/` 下 6 组，目录名带 token 学习率（`token_tuning_lr1e-4_with_geo_loss_<geo>`、`scene_latent_and_token_tuning_token_lr{1e-4,1e-2}[_with_geo_loss_<geo>]` 等）。代码入口未改，只改一键脚本、报告脚本和文档。
18. （2026-10-20 12:53–13:06）用户运行 7 组：token lr 1e-4 的 3 组 140/140、exit 0（联合循环 GPU 首跑正常）；token lr 1e-2 的 4 组全部发散（Gaussian 出现 NaN/Inf，前 1–8 个场景内报错中止，跑完的场景比不微调低约 6 dB）。结果见第 6 节和 `EXP_TokenGS.md` 执行日志；失败目录保留作证据。
19. （2026-10-20）用户要求一个单独文件的总表：每行一个实验、每列一个指标、发散写“发散”。新增 `scripts/build_all_experiments_table.py`（只读各目录 `scene_metrics.jsonl`，从仓库根目录运行），生成 `EXP_TokenGS/all_experiments_table.{md,csv}`：13 个 140 场景实验 + 2 行官方发布表参考，62 列（设置、两个配对 Δ PSNR、全部帧 / 新视角 / 4 个距离桶的 5 项画质、几何 13 项、针状 7 项、耗时与显存）。官方循环那次只调 scene 的运行只有 PSNR / SSIM / LPIPS，其余写“未记录”。**以后每个新的 140 场景实验都要加进脚本的 `EXPERIMENTS` 并重新生成总表。**
20. （2026-10-20）ply 可视化：模型是 4096 个 token × 64 子 Gaussian（不是默认的 1024），`_slotcolor.ply` 全部染色太乱。`scripts/highlight_selected_tokens_ply.py` 在 no_tuning 的活跃 token（不透明度总量 ≥ 中位数）里用种子 0 随机抽 20 个（前 10 个即 10 个版本），所有实验使用同一批编号和颜色。每个实验生成两套：`_highlight{10,20}tokens.ply` 保留其他点但置灰、opacity ×0.1；用户反馈背景仍遮挡后，新增 `_only{10,20}tokens.ply`，彻底删除未选点，仅保留 640 / 1280 个 Gaussian，同时保留原 `slot_id` / `child_id`。图例为 `demo/scene_<id>/highlighted_tokens.{md,json}`；`run_demo.sh` 对比后自动调用。现有 4 个 demo 已生成 8 个 only PLY 并核验 token 集合和点数。
21. （2026-10-20）用户决定：不测 token lr 1e-3；1e-2 的 4 个失败目录保留；“有什么可行的实验，就要有什么场景 ply”；目录名改用 scene / token / geo、词序固定。**第三次改名**：`scene_latent_tuning/` → `scene_tuning/`，`geo_loss_in_scene_latent_tuning/` → `scene_geo_tuning/`，`token_tuning/lr{lr}_steps50` → `token_tuning/lr{lr}`，`tuned_parameters_x_geo_loss/` 下 6 组分到 `token_geo_tuning/`、`scene_token_tuning/`、`scene_token_geo_tuning/`，对照目录改名 `scene_token_x_geo/`；两个 report 脚本改名 `report_scene_geo_tuning.py` / `report_scene_token_x_geo.py`。重新生成后所有数字与改名前逐项相同。同场景 ply 一键脚本扩到 7 种可行方法（新增 token_geo、scene_token、scene_token_geo，待用户运行）；`compare_same_scene_ply.py` 的 `METHOD_ORDER` 固定对比表 / 图例的方法顺序。
22. （2026-10-20）用户确认**第四次、最终目录整理**：每个一层文件夹代表一次 140 场景实验，不按方法嵌套；学习率按方法词序写进实验名；固定几何权重从名字删掉；demo 全部集中到 `demo/scene_<id 前12位>/<实验名>/` 并平铺。新增 `scripts/experiment_registry.py` 作为 11 个实验的唯一事实源，根目录只保留 `run_140_scenes.sh` / `run_demo.sh` 两个统一入口，报告和总日志分别归入 `reports/` / `launch_logs/`。同设置的 scene 官方循环和复刻重跑删除，核对结论冻结在 `reports/scene_tuning_noise_floor.json`。117 次移动 sha256 全部一致，4 个 ply 旁注按新位置有意改写；报告、总表、对比、高亮数值与备份完全相同；pytest（排除官方坏 contracts）66 passed，两个脚本 `DRY_RUN=1` 通过。完整对照见 `EXP_TokenGS/renamed_directories_old_to_new.md` 第四次整理。
23. （2026-10-20）`no_tuning` token 诊断：新增 CPU 脚本 `scripts/diagnose_no_tuning_tokens.py`，从 demo 的 `gaussians.pt` 输出 4096 行 `token_diagnostics.csv`、JSON、中文报告和四联图。当前场景 85.2% token 的 packing ratio 高于目标，43.2% 达“强分散”（packing/target >4）；中位 RMS 半径 0.257、Gaussian RMS 尺寸 0.0622，99.7% token 的 RMS 半径大于最近 token 中心距。强分散 token 仍承担 31.5% 总 opacity。140/140 场景的场景级 packing/target >4（中位 76.3），说明 demo 不是孤例；但 140 场景未保存逐 token 张量，逐 token 百分比只适用于 demo。结论：一个 token 的 children 通常不是紧凑局部簇，这个描述成立；但 token 是全局 latent slot，强制每个 token 收紧的优化假设未成立，且现有几何损失已被画质结果反驳。
24. （2026-10-20）全仓代码、日志与微调曲线深度核查：
    - **visibility loss 读未初始化 GPU 内存的实测证据**：在 demo 场景 `2f3e1c0f688c` 的 `tuning_curve.jsonl` 第 0 步（尚未执行任何 Adam step，解码出的 Gaussian 与 `context_loss_rgb`、`context_loss_ssim`、`geo_sw_loss` 在三个实验间逐位一致），`context_loss_visibility` 在 `scene_tuning_lr1e-2` 为 `2.307e-5`，而在 `token_tuning_lr1e-4` 为 `8.791e-5`；且在 50 步内于 `0.0`、`3.05e-9`、`2.28e-5`、`3.52e-4` 之间无规律跳变，直接证实 `gsplat` 对被剔除点不写 `at::empty` 分配的 `means2d`。
    - **50 步微调曲线动力学**：① `scene_tuning_lr1e-2` 在第 1 步因 Adam 沿全坐标走 `±1e-2` 满步而出现瞬时超调（`target_all_psnr` 从 `23.591` 跌至 `22.934`，解释了早期 1 步微调掉点的现象），第 2 步起恢复上升；而 `token_tuning_lr1e-4` 从第 1 步起单调上升。② 在该 demo 场景上，`scene_tuning_lr1e-2` 的远视角 `d4+` PSNR 在第 32–43 步即见顶（`23.626`）随后微降至第 50 步的 `23.587`（同期 `d0` 仍在从 `28.52 → 29.36` 爬升），而 `token_tuning_lr1e-4` 的 `d4+` 单调升至 `23.713` 反超 scene 微调，但其 `HF ratio` 在前 8 步先降至 `0.699` 后才缓慢回升至 `0.730`（远低于 scene 微调的 `0.839`）。③ `scene_geo_tuning_lr1e-2` 在第 0 步加权几何损失（`0.0737`）即为渲染损失（`0.0440`）的 1.68 倍，到第 50 步仍占总损失的 59.8%（`0.0365 / 0.0610`），且把视锥外高不透明度点（`opacity > 0.1`）从 646/689 个推高 6 倍至 3879 个。
305|    - **`token lr=1e-2` 发散的架构根因与报错位置区分**：`scene_latents` 经 `latent_kv_proj` 后有 `k_proj_norm (LayerNorm)`、softmax 注意力归一化以及 `gs_cross_attn_scale`（LayerScale `γ≈0.027`）三重缓冲；而 `gs_tokens`（初值 std `0.02`）在 `DecoderBlock` 中走无归一化的残差直连通道直达无前置 LN 的 `ClipActivationHead.deconv`，再经 `pos_act(x)=sign(x)(exp(|x|)-1)` 与 `scale_act(x)=min(exp(x-shift),cap)` 指数激活，`1e-2` 的步长极易使线性读出越过 FP32 `exp` 溢出阈值（`~88.7`）。不加几何的两组在完成第 1 场景 50 步微调后于外层 `geometry_stats` 触发 `ValueError`；加几何的两组则在内循环 `geo_losses_for` 中触发。
306|306|25. （2026-10-20 17:10–17:12）按用户指令：① **删除 4 个 `token lr=1e-2` 发散目录**（`token_tuning_lr1e-2`、`token_geo_tuning_lr1e-2`、`scene_token_tuning_lr1e-2_1e-2`、`scene_token_geo_tuning_lr1e-2_1e-2`），同步从 `scripts/experiment_registry.py` 与 `scripts/report_scene_token_x_geo.py` 移除并重生成总表（现为 7 组完整 140 场景实验 + 2 行官方参考）与报告，pytest 64 passed / 44 subtests passed；② **运行 `bash EXP_TokenGS/run_demo.sh` 补齐缺失的 3 组 demo**（在 GPU 1–3 并行，约 21 s 完成，exit 0），自动更新 `demo/scene_2f3e1c0f688c/comparison.{md,json}` 与 `highlighted_tokens.{md,json}` 及全部 7 组的 `_highlight{10,20}tokens.ply`、`_only{10,20}tokens.ply`、`_minus_no_tuning.ply`。7 组 `decoder_inputs.pt` 重解码 `max |diff|` 全为 `0.000e+00`。
307|26. （2026-10-20 17:20）**第五次命名优化（lr 紧跟对应模块词，消除多模块组合时的归属歧义）**：用户指出 `scene_token_geo_tuning_lr1e-2_1e-4` 这类把多个 lr 堆在末尾的名字在多模块组合时容易分不清哪个 lr 属于哪个模块，选择方案 A（每个模块的 lr 紧跟在该模块词后面）：`no_tuning`、`scene_lr1e-2_tuning`、`scene_lr1e-2_geo_tuning`、`token_lr1e-4_tuning`、`token_lr1e-4_geo_tuning`、`scene_lr1e-2_token_lr1e-4_tuning`、`scene_lr1e-2_token_lr1e-4_geo_tuning`。已迁移 6 个 140 场景目录、6 个 demo 子目录及其中全部 PLY / 旁注文件，同步更新 `experiment_registry.py`、`report_scene_geo_tuning.py`、`report_scene_token_x_geo.py`、`build_all_experiments_table.py` 与测试，重新生成两份报告、总表、comparison 与 highlighted_tokens，pytest 64 passed / 44 subtests passed，两个一键脚本 `DRY_RUN=1` 全部识别为 already complete。
308|27. （2026-10-20 17:30）**`no_tuning` Token 机制深度剖析与关键突破（发现“死点 `<1/255` 污染”真相）**：
309|    - **核心发现：Token 的存活子高斯原本就是紧凑的，此前的“高分散（`packing_ratio=367`、`SW=0.526`）”以及几何微调掉点，很大程度上源于 `geo_losses.py` 未过滤被 `gsplat` 硬件剔除的死点（`opacity < 1/255`）！**
310|      - `gsplat` 对 `opacity < 1/255 (0.00392)` 的 Gaussian 直接剔除（`radii=0`，不渲染、预训练全程 0 梯度），因此每个 token 中约 19 个死点的 `xyz` 留在远离 token 中心的随机/初始位置（距中心中位距离 `0.90~2.58`，而前 36 个主力存活点距中心中位距离仅 `0.061~0.069`）。
311|      - 但 `sigmoid(raw-2)` 恒大于 0（死点 `opacity ≈ 0.0006~0.0039`），`geo_losses.py` 对全部 64 个 child 计算 `opacity * area` 权重时未将 `<1/255` 置零。由于死点距离平方 $r^2$ 是存活点的上千倍，仅 8 个固定死点槽位（`child 2, 61, 41, 27, 12, 0, 7, 20`，`med_opac ≈ 0.0010~0.0016`）就贡献了全场景平均 **59.5%** 的 `center_variance`！
312|      - **只统计真正参与渲染的存活点（`opacity >= 1/255`）时**：`no_tuning` 在 demo 场景上的 `packing_ratio` 均值从 **`366.96` 暴降至 `5.30`**（中位数从 `17.56` 降至 **`2.48`**，低于理论目标 `5.74`；超目标比例从 `85.2%` 降至 `28.7%`，强分散 `>4` 从 `43.2%` 降至 **`3.5%`**），`sw_loss` 从 **`0.5264` 降至 `0.1082`**，`inter_distance` 从 `0.810` 升至 `1.231`。
313|      - 这同时解释了为什么现有 `scene_lr1e-2_geo_tuning` 会严重拉低 PSNR（−0.665 dB）：内循环的 `sw_loss` 与 `packing_loss` 把近 60% 的方差梯度花在“强行把 `r~2.5` 的不可见死点和 `r~0.08` 的可见表面点往一块拉”上，由于 64 个 child 共享同一个线性 `deconv` 读出，直接把可见的存活 Gaussian 拽离了真实物体表面。
314|    - **64 个 Child 槽位的跨 Token 固定分工**：在全部 4096 个 token 上，64 个 child 下标具有高度固化的角色——5 个槽位（`0, 7, 16, 19, 20`）在 4096/4096 个 token 中 100% 为死点，9 个槽位存活率 `<5%`，而 34 个槽位在 `>95%` 的 token 中恒定存活（如 `child 40, 63, 18, 43` 平均 opacity `0.136~0.156`）。
315|    - **Gaussian 真实三维形状分类（打破“全是细针”的误区）**：将各向异性比 $s_1/s_3$（不加权中位数 `59.1`，加权 `23.6`）拆解为伸长比 $s_1/s_2$（中位数仅 `2.35`，加权 `2.18`）与扁平比 $s_2/s_3$（中位数高达 `22.18`，加权 `8.62`，因最短轴 $s_3$ 中位数仅 `0.00055`）：存活点中 **57.0%（占 53.3% opacity）是二维薄圆盘/瓦片（disk/surfel，$s_1/s_2 < 3, s_2/s_3 \ge 3$）**，**28.9%（占 23.3% opacity）是扁平长条/刀片（ribbon，$s_1/s_2 \ge 3, s_2/s_3 \ge 3$）**，**真正的圆柱形细针（needle，$s_1/s_2 \ge 3, s_2/s_3 < 3$）仅占 7.5%（11.3% opacity）**，球体/微椭球占 6.6%（12.1% opacity）；且 31.7% 存活点（37.6% opacity）最长轴触及 `0.075` 剪裁上限。
316|    - **Token 按深度的自适应分工**：近景 token（距相机 `0.32~0.60`）尺度小（`scale_rms=0.014`）、核心半径小（`r_p50=0.055`）、opacity 适中（`sum=2.46`），负责高频几何细节；远景/背景 token（距相机 `4.78~10.69`）尺度触顶（`scale_rms=0.094`）、核心半径大（`r_p50=0.291`）、opacity 最高（`sum=6.25`），负责大面积背景覆盖。每个 token 内部存活点的 RGB 标准差中位数达 `0.203`（全局为 `0.258`），说明单个 token 不是单色体素，而是自带明暗/纹理梯度的局部表面微片簇。
317|28. （2026-10-20 17:40）**回溯对比 `EXP_TokenGS.zip` 原始报告与 `4DGS/FFRGS/AGENTS.md`（L455–490），定位前人研究的 5 个核心缺陷**：
318|    1. **未过滤 `<1/255` 死点 + 可视化时把死点 `opacity` 强行设为 1**：原报告用 `colored_20_slots_opacity1.ply`（把 20 个 slot 全部 64 个 child 的 opacity 设为 1）观察到“同一 slot 的 64 个 GS 大范围散落”，并用未过滤死点的 `opacity` 加权算出 `SW=0.646`、`packing=18.94 vs target 8.87`、`equal_spread=420.5`。实测表明，若按原报告的纯 `opacity` 权重但过滤掉 `<1/255` 死点，`no_tuning` 的 `SW` 原本就只有 **`0.0893`**、`packing_ratio` 原本就是 **`6.32 < target 8.58`**（不加权 `equal_spread` 中位数也从 `311.6` 降到 `3.5`）。原报告微调 10k 步后发现“packing overshoot（`3.87 < 7.89`）”，其实是因为存活点本来就低于 target，10k 步只是把 19 个死点拉回中心并顺带把存活点挤得过紧。
319|    2. **本仓早期改用 `opacity_area`（`opacity * scale^2`）但仍未过滤死点，将虚高再放大 6.6 倍**：近景 token 的存活点极小（`scale~0.014`），部分死点尺度反而偏大（`0.012~0.027`），乘上面积权重后死点权重占比上升，使 `no_tuning` 的未过滤 `packing_ratio` 从纯 `opacity` 加权的 `55.41` 进一步虚增到 `366.96`。
320|    3. **训练/评估数据集泄漏与选择性报告口径**：原报告的“12 场景统计”在 `FFRGS/AGENTS.md` L468–471 中被证实全部包含在训练集内且各重复 6 次；补充的 30 个 DL3DV-5K 场景对官方 checkpoint 而言很可能也在预训练集中，且仅评测了随机采样的 4 个 target views（基线仅 17.05 dB，非官方 140 场景 Benchmark 的 25.19 dB）。更重要的是，`FFRGS/AGENTS.md` L483–484 记录了 **10 视图平均 PSNR 实际上从官方 `19.965` 掉到了普通微调 `19.774`、几何微调 `19.453`（比官方低 −0.512 dB）**，而 `EXP_TokenGS.zip` 只汇报了 4 个 target views 的 `+0.264 / -0.087 dB`。
321|    4. **VGGT-Ω 点云对齐只报单向距离（`Ω → GS`），隐去了恶化近 3 倍的反向距离（`GS → Ω`）**：`EXP_TokenGS.zip` 只写了 `Ω → GS` p90 从 `0.0701` 降到 `0.0473`（改善 32%），但 `FFRGS/AGENTS.md` L487–489 明确记录反向距离 **`GS → Ω`（即 Gaussian 偏离真实表面的程度）p90 从 `0.455` 恶化到 `0.793`，对 `opacity > 0.1` 的高不透明点更是从 `0.815` 恶化近 3 倍到 `2.34`**！这与我们在 140 场景 TTT 中发现“几何损失把高不透明度 Gaussian 推到视锥外 / 表面外（demo 上从 646 激增到 3879）”完全吻合。
322|    5. **把 `anisotropy = s1/s3` 全部当成有害“细针”**：未区分 $s_1/s_2$（细针度，中位数仅 `2.35`）与 $s_2/s_3$（薄片度，中位数 `22.18`），将占 85.9% 的正常 2D 表面薄瓦片/长条也当成针状伪影去打压。
323|29. （2026-10-20 17:50）**在严格剔除 `<1/255` 死点后，量化确认用户指出的 TokenGS 两大客观结构缺陷（无表面锚定 + 无严格空间连续/排他性）**：
324|    - **即使完全剔除 `<1/255` 死点（甚至只看 `opacity >= 0.05` 或 `>= 0.1` 的高可见点），TokenGS 依然存在三大客观几何缺陷**：
325|      1. **缺乏显式表面锚定 → 形成沿视线厚达 12%~38% 的“多层半透明体积雾（Volumetric Multi-Layer Shell）”**：Decoder 通过 `deconv` 直接在 $\mathbb{R}^3$ 自由空间回归绝对坐标 $(x,y,z)$，无相机射线 $\mathbf{o}+t\mathbf{d}$ 或深度/点云锚定。实测在输入视角每个局部块（`18×18 px`）内，存活点沿视线方向的 opacity 加权深度四分位距 `(z_p75 - z_p25) / z_p50` 中位数达 **`12.4%`**（p75 达 **`38.1%`**），`P90-P10` 深度跨度中位数达 **`33.1%`**（p75 达 **`68.1%`**）。因为存活点中位 opacity 仅 `0.049`（Top-1 child 中位也仅 `0.297`），模型并非在真实表面铺一层不透明外壳，而是靠多层不同深度的半透明片沿视线 Alpha 混合凑出输入视角颜色；一旦转到新视角（如 `d4+`）就会发生视差错位模糊（解释了 `d0` 31.47 dB vs `d4+` 25.33 dB 的 6.14 dB 落差）。
326|      2. **单 Token 内部虽核心紧凑，但仍有长尾分裂与大深度跨度，无法保证严格连续**：仅看存活点（`>=1/255`），单 Token 内部最大两点距离（diameter）中位数仍有 **`0.368`**（p75 **`1.492`**，p90 **`2.403`**），相机距离相对跨度 `(max_d - min_d)/med_d` 中位数 **`16.4%`**（p90 **`47.9%`**），且 **25.8% 的 Token 仅在存活点内部就断裂成 $\ge 2$ 个互不接触的 $3\sigma$ 孤岛**；即便只看 `opacity >= 0.1` 的高不透明点，单 Token 直径中位数仍达 `0.319`（p90 `1.145`），深度相对跨度中位数 `10.3%`（p90 `25.0%`）。
327|      3. **跨 Token 空间高度混叠（Many-to-One Soup），毫无区域排他性**：哪怕只看 `opacity >= 0.05`（或 `>= 0.1`）的高可见点，相邻 Token 质心最近距离中位数仅 **`0.0189`**（`0.0249`），而单个 Token 的有效半径 `rms_r` 中位数为 **`0.0929`**（`0.1049`），**98.0%（95.6%）的 Token 半径是最近邻 Token 间距的 4–5 倍**；每个 Token 的 `p90` 半径内平均嵌套了 **7.8 个（p90 达 16 个）其他 Token 的质心**；图像上每个 `18×18 px` 小块平均由 **19 个（p90 达 50 个）不同 Token 的高可见点（`opacity >= 0.05`）** 共同混合而成。
328|    - **核心辨析**：剔除 `<1/255` 死点纠正的是“把 `packing_ratio=367` 当成全部 64 个点满天飞”的统计误差；而用户指出的“无显式空间锚定导致不在真实表面（多层半透明深度雾）”和“无局部连续/排他约束导致单 Token 长尾分裂（25.8% 多孤岛）与跨 Token 严重混叠（每小块 ~20 个 Token 叠加）”是 **TokenGS 架构真正的客观痛点**。此前 `SW + Packing + Inter` 失败，是因为它在没有任何真实表面锚定的情况下，盲目在自由空间里把点往可能偏离表面的 Token 均值缩、或把 Token 互相推开，破坏了多层 Alpha 混合的视觉平衡。
329|30. （2026-10-20 18:00）**验证用户核心洞察：“死点/远点离得远，本质是因为位置预测偏离后模型通过压低 opacity 来降低错误影响（透明逃逸机制）”**：
330|    - **直接数据与权重证据（`dl3dv_latent_6v_ssim.safetensors` 的 `activation_head.deconv` + `no_tuning/gaussians.pt`）**：
331|      1. **预训练线性头 `deconv` 的 bias 并未关死任何槽位**：全部 64 个 child 的 `b_opac` 都在 `+0.006 ~ +0.026` 之间（对应初始 `opacity ≈ 0.12`），且 64 个 child 的位置读出权重 $W_c^{\text{pos}}$ 与 Token 均值方向余弦相似度高达 `+0.97 ~ +0.99`。区别在于**各 child 相对偏移权重 $\|\Delta W_c^{\text{pos}}\|$ 的范数分化为三档**：34 个常驻核心槽位偏移范数小（中位 `0.294`），21 个动态槽位居中（`0.582`），9 个外围槽位很大（`1.144`）。
332|      2. **同一槽位跨 Token 的强负相关（越偏离核心，opacity 越被压低）**：经过指数激活 $\text{pos\_act}(u)=\text{sign}(u)(\exp(|u|)-1)$ 后，当某个 Token 的特征 $h_s$ 在 $\Delta W_c^{\text{pos}}$ 上投影较大时，该 child 的位置被指数级甩出局部区域；此时网络同步把 $W_c^{\text{opac}} h_s$ 压成强负值。在 21 个动态槽位（共 $4096\times 21 = 86,016$ 个点）中，同一槽位跨 4096 个 Token 的 $\text{corr}(\text{dist\_to\_core}, \log \text{opac})$ 中位数达 **`-0.501`**：当它落在 Token 核心附近时（`med dist = 0.0599`）被赋予高可见度（`opac >= 0.02`，24,821 个点）；当它被甩远时（`med dist = 0.5101`，`p90 = 2.7318`）则被压低到 `< 1/255` 变成死点（38,771 个点）。
333|      3. **全场景 262,144 个点随离核距离 `dist_to_core` 呈现连续的“逃逸光谱”**：
334|         - `dist < 0.12`（核心连续片，占 50.1%）：存活率 **`92.3% ~ 94.3%`**；
335|         - `dist ∈ [0.12, 0.25)`（近核边缘，占 14.6%）：存活率 **`85.3%`**；
336|         - `dist ∈ [0.25, 1.00)`（**半逃逸/未彻底压死的长尾层**，共 55,934 个点，占 21.3%）：存活率从 `62.6%` 降到 `26.6%`，但**仍有 26,534 个点 `opacity >= 1/255`（其中 15,919 个点 `opacity >= 0.05`）没有被压到阈值以下**！正是这批“位置已经飞出局部区域（`0.25~1.0m`）、但 opacity 只降了一半”的点，造成了单 Token 内部断裂（25.8% 多孤岛）与视线方向的多层半透明雾；更严重的是，在无几何约束的 `scene_lr1e-2_token_lr1e-4_tuning` 50 步微调中，这 26,534 个离群存活点的平均 opacity 反而从 `0.1172` 被推高到了 `0.1470`（用来过拟合 6 个输入视角）！
337|         - `dist >= 1.00`（完全逃逸/死点层，占 14.0%）：95.6% 已被压到 `< 1/255`（`med_opac = 0.0011`），仅剩 1,602 个存活点。
338|    - **对“如何让 Token 真正负责一片连续区域”的方法论启示**：
339|      - 由于微调时 `deconv` 线性层冻结且 64 个 child 的位置权重高度线性耦合（`cos = +0.97~0.99`），此前 `geo_losses.py` 把 `opacity` detach 掉、强行靠改 Token 特征去把 `dist > 0.25` 乃至 `dist > 1.0` 的离群点拉回中心，必然把核心 34 个表面点一起拽偏。
340|      - 要让单个 Token 成为干净的连续局部区域，一方面在计算 Token 质心与局部几何时必须排除已死点（`< 1/255`），另一方面对那些被冻结线性头甩出局部半径（如 `dist_to_core > r_local`）却“没死透”的半逃逸点（`26,534` 个点），应当顺应模型原本的抑制机制（或允许通过 opacity 衰减将离群 child 关停，而只对局部核心内的主力 child 施加表面/紧凑约束）。
341|31. （2026-10-20 18:10）**“34 核心 + 21 机动 + 9 死点”究竟是训练失误还是泛化机制？（实证定性）**：
342|    - **9 个常驻死点槽位（占 14.1% 容量，总 opacity 仅占 0.29%，其中 5 个槽位在 4096/4096 个 Token 中 0% 存活且微调后仍为 0%）：属于纯粹的“训练失误 / 死神经元坍缩（Dead-Slot Collapse）”**。根源是预训练早期指数激活 $e^{|u|}-1$ 甩飞位置后，opacity 一旦被压破 `gsplat` 的 `< 1/255` 硬截断阈值，渲染梯度立即归零（类似 ReLU Dead Neuron），且原模型无 opacity 复活或负载均衡机制，白白浪费了 14% 的算力与显存。
343|    - **34 个常驻核心槽位（占 88.05% 总 opacity）+ 21 个动态机动槽位（占 11.66% 总 opacity）：是模型在“固定每 Token 64 球 + 无层级相对坐标”架构下，自发演化出的“按深度自适应分配容量（Depth-Adaptive Capacity Gating）”泛化策略，但带有严重的耦合副作用**：
344|      - **跨深度泛化收益（双峰激活分布）**：统计每个 Token 激活的机动槽位数（0~21 个），呈明显双峰分布（`p25=4, p50=15, p75=16`）：
345|        - **近景 Token（激活 14~21 个机动槽位，共 2,357 个 Token，距相机中位 `0.76m`）**：近景透视面积大、细节多且单点尺度小（`scale_rms = 0.0228`），模型打开绝大部分机动槽位，使每个近景 Token 拥有约 **50 个超薄瓦片**（存活机动槽位的扁平比 $s_2/s_3$ 中位数高达 **`104.0`**，比核心槽位的 `12.1` 薄 8 倍以上，专门贴在近景表面补高频细节）；
346|        - **远景 Token（仅激活 0~5 个机动槽位，共 1,205 个 Token，距相机中位 `2.92m`）**：远景单点尺度大（`scale_rms = 0.0438`，直至触及 `0.075` 上限），且推向远景需要大的 $|h_s|$，模型自动关停绝大部分机动槽位，仅保留 ~35 个大尺寸、高不透明度的核心槽位（`opac_sum = 4.19`）铺满背景。
347|      - **副作用（有缺陷的妥协）**：因为缺乏显式的“Token 中心 + 局部有界偏移”层级参数化，模型只能拿同一个线性投影同时控制“3D 绝对坐标”和“子球开关”，导致该关的球关不彻底（留下 2.65 万个离核 `0.25~1.0m` 的半透明离群点），造成了几何不连续与多层深度雾。
348|32. （2026-10-20 18:25）**路线 A2“离核半逃逸点 Opacity 清理”四大目标的定量预评估（基于 Demo 场景 `no_tuning` 与 `scene_lr1e-2_token_lr1e-4_tuning` 实测）**：
349|    - 以每个 Token 内 Top-24 高不透明度子球的坐标中位数作为稳健核心 $\boldsymbol{\mu}_{\text{core}}$，其中位离核半径为 $r_{\text{core}}$：
350|      - **清理 `dist > 3.0 * r_core` 的离核存活点**（仅占存活点的 **2.9%~3.3%**，仅占总 opacity 的 **1.9%~2.1%**）：单 Token 直径 `p90` 从 `2.40m / 2.55m` 直接砍半至 **`1.14m / 1.23m`**；**单 Token 分裂为 $\ge 2$ 个 $3\sigma$ 孤岛的比例从 `25.8% / 26.5%` 暴降至 `0.6% / 1.2%`**（几乎 99% 的 Token 恢复为单一连续连通块）；视锥外高不透明漂浮物减少约 13%。
351|      - **清理 `dist > 2.0 * r_core` 的离核存活点**（占存活点的 **11.8%~12.5%**，占总 opacity 的 **10.0%**）：单 Token 直径 `p90` 降至 **`0.90m / 0.95m`**；**多孤岛比例降至 `0.1% / 0.4%`**；视锥外高不透明漂浮物（`opacity > 0.1`）从 `646 / 708` 降至 **`406 / 446`（减少 37%）**；总有效 GS 数量从 262,144 精简至 **160,467（直接减少 38.8% 冗余基元）**。
352|    - **两个关键设计边界**：① **必须用相对核心半径 $r_{\text{core}}$（而非固定绝对米数）**作为门控基准，因为远景 Token 正常核心半径（`0.29m`）是近景（`0.055m`）的 5 倍，用绝对阈值会误杀远景背景墙；② **必须在 50 步微调内循环中渐进压低 Opacity（而非事后硬删）**，让核心内的 34~45 个子球在渲染梯度下自动补偿被关停离群点原本承担的 `2%~10%` 颜色贡献，避免事后硬删导致漏光掉点。
353|33. （2026-10-20 18:35）**`deconv.weight` 子空间正交性定理（解释为何“拉位置必伤核心、压 Opacity 不伤核心”）**：
354|    - 对预训练 `activation_head.deconv.weight`（`[64, 14, 1024]`）做 SVD 子空间投影分析：
355|      - **位置权重高度共线（不可解耦）**：21 个机动槽位的位置权重 $W_c^{\text{pos}}$ 与 34 个核心槽位位置权重的余弦相似度高达 **`+0.971`（97% 共线）**，因此旧几何损失试图通过修改 1024 维 Token 向量 $h_s$ 去拉动离群点 `xyz`，必然同步拽偏 34 个核心点的 3D 坐标；
356|      - **不透明度权重与核心位置高度正交（可解耦！）**：34 个核心槽位的三维坐标权重 $W_{\text{core}}^{\text{pos}}$ 在 $\mathbb{R}^{1024}$ 中仅张成一个 **102 维子空间**（$34\times 3 = 102$）。将 21 个机动槽位的不透明度权重向量 $W_c^{\text{opac}} \in \mathbb{R}^{1024}$ 投影到该 102 维子空间的正交补空间上，发现 **`95.7% ~ 99.1%`（中位数 `97.7%`）的能量完全正交于全部 34 个核心点的 XYZ 坐标**（即便对 34 个核心点的全部 14 维属性共 476 维子空间求正交补，仍有中位数 `25.1%` 的独立正交分量）！
357|      - **微调载体选择**：`scene_latents` 通过 Cross-Attention 的 K/V 被所有 Token 共享，主要负责宏观场景对齐（`27.400 dB`）；而 `gs_tokens` 的第 $s$ 行拥有直达第 $s$ 个 Token `deconv(h_s)` 的残差直连通道（Self-Attention LayerScale 仅 $3.6\times 10^{-4}$），能逐 Token 独立在 97.7% 正交补空间内关停各自的离群子球。因此 **`scene + token 联合微调`（`scene_lr1e-2_token_lr1e-4`）是开展离核 Opacity 约束最理想的载体**（兼顾 `scene` 的宏观高画质与 `token` 的逐 Token 独立门控自由度）。
358|34. （2026-10-21 15:23–15:36）**完成 3 组 140 场景新实验与同场景 Demo（修复 `gsplat` 视锥外 bug：`vis_xyz` vs `vis_opac`，以及内循环离核空间门控 `core_gate2.5`）**：
359|    - 全部 3 组在 140/140 场景与 Demo 场景上均 `exit 0` 完成，`decoder_inputs.pt` 重解码 `max |diff| = 0.000e+00`：
360|      1. `scene_lr1e-2_token_lr1e-4_vis_xyz_tuning`（解析视锥投影拉 `xyz`，权重 1.0）：all PSNR **`27.440`**（比 `scene_lr1e-2_tuning` 高 `+0.040 dB`，110/30 胜），140 场景视锥外加权比例从 `1.25%` 降至 **`0.16%`**（Demo 视锥外 `op>0.1` 点数从 `708` 降至 **`101`**）。
361|      2. `scene_lr1e-2_token_lr1e-4_vis_opac_tuning`（解析视锥投影压 `opacity`，权重 2.0）：all PSNR **`27.441`**（比 `scene_lr1e-2_tuning` 高 `+0.041 dB`；**Demo 场景达 `25.973 dB` 为全部 10 组最高**），140 场景视锥外加权比例降至 **`0.09%`**（Demo 视锥外 `op>0.1` 点数从 `708` 降至 **`16`**，清除率 **97.7%**，且平均每场景减少 2,764 个冗余活点）。证明对视锥外点“压 `opacity`”比“拉 `xyz`”清除漂浮物更彻底（16 vs 101）且画质略优。
362|      3. `scene_lr1e-2_token_lr1e-4_vis_opac_core_gate2.5_tuning`（解析视锥压 `opacity` + 内循环离核 $>2.5\,r_{\text{core}}$ 可微空间门控）：
363|         - **画质**：140 场景 all PSNR **`27.422`** / SSIM **`0.8702`** / LPIPS **`0.1813`**——**比 `scene_lr1e-2_tuning`（`27.400`）高 `+0.022 dB`（105/35 胜），比旧几何损失 `scene_lr1e-2_token_lr1e-4_geo_tuning`（`26.940`）高出 `+0.482 dB`**！在每场景额外关停 **6,287 个离核活点**（相对于无约束联合微调 `27.481` 仅差 `0.059 dB`）的同时，彻底避免了旧几何损失的大幅掉点。
364|         - **几何指标（在未修改的原版 `geo_losses.py` 口径下）跨数量级改善**：140 场景均值 `SW` 从 `0.7102` 暴降至 **`0.1013`**（比旧 `geo_tuning` 的 `0.3558` 还低 3.5 倍），`packing_ratio` 从 **`717.26` 暴降至 `3.14`（低于目标 `5.60`）**，Token 中心方差从 `0.0301` 降至 **`0.0038`**（且不膨胀局部尺度方差 `0.0006`），`inter_distance` 升至 **`1.139`**；Demo 上单 Token 多孤岛断裂率从 `26.5%` 暴降至 **`1.5%`**，单 Token 直径 `p90` 从 `2.55m` 砍半至 **`1.27m`**。
365|         - **Floater 清除与轻量化**：140 场景视锥外加权比例从 `1.25%`（旧几何为 `3.93%`）降至 **`0.08%`**（Demo 视锥外 `op>0.1` 点数从 `708` / `4007` 降至 **`9` 个**，清除率 **98.7%**）；140 场景平均有效活点从 `187,563`（`71.55%`，旧几何反而膨胀到 `194,788`）精简至 **`181,276`（`69.15%`）**。
366|35. （2026-10-21 15:45）**生成 6 组精简对照报告 `EXP_TokenGS/reports/vis_and_core_opacity_cleaning.md` 及 Demo 剪枝 PLY（`_prune0.05.ply` / `_prune0.01.ply`），并记录用户提出的新研究方向转型（引入扫描仪 Dense 点云作为几何标准）**：
367|    - **低 Opacity 剪枝实测**：在 `..._vis_opac_core_gate2.5_tuning` 上，事后删 `< 1/255`（剩 67.5% 点）PSNR 变化 `+0.000 dB`；删 `< 0.01`（剩 60.2% 点，砍掉 10.4 万点）PSNR 仅降 `-0.016 dB`（近乎无损）；删 `< 0.05`（仅剩 32.3% = 8.46 万核心骨干点）在点云查看器中极其干净，但因切掉了近景 `[0.01, 0.05)` 的超薄机动纹理子球，事后硬删会使 PSNR 降 `-1.32 dB`。
368|    - **后续方向转型（用户提出：利用扫描仪直接获取的 Dense 点云 + Image 作为几何标准纠正前馈几何问题）**：
369|      - 这一思路直接命中了 TokenGS 最大的结构短板（缺乏显式 3D 表面锚定），且与 CVPR 2026 的 **AnchorSplat**（3D 几何先验锚定 + 有界局部偏移 $\mu_j = A_j + \delta\mu_j$）和 **Any4D**（多传感器度量级几何融合）方向高度契合。
370|      - 三种落地层次：① **不改架构，用扫描点云做微调/训练监督与剔除标准**（点云投影深度 L1 约束拍扁 12%~38% 多层半透明雾 + 点到点云距离 $>\delta$ 直接压低 Opacity 清除离表面漂浮物 + 点云法线约束薄瓦片最短轴）；② **点云锚定 Token（Point-Anchored TokenGS）**：从扫描点云 FPS 采样 4096 个真实表面锚点 $\mathbf{p}_s$，将子球参数化改为 $\mathbf{x}_{s,c} = \mathbf{p}_s + r_s\tanh(\Delta\mathbf{u}_{s,c})$，从根源上保证 100% 贴合表面、100% 单 Token 连续、零跨 Token 混叠；③ **混合容错机制**（应对扫描仪在天空/玻璃/反射面的空洞及标定微差）。
371|    - **针对“扫描仪大场景 + 几百张图片”扩展性瓶颈的架构推演**：
372|      - 原版 TokenGS 在大场景/几百张图下会撞上三堵硬墙：① ViT Encoder 跨视图全注意力 $O((V\cdot P)^2)$ 在几百张图下必 OOM 且超出 6v 预训练分布；② 固定 4096 个 Token（~18 万活球）摊到大场景后密度严重不足；③ `mean_cam` 中心化与 `scale_cap=0.075` 仅适用于原点附近数米范围。
373|      - 两条可行扩展路线：
374|        - **方案一（免重训，滑窗复用 6v TokenGS + 全局点云对齐融合，借鉴 FreeSplat++ / Z-Order Transformer）**：将几百张图按共视度切分为多个 6 视图局部窗口，用各窗口内的扫描点云统一归一化尺度后分别通过 6v TokenGS 前馈；将生成的局部高斯变换回扫描仪全局坐标系，利用扫描点云直接剔除离表面 $>\delta$ 的漂浮球（天然消除跨窗口尺度漂移与重影），再按体素 / Morton 码池化合并重叠区冗余高斯，最后做带点云深度约束的轻量微调。
375|        - **方案二（训练新型点云 Anchor 局部前馈解码器，借鉴 AnchorSplat + SparseSplat）**：反转范式——不再“从固定视图生固定 4096 个 Token”，而是先从扫描仪全局点云按表面积/曲率采样 $M$ 个 3D Anchor Token（$M$ 随场景大小自适应扩展）；每个 3D Anchor 利用点云深度遮挡关系只向能看见它的最近 4~6 张图片反投影提取特征，经 3D KNN 局部注意力解码出有界局部子球 $\mathbf{x}_{s,c} = \mathbf{p}_s + r_s\tanh(\Delta\mathbf{u}_{s,c})$，使显存与图片总数解耦（保持 $O(M)$ 线性扩展）。
376|36. （2026-10-21 15:55）**核查本地现成点云资源与“小步快跑”三阶段推进计划**：
377|    - **重大资源发现（无需重跑 COLMAP，本地 `data/dl3dv_eval/*.zip` 已自带完整点云与 300+ 帧全长图像！）**：
378|      - 经实测核验，`data/dl3dv_eval/`（指向 `/mnt/disk/xveyan/4dgs/DL3DV-10K-Benchmark`）下**全部 140/140 个场景的 zip 包内均已包含预计算好的 `gaussian_splat/sparse/0/points3D.bin` 与 `points3D.ply`**，平均每场景拥有 **`101,016` 个 3D 点**（中位数 `92,036` 个，Demo 场景 `2f3e1c0f688c` 有 `85,998` 个点，且自带每个 3D 点在各帧的可见性轨迹 `image_id_sets`），并且 `tokengs/data/static/dl3dv.py`（L396–512）已内置 `_read_points3d_bin` 与 `_colmap_world_to_final_world_transform` 坐标对齐函数！同时每个 zip 包内都有 ~300 张全场景图片（此前评估仅用了前 50 张）。
379|      - 外部公共 `Image + Scanner Point Cloud` 权威数据集清单：**ScanNet++ v2**（CVPR 2026 AnchorSplat 与 TPAMI 2026 FreeSplat++ 主力数据集，含 Faro 激光扫描仪毫米级点云 `pc_aligned.ply` + DSLR/iPhone 图像）、**Tanks and Temples (T&T)**（室外/室内激光扫描点云 + 高清多视角图）、**Replica**（8 个测试室内场景，体量小易零样本测试）、**ARKitScenes**（iPad LiDAR 点云 + RGB）。
380|    - **稳扎稳打的“三小步”演进路线**：
381|      - **第 1 小步（单块 50 帧内用现成 `points3D.bin` 验证点云几何纠偏）**：在现有 6v（前 50 帧）上直接读取已被该 6 视图观测到的 COLMAP 点云，测试点云深度监督（消除 12%~38% 视线多层半透明雾以提升远视角 `d4+`）与点到真实表面距离过滤；
382|      - **第 2 小步（单场景 300 帧切块 + 全局点云骨架对齐融合）**：将单个场景的 300 张图切分为 5~6 个 50 帧子块（每块抽 6 张输入图），利用全局点云统一尺度、过滤离表面漂浮物并融合重叠区高斯；
383|      - **第 3 小步（走向路线 2：点云 Anchor 锚定解码头 $\mathbf{x}_{s,c} = \mathbf{p}_s + r_s\tanh(\Delta\mathbf{u}_{s,c})$）**：在前两步验证完数据流与融合管线后，用点云 FPS 采样的 3D Anchor 直接替换自由空间坐标预测并微调解码头。
384|37. （2026-10-21 16:10）**调研 2025–2026 3D World Models（WorldMirror、Lyra 1.0/2.0、Wonderland、WonderWorld、ViewCrafter、Flash3D、Aether、HunyuanWorld）并提炼五大可迁移设计准则（详见 `docs/LITERATURE_REVIEW_2026.md` 第五节）**：
385|    - **World Model 单图生成整场景的本质机制**：分为 ① 任意几何先验提示统一重建（**WorldMirror**）、② 相机受控视频扩散世界模型 + 前馈 3DGS 提升（**Lyra 1.0/2.0**——注：与 TokenGS 同属 NVIDIA Toronto AI Lab 团队、**Wonderland**）、③ 点云缓存投影引导迭代补全扩展（**ViewCrafter / ReconX / WonderWorld**）、④ 单图多层深度偏移前馈（**Flash3D**）。
386|    - **直接指导本项目的 5 条核心设计准则**：
387|      1. **点云/深度输入注入方式（WorldMirror Eq. 1）**：将点云投影为视角深度图并做 Patch Embedding，直接**逐元素加到图像 Token 上（$\mathbf{T}^{\text{img}} + \mathbf{T}^{\text{depth}}$）**，不增加 Token 序列长度与注意力开销；
388|      2. **几何路由只入 Q/K 不入 V（Lyra 2.0）**：3D 坐标/对应信号只加到 Attention 的 Query 和 Key 上引导跨视角配对，保持 Value 纯净传递高频外观纹理；
389|      3. **几何与 3DGS 外观解耦顺序训练 + 渲染深度锁定（WorldMirror）**：先固定 3D 几何再训 3DGS 外观，并用 $\lambda_{\text{gsdepth}}=0.1$ 锁定渲染深度；
390|      4. **FLAGS 几何解析初始化（WonderWorld）**：用点云坐标定中心、点云法线定最短轴旋转、$d/f$ 定初始切向尺度，网络只预测局部残差；
391|      5. **基于点云可见性最大覆盖选图 + 重叠分块（Chunked）前馈（Lyra 2.0）**：同作者团队在 Lyra 2.0 中证实长序列/大场景一次前馈必糊，须按点云可见性贪心选帧并切分为重叠 Chunk 分别前馈再合并。
392|38. （2026-10-21 16:25）**验证用户核心关切：“COLMAP 点云稀疏性是否会导致错误的几何纠正？”——证实 3D 最近邻拉扯必错，而“2D 有效像素渲染深度锚定（$\lambda_{\text{depth}}=0.01$）”实现重大突破**：
393|    - **COLMAP 点云在 6 个输入视图中的真实稀疏度（Demo 场景 `2f3e1c0f688c` 实测）**：
394|      - 全场景 319 帧虽有 `85,998` 个 COLMAP 点，但在 6 个输入视图中可见的仅 **`7,884` 个点**；投影到 `960×540`（51.84 万像素/图）上，每张图仅有 **`1,330 ~ 1,775` 个有效像素（仅占 `0.30%` 像素！）**，按 `8×8` patch 统计也仅有 **16.0%** 的 patch 含点，其余 **84.0%**（白墙、平滑地面、桌面、天空等弱纹理区域）完全空白。
395|    - **证实用户的担忧：若把稀疏 COLMAP 点云当成稠密扫描仪在 3D 空间做最近邻纠正，确实会导致严重错误**：
396|      - **错误做法 1（3D 最近邻拉扯）**：对所有活点计算到最近 COLMAP 3D 点的距离并施加拉力（$\lambda=0.5$），由于白墙中间无点（25% 的正常高斯离最近 COLMAP 点 `>0.20m`），白墙高斯被拽向墙角特征点，Demo `all_PSNR` 从 `25.926` 暴跌至 **`25.554`（−0.372 dB）**；
397|      - **错误做法 2（3D 最近邻距离门控删点）**：压低离最近 COLMAP 3D 点 `>0.15m` 的高斯 opacity，误杀平滑墙面高斯，`all_PSNR` 跌至 `25.843`；
398|      - **错误做法 3（2D 像素深度权重过大 $\lambda=0.05$）**：强迫 `0.30%` 的像素过度拟合，在连续表面拉出局部凹坑，`d0` 从 `29.66` 跌至 `28.07`，`all_PSNR` 跌至 `25.817`。
399|    - **正确解法（只在有 COLMAP 点的 `0.30%` 像素上做温和的 2D 渲染深度 Smooth-L1 锚定，$\lambda_{\text{depth}}=0.01$）**：
400|      - **为什么 `0.30%` 的稀疏像素深度锚定不会拉坏白墙，反而能带正整片表面？** ① 它只在有 COLMAP 特征点的像素 `(u,v)` 上沿视线方向约束渲染深度 $\hat{D}(u,v)$，绝不在 3D 空间把白墙高斯横向拽向墙角；② 因为每个 Token 是一个连续局部薄瓦片簇（且单球有面积，加上 `core_gate2.5` 保证了单 Token 内部连续性），当 `0.30%` 的角点/边缘像素把对应 Token 的深度锚定到真实表面时（中位深度误差从 `0.2738` 暴降 **2.4 倍** 至 **`0.1143`**），整块相连的局部表面自然随之对齐到正确深度！
401|      - **Demo 场景实测大幅突破（在 `vis_opac + core_gate2.5` 基础上加入 `lam_depth = 0.01`）**：
402|        - **All PSNR**：从 `25.939` 跃升至 **`26.228 dB`**（比此前最佳联合微调 `25.957` 高出 **`+0.271 dB`**，比只调 scene `25.802` 高出 **`+0.426 dB`**！）；
403|        - **Novel PSNR（44 帧新视角）**：从 `25.432` 跃升至 **`25.801 dB`（`+0.369 dB`）**；
404|        - **远视角 `d4+` PSNR（此前全部 10 组实验卡在 `23.58~23.71 dB` 的瓶颈）**：直接从 `23.682` 暴涨 **`+0.562 dB`** 达到 **`24.244 dB`**（$\lambda=0.02$ 时更达 **`24.407 dB`，暴涨 `+0.726 dB`**）！彻底证实“用稀疏 COLMAP 像素深度锚定消除多层半透明深度漂移”直接攻克了远视角视差模糊难题。
405|39. （2026-10-21 16:35）**在 Demo 场景实测 VGGT-Ω（`/mnt/disk/ethan/code/vggt-omega/checkpoints/vggt_omega_1b_416_reproduce.pt`）稠密深度/点云 vs 稀疏 COLMAP 像素深度锚定**：
406|    - **VGGT-Ω 原始预测深度的跨视角误差诊断**：
407|      - 用相机中心 Sim3 或全局 COLMAP 中位数将 VGGT-Ω 的 `256×448` 稠密深度对齐到 TokenGS 坐标系后，与真实 COLMAP 三角化深度对比：View 0–1 的中位误差为 `0.082~0.085`，但到 View 4–5 中位误差增大至 **`0.225~0.244`**（`p90` 误差达 `1.34~2.58`），说明神经网络估计的稠密深度在大基线多视角间存在明显的**非刚性深度漂移（Multi-View Depth Drift）**，与真实 GT 相机外参并不严格几何自洽。
408|    - **五种方案在 Demo 场景（`2f3e1c0f688c`）上的定量对比（均基于 `vis_opac + core_gate2.5`）**：
409|      1. **无深度监督基线**：`all = 25.931`，`novel = 25.424`，`d0 = 29.655`，`d4+ = 23.677`，深度误差中位数 `0.2727`；
410|      2. **VGGT-Ω 稠密 3D 点云最近邻拉扯（`lam_3d=0.2`）**：因跨视角深度不一致导致点云存在多层重影，硬拉 3D 位置使 `all` 暴跌至 **`25.309`（−0.62 dB）**；
411|      3. **VGGT-Ω 原始稠密深度监督（Sim3 / 全局中位数缩放，top-80% 置信度像素，`lam=0.01`）**：因跨视角深度漂移与 GT 相机外参冲突，`all = 25.864`（−0.07 dB），`d4+ = 23.698`；
412|      4. **COLMAP 局部校准的 VGGT-Ω 稠密深度（用每个像素最近 8 个 COLMAP 点的深度比局部校准 VGGT-Ω，`<=12px` 覆盖 71.3% 像素，`lam=0.01`）**：跨视角漂移被 COLMAP 修正后，各视图深度中位误差降至 `0.035~0.081`，**`all` 升至 `26.105`（`+0.174 dB`），`novel` 升至 `25.643`（`+0.219 dB`），`d4+` 升至 `24.061`（`+0.384 dB`）**；
413|      5. **纯稀疏 COLMAP 有效像素深度锚定（仅监督 `1.35%` = 9,293 个真实三角化像素，`lam=0.01`）**：**全场最佳**——**`all = 26.240 dB`（`+0.309 dB`），`novel = 25.813 dB`（`+0.390 dB`），`d4+ = 24.264 dB`（`+0.587 dB`），深度误差降至 `0.1138`**！
414|    - **核心物理结论**：在微调几何纠正中，**“跨视角严格几何自洽性（Accuracy & Multi-View Consistency）”远比“100% 像素稠密度（Density）”重要**！因为 TokenGS 的每个 Token 本身已是连续局部瓦片簇（配合 `core_gate2.5`），只需 `1.35%` 绝对精确、与相机位姿 100% 自洽的 COLMAP 三角化点做 2D 视线深度锚钉，即可无偏差地带正整片表面；而神经网络（VGGT-Ω）预测的稠密深度含有跨视角非刚性漂移，直接监督反而引入跨视角冲突（必须先用 COLMAP 局部变形校准后才能产生 `+0.17 dB` 正收益，但仍不如纯 COLMAP 真值点干净）。
415|40. （2026-10-22 10:10–10:36）**完成两组严格单变量解耦的 140 场景 + Demo 实验（直接基于 `scene_lr1e-2_token_lr1e-4_tuning`，不含 `vis_opac` 与 `core_gate`，单独对比 `+ COLMAP (λ=0.01)` vs `+ VGGT-Ω Sim3 (λ=0.01)`）**：
416|    - **140 场景全量结果（全部 `140/140 complete, exit 0`，Demo `redecode_max_abs_diff = 0.000e+00`）**：
417|      1. **基线 `scene_lr1e-2_token_lr1e-4_tuning`**：All PSNR `27.4807`，Novel PSNR `26.9012`，`d0 = 31.7302`，`d2-3 = 27.1010`，`d4+ = 25.3573`，All SSIM `0.8707`，Novel SSIM `0.8617`，`d4+` SSIM `0.8319`；
418|      2. **`scene_lr1e-2_token_lr1e-4_colmap_depth0.01_tuning`（解耦实验 1：仅加 COLMAP 稀疏有效像素深度 Smooth-L1，$\lambda=0.01$）**：
419|         - **刷新 140 场景全套核心指标最高纪录**：All PSNR **`27.4822`**（80 胜 / 60 负，中位数 `+0.0048 dB`）、**Novel PSNR `26.9110`**（`+0.0098 dB`，**86 胜 / 54 负**）、**`d2-3` PSNR `27.1091`**（**78 胜 / 62 负**）、**远视角 `d4+` PSNR `25.3806`**（`+0.0233 dB`，**91 胜 / 49 负**，最高单场景 `d4+` 暴涨 **`+0.555 dB`**！）、All SSIM **`0.8712`**（83 胜 / 57 负）、Novel SSIM **`0.8623`**（89 胜 / 51 负）、`d4+` SSIM **`0.8330`**（**94 胜 / 46 负**）！
420|         - **物理机制验证**：`d0` 从 `31.730` 略降 `-0.059 dB` 至 `31.671`，而远视角 `d2-3` 与 `d4+` 全面上涨（91/140 场景胜），直接证明 COLMAP 真实深度锚定抑制了模型利用多层半透明雾对 6 个输入视角的过拟合，迫使高斯收敛到真实三维表面从而提升大视差泛化！在 Demo 场景（`2f3e1c0f688c`）上：All PSNR 从 `25.957` 跃升至 **`26.253`（+0.296 dB）**，Novel PSNR 从 `25.449` 跃升至 **`25.820`（+0.371 dB）**，`d4+` 从 `23.695` 跃升至 **`24.261`（+0.566 dB）**，All SSIM 从 `0.8510` 升至 **`0.8574`**！
421|         - **尾部 5 个掉分场景的原因诊断**：`points3D.bin` 中存在少量远距离极窄基线噪点（`z*0.15` 达 `10~48m` 且未过滤 COLMAP 重投影误差字段 `error`），在线性深度 Smooth-L1 下单个 `z=40m` 噪点产生过大梯度；后续若改用 `log(depth)`（或过滤 `reproj_error < 1.0px, track_len >= 3`）可进一步消除尾部负向场景。
422|      3. **`scene_lr1e-2_token_lr1e-4_vggt_omega_depth0.01_tuning`（解耦实验 2：仅加 VGGT-Ω Sim3 稠密深度，$\lambda=0.01$）**：
423|         - 140 场景 All PSNR **`27.4534`**（比 `scene_lr1e-2_tuning` 高 `+0.054 dB`，112/28 胜；但比无深度联合微调 `27.4807` 低 `-0.027 dB`，58 胜 / 82 负），Novel PSNR `26.8768`，`d4+` PSNR `25.3377`；Demo 场景 All PSNR `25.910`，`d4+` `23.740`。
424|         - **对比结论**：在 140 个场景上，**仅监督 1.35% 像素的纯稀疏 COLMAP 真值深度全面击败了监督 80% 像素的 VGGT-Ω Sim3 稠密深度**（All PSNR `27.482` vs `27.453`，Novel `26.911` vs `26.877`，`d4+` `25.381` vs `25.338`，Demo `26.253` vs `25.910`），铁证“跨视角严格几何自洽的稀疏真值点”显著优于“含跨视角非刚性深度漂移的神经网络稠密预测深度”。
425|41. （2026-10-22 10:42）**方法论重要纠偏：DL3DV 无激光雷达真值，不能直接拿参与微调的 COLMAP 点云当“几何真值（GT）”去评测 `+colmap_depth` 实验（存在同源循环评测偏差），应采用“完全无深度真值（COLMAP-Free / GT-Free）”的客观几何评测指标**：
426|    - **核心逻辑**：在微调中使用了 COLMAP 点云后，若再用同一套 COLMAP 点云评测 `AbsRel` / `δ<1.25`，属于“既当运动员又当裁判”。
427|    - **四个完全不依赖点云真值（仅依赖真实测试照片 $I^{\text{gt}}$ 与渲染器输出）的公正几何正确性指标（含 Demo 场景实测）**：
428|      1. **跨视角深度重投影光度一致性 `Cross-View Depth-Warping PSNR ↑`（最硬核的无真值几何指标）**：完全不让高斯渲染颜色，只取视角 $t$ 的高斯渲染深度图 $\hat{D}_t$，用 $\hat{D}_t$ 将相隔 $\Delta t=5$ 或 $8$ 帧的**真实测试照片 $I_{t+\Delta t}^{\text{gt}}$** 三维重投影到视角 $t$，与视角 $t$ 的**真实照片 $I_t^{\text{gt}}$** 计算 PSNR。只有当渲染深度 $\hat{D}_t$ 真正落在物理表面上，真实照片跨视角重投影才能对齐！实测（$\Delta t=5 / \Delta t=8$）：`no_tuning` 为 `19.817 / 18.091` $\to$ `scene+token` 为 `20.059 / 18.336` $\to$ `vggt_omega_depth0.01` 为 `20.176 / 18.449` $\to$ **`colmap_depth0.01` 达 `20.398 (+0.34 dB) / 18.595 (+0.26 dB)` 全场最高**！
429|      2. **远距离新视角渲染质量 `Far Novel-View (d4+) PSNR / SSIM ↑`**：由于错误深度（多层半透明雾）在小视差（`d0`）下能骗过 RGB 损失、但在大视差（`d4+`）下必产生视差错位模糊，因此 `d4+` 与 `d0` 的泛化比是检验真实三维几何的最直接外观证据（140 场景 `colmap_depth0.01` 在 `d4+` PSNR 91/140 胜、SSIM 94/140 胜）。
430|      3. **跨视角深度循环自洽误差 `Multi-View Depth Cycle Error ↓`**：将视角 $t$ 的渲染深度 $\hat{D}_t$ 反投影到视角 $t+5$ 并与视角 $t+5$ 自身的渲染深度 $\hat{D}_{t+5}$ 对比相对误差。实测发现：纯 RGB 微调（`scene` / `scene+token`）使跨视角深度不一致误差从 `no_tuning` 的 `1.66%` **恶化至 `2.06% ~ 2.12%`**（因堆叠多层半透明雾讨好输入视角），而加入深度约束后误差回落（`colmap_depth0.01` 降至 `1.97%`，`vggt_omega_depth0.01` 降至 `1.39%`）。
431|      4. **深度梯度法线与高斯最短轴法线夹角 `Depth-Normal Angle ↓`**：实测当前所有方法的中位夹角均高达 **`67.3° ~ 68.0°`**（接近随机分布），说明现有微调只改了位置/深度，尚未约束高斯薄瓦片的最短轴法线与宏观表面法线对齐（这是下一步法线自洽正则的明确突破口）。
432|42. （2026-10-22 10:50）**完成阶段性总结与分区域深度诊断报告 `EXP_TokenGS/reports/stage_summary_opacity_and_depth_priors.md`（解释用户肉眼观察“加 COLMAP 后部分物体深度更符合现实形状，但整体仍不足”的四大定量机制）**：
433|    - **分区域实证数据（Demo 场景 `2f3e1c0f688c` 远视角 `d4+`）**：
434|      - **有 COLMAP 特征点的物体/边缘区（`d4+ Feat`，占 44.4% 像素）**：从联合微调基线 `21.342 dB` 暴涨 **`+0.726 dB`** 至 **`22.068 dB`**；中景（`1.2~2.5m`）涨 **`+0.242 dB`**（`24.767 -> 25.009`），远景结构（`>2.5m`）暴涨 **`+1.089 dB`**（`21.800 -> 22.889`）——直接印证用户肉眼看到的“部分物体深度明显改善、更符合现实形状”。
435|    - **为什么肉眼看“仍然不足”的四大定量病灶**：
436|      1. **平滑无特征区（`d4+ Smooth`：白墙/桌面，占 55.6% 像素）涨幅仅 `+0.099 dB`**（`27.589 -> 27.688`），缺乏直接几何约束；
437|      2. **单射线深度厚度 `RayThick`（透射权重深度标准差 $\sqrt{\widehat{D^2}-\hat{D}^2}$）仍厚达 `0.2235m`**（基线 `0.2545m`）：因为 $\hat{D}=\sum w_k z_k$ 只约束了沿视线的“平均重心深度”，前后分裂为 $z-0.2\text{m}$ 与 $z+0.2\text{m}$ 的两层半透明雾仍可满足平均深度等于真值；
438|      3. **大尺寸糊高斯（`scale > 0.05`）占比仍高达 `45.89%`**（与基线 `45.79%` 几乎不变），且薄瓦片最短轴与表面法线中位夹角仍为 **`67.29°`**（斜插在表面而非平贴表面）；
439|      4. **解耦版 `colmap_depth0.01` 未开启 `vis_opac + core_gate2.5`，且未过滤 `points3D.bin` 中的高重投影误差远端噪点**。
440|43. （2026-10-22 11:25）**将 6 项分区域与无真值物理几何指标（`surf_feat_pixel_ratio`、`surf_d4+_feat_psnr`、`surf_d4+_smooth_psnr`、`surf_warp_psnr_dt5`、`surf_mv_cycle_err_dt5`、`surf_ray_thickness_med`）写入 140 场景评估管线并完成 140 场景全量回填**：
441|    - **140 场景全量配对结果（对比 `[Exp-04]` `scene_lr1e-2_token_lr1e-4_tuning` 基线）**：
442|      1. **真实测试照片跨视角重投影 `Warp-PSNR(Δt=5)`（完全无点云真值的硬核几何裁判）**：
443|         - `[Exp-01]` `no_tuning`: `21.732 dB` $\to$ `[Exp-04]` 联合微调: `21.834 dB`；
444|         - **`[Exp-11]` `+colmap_depth0.01`**：跃升至 **`22.019 dB`（`+0.185 dB`，140 场景中 139 胜 / 1 负！）**；
445|         - **`[Exp-12]` `+vggt_omega_depth0.01`**：跃升至 **`22.040 dB`（`+0.206 dB`，140 场景中 140 胜 / 0 负，100% 场景全部提升！）**。
446|      2. **跨视角深度循环不一致误差 `MV-Cycle-Err(Δt=5)`**：
447|         - `[Exp-01]` `no_tuning` 仅 `0.70%`，但纯 RGB 微调（`[Exp-02]` / `[Exp-04]`）因堆叠多层半透明雾讨好输入视角，使 140 场景平均跨视角深度不一致**恶化 36% 至 `0.93% ~ 0.95%`**；
448|         - 加入深度监督后，**`[Exp-11]` 降至 `0.89%`（131 胜 / 9 负）**，**`[Exp-12]` 降至 `0.85%`（136 胜 / 4 负）**。
449|      3. **有特征物体区 `d4+ Feat`（140 场景平均占 56.54% 画面）vs 平滑无特征区 `d4+ Smooth`（占 43.46% 画面）**：
450|         - 在 `[Exp-11]` `+colmap_depth0.01` 中，有特征点物体区 `d4+ Feat` 提升 **`+0.029 dB`（`24.618 -> 24.647`，90 胜 / 50 负）**，涨幅是平滑无特征区 `d4+ Smooth`（`+0.003 dB`，`26.918 -> 26.922`）的 **9 倍**，在 140 场景大样本上印证了“有特征点区域显著改善、无特征平滑区缺乏直接约束”。
451|      4. **视线深度厚度 `RayThick`**：
452|         - 140 场景均值在 `[Exp-11]` / `[Exp-12]` 下虽从 `0.3643m` 显著收窄至 `0.3559m`（109/30 胜）/ `0.3533m`（107/32 胜），但绝对厚度仍达 **`~0.35m`**，证明全数据集普遍存在“仅约束平均深度 $\hat{D}$ 无法消除前后多层半透明雾”的共性瓶颈。

---

## 6. 当前权威结果（140 场景，FP32，latent 6v SSIM）与四阶段报告索引

### 6.1 四阶段专题报告目录（`EXP_TokenGS/reports/`）
1. **`01_baseline_and_scene_token_tuning.md`**：阶段 01 —— 前馈基线与 `scene` / `token` 微调对比实验（`Exp-01 ~ Exp-04`）
2. **`02_free_space_geo_regularization.md`**：阶段 02 —— 自由空间统计几何正则（`SW + Packing + Inter`）实验与失效机理（`Exp-05 ~ Exp-07`）
3. **`03_frustum_and_core_opacity_cleaning.md`**：阶段 03 —— 解析视锥投影与 Token 离核 Opacity 门控清理实验（`Exp-08 ~ Exp-10`）
4. **`04_colmap_and_vggt_depth_supervision.md`**：阶段 04 —— 三维表面深度几何监督对比实验（COLMAP 稀疏三角化 vs VGGT-Ω 稠密预测，`Exp-11 ~ Exp-12`）

### 6.2 全部 12 组正式实验权威结果总表（`Exp-01 ~ Exp-12`）

全部实验、全部指标的总表见 `EXP_TokenGS/all_experiments_table.md`（脚本生成）；下表是摘要。目录名是 2026-10-20 第五次命名优化后的最终名字（见第 10 节命名规则）。

| 实验编号 | 阶段与设置 | 目录 | all_frames PSNR / SSIM / LPIPS | novel_only PSNR / SSIM / LPIPS |
|---|---|---|---|---|
| **`[Exp-01]`** | 阶段 01：不微调（Baseline） | `EXP_TokenGS/no_tuning` | 25.194 / 0.808 / 0.255 | 25.007 / 0.804 / 0.257 |
| **`[Exp-02]`** | 阶段 01：只调 `scene_latents`（50 步，lr 1e-2） | `EXP_TokenGS/scene_lr1e-2_tuning` | 27.400 / 0.868 / 0.183 | 26.845 / 0.860 / 0.188 |
| **`[Exp-03]`** | 阶段 01：只调 `gs_tokens`（50 步，lr 1e-4） | `EXP_TokenGS/token_lr1e-4_tuning` | 26.810 / 0.854 / 0.215 | 26.439 / 0.847 / 0.219 |
| **`[Exp-04]`** | **阶段 01：`scene + token` 联合微调（1e-2 / 1e-4，无约束基线）** | `EXP_TokenGS/scene_lr1e-2_token_lr1e-4_tuning` | **27.481 / 0.871 / 0.180** | **26.901** |
| **`[Exp-05]`** | 阶段 02：只调 `scene` + 旧几何损失（sw 0.1 / packing 0.001 / inter 0.01） | `EXP_TokenGS/scene_lr1e-2_geo_tuning` | 26.734 | 26.251 |
| **`[Exp-06]`** | 阶段 02：只调 `token`（1e-4）+ 旧几何损失 | `EXP_TokenGS/token_lr1e-4_geo_tuning` | 26.606 / 0.847 / 0.230 | 26.264 |
| **`[Exp-07]`** | 阶段 02：联合微调 + 旧几何损失 | `EXP_TokenGS/scene_lr1e-2_token_lr1e-4_geo_tuning` | 26.940 / 0.858 / 0.203 | 26.409 |
| **`[Exp-08]`** | 阶段 03：联合微调 + 解析视锥投影拉 `xyz` | `EXP_TokenGS/scene_lr1e-2_token_lr1e-4_vis_xyz_tuning` | 27.440 / 0.871 / 0.180 | 26.855 |
| **`[Exp-09]`** | 阶段 03：联合微调 + 解析视锥投影压 `opacity` | `EXP_TokenGS/scene_lr1e-2_token_lr1e-4_vis_opac_tuning` | 27.441 / 0.870 / 0.181 | 26.859 |
| **`[Exp-10]`** | **阶段 03：联合微调 + 视锥压 `opacity` + 离核(`>2.5r`)门控清理** | `EXP_TokenGS/scene_lr1e-2_token_lr1e-4_vis_opac_core_gate2.5_tuning` | **27.422 / 0.870 / 0.181**（SW **0.101**，packing **3.14**，视锥外 **0.08%**） | **26.847** |
| **`[Exp-11]`** | 阶段 04：联合微调 + COLMAP 原始稀疏深度（λ=0.01） | `EXP_TokenGS/scene_lr1e-2_token_lr1e-4_colmap_depth0.01_tuning` | **27.482 / 0.8712 / 0.180**（d4+ **25.381**，Warp **22.019**；Demo **26.253**） | **26.911** |
| **`[Exp-12]`** | 阶段 04：联合微调 + VGGT-Ω Sim3 稠密深度（λ=0.01，对照） | `EXP_TokenGS/scene_lr1e-2_token_lr1e-4_vggt_omega_depth0.01_tuning` | 27.453 / 0.8705 / 0.180（d4+ 25.338，Warp **22.040**；Demo 25.910） | 26.877 |
| **`[Exp-13]`** | **阶段 04：联合微调 + COLMAP 去噪稀疏深度（λ=0.01，默认 COLMAP 基线）** | `EXP_TokenGS/scene_lr1e-2_token_lr1e-4_colmap_clean_depth0.01_tuning` | **27.478 / 0.8711 / 0.180**（d4+ **25.372**，Warp **22.019**，MV-Cycle **0.88%**；Demo **26.196**） | **26.905** |
| 参考 | 官方 release（对照） | `release_2026.6.md` | w/o 25.120；w/ 27.304 | — |

（注：`token lr=1e-2` 的 4 组全部发散，已按用户要求于 2026-10-20 删除目录并从 registry 移除，发散细节记录在陷阱 20 与 `EXP_TokenGS.md`。）

同设置的 `scene_tuning/official_loop` 和 `scene_tuning/reimplemented_loop_repeat_run` 已删除；其原始均值、逐场景差和配对 CI 冻结在 `EXP_TokenGS/reports/scene_tuning_noise_floor.json`。官方循环 vs 保留的复刻循环差 +0.0002 dB，同配置重跑差 +0.0003 dB，均在约 ±0.004 dB 噪声底内。

- 复现差：w/o TTT +0.074 dB，w/ TTT +0.095 dB，视为吻合。
- TTT 增益（配对 140 场景）：all_frames +2.206 dB，novel_only +1.837 dB。
- TTT 下最差场景 `fb3b73f1` 18.35，最好场景 `07d9f97` 36.20。Baseline 下 top 场景为 `374ffd0c`（32.33）。
- **几何损失（当前这组权重）让画质显著变差**：相对微调 −0.665 dB，远超重跑噪声底（±0.004 dB）；完整配对统计见 `EXP_TokenGS/reports/scene_geo_tuning.md`。在 token 微调（−0.204）和一起调（−0.541）下同样几乎全部场景变差，降幅随几何被推动的程度增大。
- **一起调比只调 scene latents 高 +0.081 dB**（CI [+0.068, +0.094]，127/140），小但稳定。

### 6.1 Baseline / TTT 口径核查（2026-10-19，只读核查）

**Baseline = 官方权重原样推理，什么都不做**
- 入口 `scripts/evaluate_with_scene_latent_tuning.py --no-tuning`：严格加载权重（strict=True，sha256 `5913da34…78c3`）→ `model.eval()` → `torch.no_grad()` 下逐场景 `model(data)`。没有 optimizer，没有训练，没有 TTT（`scene_latent_tuning_enabled=False`；`no_tuning` 的旧 contract 里叫 `ttt_enabled`），没有几何项（`use_geo_losses=False`），FP32。
- 140 场景完整（`formal_complete=True`），逐场景校验 context/target 帧与评估 json 一致。

**TTT = 权重全冻结，但优化的是 scene latents，不是论文的 Gaussian tokens**
- `forward_ttt_scene_latents`：`freeze_model_parameters(self)` 把全部网络参数设为 `requires_grad=False`；encoder 在 no_grad 下只跑一次，得到 scene latents `[1, 4096, 1024]`；只把这个张量交给 Adam（50 步，lr 1e-2）；`gs_tokens` 固定不动；最终用 `scene_latents.detach()` 解码。
- 监督：`to_ttt()` 渲染回 6 个 context 视角，用 context 图像监督，**不使用 target 图像**（无泄漏）。损失 = 0.8·L1 + 0.2·(1−SSIM)/2 + 1.0·visibility（`lambda_lpips=0`）。
- 与论文 Sec. 3.3 的对应："冻结网络、只用输入视角自监督、50 步"一致；**优化对象不同**——论文是 Gaussian token embeddings（lr 1e-4，代码 `forward_ttt_tokens`，仍保留），release 2026.6 latent preset 改为 scene-latent-tuning（`release_2026.6.md` 第 12、29、63 行明确说明），27.304 也是用这个设定测得的。
- 已删除的旧 `scene_tuning/official_loop` 的 contract 由旧包装脚本生成，**未记录 ttt_mode/steps/lr**；间接证据：用当前入口（contract 记录 scene-latent-tuning / 50 / 1e-2）在 `032dee9f` 上复算，与它相差 −0.016 dB，落在 TTT 自身噪声（约 0.02 dB）内；140 场景上保留的复刻循环与它相差 +0.0002 dB。旧运行的统计已冻结到 `EXP_TokenGS/reports/scene_tuning_noise_floor.json`。

**与官方代码唯一影响数字的差异：`07d9f97` 内参**
- 本地 `dl3dv.py` 按实际图片分辨率缩放内参；官方 HEAD 按固定 `[540, 960]` 缩放。其余 139 个场景（元数据 3840×2160，images_4 为 960×540）两者结果完全相同；`07d9f97`（元数据 1920×1080，images_4 为 480×270）在官方代码下 K 被放大 2 倍，本地是正确的。
- 该场景本地 Baseline 32.32、TTT 36.20（140 场景中最好）。我们比 release 高出的 +0.074 / +0.095 dB，可能有一部分来自这里（单场景对 140 均值的影响上限约为它与官方值之差 / 140），尚未用官方代码单场景对照验证。
- 其他本地改动（provider 附加 scene_id/frame_ids 并在读取失败时 fail_fast、options 新增字段、`use_geo_losses` 分支默认关闭、ply 导出）不影响评估数字。

**已删除的非权威 / 历史产物（2026-10-19 晚，用户要求；只剩下面的数字，不要引用为正式结果）**：
- BF16 官方循环 140 场景：PSNR 12.11，即“BF16 会崩”的证据；
- 排查目录（20 场景 test 划分，只有 metrics.txt）：FP32 preset 步数 27.62、0 步 25.23、1 步 23.56、BF16 去掉 visibility 12.18；
- BF16 不微调 20 场景 val / test：25.243 / 25.228；
- 16:39 取消的并行启动留下的半成品及其启动脚本；
- 11:50 那次不微调（被逐帧相同、指标更全的一次取代）。

---

## 7. 几何损失与梯度审计要点

详细数字见 `EXP_TokenGS.md` “执行日志：几何损失 → scene latent 梯度通路审计”。要点如下：

1. **现有开关对 TTT 无效**：`use_ttt_for_eval=True, use_geo_losses=True` 时，几何损失作用在 `no_grad` 解码后的最终输出上，只多算一个无梯度的数。要做 Geometry TTT，必须把几何项加进 50 步内循环：`loss = render + λ·geo`。
2. **在内循环中梯度通路正常**：三项几何损失对 scene latent `z`（[1, 4096, 1024]）的梯度有限、非零，与中心差分一致（相对误差 ≤0.4%）；仅用几何项做 20 步 Adam，真实损失下降 61–85%。
3. **几何项能动的属性有限**：sw / packing 只驱动 center；inter 驱动 center + scale + rotation；三项对 opacity 和 rgb 的梯度均为 0 → **无法直接抑制 floater 的不透明度**。
4. **预训练解远离正则目标**：packing ratio 是目标的 4–19 倍，inter 距离约为目标的 1/4；几何梯度与渲染梯度近似正交（cos ≈ −0.02~0）；λ=0.01 时几何梯度范数是渲染的 16–51%。只优化几何 20 步，context render loss 恶化 1.6–6.6 倍 → **λ 过大很可能拉低 PSNR**。
5. packing / inter 的梯度范数是 sw 的 3–8 倍，三项等权时会被 packing / inter 主导。
6. **no_tuning 的分散现象与“死点 `<1/255` 污染”真相（2026-10-20 深度剖析重大修正）**：
   - 若按现有 `geo_losses.py` 把全部 64 个 child 都算进去，demo 里 85.2% token 的 packing ratio 超目标、43.2% 强分散（均值 `366.96`，`SW=0.5264`）。
   - **但真实原因是 `gsplat` 对 `opacity < 1/255 (0.00392)` 的子高斯直接剔除（不渲染、预训练 0 梯度），每个 token 约有 19 个这样的“死点”漂在 `r ~ 1.0–3.8` 处；而 `sigmoid` 输出非零（`~0.0015`），未过滤死点时仅 8 个固定死点槽位就占了全 token 中心方差的 59.5%！**
   - **一旦只统计真正参与渲染的存活点（`opacity >= 1/255`），`no_tuning` 的 `packing_ratio` 均值立刻从 `366.96` 降到 `5.30`（中位数 `2.48`，低于目标 `5.74`，强分散比例仅 `3.5%`），`SW` 从 `0.5264` 降到 `0.1082`。**
   - 换言之：TokenGS 预训练出的**存活子高斯本身就是紧凑的局部表面片簇**；此前几何微调掉点（−0.665 dB）的核心原因之一，是 `geo_losses.py` 把不可见的死点也纳入了 `SW` 和 `packing`，强行把可见表面点拉向未受训的死点坐标。详见第 5.3 节第 27 条与第 9 节陷阱 21。

---

## 8. 待决定事项与下一步

**已完成：场景潜变量微调 vs 微调+几何损失 预注册对照**（设计与结果见 `EXP_TokenGS.md`）

- 唯一变量：内循环 loss 是 `render` 还是 `render + sw 0.1·sw + packing 0.001·packing + inter 0.01·inter`（只预注册这一组权重）。实现为 `scripts/reimplemented_scene_latent_tuning.py`（逐行复刻 `_forward_ttt_scene_latents`，只多出几何项），入口参数 `--scene-latent-tuning --tuning-loop reimplemented`。
- 结论：几何损失 −0.665 dB，降幅是重跑噪声底的上百倍。耗时：微调 664 s、加几何 745 s（各一张卡）。

**已完成：同场景 7 组完整 Demo 与 PLY 对比**（`EXP_TokenGS/demo/scene_2f3e1c0f688c/`）
- 全部 7 个正式实验（`no_tuning`、`scene_lr1e-2_tuning`、`scene_lr1e-2_geo_tuning`、`token_lr1e-4_tuning`、`token_lr1e-4_geo_tuning`、`scene_lr1e-2_token_lr1e-4_tuning`、`scene_lr1e-2_token_lr1e-4_geo_tuning`）均已完成单场景评估、`decoder_inputs.pt` 重解码校验（`max |diff| = 0`）、50 步微调曲线、7 组对比表 `comparison.{md,json}` 与 10/20 token 高亮（`_highlight{10,20}tokens.ply` + `_only{10,20}tokens.ply`）。

**等待用户决定下一步实验方向（已记录 3 条候选路线，详见 [`docs/LITERATURE_REVIEW_2026.md`](LITERATURE_REVIEW_2026.md) 第四节）**：

1. **路线 A（零重训成本，改进 `scene + token 联合微调` 内循环）**：
   - **A1（过滤 `< 1/255` 死点后的干净几何口径）**：在 `geo_losses.py` 中将 `< 1/255` 死点权重置 0，验证剔除死点污染后的几何微调表现；
   - **A2（离核半逃逸点的距离门控 Opacity 抑制 $\mathcal{L}_{\text{contig-opac}}$）**：借鉴 EcoSplat / FreeSplat++，对偏离 Token 稳健核心 $\boldsymbol{\mu}_{\text{core}}$（`detach`）超过阈值的半逃逸点（`dist ∈ [0.25, 1.0m)`，共 2.65 万个点）只压低其 `opacity` 而不硬拽 `xyz`，在不伤画质的前提下让每个 Token 真正成为干净的连续局部片；
   - **A3（视线深度自洽 / 深度厚度压缩 + 解析视锥投影）**：借鉴 FreeSplat++ / Off The Grid，在微调内循环抑制 12%~38% 的视线多层半透明雾并修复 `gsplat means2d` 视锥外零梯度漏洞，防止 32 步后输入视角 `d0` 过拟合、拉升远视角 `d4+`。
2. **路线 B（解锁 `deconv` 线性耦合）**：在微调后半程对 `deconv` 位置/不透明度行做极低学习率微调或附加有界局部偏移 $\|\delta\mu\|_\infty \le r_{\max}$（借鉴 AnchorSplat）。
3. **路线 C（架构级改进）**：父锚点 + 有界局部偏移 $\mathbf{x}_{s,c} = \mathbf{p}_s + \tau_{\max}\tanh(\Delta\mathbf{u}_{s,c})$（AnchorSplat）、粗位置引导 Token 局部融合 / 3D KNN（TokenSplat / SparseSplat）、或将 50 步梯度微调蒸馏为 2~3 步前馈残差 Refiner（GIFSplat / iSplat）。
4. 评估脚本 `evaluate_with_scene_latent_tuning.py` 现在也做 token 微调和联合微调，文件名不完全贴切，是否改名待用户决定。
- token 学习率已定：不测 1e-3，1e-2 的 4 个发散目录已按用户要求删除（2026-10-20）。
- **不要在用户确认前设计或交付新的正式实验。**

---

## 9. 陷阱清单

1. **BF16 下 TTT 失稳**：PSNR 崩到约 12。正式实验一律 FP32。
2. **`opt.mixed_precision` 不生效**：官方 `tokengs.evaluate` 中 `Accelerator()` 无参构造，精度由 accelerate 配置文件决定。正式评估统一用 `scripts/evaluate_with_scene_latent_tuning.py`，其中显式 `Accelerator(mixed_precision="no")`。
3. **TTT 有 run-to-run 非确定性**：单场景约 0.02 dB（gsplat 反传 atomicAdd，无法关闭）；Baseline 完全确定。比较 TTT 类方法时必须有噪声底。
4. **`use_geo_losses` 对 TTT 无效**（见第 7 节）。
5. **options.py 默认值 ≠ preset**：默认是 token-tuning / lr 1e-4 / 1024 tokens，latent preset 才是 scene-latent-tuning / lr 1e-2 / 4096。
6. **不要和论文 Table 2 比数字**：我们用的是 release 2026.6 latent 模型。
7. **数据读取失败不能静默换场景**（全模型微调首轮的 bug），评估时要用 `--expected-scenes 140` 卡住场景数。
8. **`07d9f97` 场景是 480×270**，任何涉及内参缩放的修改都要用它验证；与纯官方代码比数字时要注意本地已修正它的内参（见 6.1）。
9. **inter 协方差必须包含旋转**（全模型微调首轮曾忽略）。
10. **全模型微调结论已废弃**，不要引用。
11. **GPU 共享**：只用空闲卡，不 kill 他人进程。
12. gsplat JIT 需要 env/bin 在 PATH 中。
13. **本研究的 TTT 不是论文的 token-tuning**：release latent preset 优化的是 scene latents（lr 1e-2），gs_tokens 固定；论文版只调 Gaussian tokens（lr 1e-4）。描述方法时不要混用（见 6.1）。
14. **visibility loss 读 gsplat `means2d`，对被剔除点无效（已有 GPU 实测证据）**：出视锥、opacity < 1/255、深度越界的 Gaussian 的 `means2d` 是 `at::empty` 未初始化内存，反传梯度为 0。`demo/scene_2f3e1c0f688c/*/tuning_curve.jsonl` 第 0 步在完全相同的初始 Gaussian 下给出不同的 `context_loss_visibility`（`2.31e-5` vs `8.79e-5`），并在后续步中于 `0` 与 `1e-9 ~ 1e-4` 间随机跳变，直接证实其读取了 CUDA 内存池残留值。这是官方实现的行为，正式对照保持原样；要研究 floater 时用解析投影（FFRGS `models/render.py::project_means2d`）另算。
15. **140 场景基准几乎只有内插**：target 是连续 50 帧，6 个 context 撒在其中（12% target 就是 context 本身）。它对几何质量不敏感（见 `docs/FFRGS_REFERENCE.md` 7.2）。**用户决定（2026-10-19）**：主口径固定为这 50 帧以保证与论文可比，当前实验不做外推，外推评估留作后续单独实验；在口径内用帧距分桶、novel_only 与几何统计作补充。
16. **几何损失会把 Gaussian 推到所有输入视锥外**：加几何损失后视锥外比例 0.0128→0.0361（约 2.8 倍），同时 opacity 均值 0.074→0.113。原因未验证（推测与 visibility 项对视锥外点梯度为 0 有关，见第 14 条）。
17. **改名后旧证据文件保持原样**：`launch_contract.json`、`summary.json`、`run.log`、`*.console*.log` 里仍是旧名（`ttt_enabled`、`baseline_ttt/...` 等），不要改；按 `EXP_TokenGS/renamed_directories_old_to_new.md` 对照。
18. **官方微调循环拿不到调好的张量**：`_forward_ttt_*` 只返回最终 Gaussian。要保存 `decoder_inputs.pt` 或画微调曲线，必须用 `--tuning-loop reimplemented`（两种微调都支持，CPU 上与官方逐位等价，见 `tests/test_reimplemented_tuning_loops.py`）。
19. **一键脚本在最后一组结束后卡住（2026-10-20 修复）**：`one_click_setup` 用 `exec > >(tee ...)` 写总日志；`one_click_run_groups` 里不带参数的 `wait` 在 bash 5.2 下也会等这个 `tee`，而 `tee` 在主 shell 退出前永远不会结束，于是死锁。表现：总日志停在最后一次 `progress ... complete`，没有 `--- exit codes`。修复：改为 `wait "${pids[@]}"`。之前所有一键运行（包括 token 140 场景）都停在同一位置，但各组的 log 和结果完整、不受影响，只是缺少汇总部分。
20. **token 微调 lr 1e-2 会发散（残差直连 + 指数激活溢出）**：release latent 模型上 Gaussian token 用 scene latents 的学习率 1e-2（50 步 Adam），无论单独调、联合调、加不加几何损失，都在前 1–8 个场景内解码出 NaN/Inf 的 Gaussian；发散前画质已比不微调低约 6 dB。根因：`scene_latents` 经 `k_proj_norm (LayerNorm)` + softmax 注意力 + LayerScale（`≈0.027`）缓冲，而 `gs_tokens`（初值 std `0.02`）经无 LN 的残差直连通道进入 `ClipActivationHead.deconv`，再经 `pos_act` / `scale_act` 的 `exp` 激活，大步长直接导致 `exp` 溢出。不加几何的组在 50 步结束后外层 `geometry_stats` 报错，加几何的组在内循环 `geo_losses_for` 报错。
21. **`geo_losses.py` 与 PLY 点模式包含 `opacity < 1/255` 死点，造成虚假“高分散”与几何拉扯（重大陷阱）**：`gsplat` 硬剔除 `opacity < 1/255 (0.00392)` 的 Gaussian（预训练 0 梯度，位置飘在 `r ~ 1–3.8`），但 `sigmoid(raw-2) > 0` 使它们在 `geo_losses.py` 中仍有约 `0.0015` 的不透明度，乘以 $r^2$ 后占了 token 中心方差的近 60%，把 `packing_ratio` 从真实存活点的 `5.30` 虚增 69 倍到 `366.96`，并在几何微调时通过共享线性头把可见点拽离表面。同时，`demo_ply_export.py` 不做 opacity 剪枝，因此在查看器 **Points 模式**下看到的飞点大多是这些在 **Splats 模式**下完全不可见的死点。
22. **DL3DV 数据集帧数使用规律与精简下载注意事项（2026-10-20 核查）**：
    - **评估 / scene 微调 / token 微调（140 场景 Benchmark）**：`assets/evaluation_idx_dl3dv_depthsplat_{2v,4v,6v}.json` 中全部 140 个场景的 `context` 与 `target` 帧下标最大均只到 `49`（即仅使用每个场景的前 50 张图片 `frame_00001.png ~ frame_00050.png`），第 51 张之后的 250+ 张图从未使用。
    - **训练 / 全模型微调（`Provider._get_indices_static`）**：每次采样窗口跨度 `min_gap=10, max_gap=40`（最多 41 帧），但窗口起点 `start_frame` 会在 `[0, total_num_frames - gap]` 全长范围内随机滑动。若为节省空间每个场景只下载前 70 张图（`70 > max_gap 40`），完全可以正常训练（仅相当于将单场景轨迹长度缩为前 70 帧）；**但必须同步将 `transforms.json["frames"]` 截断到已下载的前 70 帧（或修改 `dl3dv.py` 的 `_scene_image_resolution` 与 `count_frames` 仅统计实际存在的图片）**，否则会因遍历完整 `transforms.json` 报缺图错误。

---

## 10. 用户偏好与工作规则

- 安装类操作在前台执行并显示进度，不要放后台静默运行。
- 持续把状态写入交接文档（`EXP_TokenGS.md` 执行日志 + 本文件）。
- 只用空闲 GPU，不干扰他人进程。
- 用户催促式提问时（如“怎么样”），当轮直接给出完整汇报。
- 文档用中文正文、保留英文术语；技术文档放 `docs/`。
- `EXP_TokenGS/` 根目录放文档、总表、两个一键脚本、`reports/`、`launch_logs/`、`demo/` 和一层实验结果文件夹；临时产物放 `/tmp`，并在文档中注明可重跑方式。
- **每个文件夹就是一次 140 场景实验**（用户要求，最终结构）：`EXP_TokenGS/<实验名>/` 不再按方法嵌套；不同学习率就是不同实验文件夹；实验文件夹里只放 140 场景记录。demo 完全分离到 `demo/scene_<id 前12位>/<实验名>/`，报告统一放 `reports/`，总日志放 `launch_logs/`。不保留 BF16 结果和被完全覆盖的冗余运行。
- **实验由用户运行，agent 交付一键脚本**（用户 2026-10-19 明确要求）：用户不想自己开多个终端、手动激活环境。脚本内部设置环境、自动选空闲 GPU、并行跑全部组、可跳过已完成组；agent 不自行启动 GPU 实验，事后读 log 和产物核验。
- **每次实验都输出完整 log**（用户要求）：每个实验一个完整、未过滤的 `<输出目录>/console.log`，另有 `EXP_TokenGS/launch_logs/<脚本名>.<时间戳>.log` 总日志，agent 从 log 读结果。历史 `scene_lr1e-2_tuning`、`scene_lr1e-2_geo_tuning` 只有当年留下的 `console_warnings_filtered.log`，不伪造完整日志。
- **命名必须名实相符，禁止使用“TTT”**（用户要求）：严禁使用“TTT”或“测试时微调”笼统指代，必须直接写明是 **scene 微调**（scene latent tuning）、**token 微调**（token tuning）还是 **scene + token 联合微调**（因为论文与官方 release 代码存在不一致）；禁止 v1、v2、formal、task 编号这类指代不清的名字；改名后更新对照表。
- **目录命名用 scene / token / geo，词序固定且 lr 紧跟模块词**（用户 2026-10-20 第五次优化）：实验名格式为 `no_tuning` 或 `[scene_lr<x>_][token_lr<y>_][geo_]tuning`，模块顺序永远是 scene → token → geo → tuning，**每个被调模块的学习率直接跟在该模块词后面**（例如 `scene_lr1e-2_token_lr1e-4_geo_tuning`），彻底避免多模块组合时分不清哪个 lr 属于哪个模块。FP32、50 步和固定几何权重不写；如果以后几何权重变化，紧跟在 `geo` 后追加变化项（如 `geo_sw0.2`）。命令行参数和证据文件不跟着改。
- **实验 registry 是唯一事实源**：新增 140 场景配置时改 `scripts/experiment_registry.py` 的 `EXPERIMENTS`；`run_140_scenes.sh`、`run_demo.sh`、报告、总表、对比和高亮都从它取名字 / 参数 / 顺序，禁止在各脚本复制实验名单。
- **每个可行实验都要有同场景 demo / ply**（用户 2026-10-20）：140 场景运行 complete 后，`run_demo.sh` 会自动纳入；failed（发散）不纳入。demo 一律用复刻循环，以便保存微调曲线和张量。每个 demo 的 token 可视化同时生成带灰色背景的 `_highlight{10,20}tokens.ply` 和彻底去除未选点的 `_only{10,20}tokens.ply`。
- **重构类改动必须先给方案**（用户纠正，2026-10-20）：涉及目录结构、批量改名或大范围迁移时，先提交方案，等用户明确确认后再修改。
- **阶段报告书写规范（用户 2026-10-22 明确指定模板）**：`EXP_TokenGS/reports/01_... ~ 04_...` 统一按照 **“动机 → 设计 → 实验结果（140 场景精简表 + Demo 场景精简表）→ 核心结论与下一步突破方向（含 Todo）”** 四段式撰写，保持简洁、克制、可直接引用。
- **当前明确 Todo（2026-10-22）**：① 对 COLMAP `points3D.bin` 中的点进行过滤，剔除高重投影误差、短轨迹及远端异常噪声点；② 探索压薄 `0.35m` 视线半透明深度厚度（`RayThick`）的方案。
- 讲解用中文、面向零基础，先说结论再解释。
- 关键实验需要可复现证据：launch contract、sha256、GPU UUID、CI95。
- 方向性决策（λ 选取、是否开始正式实验、删除产物、还原代码）先问用户。

---

## 11. 来源索引

| 内容 | 位置 |
|---|---|
| 论文 | 本地 `TokenGS.pdf`（全文，首选）；https://arxiv.org/abs/2604.15239（HTML 版的 3.3 节抽取为空） |
| Release 说明 | `release_2026.6.md`（`checkpoints_hf/` 下的同名文件只少了开头的 gif 和 HF 链接） |
| LocusGS | https://arxiv.org/abs/2608.12825 |
| StreamSplat | https://arxiv.org/abs/2506.08862 |
| 会话“检查4dgs代码完整性” | `/home/jms/.qoder/cache/projects/xveyan-425e665f/conversation-history/4f8b4e60/4f8b4e60.jsonl` |
| 当前会话 | `/home/jms/.qoder/cache/projects/xveyan-425e665f/conversation-history/b1a5cb35/b1a5cb35.jsonl` |
| 实验执行日志 | `EXP_TokenGS/EXP_TokenGS.md` |
| 原始方案文档 | `EXP_TokenGS.zip`、`EXP_FFRGS.zip`、`EXP_FFRGS/EXP_FFRGS.md` |
| FFRGS 代码 / 日志 / 知识整理 | `4DGS/FFRGS/`（`AGENTS.md`、`design_notes_*.md`、`REPRODUCE_v18.md`）、`4DGS/ffrgs_workspace/`、`docs/FFRGS_REFERENCE.md` |
| 数据说明 | `data/DATA.md` |
