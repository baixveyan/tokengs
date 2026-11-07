# LocusGS 诊断：解决了什么、暴露了什么（2026-11-05）

## 设置

- 模型（全部冻结、FP32、TF32 关闭，不训练）：
  - `locusgs_finetune_4v`：LocusGS DL3DV 微调权重（4096 个 token × 64 个高斯，4 视角训练，256×448），`/mnt/disk/xveyan/checkpoints/locusgs/checkpoints/dl3dv/finetune/model.safetensors`，SHA256 `a554f552…`。从用户下载的 `checkpoints-20261105T024314Z-1-00{1,2}.zip` 中只解出 DL3DV 的 base / finetune 两个权重，压缩包没有改动。
  - `tokengs_paper_4v`：官方**论文版** TokenGS `checkpoints_hf/checkpoints/dl3dv_4v.safetensors`（SHA256 `0d6012d1…`），3 层编码器 + 12 层解码器，4 视角训练。LocusGS 论文表 1 中的 TokenGS 就是这个权重（见 `4DGS/LocusGS/scripts/dl3dv/tokengs_official.sh`）。它和 LocusGS 的结构、token 预算、训练视角数都相同，是**对等对照**。
  - `tokengs_release_latent_6v`：此前诊断的 release 隐变量版（12 层编码器 + 4096 个场景隐变量 + 1 层解码器，6 视角训练）。结构和训练都不同，只作参照。
- 代码：LocusGS 和论文版 TokenGS 都用 `4DGS/LocusGS` 仓库里的模型类加载（LocusGS 权重里多存了一组 `anchor_head_cross_attn`，这组参数只在 `learned_positional` 变体下才会用到；权重里含 `geo_scale`，说明它按 `geometric_positional` 训练，所以这组多出来的参数没有用，已核对没有缺失的键）。数据沿用 TokenGS 仓库的 ZIP 加载器，归一化方式与各模型的训练一致：LocusGS 和论文版用 first_cam，release 用 mean_cam，缩放统一为 0.15。
- 为了拿到注意力，诊断时重放了解码器，最后一层用手写 einsum 代替 SDPA。它和标准前向的差：PSNR 在小数点后四位一致，xyz 的 p99.9 差约 1e-4。
- DL3DV：140 个场景，官方 6 视角索引，口径同 `diag_dl3dv_report.md`，几何参照为 COLMAP 稀疏点，只报相对指标。脚本 `scripts/diag_dl3dv.py --model …`，一键脚本 `EXP_DL3DV/run_diag_dl3dv.sh`，结果 `diag_dl3dv_summary_<模型>.json`、`diag_dl3dv_probe_<模型>.json`。
- ScanNet++：50 个验证场景、激光真值 G1–G4，协议同 `evaluate_scannetpp.py`，只是相机和网格按 first_cam 归一化（脚本 `scripts/evaluate_scannetpp_query_models.py`），结果在 `EXP_ScanNetPP/baseline_locusgs_dl3dv_4v/` 和 `EXP_ScanNetPP/baseline_tokengs_paper_dl3dv_4v/`。尺度门限：中位深度比 1.020 / 1.036，在 [0.90, 1.10] 内。

## 画质

| | 论文版 TokenGS 4v | LocusGS | release 6v |
|---|---:|---:|---:|
| DL3DV PSNR（50 个目标） | 23.77 | **25.29** | 25.19 |
| LocusGS 论文表 1（6 视角） | 24.16 | 25.78 | — |
| ScanNet++ PSNR / LPIPS（零样本） | 28.13 / 0.237 | 29.52 / 0.195 | **30.50 / 0.180** |

我们测出的绝对值比 LocusGS 论文低约 0.4–0.5 dB，可能是因为论文用 images_8（480×270），我们用 images_4（960×540）再缩放。但 +1.5 dB 的差距复现了。

## LocusGS 解决了什么（和对等的论文版 TokenGS 比）

| 问题 | 论文版 TokenGS 4v | LocusGS | 说明 |
|---|---:|---:|---|
| 同一个 token 的 64 个高斯散开（LocusGS 论文表 3 的 C_centroid，归一化单位，均值 / 中位） | 0.689 / 0.622 | **0.040 / 0.021** | 论文报告 0.588 / 0.553 → 0.043 / 0.038，**复现** |
| 厚度里来自 token 内部散开的部分（DL3DV） | **38%** | 0.8% | 论文版 TokenGS 确实有大量 token 内部散开，LocusGS 的归因对它成立 |
| 透明且远离表面的休眠槽位（DL3DV） | 47.5% | **5.3%** | release 为 15.5% |
| 在 ≥99% 的 token 里都死亡的子槽位 | 14 个 | **0 个** | release 为 8 个 |
| 最后一层注意力的局部头（16 个头中） | 4.8 | **15.9** | 有效图像块约 6 个 |
| token 注意力的 90% 集合（中位） | 64 块以上（前 64 块只占 52% 的注意力） | **9 块** | 每个图像块被 21 个 → 3 个 token 覆盖 |
| token 高斯沿视线 / 横向的展开比 | 3.39 | **1.20** | release 为 1.35 |
| token 中心相对深度误差，中位 / p90（DL3DV，对 COLMAP） | 0.117 / 0.455 | **0.024 / 0.242** | 和 release 相当（0.023 / 0.225） |
| 读出头丢几何：探针 vs 实际输出（相对深度误差中位） | 探针 0.089 **优于**输出 0.114 | 输出 0.023 优于探针 0.083 | 论文版 TokenGS 上“读出头丢几何”成立；LocusGS 消除了这个差距 |
| ScanNet++ 激光：AbsRel / Accuracy / F@5cm / 法线中位 | 0.188 / 19.9 cm / 0.397 / 39.4° | **0.131 / 16.3 cm / 0.493 / 30.4°** | 也优于 release（0.174 / 22.4 cm / 0.445 / 37.3°） |
| ScanNet++ 激光：视线厚度中位 / p90 | 20.5 / 48.2 cm | **12.9 / 35.0 cm** | |

## 新暴露 / 没解决的问题

1. **厚度全部变成了 token 之间的叠放。** DL3DV 上，词元间方差占比在论文版 TokenGS 是 0.62，在 LocusGS 是 **0.992**（逐像素中位 0.96）。每个 token 自己收得很紧，但多个 token 在同一条视线上前后叠放；体素合并照样不能变薄（0.992 不变，PSNR 25.10 → 22.20）。
2. **厚度的长尾仍然很重，而且比 release 厚。** 厚度 / 深度：中位 0.10，p90 **0.42**（论文版 0.135 / 0.40，release 0.057 / 0.25）。ScanNet++ 激光上的视线厚度为 12.9 / 35.0 cm，release 为 8.6 / 25.7 cm。
3. **锚点严重扎堆（这就是你担心的“多个 token 落在同一个面上”）。** 每个锚点自身半径内，中位有 **52 个**其他锚点（p90 234 个）；支撑球与其他 **241 个**锚点重叠（p90 812 个）；最近的锚点只在 0.17 倍半径处。“每个 token 负责一块局部区域”的分工并没有形成，而是很多 token 挤在同一片区域。
4. **画的范围远大于看的范围。** 渲染权重落在注意力 90% 区域内的比例：中位 **0.57**，按渲染量加权 **0.47**（论文版 1.00 / 0.96，release 0.84 / 0.79）。每个 token 只看约 9 个图像块，却画到这些块以外的地方。注意，论文版的 1.00 部分来自它的注意力本身很散。
5. **半径不能约束子高斯，也不是不确定度。** 存活子高斯离锚点的距离 / 半径：中位 0.45，p90 **2.1**，p99 **4.3**；10% 的 token，所有存活高斯都在 1 倍半径之外。半径 / 深度与视差的 Spearman 为 **−0.14**（低视差处半径并不更大，和论文的定性描述不符），与误差的相关只有 0.17。
6. **锚点本身不是好的几何估计。** 锚点的相对深度误差中位 0.045 / p90 0.38，比它自己的高斯中心（0.024 / 0.24）更差；锚点到高斯中心的距离 p90 达到深度的 33%。
7. **浪费从“子槽位”转到了“整个 token”。** 在输入视角里渲染权重不足 1 个像素的 token 占 5.5%（论文版 0.15%）。
8. **深度长尾。** DL3DV 上渲染深度对 COLMAP 点的 AbsRel：中位更好（0.046，论文版 0.060），均值反而更差（0.205，论文版 0.185，release 0.156）。说明有少数像素错得很离谱。ScanNet++ 激光上的 AbsRel 均值是好的（0.131），这一条可能和 DL3DV 的场景或评测口径有关，需要再核对。
9. **单视角 token 变多。** 注意力更局部之后，只看一个视角的 token 占 12%（论文版 0.5%），视差 < 2° 的占 25%。误差仍随视差变大：单视角 p90 0.41，>20° 为 0.18。

## 结论

- LocusGS 解决的是论文版 TokenGS 的 **token 内部问题**：高斯散开、死槽位、注意力不聚焦、读出头丢几何。几何和画质都明显变好。
- 它没有处理、并且把问题集中暴露出来的是 **token 之间的组织**：锚点扎堆、支撑互相重叠、多个 token 在同一视线上叠放，造成厚度长尾。这和我们在 release 版上的结论（厚度主要来自词元间叠放）一致，说明这是查询式前馈 GS 的共性问题，而不只是某个版本的缺陷。
- ⑫（ScanNet++ 上加激光监督微调的 release 版）的几何仍然远好于 LocusGS（AbsRel 0.066，厚度 5.6 / 13.6 cm，Accuracy 6.9 cm），但它用了 ScanNet++ 训练，不能和 LocusGS 的零样本结果直接比较。
