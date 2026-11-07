# 查询式前馈 GS 的 demo 可视化诊断（2026-11-05）

把 `EXP_DL3DV/diag_dl3dv_report.md` 和 `EXP_DL3DV/diag_locusgs_report.md` 里发现的每个问题，都导出成可以直接在三维里查看的 demo（PLY），再配一张索引图。没有训练任何模型。

## 设置

- **场景**
  - ScanNet++ `7b6477cb95`：项目固定的 demo 场景，有激光网格。输入、目标帧与正式 50 场景评测相同。
  - DL3DV `2f3e1c0f688c…`：`EXP_TokenGS/demo/demo_scene.json` 预先登记的 demo 场景，没有激光，几何参照为 COLMAP 点。
- **模型**（全部冻结、FP32、TF32 关闭）
  - `tokengs_release_latent_6v`：官方 release 隐变量版
  - `tokengs_paper_4v`：官方论文版 `dl3dv_4v`，LocusGS 论文的基线
  - `locusgs_finetune_4v`：LocusGS
  - `sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20`（⑫）：只在 ScanNet++ 场景导出
- **统一坐标系**：所有 PLY 都放在 release 版的 mean_cam 归一化坐标系里（ScanNet++ 缩放 0.3，DL3DV 缩放 0.15）。LocusGS 和论文版原本用 first_cam 坐标系，这里把它们的高斯中心和旋转一起做刚体变换。变换前后，同一视角渲染出的图像最大差约 0.002。因此，同一场景下不同模型的 PLY、激光网格、COLMAP 点和相机都可以直接叠在一起看。
- **核对**：ScanNet++ 场景上四个模型的 PLY 渲染 PSNR（29.477 / 28.716 / 29.698 / 29.999）与正式评测 `scene_metrics.csv` 一致。
- **对照方式**：选 token 和固定射线探针用的像素，都是与模型无关的固定网格，所以不同模型之间可以逐一对照。“最厚像素”探针按各模型自己的结果挑选。
- **重跑**：`GPU_A=1 GPU_B=2 bash EXP_QueryGS_Diagnosis/run_demo.sh`；底层脚本 `scripts/demo_query_diagnosis.py`。

## 怎么看

- 用 VS Code 插件 kleinicke.ply-visualizer 打开（和 `EXP_ScanNetPP/demo` 相同）。
  - `*.ply` 里的高斯文件是标准 3DGS 格式，可以按 splat 模式看，也可以按 Points 模式按标量字段着色。
  - `*_points.ply` 和带颜色的点云文件，直接按 RGB 看。
- 场景目录下的共享参照：
  - ScanNet++：`gt_lidar_mesh_frustum.ply`（输入视锥内的激光网格顶点）
  - DL3DV：`colmap_points.ply`（按被几个输入视角看到着色）
  - 两个场景都有 `cameras.ply`（红色为输入相机，绿色为目标相机）
  - `tokens20_pixels_input_view0.png`：20 个固定像素的位置，以及它们的编号和颜色
- 每个模型目录里的高斯 PLY 带这些标量字段：
  - `token_id`、`child_id`
  - `dist_to_surface_m`（ScanNet++，到激光网格的距离）或 `dist_to_surface_norm`（DL3DV，到最近 COLMAP 点的距离）
  - `is_junk`
  - `token_parallax_deg`、`token_centre_dist_*`、`token_c_centroid`
  - LocusGS 还有 `anchor_radius`、`offset_over_radius`

## 问题 → demo 文件 → 看什么

| # | 问题 | 文件（在 `scene_*/<模型>/` 下） | 看什么 |
|---|---|---|---|
| 1 | 整体重建、厚雾 | `gaussians.ply`、`gaussians_prune0.05.ply`，叠加 `../gt_lidar_mesh_frustum.ply` | 墙面、桌面前后是否有一层半透明的“雾” |
| 2 | 同一 token 的高斯散开（LocusGS 解决的问题） | `tokencolor_alive.ply`（每个 token 一种颜色） | 论文版 TokenGS 同色点散满全屋；LocusGS 同色点聚成小团 |
| 3 | 20 个 token 各自负责什么 | `tokens20_points.ply`（暗色 = 已死子高斯）、`tokens20_gaussians.ply`、`tokens20_centres.ply`；LocusGS 另有 `tokens20_anchor_spheres.ply`（锚点半径球面）| 子高斯是否落在半径内（LocusGS 不少逃出半径）、死槽位在哪 |
| 4 | 看的地方 vs 画的地方（渲染归属） | `tokens20_attention_vs_render.png`：每个 token 两格，左边红色 = 注意力，右边绿色 = 渲染权重，标注渲染权重落在注意力 90% 区域内的比例 | LocusGS 注意力只是一个小点，渲染范围大得多，而且常常偏开 |
| 5 | 透明且远离表面的休眠槽位 | `junk_slots_points.ply`（红色 = 不透明度 < 0.05 且离面 > 30 cm 或 0.09 归一化单位；灰色 = 可见高斯，抽样） | 论文版红点漫天，LocusGS 和 ⑫ 很少 |
| 6 | 离面误差的空间分布 | `surface_distance_heat_points.ply`（不透明度 ≥ 0.05 的高斯；蓝 = 贴面，红 ≥ 30 cm） | 误差集中在哪些地方（远处、边缘、白墙） |
| 7 | 视线厚度来自 token 间叠放 | `render_strip.png` 第 7–9 列：视线厚度、词元间方差占比、每个像素由几个 token 贡献 ≥1% | LocusGS 的“词元间占比”整片接近 1（红色） |
| 8 | 一个像素由多个 token 在不同深度叠出来（射线探针） | `ray_probes_pixels.png`（青色 = 固定像素，品红 = 本模型最厚的像素）；`ray_probes_rays.ply`（白色 = 视线，绿色 = 激光真值落点，黄色 = 渲染深度）；`ray_probes_tokens*.ply`（对该像素贡献 ≥1% 的 token 的全部子高斯，按 token 着色）；`ray_probes.json`（每个像素有哪些 token、权重多少、token 中心深度） | 沿同一条视线，多团不同颜色的高斯前后排开 |
| 9 | 渲染、深度误差总览 | `render_strip.png`，每行一个目标视角：真值、预测、RGB 误差、预测深度、真值深度、AbsRel、厚度、词元间占比、token 数 | |
| 10 | 视差与几何误差 | `token_centres_parallax.ply`（0–20° 着色）、`token_centres_surface_distance.ply` | 低视差（蓝色）区域是否对应大误差（红色） |
| 11 | LocusGS 锚点扎堆 | `anchors_crowding.ply`（按半径内其他锚点数着色，log 刻度，红色 ≈ 300 个） | 锚点在哪里扎堆 |
| 12 | LocusGS 半径不是不确定度 | `anchors_radius.ply`（半径 / 深度，0.05–0.25） | 半径大小和视差、误差的关系 |
| 13 | LocusGS 锚点和自己的高斯脱节 | `anchor_to_gaussian_centre_offsets.ply`（橙色线段：锚点 → 该 token 的高斯中心，只画距离大于深度 10% 的 token） | 线段越长，锚点越不代表自己高斯的位置 |

## 场景内数字（`scene_*/summary.json`）

ScanNet++ `7b6477cb95`（厚度单位为 m，在 3 个展示视角上统计；AbsRel 只用 3 个视角，与 50 场景评测不同）：

| | release 6v | 论文版 4v | LocusGS | ⑫ |
|---|---:|---:|---:|---:|
| PSNR（8 个目标） | 29.48 | 28.72 | 29.70 | 30.00 |
| AbsRel（3 个展示视角） | 0.083 | 0.087 | 0.044 | 0.033 |
| 视线厚度中位 / p90 | 0.069 / 0.159 | 0.118 / 0.326 | 0.089 / 0.186 | 0.052 / 0.115 |
| 休眠槽位占比 | 21.2% | 72.5% | 5.3% | 1.2% |
| C_centroid（同一 token 的散开程度） | 0.41 | 0.60 | 0.041 | 0.039 |
| 20 个 token 的渲染权重落在注意力 90% 区域内（中位） | 0.88 | 1.00 | **0.34** | 0.88 |
| 每个射线探针像素上贡献 ≥1% 的 token 数 | 9–24 | 9–15 | 7–17 | 8–24 |
| LocusGS：半径内其他锚点数（中位） | — | — | **87** | — |
| LocusGS：存活子高斯落在 1 倍半径外的比例 | — | — | **58%** | — |

DL3DV `2f3e1c0f688c`：PSNR 23.59 / 22.71 / 24.49；休眠槽位占比 21.9% / 33.0% / 14.6%；C_centroid 0.56 / 0.85 / 0.080；厚度 / 深度中位 0.11 / 0.21 / 0.20；LocusGS 半径内其他锚点 23 个，存活子高斯 18% 在 1 倍半径外。

说明：射线探针里不同 token 的“中心深度”差异，包含了 token 中心的横向偏移，只能作为直观参考，不是严格的叠放量。严格的词元间占比请看 `render_strip.png` 第 8 列，以及 `EXP_DL3DV` 的 140 场景统计。
