# 网格占据 demo：7b6477cb95（验证场景）

由 `scripts/demo_grid_occupancy.py` 生成。探针与特征来自 `EXP/grid_occupancy_10cm`，阈值为概率 0.5。该场景未参与训练和阈值校准。

## 场景与计数

- 输入帧 101、104、107、111、114、118；网格 58×41×47 = 111,766 个格子，每格 10 cm。
- 激光只用于画标签和参考表面，不输入探针。

| 模型 | 找对的表面 TP | 误报 FP | 漏报 FN | 正确判空 TN | 召回 | 精确率 | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 仅相机几何 | 3,871 | 36,690 | 2,267 | 68,938 | 63.1% | 9.5% | 0.090 |
| 加图像特征 | 4,032 | 13,485 | 2,106 | 92,143 | 65.7% | 23.0% | 0.205 |

这个场景比验证集中位数（IoU 0.160）略好。

图像探针的误报离最近真实表面格子的距离（Chebyshev）：1 格内 49.2%，2 格内 75.9%，3 格内 89.2%。

## 怎么看

| 文件 | 内容 |
|---|---|
| `input_views.png` | 6 张输入图 |
| `confusion_slices.png` | 5 个水平切片（x 向右、z 向前）：原始激光扫描（RGB）、标签、预测、TP 绿/FP 紫/FN 红并叠加激光（青色） |
| `confusion_topdown.png` | 俯视：每个竖直柱内的表面格数，标签、预测、误报、漏报；三角为输入相机 |
| `confusion_3d.png` | 三个视角：第一行原始激光扫描（网格内 8 万点），第二行真实表面格（绿找到、红漏掉），第三行误报按离表面距离着色（浅=1 格，深=超过 3 格） |
| `laser_scan_points.ply` | 原始激光扫描全部点（`mesh_aligned_0.05.ply` 顶点，原 RGB），含输入看不到的部分；`laser_scan_points_in_grid.ply` 只保留网格内 |
| `cells_confusion.ply` | TP 绿、FP 紫、FN 红的立方体，可直接在三维查看器中打开 |
| `cells_tp.ply`、`cells_fp.ply`、`cells_fn.ply` | 三类分别单独存放 |
| `cells_fp_by_distance.ply` | 误报按离表面距离着色 |
| `cells_label.ply`、`cells_predicted.ply` | 标签（金）与预测（蓝） |
| `laser_input_surface.ply` | 输入视角可见的激光表面（灰点），用于叠加对照 |
| `input_cameras.ply` | 输入相机位置（黑色小八面体） |
| `confusion_demo.json` | 上述计数 |

PLY 在归一化 mean-camera 坐标系（0.3 单位/米），PNG 坐标轴为米。

`grid_labels.ply`、`grid_predictions.ply`、`laser_hole_cells.ply`、`local_section.png` 是主实验在校准 99% 召回阈值（logit −33.2）下的旧可视化，阈值不同，不要与本页混看。`laser_input_surface.ply` 已由本脚本按同一数据重新生成为灰色。

## 现象

1. 房间的整体布局（几面墙、地面和桌面附近）基本被找到了，但预测出的表面是 30–80 cm 厚的带状区域，真实表面是其中一条细线。
2. 误报大多紧贴真实表面，是“画厚”而不是凭空出现；这与沿视线深度难以分辨一致。
3. 漏报集中在离相机较远、视角较斜的墙面，尤其是右侧墙和远端角落。
