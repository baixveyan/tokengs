# `scene_7b6477cb95` 20 个高亮 Token 的纯相机射线三角化点 vs 真实表面点对比

## 生成的 PLY 文件（均位于 `EXP_ScanNetPP/demo/scene_7b6477cb95/baseline_pretrained_6view/`，与 `gt_lidar_mesh_frustum.ply` 及 `baseline_pretrained_6view_only20tokens.ply` 坐标系、颜色 100% 对齐）

1. `baseline_pretrained_6view_only20tokens_triangulated_anchors.ply`：仅包含 **20 个纯 RGB 注意力射线三角化锚点**（0 点云、0 真值深度，与 20 个 token 一一同色）。
2. `baseline_pretrained_6view_only20tokens_gt_surface_anchors.ply`：仅包含 **20 个注意力盯的真实网格表面点**（同色，代表该 token 眼睛看到的真墙面位置）。
3. `baseline_pretrained_6view_only20tokens_with_triangulated_and_gt.ply`：**三合一叠加文件**（包含原 20 个 token 的 1,280 个子球 `point_role=0` + 同色十字标记的纯 RGB 三角化点 `point_role=1` + 纯白色标记的真表面点 `point_role=2`；在点云软件中可按 `point_role` 或颜色直接对比）。
4. `baseline_pretrained_6view_all4096_triangulated_anchors.ply`：全场景 **4,096 个 token 的纯 RGB 三角化点**。

## 全场景 4,096 个 Token 汇总统计

- **注意力真表面读取点 → 真墙面中位距离**：`3.20 cm`（p90 `11.75 cm`）
- **纯 RGB 射线三角化点（0 真值）→ 真墙面中位距离**：`7.84 cm`（p75 `17.31 cm`，p90 `33.51 cm`）
- **纯 RGB 射线三角化点 → 注意力真表面读取点中位距离**：`15.68 cm`
- **原 `deconv` 64 子球中心 → 真墙面中位距离**：`5.43 cm`（p90 `26.16 cm`）

## 20 个高亮 Token 逐一对比（单位：厘米 cm）

| 序号 | token | 颜色 | 注意力真表面点→真墙面 (cm) | 纯RGB三角化点→真墙面 (cm) | 纯RGB三角化点→注意力真表面点 (cm) | 原64子球中心→真墙面 (cm) |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 3671 | red `#e6194b` | 2.2 | **65.7** | 65.7 | 12.0 |
| 1 | 867 | green `#3cb44b` | 0.5 | **5.9** | 9.9 | 8.1 |
| 2 | 755 | yellow `#ffe119` | 2.1 | **11.2** | 11.2 | 3.2 |
| 3 | 3786 | blue `#4363d8` | 4.2 | **20.9** | 39.7 | 4.9 |
| 4 | 1739 | orange `#f58231` | 1.8 | **10.4** | 15.6 | 4.6 |
| 5 | 2205 | purple `#911eb4` | 3.1 | **15.4** | 71.4 | 29.0 |
| 6 | 1989 | cyan `#42d4f4` | 2.9 | **7.2** | 10.5 | 0.7 |
| 7 | 3938 | magenta `#f032e6` | 3.8 | **5.9** | 5.0 | 4.2 |
| 8 | 2325 | lime `#bfef45` | 1.6 | **7.3** | 6.7 | 3.0 |
| 9 | 2423 | pink `#fabed4` | 6.0 | **4.9** | 18.9 | 6.9 |
| 10 | 491 | teal `#469990` | 2.5 | **3.1** | 3.7 | 1.9 |
| 11 | 3196 | lavender `#dcbeff` | 27.5 | **48.2** | 273.3 | 42.8 |
| 12 | 2324 | brown `#9a6324` | 34.8 | **7.8** | 74.2 | 2.1 |
| 13 | 43 | beige `#fffac8` | 7.2 | **37.7** | 67.3 | 5.6 |
| 14 | 1921 | maroon `#800000` | 3.1 | **2.8** | 1.4 | 2.1 |
| 15 | 210 | mint `#aaffc3` | 5.8 | **6.1** | 11.8 | 4.6 |
| 16 | 3357 | olive `#808000` | 0.5 | **25.4** | 25.6 | 2.6 |
| 17 | 1722 | apricot `#ffd8b1` | 3.5 | **45.1** | 190.9 | 5.5 |
| 18 | 2797 | navy `#000075` | 13.6 | **21.0** | 39.5 | 1.3 |
| 19 | 118 | black `#000000` | 2.2 | **2.2** | 1.6 | 4.2 |
