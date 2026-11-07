# SfM 点 demo：e7af285f7d

输入帧 [394, 397, 400, 404, 407, 411]（验证协议的 6 个输入视角）。所有 PLY 在 TokenGS 归一化坐标系，可与
`EXP_ScanNetPP/demo/scene_7b6477cb95/*/` 的高斯 PLY 直接叠加（仅 7b6477cb95 有这些高斯 demo）。

| 文件 | 看什么 |
|---|---|
| `sfm_points_status.ply` + `gt_lidar_mesh_frustum.ply` | 所有三角化点。绿 = anchor，蓝 = reliable，橙 = rejected: weak，红 = rejected: isolated |
| `sfm_points_laser_error.ply` | 同一批点按离激光网格距离着色：绿 0 → 黄 5 cm → 红 ≥ 10 cm |
| `sfm_anchors.ply` | 最终作为 token 固定位置的点（小八面体） |
| `anchor_coverage_laser.ply` | 输入视角看到的激光表面，按离最近锚点距离着色：绿 < 10 cm，黄 10–20 cm，红 > 20 cm |
| `cameras.ply` | 6 个输入相机中心（红） |
| `input_views_sfm.png` | 输入图上每个观测点，颜色同上（锚点带黑边） |
| `input_views_coverage.png` | 输入图上激光像素离最近锚点的距离（灰 = 无激光） |

| 点的类别 | 数量 | 离激光面 中位 / p90 (cm) |
|---|---:|---:|
| anchor (used as token position) | 1024 | 1.4 / 5.4 |
| reliable, not selected (beyond 1024 or FPS redundant) | 326 | 0.6 / 2.2 |
| rejected: weak (2 views and angle < 3 deg) | 883 | 1.9 / 7.3 |
| rejected: isolated (5-NN mean > 10 cm) | 70 | 13.5 / 23.2 |

输入可见激光表面离最近锚点：< 10 cm: 62%，10-20 cm: 19%，> 20 cm: 20%。
