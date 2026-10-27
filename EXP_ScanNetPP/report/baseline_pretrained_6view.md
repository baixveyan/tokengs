# 官方 6 输入预训练模型基线

状态：已完成，50/50 场景、400 个新视角；目录整理未重跑模型、未修改指标。

## 一、动机

建立重新设计训练的 RGB 前馈基线。训练方案的最终目标是训练时利用点云、相机和 RGB 学习空间组织，评估时仅用相机和 RGB；本基线不训练，不验证点云引导方案是否有效。

## 二、设计

- 官方 `dl3dv_latent_6v_ssim` 权重，严格加载；SHA256：`5913da347475049cb58e552634ee1c527293c2571576806db0d4da702ce978c3`。
- ScanNet++ 官方 50 个验证场景；固定 18 帧窗口，6 输入、8 个不重叠的新视角，分辨率 H=256、W=448，seed=42。
- FP32，TF32 关闭；模型只输入 RGB 与相机，不加载点云，不进行权重微调或单场景优化。
- PSNR、SSIM（fused-ssim，train=False）、LPIPS-VGG（normalize=True）；先每场景 8 帧平均，再对 50 场景等权平均。
- 运行时间：2026-10-26 11:54:29 至 11:55:30，GPU 1，退出码 0。目录整理前名称为 `pretrained_6view_rgb_fp32_50val`，日志保留原始命令；本次只是整理产物，没有重跑或改变精度。

## 三、结果（50 验证场景，400 个新视角）

### 3.1 图像质量指标

| PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|
| 30.5033 | 0.9270 | 0.1801 |

### 3.2 三维几何指标（G1–G4，真实米/厘米尺度，预测完成后独立评分）

| 深度覆盖率 ↑ | Depth AbsRel ↓ | Depth RMSE (m) ↓ | δ<1.05 ↑ | δ<1.10 ↑ | 视线厚度中位 (m) ↓ | 视线厚度 p90 (m) ↓ | Accuracy (cm) ↓ | Completeness (cm) ↓ | F-score@3cm ↑ | F-score@5cm ↑ | 法线均值 (°) ↓ | 法线中位 (°) ↓ | 法线<11.25° ↑ | 法线<30° ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9988 | 0.1744 | 0.3394 | 0.4698 | 0.6239 | 0.0859 | 0.2568 | 22.43 | 10.21 | 0.3212 | 0.4453 | 40.65 | 37.32 | 0.1201 | 0.4127 |

- 尺度对齐门禁核验：50 场景 `depth_scale_ratio_med` 中位数为 **`1.0036`**（均值 `1.0163`），与真实物理尺度高度自洽。
- 结果在 `../baseline_pretrained_6view/`：`summary.json`、`scene_metrics.csv`、`frame_metrics.csv`、`run.log`。CSV 保存实际输入/目标帧身份，指标已核对聚合一致；实验汇总表位于 `../all_experiments_table.md` 与 `../all_experiments_table.csv`。

Demo 位于 `../demo/scene_7b6477cb95/baseline_pretrained_6view/`（与 `EXP_TokenGS/demo/scene_<id>/<实验名>/` 结构对齐）：包含完整未剪枝 262,144 高斯点云 `baseline_pretrained_6view.ply`、剪枝版本 `_prune0.01.ply` / `_prune0.05.ply`、按 Token 染色 `_slotcolor.ply`、固定随机种子 0 抽样的 10/20 个活跃 Token 点云 `_highlight{10,20}tokens.ply` 与 `_only{10,20}tokens*.ply`、相机位姿 `cameras.ply`、8 帧新视角渲染对比图 `render_strip.png` 及 `needle_stats.json`；场景层 `../demo/scene_7b6477cb95/` 存放真值视锥激光雷达点云 `gt_lidar_mesh_frustum.ply`、跨实验对比表 `comparison.md` 与 `highlighted_tokens.md`。

后续重跑与汇总入口：`bash EXP_ScanNetPP/run_50_scenes.sh`（50 场景评测与总表生成）、`bash EXP_ScanNetPP/run_demo.sh`（同场景 Demo 导出与对比）、或单独运行 `bash scripts/run_scannetpp_baseline.sh`。脚本自动选择空闲 GPU，已完成的实验自动跳过、拒绝覆盖已有结果；本基线没有另存模型。

## 四、结论与已锁定的三维几何评测协议（G1–G4）

当前已锁定三维几何评测协议 G1–G4（代码已接入 `scripts/scannetpp_evaluation_protocol.py` 与 `scripts/evaluate_scannetpp.py`，使用 Open3D `RaycastingScene` 对真值三角网格 `mesh_aligned_0.05.ply` 求交，恢复原始米制尺度 $\div 0.3$，不做额外尺度对齐；在首场景 `7b6477cb95` 上已通过尺度对齐检验 `median(D_pred/D_gt) = 1.0024`）：

1. **G1 新视角深度误差**：在有效掩码 $M_v = (D^*_v > 0) \land (\hat{A}_v > 0.5)$ 上计算 `深度覆盖率`、`AbsRel`、`RMSE (m)`、`δ<1.05`、`δ<1.10`。
2. **G2 视线表面厚度（修正版）**：使用零背景、按当前相机坐标系计算的 $[z_{\text{cam}}, z_{\text{cam}}^2]$ 双通道光栅化求视线深度标准差 $\sigma_z$（单位米），报告 $M_v$ 上的中位数与 p90（CPU 合成单层 $\alpha=0.5 \to \sigma_z=0$、双层间距 $d \to \sigma_z=d/(2s)$ 测试已通过）。
3. **G3 双向三维表面重建质量**：将 8 个新视角下的预测深度与真值深度分别反投影至三维空间并按 1 cm 体素去重，计算双向 `Accuracy (cm)`、`Completeness (cm)` 及 `F-score @ 3cm / 5cm`（归一化阈值 `0.009 / 0.015`）。
4. **G4 表面法线角误差**：由预测深度图中心差分计算表面法线（剔除相邻深度突变 $>5\%$ 的边缘），与 Open3D 投射的真实三角面片法线比较夹角，报告均值、中位数及 `<11.25°`、`<30°` 占比。

执行 `bash EXP_ScanNetPP/run_50_scenes.sh baseline_pretrained_6view` 即可利用空闲 GPU 从 50 场景缓存补算完整 G1–G4 指标并自动写入总表。真值几何只用于预测完成后的独立评分，不进入模型推理。
