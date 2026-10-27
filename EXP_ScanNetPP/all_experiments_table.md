# EXP_ScanNetPP 全部实验总表（50 验证场景）

由 `python scripts/scannetpp_evaluation_protocol.py build-table` 从各实验目录的 `scene_metrics.csv` 与 `summary.json` 自动生成，不要手改；新实验跑完后重新运行即可更新。同一份数据另存为 `all_experiments_table.csv`。

- 每行一个实验，第一列对应实验文件夹 `EXP_ScanNetPP/<实验名>/`；详细配置与分析见 `EXP_ScanNetPP/report/<实验名>.md`。
- 评估固定协议：ScanNet++ v2 官方 50 个验证场景（`nvs_sem_val.txt`），`window_size=18`（6 输入视角 → 8 个不重叠新视角，共 400 帧），分辨率 `256×448`，严格 FP32（TF32 关闭），推理仅输入 RGB 与相机，0 步单场景优化。
- 同场景三维高斯点云与可视化对比位于 `EXP_ScanNetPP/demo/scene_7b6477cb95/<实验名>/`。

| 实验 | 说明 | 训练输入 | 推理输入 | 完成场景 | Δ 新视角 PSNR 对 baseline_pretrained_6view（95% CI，胜/负） | 新视角 PSNR ↑ | 新视角 SSIM ↑ | 新视角 LPIPS ↓ | 深度覆盖率 ↑ | Depth AbsRel ↓ | Depth RMSE (m) ↓ | δ<1.05 ↑ | δ<1.10 ↑ | 视线厚度中位 (m) ↓ | 视线厚度 p90 (m) ↓ | Accuracy (cm) ↓ | Completeness (cm) ↓ | F-score@3cm ↑ | F-score@5cm ↑ | 法线均值 (°) ↓ | 法线中位 (°) ↓ | 法线<11.25° ↑ | 法线<30° ↑ | 每场景前馈耗时 s | 显存峰值 MiB | 报告 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_pretrained_6view` | 官方 6 输入预训练模型直接前馈基线（不微调） | RGB + 相机（DL3DV 预训练） | RGB + 相机（0 点云、0 优化步） | 50/50 | 0 | 30.5033 | 0.9270 | 0.1801 | 0.9988 | 0.1744 | 0.3394 | 0.4698 | 0.6239 | 0.0859 | 0.2568 | 22.43 | 10.21 | 0.3212 | 0.4453 | 40.65 | 37.32 | 0.1201 | 0.4127 | 0.70 | 3828 | `report/baseline_pretrained_6view.md` |
| `sft_rgb_only` | 对照 C0：ScanNet++ 纯 RGB 光度后训练（解冻 latent_blocks.8..11 + KV 投影） | RGB + 相机（无点云） | RGB + 相机（0 点云、0 优化步） | 50/50 | +0.6842 [+0.5075, +0.8608] 50/0 | 31.1874 | 0.9305 | 0.1760 | 0.9991 | 0.1645 | 0.3251 | 0.4730 | 0.6251 | 0.0784 | 0.2323 | 20.62 | 10.40 | 0.3230 | 0.4446 | 40.16 | 36.62 | 0.1253 | 0.4219 | 0.70 | 3836 | `report/sft_rgb_only.md` |
