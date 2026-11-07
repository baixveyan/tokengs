# ScanNet++ 实验改名对照（2026-11-03）

命名规则：`sft_<放开训练的部分>_<损失>[_<额外结构>]`。目录、workspace、外推子目录（`eval_extrapolation/ext_near|ext_far/`）、Demo 目录及其中以实验名开头的文件、报告、实验注册表、训练预设（`sft_scannetpp_<名字去掉 sft_>`）、`summary.json` 的 `name` 字段均已改为新名。`run.log`、workspace 内 `config.yaml`、CSV 等原始记录**保留旧名**，按下表对照。

| 新名 | 旧名 | 旧代号 | 内容 |
|---|---|---|---|
| `baseline_pretrained_6view` | 同左 | 基线 | 官方 6 输入预训练模型直接前馈 |
| `sft_latenttail_rgb` | `sft_rgb_only` | C0 | 放开 scene_latent 尾段，只用画质损失（30 epoch） |
| `sft_readout_rgb` | `sft_readout_rgb_only` | R0 | 放开读出链路，只用画质损失（10 epoch） |
| `sft_readout_rgb_depth` | `sft_readout_depth` | R1 | + 渲染对数深度 L1 |
| `sft_readout_rgb_depth_thick` | `sft_readout_depth_thick` | R2 | + 视线厚度损失 |
| `sft_readout_rgb_depth_thick_sh1` | `sft_readout_depth_thick_sh1` | R2+SH1 | + 1 阶球谐颜色 |
| `sft_readout_rgb_depth_thick_posmlp` | `sft_readout_mlp_geo` | 2a / H2a | + 零初始化非线性位置头 |
| `sft_readout_rgb_depth_thick_posmlp_surfel_binar` | `sft_readout_surfel` | 2b / H2b | + 面元约束 + 不透明度二极化 |
| （已删除） | `sft_readout_depth_headlr1` | B1 | 读出头全学习率，停在第 3 epoch，2026-11-03 经用户同意删除 |

诊断脚本与结果：

| 新名 | 旧名 |
|---|---|
| `scripts/eval_by_visibility.py`，缓存 `workspace_scannetpp/eval_by_visibility/` | `scripts/scannetpp_breakdown.py`，缓存 `breakdown/`（旧代号 A1/A2） |
| `scripts/diag_thinning_and_parallax.py thinning_cost \| parallax` | `scripts/diag_h3_h5.py h3 \| h5` |
| `docs/DIAG_thinning_image_cost.json` | `docs/DIAG_h3_fog_hedging.json` |
| `docs/DIAG_parallax_vs_geometry_error.json` | `docs/DIAG_h5_parallax.json` |
| 读出头容量探针 `scripts/probe_readout_capacity.py` | 同名（旧代号 V1） |
