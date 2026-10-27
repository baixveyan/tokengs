# AGENTS.md

本仓库是官方 TokenGS（nv-tlabs/TokenGS）的本地研究副本。**当前重心是从头重新设计 ScanNet++ 实验：训练阶段网络可以使用点云、相机和 RGB，评估/推理阶段模型只接收相机与 RGB；目标是把点云的空间组织知识学入可独立工作的 RGB 路径，而不是只拟合图像。** 不得再沿用“训练阶段也只输入 RGB、点云只能算输出损失”的早先理解；训练专用点云分支、几何教师或其他机制的选择需与用户讨论。

用户已亲自清空过旧 `EXP_ScanNetPP/`，不得恢复旧实验或自动续训旧 workspace。现已按用户明确授权重新运行官方 6 输入预训练基线，结果位于 `EXP_ScanNetPP/baseline_pretrained_6view/`。不使用单场景 scene/token 内循环微调，不自动删除其他 checkpoint、训练日志或数据集。

**实验目录必须轻量且对齐 `EXP_TokenGS` 习惯**：
- `EXP_ScanNetPP/<实验名>/` 单层存放 50 场景评测结果（`summary.json`、`scene_metrics.csv`、`frame_metrics.csv`、`run.log`），只有模型微调实验才保存模型；
- `EXP_ScanNetPP/all_experiments_table.{md,csv}` 存放全部实验汇总总表（由 `python scripts/scannetpp_evaluation_protocol.py build-table` 自动生成）；
- `EXP_ScanNetPP/run_50_scenes.sh` 与 `EXP_ScanNetPP/run_demo.sh` 作为顶层一键运行入口，底层实现统一留在仓库 `scripts/`；
- `EXP_ScanNetPP/report/<实验名>.md`（保持单数目录 `report/`）与实验同名且不再嵌套，记录参数、监督路径和指标口径；
- `EXP_ScanNetPP/demo/scene_7b6477cb95/<实验名>/` 按场景归拢存放同场景高斯点云（完整/剪枝 PLY、10/20 Token 抽样 PLY、`cameras.ply`、`render_strip.png`）及场景级 `gt_lidar_mesh_frustum.ply`、`comparison.md`、`highlighted_tokens.md`。不建立 configs、launch_logs、source_snapshot、逐场景张量缓存等目录。

此前 DL3DV 的 scene 微调、token 微调及联合微调作为历史研究保留，不因 ScanNet++ 方向调整而删除。禁用“TTT”笼统简称；历史路径中的旧术语不等于当前研究方向。最新输入协议见共享记忆第 14 节，目录规范、G1–G4 几何评测协议、C0/S1 第一批实验与新基线见第 15 节（15.3），**下一阶段已选定方向（“原模型 + 点云”自蒸馏老师 → RGB 学生、网格可信度标签作为第一个中间产物、agent 作为区域级编辑主编）见 15.4**，旧审计教训见第 13 节。

接手前必读：

1. [`docs/AGENT_MEMORY.md`](docs/AGENT_MEMORY.md)：共享背景记忆，包括论文要点、release 2026.6 差异、实验历史、已废弃方向、当前权威结果、陷阱清单、用户偏好。
2. [`EXP_ScanNetPP/report/baseline_pretrained_6view.md`](EXP_ScanNetPP/report/baseline_pretrained_6view.md)：新基线的实际参数、结果与待讨论指标；每个新实验都按同名文件写报告，不恢复旧文件。
3. [`EXP_TokenGS/EXP_TokenGS.md`](EXP_TokenGS/EXP_TokenGS.md)：历史 DL3DV 实验执行日志与原始数字。
4. [`TokenGS.pdf`](TokenGS.pdf)：论文原文（arXiv 2604.15239），需要逐字引用时查阅。

硬性规则（详见 AGENT_MEMORY 第 3、9、10 节）：

- 启动前 `export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH`。
- 只用空闲 GPU（显存 ≤100 MiB 且无 compute app），不干扰他人进程。
- 正式评估一律 FP32。DL3DV 入口为 `scripts/evaluate_with_scene_latent_tuning.py`；ScanNet++ 新基线入口为 `scripts/evaluate_scannetpp.py`，一键脚本为 `scripts/run_scannetpp_baseline.sh`。旧 full_metrics 入口的 RayThick 和厘米阈值未经修正核验不得引用。
- 实验由用户运行：agent 交付一键脚本（内部设环境、**每个任务启动前实时检查并只用空闲 GPU**、任务顺序执行、训练任务用全部空闲卡多卡并行、每组一个完整未过滤的 log，共用 `scripts/one_click_launch_common.sh`），事后读 log 与产物核验。
- ScanNet++ 名称简洁、说明方法（如 `baseline_pretrained_6view`）；精度、场景数、划分等固定条件写在同名报告。禁止 v1、v2、formal、task 编号之类名字；DL3DV 旧名对照见 `EXP_TokenGS/renamed_directories_old_to_new.md`。
- 工作区有未提交的本地修改，不要还原；方向性决策先问用户。
- 目录重构先提交明确方案并确认；删除已有结果须列出具体范围并再次确认。ScanNet++ 审计建议逐条讨论，禁止未经确认批量修复或重跑实验。
- 完成阶段性工作后，更新 `docs/AGENT_MEMORY.md` 与对应实验报告；不为记录参数额外建立配置文件或多层实验管理结构。
