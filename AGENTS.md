# AGENTS.md

本仓库是官方 TokenGS（nv-tlabs/TokenGS）的本地研究副本，当前研究方向是在 **scene 微调**（scene latent tuning）、**token 微调**（token tuning）及 **scene + token 联合微调** 的内循环里研究几何机制与改进方案（注：由于论文与官方代码对 TTT 的定义不一致，本项目禁用“TTT”这一笼统简称，一律直接指明是 scene 微调还是 token 微调）。

接手前必读：

1. [`docs/AGENT_MEMORY.md`](docs/AGENT_MEMORY.md)：共享背景记忆，包括论文要点、release 2026.6 差异、实验历史、已废弃方向、当前权威结果、陷阱清单、用户偏好。
2. [`EXP_TokenGS/EXP_TokenGS.md`](EXP_TokenGS/EXP_TokenGS.md)：实验执行日志与原始数字。
3. [`TokenGS.pdf`](TokenGS.pdf)：论文原文（arXiv 2604.15239），需要逐字引用时查阅。

硬性规则（详见 AGENT_MEMORY 第 3、9、10 节）：

- 启动前 `export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH`。
- 只用空闲 GPU（显存 ≤100 MiB 且无 compute app），不干扰他人进程。
- 正式评估一律 FP32，入口为 `scripts/evaluate_with_scene_latent_tuning.py`。
- 实验由用户运行：agent 交付一键脚本（内部设环境、自动选空闲 GPU、并行跑全部组、每组一个完整未过滤的 log，共用 `scripts/one_click_launch_common.sh`），事后读 log 与产物核验。
- 命名写清方法 / 精度 / 场景数 / 划分，禁止 v1、v2、formal、task 编号之类的名字；旧名对照见 `EXP_TokenGS/renamed_directories_old_to_new.md`。
- 工作区有未提交的本地修改，不要还原；方向性决策先问用户。
- 完成阶段性工作后，更新 `docs/AGENT_MEMORY.md` 与 `EXP_TokenGS/EXP_TokenGS.md`。
