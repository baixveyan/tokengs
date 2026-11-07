# GeometryGS 项目规则

遵循父目录 `/home/jms/xveyan/AGENTS.md`。接手先读 `docs/CURRENT_STATE.md` 和 `docs/KNOWLEDGE_INDEX.md`，再按需要查 `docs/legacy_tokengs/`。历史副本中的旧方案、运行命令和状态不自动成为本项目当前指令。

## 目标与范围

- 用户最新决定：主模型容量直接对标 TokenGS，1024 维、12 层图像编码、12 层 scene 聚合，不主动缩小。完整迁移官方图像/scene 上游权重，新建 SfM 融合、空间 GS decoder 与局部头；SfM 与空间编码为基础，软链可选。旧“全部随机初始化”和“75M 作主模型”均被最新讨论替代，缩小留作后续消融。旧 `sfm_anchor_k1024` 保留作历史模型。
- 同一模型最终支持 RGB+相机和 RGB+相机+可选点云，两种模式独立评测。无点云模式不得访问点云、缓存点云特征或在线几何教师。
- 学习正确表面的位置、形状与跨 token 一致性，同时保持完整度和画质。紧凑度、低厚度、PSNR 不能单独证明几何正确。
- 当前已授权架构见 `docs/SPATIAL_ARCHITECTURE.md`：上游按完整匹配的模块严格加载，原 GS query/decoder/deconv 不加载；禁止以形状碰巧匹配或通道截断代替迁移。重大研究方向改变仍先讨论。
- 不做单场景 scene/token 内循环优化，不恢复已清空的旧 ScanNet++ workspace，不自动续训旧任务。

## 运行与数据

- 环境沿用已知 TokenGS 环境，执行前确认：
  `export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH`
- 只使用显存 ≤100 MiB 且无 compute app 的空闲 GPU，每个任务启动前重新检查。正式评测 FP32。
- 默认交付用户运行的一键脚本：设置环境、任务顺序执行、训练使用全部空闲卡多卡并行、每组保留完整未过滤日志。优先复用原仓库 `scripts/one_click_launch_common.sh`；移植时检查其路径依赖。
- 数据根 `/home/jms/xveyan/data/4dgs`，不要全量扫描、修改或删除。模型、日志、数据和归档不自动删除或覆盖。
- 模型代码变更后运行相关测试及仓库要求的 pytest；本次文档迁移不能当成新模型已通过测试。
- 不自动调用 reviewer；仅用户明确要求审查时适用父目录分工。

## 实验与知识维护

- 逐阶段、尽量单变量验证；正式对照匹配初始化、训练预算、输入视角、分辨率及监督条件。
- 同表面一致性不能等同于所有邻居深度相等；排除遮挡边界、多层表面，保留绝对表面监督和覆盖检查。
- 必须展示可直接检查的 demo：激光叠加、token 着色、局部截面与 render strip；同时报告失败案例。
- 新实验放在 `EXP/`：`EXP/<实验名>/` 单层存放 `summary.json`、`scene_metrics.csv`、`frame_metrics.csv`、`run.log`（训练实验另存 `model.safetensors`，包括从头训练）；报告 `EXP/report/<实验名>.md`；demo `EXP/demo/scene_7b6477cb95/<实验名>/`；一键入口 `EXP/run_train.sh <preset>`，默认 `spatial_sfm`。训练 workspace 放 `/mnt/disk/xveyan/workspace_scannetpp/geometrygs_<实验名>`，验证深度缓存按协议设置区分。
- 代码在 `geometrygs/`，以注册模型、子类化 Options/Provider 的方式复用 `../TokenGS`，不改 TokenGS 仓库。新预设加在 `geometrygs/options.py`。改代码后运行 `PYTHONPATH=$PWD /mnt/disk/jinsoo/anaconda3/envs/tokengs/bin/python -m pytest -q tests`。
- SfM 依赖 kornia 0.9.0 / kornia-rs 0.2.0（已装入 tokengs 环境）。
- DA3：官方代码 `/home/jms/xveyan/4DGS/Depth-Anything-3`（只加 `src` 到 PYTHONPATH，不 pip 安装包本身），权重 `/mnt/disk/xveyan/checkpoints/DA3-GIANT-1.1`（CC BY-NC 4.0）。其推理依赖已以固定版本装入 tokengs 环境；不要安装 xformers（会牵动 torch）。DA3 训练用过约 230 个未公开的 ScanNet++ 场景，只有其基准 20 个场景（`SCANNETPP_SCENES`，均在官方 val 中）确认未见，DA3 相关评测以这 20 个为主，demo 用 `fb5a96b1a2`。`EXP/run_train.sh` 只用无 compute 进程的严格空闲 GPU；续训要求相同 GPU 数（world_size 是恢复校验项）。
- `docs/legacy_tokengs/` 是原始知识快照，不在其中续写新实验；新决定和结果更新 `docs/CURRENT_STATE.md` 及对应新报告。
- 旧文献笔记是检索线索，正式引用、公式与新颖性结论须回到原文确认。
