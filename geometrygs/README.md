# GeometryGS

保留 scene latent / GS token 两级表示、对齐 TokenGS 上游容量的空间高斯模型。`GeometryGS` 暂为工作名，不代表论文正式命名。主模型复用预训练图像/scene 聚合，重新训练空间生成路径；旧 SfM-TokenGS 后训练版本仍然保留。

目标是利用真实几何训练出更准确、薄且完整的高斯表面，保持图像细节，并让同一模型支持：

- RGB + 相机；
- RGB + 相机 + 可选点云。

当前已迁入旧知识，以及按路线筛选的历史小实验、最强模型权重和相关 demo（未选中的仍在原 TokenGS 目录）。已有结果与路线对应关系见 [结果索引](docs/RESULTS_INDEX.md)。

当前主模型 `geometrygs_spatial` 已实现：官方预训练图像/射线及 scene 聚合 → 恒等初始化的 SfM 融合 → 具有三维中心的锚定/自由 GS tokens → 空间 decoder → 局部 child MLP。基础部分含 SfM 与空间编码，软链默认关闭。默认 **454.51M 参数（340.78M 上游迁移，113.73M 新参数）**、1024 维、4096 scene latents、4096 GS tokens、每 token 64 个高斯；没有 1024→512 压缩。

复用 `../TokenGS` 的原图像/scene 块及训练、数据、渲染和评测接口，不使用旧 GS decoder/线性头，不修改 TokenGS 仓库。上游维度、归一化、LayerScale、多尺度路径完整保留并严格加载。实现、参数和测试范围见 [架构说明](docs/SPATIAL_ARCHITECTURE.md)。

```bash
bash EXP/run_train.sh spatial_sfm              # 默认：完整宽度、预训练上游、新空间生成器
bash EXP/run_train.sh spatial_free             # 对照：无 SfM
bash EXP/run_train.sh spatial_sfm_chain        # 可选：增加同表面法向软链
# 每条命令独立运行；使用全部严格空闲 GPU，结束后 50 场景 FP32 评测 + demo
# GPUS="1 7" bash EXP/run_train.sh spatial_sfm  # 限制允许使用的卡，仍检查空闲
# bash EXP/run_train.sh sfm_anchor_k1024        # 旧后训练模型
# bash EXP/run_train.sh spatial_small_sfm      # 旧 75M 随机初始化版本，仅容量消融
PYTHONPATH=$PWD python -m pytest -q tests       # 单元测试（tokengs 环境）
```

当前 454.51M 模型已完成工程冒烟，尚未正式训练，记录在 `EXP/spatial_tokengs_smoke/`。旧 `EXP/spatial_full_smoke` 和 `EXP/demo/scene_7b6477cb95/spatial_smoke` 属于 75M 小模型调试产物；两组 smoke 都仅在验证 demo 场景做过一步更新，不能作为泛化评测或方法效果引用。

## 接手阅读顺序

1. [项目规则](AGENTS.md)。
2. [当前状态、可靠结论与更正](docs/CURRENT_STATE.md)：区分用户确定的目标、架构建议、已经验证的结果。
3. [旧知识索引与外部依赖](docs/KNOWLEDGE_INDEX.md)：按问题定位历史文档、诊断数据和原仓库证据。
4. [架构路线草案](docs/legacy_tokengs/ARCHITECTURE_ROADMAP_2026.md)：先读顶部新路线；后半段是历史方案。
5. [完整历史记忆](docs/legacy_tokengs/AGENT_MEMORY.md)：有冲突时结合末尾更正与当前状态解读，不把早期“最新方向”当现行决定。

## 目录

```text
GeometryGS/
├── README.md
├── AGENTS.md
├── geometrygs/                 # 新模型：sfm.py（GPU SfM）、model.py、data.py、options.py、train.py、evaluate.py
├── scripts/check_sfm_quality.py
├── tests/
├── EXP/                        # 新实验：<实验名>/ 结果、report/<实验名>.md、demo/、run_train.sh
├── EXP_ScanNetPP/              # 筛选后的旧结果、最强模型权重、报告、外推及 demo
├── EXP_QueryGS_Diagnosis/      # 多模型三维诊断
├── EXP_LocusGS_Diagnosis/      # LocusGS 诊断与 demo
├── EXP_GeometryCorrectability/ # 可恢复性验证与 demo
└── docs/
    ├── CURRENT_STATE.md
    ├── KNOWLEDGE_INDEX.md
    ├── RESULTS_INDEX.md
    ├── MIGRATION.md
    ├── results_migration_manifest.json
    ├── legacy_tokengs.sha256
    └── legacy_tokengs/        # TokenGS/docs 的全部 32 个原文件
```

迁移采用独立复制，原 TokenGS 文档保留不动。旧文档中的历史相对路径可能依赖原仓库目录；定位方式见知识索引。迁移清单与完整性核验见 [MIGRATION.md](docs/MIGRATION.md)。
