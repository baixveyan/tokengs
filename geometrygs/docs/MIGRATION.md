# 知识迁移记录

## 范围

- 来源：`/home/jms/xveyan/4DGS/TokenGS/docs/`。
- 目标：`/home/jms/xveyan/4DGS/GeometryGS/docs/legacy_tokengs/`。
- 完整复制 32 个文件，总文件内容 752,946 bytes，包含全部 Markdown 与诊断 JSON。
- 复制方式为独立文件副本，保留原目录结构和原文；不是符号链接。源目录没有修改或删除。
- 根 README、AGENTS、CURRENT_STATE、KNOWLEDGE_INDEX 和本文为新编交接材料，不属于旧文档快照。

## 完整性

`legacy_tokengs.sha256` 记录快照各文件 SHA-256，路径相对本 `docs/` 目录。迁移时比较源和目标的全部相对文件名、逐文件大小及 SHA-256，并检查诊断 JSON 可解析。

后续检查快照：

```bash
cd /home/jms/xveyan/4DGS/GeometryGS/docs
sha256sum --check legacy_tokengs.sha256
```

哈希只验证迁移完整性，不证明历史研究解释正确。新模型没有代码变更或测试结果；无需把这次文件校验记成模型 pytest 通过。

## 边界

第一批只迁移知识。第二批按用户要求从原仓库 `EXP_ScanNetPP`、`EXP_QueryGS_Diagnosis`、`EXP_LocusGS_Diagnosis`、`EXP_GeometryCorrectability` 中**按路线筛选**复制：11 组评测、1 个最强 SFT 权重、相关报告/外推/demo，共 344 个文件。筛选标准和未迁入清单见 RESULTS_INDEX。（中途曾完整复制一次，按用户纠正删除了多余副本，删除前逐个核对原文件存在且大小一致；原目录未动。）

保留文件与源文件的 SHA-256 记录在 `results_migration_manifest.json`。旧入口脚本未迁入。

模型源码、数据集、官方预训练基线权重及外部工作区缓存仍未迁入。PLY、图片和报告可以在新目录直接查看；运行模型仍需后续源码与依赖移植。

新决定与结果写入新项目文档，不续写 `legacy_tokengs/`，以便保持来源可追溯。目录名 `GeometryGS` 是暂用工作名。
