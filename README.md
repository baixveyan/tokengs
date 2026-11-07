# TokenGS & GeometryGS Research Workspace

本仓库包含两个并列的研究目录：

- **`tokengs/`**：基于官方 TokenGS (`nv-tlabs/TokenGS`) 的本地研究副本、全套三维几何与注意力诊断（`docs/DIAG_*`、`EXP_QueryGS_Diagnosis/`、`EXP_LocusGS_Diagnosis/`、`EXP_GeometryCorrectability/`）、ScanNet++ 与 DL3DV 实验报告与评测脚本，以及共享背景记忆 [`tokengs/docs/AGENT_MEMORY.md`](tokengs/docs/AGENT_MEMORY.md)。
- **`geometrygs/`**：保留 `scene_latents` / `gs_tokens` 两级表示、对齐 TokenGS 上游容量的空间高斯生成模型（`geometrygs/`）、GPU SfM 锚点模块、网格占据与 DA3 占据/飞点探针、实验结果（`EXP/`）以及项目当前状态记忆 [`geometrygs/docs/CURRENT_STATE.md`](geometrygs/docs/CURRENT_STATE.md) 与知识索引 [`geometrygs/docs/KNOWLEDGE_INDEX.md`](geometrygs/docs/KNOWLEDGE_INDEX.md)。

## 快速测试

```bash
# 运行 geometrygs 单元测试（自动定位同级 ../tokengs 或 ../TokenGS）：
cd geometrygs
PYTHONPATH=$PWD python -m pytest -q tests

# 运行 tokengs 单元测试：
cd ../tokengs
PYTHONPATH=$PWD python -m pytest -q tests
```
