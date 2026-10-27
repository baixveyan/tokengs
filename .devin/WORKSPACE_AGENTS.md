# xveyan 工作区规则

## 工作区结构

- 根目录不是 git 仓库。子项目各自是独立仓库，例如 `3DGS/3dgs`、`3DGS/cloud-refiner`、`3DGS/fused_pointcloud`、`4DGS/FFRGS`、`4DGS/StreamSplat`、`4DGS/TokenGS`、`4DGS/WAFT-Stereo`。
- `data/` 是指向 `/mnt/disk/xveyan` 的数据目录，不要全量扫描、修改或删除其中内容。
- 不要解压、覆盖或删除 `*.tar.gz` 归档文件，除非用户明确要求。

## 双 Agent 分工

用户已明确要求使用以下分工：

- 主 Agent（本会话，默认 Gemini 4 Argon）：与用户讨论方案、探索代码与论文思路、编写代码、运行测试。
- `gpt-reviewer` 子 Agent（GPT-6.1 Sol XHigh）：只读审查代码变更和研究方案。
- 如果主会话本身已由 GPT 主导（例如以 GPT 为 Lead 的 Fusion），不需要再调用 `gpt-reviewer`，由主 Agent 自行审查。

### 何时调用 gpt-reviewer（仅限用户明确要求，严禁自动触发）

- **默认一律不调用 `gpt-reviewer`**：任何代码编写、修改、重构、实验设计与脚本交付完成后，主 Agent 通过本地测试（如 `pytest`）自行验证并**立即直接向用户汇报**，**如果用户没有明确要求 review，绝对不要调用 `gpt-reviewer`**。
- **仅当用户明确发出指令**（如“审一下”“调用 gpt-reviewer”“复核方案”）时，才调用一次 `gpt-reviewer`。

### 委派与单轮闭环（仅在用户要求时适用）

1. **直接提供文件路径与行号（免去多轮 `git diff`）**：委派时直接列出修改文件的**绝对路径与关键行号范围**、研究目标及核心数学/实验约束，让 `gpt-reviewer` 第一轮直接并行 `read` 一步读完。
2. **单轮闭环，严禁二审**：`gpt-reviewer` 返回结果后，主 Agent 自行核实每条发现、修复确认的问题并跑通 `pytest` 即可直接向用户汇报，**不再调用第二轮复审**。
