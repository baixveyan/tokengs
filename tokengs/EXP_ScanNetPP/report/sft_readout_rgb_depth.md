# 读出链路 + 单项渲染深度损失（`sft_readout_rgb_depth`）

状态：已完成（2026-10-27 训练，50 场景结果见 `EXP_ScanNetPP/all_experiments_table.md` 与 `docs/AGENT_MEMORY.md` 15.13；下文为训练前写的设计与预期）。

## 一、动机

读出链路阶梯第一步（`docs/AGENT_MEMORY.md` 15.12）：在 `sft_readout_rgb` 的完全相同设置上**只加一项**渲染深度损失，回答“结构不动、读出链路可训时，一个最简单的几何监督能把几何教到什么程度、画质是否守得住”。与 `sft_readout_rgb` 的差异仅在损失项，其余逐项一致。

## 二、设计（与 `sft_readout_rgb` 的差异）

- **新增损失**：渲染期望深度与网格真值深度的**对数 L1**，只在有真值的像素上计算（`gt_ok`），6 个输入视角 + 4 个目标视角都算；权重 `lambda_render_depth = 0.1`。
- **升温**：几何损失总权重在第 1 个 Epoch 内从 0 线性升到 1（`render_geo_warmup_epochs = 1.0`），避免开局画质骤降。
- **未启用**：alpha 实心项、视线厚度项、法线项（权重 0）；它们留给后续读出层步骤。
- 其余（初始化、解冻范围与头部学习率缩放、数据、轮数、学习率、精度、验证与评测）与 `sft_readout_rgb` 完全相同。

## 三、事先定好的判读规则

| 观察 | 结论 | 下一步 |
|---|---|---|
| 深度 RMSE / AbsRel 明显下降，PSNR 掉幅 ≤ 0.2 dB（相对 `sft_readout_rgb`） | 路线成立 | 进下一步（自由空间惩罚；实际执行时改为视线厚度损失，即 `sft_readout_rgb_depth_thick`） |
| 深度下降但 PSNR 掉 > 0.5 dB | 几何与画质拉锯 | 先按区域分析掉在哪，再决定调权重或转稠密读出头 |
| 深度不降 | 梯度未到位或权重太小 | 查 `loss_render_depth` 曲线后再议 |
| 厚度 p90 不动 | 符合预期（期望深度对了不代表雾散了） | 这正是下一步（厚度/自由空间约束）的立项依据 |

跑完后在 Demo 场景重跑解码器读取诊断（`scripts/diagnose_decoder_attention_geometry.py`）与 20 个高亮 token 的 PLY，直观看子球是否收向墙面。

## 四、结果

待运行完成后自动写入 `../sft_readout_rgb_depth/` 与 `../all_experiments_table.md`。
