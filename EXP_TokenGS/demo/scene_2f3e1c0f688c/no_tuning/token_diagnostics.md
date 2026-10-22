# no_tuning Gaussian token 详细诊断

- 数据：`EXP_TokenGS/demo/scene_2f3e1c0f688c/no_tuning/gaussians.pt`，4096 个 token × 64 个 child = 262,144 个 Gaussian。
- 本报告逐 token 数字来自当前预注册 demo 场景；140 场景部分来自正式 `scene_metrics.jsonl` 的场景级聚合，140 场景运行没有保存每个场景的 4096-token 明细。

## 结论

1. **“一个 token 解出的点会很散”这个观察成立。** 3489/4096 (85.2%) 个 token 的 packing ratio 高于几何目标；1771/4096 (43.2%) 达到“强分散”标准。
2. 中位 token 的质量加权 RMS 散开半径为 0.2573，Gaussian RMS 尺寸为 0.06215；相对几何目标，spread/scale 仍多出 1.74 倍。
3. token 云之间高度交叠：4085/4096 (99.7%) 的 RMS 半径已经大于最近 token 中心距离；按 p90 半径计算是 99.9%。
4. 分散不是只有透明点造成：强分散 token 承担 31.5% 的总 opacity；最高 opacity 四分位中仍有 21.2% 强分散。
5. **但这不等于“把每个 token 强行收紧一定正确”。** Gaussian token 是全局 latent slot，官方架构没有规定一个 token 必须对应一个局部物体或空间块；现有几何损失确实收紧了统计量，却在 140 场景上降低画质。所以描述性出发点正确，作为优化假设仍未被验证。

## 排除低 opacity child 后重新统计

- 此处只在分析中保留 `opacity > 1/255` 的 child；共保留 183,220/262,144 (69.9%) 个 Gaussian。有 1 个 token 完全没有超过阈值的 child，故不参与几何复算。
- 每个 token 的中心、Gaussian 尺寸、packing target 和最近 token 中心都用过滤后的 child 从头计算。

| 指标 | 原始 64 child | 过滤后 |
|---|---:|---:|
| packing excess 中位数 | 3.03 | 0.4694 |
| packing 高于目标的 token | 85.2% | 28.7% |
| 强分散 token | 43.2% | 3.5% |
| RMS 散开半径中位数 | 0.2573 | 0.08077 |
| Gaussian RMS 尺寸中位数 | 0.06215 | 0.0625 |
| RMS 半径大于最近 token 中心距 | 99.7% | 99.3% |

过滤后结果用于检验结论是否由低 opacity 远端点驱动；该阈值是分析口径，不代表 feed-forward 或渲染器执行了清理。

## “散”的量化口径

- token 中心及 spread 均按 `opacity × (scale_x scale_y scale_z)^(2/3)` 加权，与现有 geometry loss 完全一致。
- `packing_ratio = 子中心每轴方差 / Gaussian 每轴尺度方差`；目标是 `effective_children^(2/3)`。
- `packing_excess = packing_ratio / target`。大于 1 表示比目标散；大于 4 表示 spread/scale 超目标 2 倍，本报告把它称为“强分散”。这是一条透明、可复算的工程阈值，不是论文给出的自然定律。
- opacity 总量只是 64 个 activated opacity 的和；未考虑投影面积、遮挡和视角，因此不能直接当作最终渲染贡献。

## 4096 个 token 的分布

| 指标 | p10 | p50 | p90 | p99 | 均值 |
|---|---:|---:|---:|---:|---:|
| 每 token opacity 总量 | 0.9267 | 3.261 | 6.733 | 10.18 | 3.631 |
| 存活 child 数 | 37 | 49 | 51 | 53 | 44.73 |
| 有效 child 数（opacity-area） | 7.97 | 15.07 | 22.49 | 28.39 | 15.21 |
| RMS 散开半径 | 0.143 | 0.2573 | 0.5005 | 0.8095 | 0.2951 |
| child 中心最大直径 | 2.58 | 3.269 | 7.032 | 13.82 | 4.144 |
| Gaussian RMS 尺寸 | 0.01135 | 0.06215 | 0.09347 | 0.1096 | 0.05494 |
| packing ratio | 4.482 | 17.56 | 1192 | 4371 | 367 |
| packing target | 3.99 | 6.1 | 7.968 | 9.307 | 6.044 |
| packing excess | 0.7995 | 3.03 | 207.2 | 722.1 | 60.57 |
| spread/scale 超目标倍数 | 0.8941 | 1.741 | 14.39 | 26.87 | 4.747 |
| 最近 token 中心距离 | 0.004301 | 0.0181 | 0.09932 | 0.2654 | 0.039 |
| RMS 半径 / 最近中心距 | 2.799 | 12.97 | 66.59 | 219.5 | 30.73 |

## opacity 在 4096 个 token 中如何分配

- 全部 child opacity 总和：14872.534；token opacity Gini：0.351。
- opacity 最高的前 10% token 承担 22.7% 的总 opacity。
- 累计 50% opacity 需要 1118/4096 个 token；累计 90% 需要 2873/4096 个 token。
- 全体 child 存活率（opacity > 1/255）：69.9%；64 个 child 全部存活的 token 为 0，无存活 child 的 token 为 1。
- log(opacity 总量) 与 log(packing excess) 的 Spearman 相关：-0.377。

## opacity 总量最高的 10 个 token

| token | opacity 总量 | 存活 child | packing excess | spread/scale 超目标倍数 | RMS 半径 / 最近中心距 |
|---:|---:|---:|---:|---:|---:|
| 604 | 15.16 | 39 | 2.319 | 1.523 | 8.152 |
| 64 | 13.62 | 42 | 3.415 | 1.848 | 5.005 |
| 947 | 13.12 | 37 | 2.46 | 1.569 | 3.293 |
| 1088 | 12.76 | 39 | 2.678 | 1.636 | 8.672 |
| 3136 | 12.71 | 40 | 3.664 | 1.914 | 5.131 |
| 2112 | 12.67 | 39 | 2.355 | 1.535 | 8.152 |
| 472 | 12.55 | 38 | 4.319 | 2.078 | 2.813 |
| 136 | 12.43 | 38 | 2.856 | 1.690 | 6.181 |
| 543 | 12.09 | 39 | 2.43 | 1.559 | 6.325 |
| 1567 | 11.89 | 39 | 2.204 | 1.485 | 7.672 |

## 140 场景总体证据（场景级聚合）

- 140 个场景中，packing ratio/target 全部高于 1：140/140；全部高于强分散阈值 4：140/140。
- 场景级 packing ratio：p10 144.62 / p50 470.36 / p90 1387.98；目标中位数仅 6.08。
- 场景级 packing excess 中位数 76.3；对应 spread/scale 超目标中位数 8.74 倍。
- inter 最近邻 Mahalanobis 距离均值的场景中位数为 0.472（目标 2.0）；过近边比例的场景中位数为 91.9%。
- 这说明 demo 不是孤例；但 140 场景只保留每场景聚合，不能声称每个场景的每个 token 都满足同一结论。

## 文件

- `token_diagnostics.csv`：4096 行逐 token 完整明细。
- `token_diagnostics_alive_filtered.csv`：排除 opacity ≤ 1/255 后的逐 token 明细。
- `token_diagnostics.json`：定义、原始/过滤后全局统计和 140 场景摘要。
- `token_diagnostics.png`：packing、opacity、重叠和二者关系的四联图。
