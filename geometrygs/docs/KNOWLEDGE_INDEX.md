# 旧知识索引

源目录：`/home/jms/xveyan/4DGS/TokenGS/docs`。

`legacy_tokengs/` 完整保留源目录全部 32 文件，不只摘选 Markdown。JSON 保留原始诊断数值和统计口径，未重算。先读 [CURRENT_STATE.md](CURRENT_STATE.md) 获取最新边界和更正。

## 按问题查阅

当前完整宽度、预训练上游与新空间生成器的实现及验证记录优先看 [SPATIAL_ARCHITECTURE.md](SPATIAL_ARCHITECTURE.md)。75M 全随机版本已降为容量消融，旧路线与最新决定有冲突时以当前文档为准。

三维 query 的下一步参考：[AnchorSplat 源码学习](ANCHORSPLAT_CODE_STUDY.md)。包括官方版本、投影采样、KV/生成点分工、固定坐标、深度依赖及与当前模型的差异；属于研究方案，尚未替换当前实现。

DA3 可部署改进：[da3_refined_occupancy_10cm](../EXP/report/da3_refined_occupancy_10cm.md)。SfM 尺度是主要收益；AnchorSplat 式一致性规则会删掉单视角区域。

DA3 飞点诊断：[da3_flying_pixels](../EXP/report/da3_flying_pixels.md)。边界深度平滑过渡→点沿视线连接前后表面；可部署的边缘过滤及其代价。

预训练深度模型做网格占据：[da3_occupancy_10cm](../EXP/report/da3_occupancy_10cm.md)。DA3 深度→10 cm 格，IoU 0.326（探针 0.132），误差多为前后偏一格；含 DA3 环境、许可证、训练集重叠（只有 20 个验证场景确认未见）与激光噪声参照。

网格占据的正式协议：[grid_occupancy_10cm](../EXP/report/grid_occupancy_10cm.md)。激光包围盒＋10 cm 固定格，输入可见表面为正、其余为负；冻结 TokenGS 特征只能定方向、难定深度，高召回下关闭率低。

网格方案的首轮实测：[grid_activation](../EXP/report/grid_activation.md)。包含占据/未知标签、官方有效高斯容量分布、冻结特征共享探针、独立阈值校准及失败案例。排序有信号，但高召回硬关闭尚未成功；不是新的 GS 模型评测。

两个 token 层级的综合证据与旧结论纠正，优先看 [TOKEN_HIERARCHY_DIAGNOSTIC_REVIEW.md](TOKEN_HIERARCHY_DIAGNOSTIC_REVIEW.md)：重新核对了注意力区域、几何探针、渲染归属、读出容量、跨 token 冲突和历史场景内干预，区分实验结果与过度推断。

| 问题 | 本地知识 |
|---|---|
| 全部历史、环境、数据、checkpoint、协议与更正 | [AGENT_MEMORY.md](legacy_tokengs/AGENT_MEMORY.md)，重点第 13–15 节及末尾更新 |
| 本轮增量架构与历史失败 | [ARCHITECTURE_ROADMAP_2026.md](legacy_tokengs/ARCHITECTURE_ROADMAP_2026.md)，顶部当前草案、后部历史 |
| 旧 SFT 参数和监督作用位置 | [SFT_ARCHITECTURE_REPORT.md](legacy_tokengs/SFT_ARCHITECTURE_REPORT.md)，属于历史配方，不等同于最新最强模型 |
| 文献、点云教师、WorldMirror 等线索 | [LITERATURE_REVIEW_2026.md](legacy_tokengs/LITERATURE_REVIEW_2026.md) |
| FFRGS 参考与下载同步经验 | [FFRGS_REFERENCE.md](legacy_tokengs/FFRGS_REFERENCE.md)、[SYNC_AND_DOWNLOAD_GUIDE.md](legacy_tokengs/SYNC_AND_DOWNLOAD_GUIDE.md) |
| latent 注意力与几何 | `DIAG_latent_attention_geometry.md`、同名 JSON、`DIAG_latent_attention_geometry_heads.json` |
| decoder 读取位置与几何 | `DIAG_decoder_attention_geometry*.json` |
| latent 定位、覆盖、重叠和容量 | `DIAG_scene_latent_*.json` |
| token 形状、归属与高斯表面距离 | `DIAG_token_*.json`、`DIAG_gs_token_*.json` |
| 读出容量、无效槽位与旧平面头 | `DIAG_readout_capacity.json`、`DIAG_junk_slots.json`、`DIAG_surface_head_smoke.json` |
| 厚度分解、压薄的画质代价 | `DIAG_thickness_split*.json`、`DIAG_thinning_image_cost.json`、`DIAG_voxel_snap.json` |
| 视差与几何误差 | `DIAG_parallax_vs_geometry_error.json` |

以上所有 `DIAG_*` 均位于 `legacy_tokengs/`。完整逐文件清单及摘要值见 [legacy_tokengs.sha256](legacy_tokengs.sha256)。

## 原仓库外部证据

下列内容不在旧 `docs/` 内。后续按用户要求，`EXP_ScanNetPP`、`EXP_QueryGS_Diagnosis`、`EXP_LocusGS_Diagnosis`、`EXP_GeometryCorrectability` 已按路线筛选后复制到本项目同名目录，见 [RESULTS_INDEX.md](RESULTS_INDEX.md)。下表保留原来源路径以便追溯；源码、DL3DV 实验日志及数据仍依赖原位置。

| 内容 | 已核实的源位置 |
|---|---|
| 官方 ScanNet++ 新基线报告 | `/home/jms/xveyan/4DGS/TokenGS/EXP_ScanNetPP/report/baseline_pretrained_6view.md` |
| 几何可恢复性协议、结果与 demo 索引 | `/home/jms/xveyan/4DGS/TokenGS/EXP_GeometryCorrectability/README.md` |
| LocusGS 深度诊断及其 demo | `/home/jms/xveyan/4DGS/TokenGS/EXP_LocusGS_Diagnosis/README.md`，结合 CURRENT_STATE 更正阅读 |
| 查询模型可视化诊断 | `/home/jms/xveyan/4DGS/TokenGS/EXP_QueryGS_Diagnosis/README.md` |
| DL3DV 历史实验日志 | `/home/jms/xveyan/4DGS/TokenGS/EXP_TokenGS/EXP_TokenGS.md` |
| 既有读出头和残差分支 | `/home/jms/xveyan/4DGS/TokenGS/tokengs/models/activations.py` |
| ScanNet++ 评测协议与总表生成 | `/home/jms/xveyan/4DGS/TokenGS/scripts/scannetpp_evaluation_protocol.py` |
| 空闲 GPU 与一键启动公共工具 | `/home/jms/xveyan/4DGS/TokenGS/scripts/one_click_launch_common.sh` |
| 几何可恢复性诊断实现 | `/home/jms/xveyan/4DGS/TokenGS/scripts/verify_geometry_correctability.py` |

历史最强 SFT 数字、权重和 demo 已在本地 `EXP_ScanNetPP` 对应目录。按命名猜测的 `EXP_ScanNetPP/report/sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20.md` 在迁移检查时不存在，因此不生成虚假报告链接；查看实际 summary、总表、日志及旧记忆。

## 大型外部资产

- 主模型与旧实验仓库：`/home/jms/xveyan/4DGS/TokenGS`。
- 对照模型仓库：`/home/jms/xveyan/4DGS/LocusGS`。
- 数据根：`/home/jms/xveyan/data/4dgs`；ScanNet++ 位于其 `scannetpp/`，DL3DV 位于其 `DL3DV-10K-Benchmark/`。
- LocusGS 权重历史位置：`/mnt/disk/xveyan/checkpoints/locusgs/`；本次未加载或检查权重内容。
- 官方 TokenGS 与 SFT 权重路径在 AGENT_MEMORY 和原加载脚本中核对；不凭历史简写选择 checkpoint。

## 历史链接的处理

原文保持逐字节一致，所以其中 `docs/...`、`EXP_.../...` 等仓库路径仍相对原 TokenGS 根解析；Markdown 相对链接也可能指向原目录结构。不要为修复链接而修改快照。新文档使用当前项目可用的相对链接，外部依赖使用原仓库绝对路径。
