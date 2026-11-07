# 按路线筛选迁入的实验结果与 demo

只迁入与新路线直接相关的历史小实验。源目录 `/home/jms/xveyan/4DGS/TokenGS/` 原样保留，未选中的实验仍可在那里查看。迁入文件与源文件逐一 SHA-256 一致，清单见 `results_migration_manifest.json`（344 个文件，约 2.1 GiB）。迁入操作本身没有重新训练或评测。

## 新增本地实验

- [da3_refined_occupancy_10cm](../EXP/report/da3_refined_occupancy_10cm.md)：DA3 + 边缘过滤 + 输入视角 SfM 尺度，20 个未见场景 IoU 0.429（原始 0.326），深度误差中位 2.8 cm；多视角一致性检查无增益。
- [da3_flying_pixels](../EXP/report/da3_flying_pixels.md)：DA3 沿视线连接前后表面的原因是深度跳变处的飞点；25% 的远处误报格只由飞点产生；DA3 边缘过滤 τ=0.1 使 IoU 0.326→0.360。
- [da3_occupancy_10cm](../EXP/report/da3_occupancy_10cm.md)：预训练 DA3-GIANT-1.1 深度体素化，20 个 DA3 未见场景 IoU 0.326、召回 63.5%、精确率 40.1%；深度误差中位 4.7 cm；demo `fb5a96b1a2`。
- [grid_occupancy_10cm](../EXP/report/grid_occupancy_10cm.md)：128/32/50 场景，10 cm 网格二分类（输入可见激光表面为正，无未知）。图像探针验证 AUC 0.775（相机 0.680），校准 99% 阈值下召回 96%、负格关闭 17%、表面附近 6%；失败主因是沿视线深度不可分辨。
- [grid_activation](../EXP/report/grid_activation.md)：16/4/8 场景的网格统计与冻结特征激活探针。三维投影图像特征的独立验证 AUC=0.8964；独立校准的高召回门控未实现有效关闭。包含有效高斯分布及三维叠加 demo，不是完整重建模型训练。

## 路线阶段 → 保留的证据

| 路线阶段 | 保留的实验（`../EXP_ScanNetPP/<名>/`） | 它回答什么 |
|---|---|---|
| 0 对照 | `baseline_pretrained_6view` | 起点 |
| 0 对照 | `sft_latenttail_rgb`、`sft_readout_rgb` | 只用画质损失的微调能带来多少（域适应、画质上限参考） |
| 0 对照 | `sft_readout_rgb_depth` | 直接深度监督的收益；与 `sft_readout_rgb` 解冻范围相同，可直接配对 |
| 1 几何读出 | `sft_readout_rgb_depth_surface3d` → `sft_readout_rgb_depth_surface3d_posmlp` | 三维贴面监督、位置残差 MLP 各自的作用 |
| 1 几何读出 | `sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20`（**含权重**） | 历史最强几何模型，新方法必须超过它；也可作为热启动或教师 |
| 2 厚度 | `sft_readout_rgb_depth_thick` | 渲染厚度损失能压薄，但画质、完整度变差（负面证据） |
| 2 厚度 | `sft_readout_rgb_depth_thick_posmlp_surfel_binar` | 面元法向限幅 + 不透明度二极化进一步伤画质（一次改两项，只作警示） |
| 3 跨 token 一致性 | `../EXP_GeometryCorrectability/` 的 E1 | 同表面冲突的定义、数值与可视化，可直接作为阶段 3 度量 |
| LocusGS 参考 | `baseline_locusgs_dl3dv_4v`、`baseline_tokengs_paper_dl3dv_4v` | 外部基线在 ScanNet++ 上的指标 |
| LocusGS 参考 | `../EXP_LocusGS_Diagnosis/` 的 D3、D5、D7、D8 | LocusGS 如何把几何注入注意力、sigma 下限、表面覆盖、锚点扎堆 |

每个保留的实验目录有 `summary.json`、`scene_metrics.csv`（50 场景）、`frame_metrics.csv`（400 帧）、`run.log`。只有最强模型保留了 `model.safetensors`；其他对照不作为新实验起点，权重留在原目录。

## 关键数字（旧 summary 均值）

厚度为旧渲染中心深度分散指标，不等同于完整高斯体积厚度。

| 模型 | PSNR ↑ | AbsRel ↓ | 厚度 p90 (cm) ↓ | Accuracy (cm) ↓ | Completeness (cm) ↓ | F@5 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 官方 release | 30.50 | 0.174 | 25.7 | 22.4 | 10.2 | 0.445 |
| 读出 + 深度 | 30.50 | 0.103 | 28.0 | 11.3 | 8.2 | 0.557 |
| 读出 + 深度 + 厚度 | 29.85 | 0.110 | 7.1 | 11.7 | 11.8 | 0.500 |
| 最强：latent 尾段 + 读出 + 深度 + 贴面 + 位置 MLP，20 epoch | 30.69 | 0.066 | 13.6 | 6.9 | 7.4 | 0.641 |

厚度损失能压薄，但相对单纯深度监督，画质、完整度与 F@5 都变差。因此阶段 2 不应重做“加厚度损失”，而要检验表面法向约束能否避开这些代价。

其余实验数字见原总表 [all_experiments_table.md](../EXP_ScanNetPP/all_experiments_table.md)（含未迁入实验的行）；外推见 [extrapolation_table.md](../EXP_ScanNetPP/eval_extrapolation/extrapolation_table.md)，只迁入了保留实验的外推子目录。

## demo

- 标准场景 `../EXP_ScanNetPP/demo/scene_7b6477cb95/`：官方基线、读出 + 深度、读出 + 深度 + 厚度、最强模型四组，配 `gt_lidar_mesh_frustum.ply` 激光参照。每组有 `render_strip.png`、完整/剪枝/slotcolor PLY、`cameras.ply`、TSDF mesh。`comparison.md` 是旧的多模型对比表，会提到未迁入的模型。
- [查询模型诊断](../EXP_QueryGS_Diagnosis/README.md)：只保留 ScanNet++ 场景（release、论文版、LocusGS、最强 SFT 四模型的 token 着色、ray probes、注意力与渲染归属、表面距离）。README 里提到的 DL3DV 场景未迁入。
- [LocusGS 诊断](../EXP_LocusGS_Diagnosis/README.md)：保留 `stats_50_scenes.json` 与 D3/D5/D7/D8 demo；README 中 D1、D2、D4、D6、D9–D11 的链接在本目录失效，需要时到原目录查看。阅读时以 CURRENT_STATE 的更正为准。
- [几何可恢复性](../EXP_GeometryCorrectability/README.md)：保留 E1 冲突图与平面点 PLY、`results.json`（含 E1–E3 数字）、分散-误差事后分析；E2/E3 的 demo 未迁入。

PLY 可用 VS Code 的 `kleinicke.ply-visualizer` 查看。跨目录叠加前先核对各 README 的坐标系。

## 未迁入的内容及原因

新增实验 `sfm_anchor_k1024` 已于 2026-11-06 完成：50 场景/400 帧，PSNR 30.6237、AbsRel 0.07019、厚度 p90 13.42 cm、F@5 0.62705。结果及权重在 `EXP/sfm_anchor_k1024/`，报告在 `EXP/report/sfm_anchor_k1024.md`，demo 在 `EXP/demo/scene_7b6477cb95/sfm_anchor_k1024/`。本次仍带贴面损失；锚定/自由高斯分工诊断尚未完成。

| 未迁入 | 原因 |
|---|---|
| `sft_readout_rgb_depth_surface3d_posmlp_ep20`、`sft_latenttail_readout_rgb_depth_surface3d_posmlp`（10 epoch） | 被保留的对照对和最强 ep20 覆盖 |
| `sft_readout_rgb_depth_thick_sh1`、`sft_readout_rgb_depth_thick_posmlp` | 与厚度结论重复 |
| `..._attrmlp_ep20`、`..._opamlp` | 已停止，无正式结果（历史见旧记忆） |
| 除最强模型外的所有权重 | 新实验从官方预训练开始，不需要 |
| 旧一键脚本、`renamed_old_to_new.md`、`TokenGS分析汇报`、旧配图、三角化 token 报告 | 依赖旧仓库或不在当前路线 |
| QueryGS 的 DL3DV 场景、LocusGS 其余 D 项、E2/E3 demo | 不在当前路线，数字已在记忆和 results 中 |
