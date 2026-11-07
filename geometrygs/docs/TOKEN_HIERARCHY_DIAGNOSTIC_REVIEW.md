# Scene latent 与 GS token：历史诊断回顾

本次仅回顾已有源码、报告、JSON 和实验定义，没有训练或重新运行诊断。主对象是 TokenGS release latent 6v；论文版 TokenGS / LocusGS 的结构不同，不能直接混为同一模型。

## 当前解释

scene latent 是从多视角图像特征中形成的、局部聚焦但彼此重叠的场景表示，包含可解码的空间信息和全局上下文。它不是纯二维 patch，也不是已知坐标的准确表面点。

GS token 是带有可学习查询/残差状态的高斯生成槽位。解码器让它选择、组合 scene latent 内容，生成特征 h，再由共享读出头生成 64 个子高斯。实测读取区域与渲染贡献区域高度吻合，但区域不排他，三维几何存在深度长尾、局部离群和跨 token 错层。

必须区分初始可学习 GS query 与读取场景后的 h；后者已经携带当前场景的信息。两级都含几何和外观，不能二分为“scene latent 只有外观、GS token 才有几何”。scene latent 与 GS token 编号不对应。

## 1. 注意力与局部区域

- D0：demo 场景第 3/7 层 16 个头中 13/12 个为局部头，局部头三维 RMS 分散度约 7–15 cm；第 11 层局部头降到 7 个，出现更多全局读取。局部头跨视角严格 hit@8 约 0.40。三维落点由激光赋深度，不能据此宣称网络预测深度达到 2–3 cm。
- 50 场景区域/重叠诊断：每个 latent 的局部头注意力 90% 集合中位 17 个图像 patch；每个 patch 被中位 6 个 latent 覆盖；18.9% latent 有注意力余弦 >0.8 的近重复伙伴。此处不是完整特征向量重复。
- 区域三维 RMS 中位 8.3 cm。9.3% latent 跨越满足两侧至少各 10% 权重的 >15 cm 深度空隙；不等于语义物体划分错误。早期以深度跨度判断“64% 多表面”的口径会混入斜面，不采用。
- 140 场景 DL3DV 复测：每 latent 15 patch、每 patch 6 latent、16% 近重复，局部+重叠现象复现。

来源：TokenGS `docs/DIAG_latent_attention_geometry.md`、`DIAG_scene_latent_region.json`、`DIAG_scene_latent_overlap.json`；`EXP_DL3DV/diag_dl3dv_report.md`。

## 2. 两级的可解码几何

对冻结特征训练额外探针，使用 ScanNet++ 853 训练场景 × 3 窗口，50 验证窗口评测。最终激光目标为单元注意力主视角射线的首个网格交点，不再用注意力平均点作为目标。集合 Transformer 探针读取同窗口的全部单元。

| 对象 | 到网格距离中位/p90 (cm) | 沿预测射线深度误差中位/p90 (cm) |
|---|---:|---:|
| scene latent 探针 | 2.9 / 13.1 | 5.5 / 31.3 |
| GS 特征 h 探针 | 3.0 / 14.5 | 5.8 / 38.4 |
| 官方 64 子高斯不透明度加权中心 | 3.4 / 31.3 | 7.1 / 68.2 |

这是相同评价定义下的不同单元集合，不能把分位数相减理解成逐单元因果损失。实际输出中心也不是渲染深度或表面，更不是每个子高斯的位置；它可能被离群子高斯影响。

scene latent 探针中，注意力视角数为 1 / ≥4 时深度中位为 10.9 / 4.2 cm，p90 为 53.2 / 19.6 cm；GS h 对应中位 13.8 / 4.3 cm。注意力视角数是注意力质量阈值统计，不是完整的真实可见性计数。

探针容量/数据曲线趋平、深度长尾与少视角/远距离关联，说明在所测试解码器和训练预算下难以读出准确深度，不证明特征的理论信息上限。“冻结 latent 深度只能到 5 cm”“上游信息必然缺失”应降格为经验发现。

最终表面口径的 lateral≈0.2 cm 是预测射线网格命中点到自身注意力表面区域的最近距离，不是到唯一真实三维点的横向定位误差。

来源：TokenGS `docs/DIAG_scene_latent_surface_metrics_lasertrained.json`、`DIAG_gs_token_surface_metrics.json`；`scripts/probe_gs_token_limit.py`、`probe_scene_latent_region.py`。

## 3. GS token 选择与生成的关系

- D1：解码器单 head 常集中读取 1–1.5 个有效 latent；20 场景 head 平均后的有效数约 4.1。不能说整个 GS token 只读取一个 latent。
- 50 场景精确 one-hot token 光栅化：每 token 渲染权重中位 87.2% 在其注意力 90% patch 集合内，按渲染量加权 83.3%，反向约 85.7%。attention 用 A_dec × 第 7 层局部头 A_lat 构成，是责任区域代理，不是严格的全网络信息流分解。
- 两个区域的激光质心距离中位 2.5 cm，只证明区域对应，不证明高斯中心距真值 2.5 cm。仅激光有效像素中，约 39% 合成权重来自离网格 >10 cm 的高斯。
- DL3DV 复测区域吻合中位 84%；每像素贡献 ≥1% 的 token 数中位 11，凑满 90% 权重需中位 8 个。局部表达存在，但不排他。

来源：TokenGS `docs/DIAG_decoder_attention_geometry_20scenes.json`、`DIAG_token_render_ownership.json`；`scripts/diag_token_render_ownership.py`；`EXP_DL3DV/diag_dl3dv_report.md`。

## 4. 输出头与容量

冻结官方 h，离线训练 64 点位置头（直接双向三维区域监督，不训练不透明度、不渲染）：

| 头 | 子点到责任区域中位/p90 (cm) | 区域 3 cm recall |
|---|---:|---:|
| 微调线性头 lr=1e-4 | 7.2 / 46.1 | 0.382 |
| 小残差 MLP | 4.55 / 29.1 | 0.458 |
| 两层集合 Transformer 残差 | 4.60 / 22.7 | 0.469 |

支持现有 h 可以通过不同读出/监督生成更好的三维布局，集合交互在长尾上有额外收益。不是已经完成的渲染模型结果，也不能证明线性头是唯一瓶颈。此处全部 64 个位置都计分，低透明度远点也会计入；不能把官方全点 p90 403 cm 当可见表面厚度。

DL3DV 140 场景的中心探针未复现“探针优于实际输出”：实际相对深度中位 0.022，h 的 MLP 0.054；探针数据较少且过拟合，目标为稀疏 COLMAP 质心。结论必须限制在具体实验设置。

旧子槽位诊断还发现固定 child 编号的存活差异。DL3DV 140 场景有 8 个 child 槽位在 ≥99% token 中低于存活阈值，说明 64 个输出槽位利用不均；仅凭最终权重不能反推它们何时死亡、预训练全过程是否一直无梯度。

来源：TokenGS `docs/DIAG_readout_capacity.json`；`scripts/probe_readout_capacity.py`；`EXP_DL3DV/diag_dl3dv_report.md`。

## 5. 几何协作与干预证据

- 官方 ScanNet++ 厚度方差合并后约 90% 来自 token 间，后续逐像素占比中位约 72%；两者分母和聚合不同，不能解释为每个像素都 90%。这是深度分布的方差分解，不是全部几何误差。
- 激光平面内部的 E1 将真实法向均方误差分成共同偏差/token 间分歧/token 内分散：release 76/22/2%，最强 SFT 71/17/12%，LocusGS 58/39/4%（四舍五入）。release 共同偏差中位 10.3 cm、token 间分歧 6.9 cm；SFT 降为 2.7/3.0 cm。共同位置错误和彼此错层都存在。
- LocusGS 自注意力确实偏向附近锚点，但仍扎堆；这说明交流存在不足以保证正确几何协调。中间扰动会恢复原预测，也不等于恢复真值。不要复用旧“门控锁死、改不动”的解释。
- 历史 DL3DV 140 场景、50 步场景内优化（仅作为历史干预，不恢复该路线）：all-view PSNR 无优化 25.194、只调 scene latent 27.400、只调初始 GS query 26.810、两者一起 27.481。novel PSNR 对应 25.007/26.845/26.439/26.901。说明两个状态都是可用控制通道且有互补性；学习率不同（1e-2 vs 1e-4），不构成层级重要性或几何准确度排名。

来源：TokenGS `docs/DIAG_thickness_split.json`；`EXP_DL3DV/diag_dl3dv_report.md`；`EXP_GeometryCorrectability/README.md`；`EXP_LocusGS_Diagnosis/README.md`（结合后续纠正）；`EXP_TokenGS/EXP_TokenGS.md` 593–626 行。

## 6. 不应沿用的解释

1. “scene latent 只做二维切块”：不成立，它跨视角局部聚合并含可解码几何。
2. “scene latent 已经几何全对，只有读出坏了”：早期 2–3 cm 来自激光辅助注意力落点；最终探针仍有深度长尾。
3. “GS token 只是决定看哪里”：h 本身含场景几何，且通过初始查询的残差路径与读出头共同决定输出。
4. “每个 token 是互不重叠的一片表面”：两级都有重叠；GS 渲染也由多个 token 共同贡献。
5. “scene latent 不会分平面/折角”：旧区域类型探针把同编号 scene latent 与 GS 标签配对，源码明确它只是未对齐的随机控制；近随机结果不能当 scene latent 能力结论。GS h 的小样本类型探针也只有约 52–53% balanced accuracy，远未证明可靠的表面类型理解。
6. “加 attention / 三维 PE 自然消除分层”：未验证；原特征已经含位置，新增结构的作用应是让相对空间关系显式可用，并约束实际输出一致性，不能只凭新增模块宣称改进。

当前 SfM 方案的准确定位：将部分 GS 生成槽位的中心显式绑定到可靠测量，限制局部输出；自由槽位保持补充覆盖能力。它并非从零赋予模型空间信息，而是在已有隐式区域组织上增加可靠的绝对位置参照和输出约束。是否还需改变 scene latent，应由受控结果决定。
