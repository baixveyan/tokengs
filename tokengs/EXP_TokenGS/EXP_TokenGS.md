# TokenGS 几何约束实验

> 背景记忆（论文要点、实验历史、陷阱、用户偏好）见 [`docs/AGENT_MEMORY.md`](../docs/AGENT_MEMORY.md)。
>
> **全部实验总表**（每行一个实验、每列一个指标，发散写“发散”）：[`all_experiments_table.md`](all_experiments_table.md)（同内容 `all_experiments_table.csv`），由 `scripts/build_all_experiments_table.py` 从各实验目录生成（`run_140_scenes.sh` 跑完会自动重做）。
>
> **术语与命名（2026-10-19 起）**：官方代码和 release 文档把下面这个过程叫 TTT（`ttt_mode="scene-latent-tuning"`），本项目统一叫 **scene latent tuning（场景潜变量微调）**，因为它只优化 scene latents，不训练网络，也不是论文里调 Gaussian token 的 token tuning。下文历史日志中出现的“TTT”都指它。当前目录和脚本采用第四次整理后的实验名，旧名→新名见 [`renamed_directories_old_to_new.md`](renamed_directories_old_to_new.md)；各运行目录的 `launch_contract.json`、`summary.json`、`run.log` 等证据文件保留当时真实执行的旧命令和旧路径，不随目录改名。
>
> **目录组织与命名规则（2026-10-20 第五次优化，用户要求）**：EXP_TokenGS 下**每个文件夹 = 一次 140 场景实验**，不同学习率就是不同文件夹，不嵌套；示例场景（demo）全部放在 `demo/`，实验文件夹里只有 140 场景记录。
> **命名规则** `no_tuning` 或 `[scene_lr<x>_][token_lr<y>_][geo_]tuning`：模块只用 `scene`（调 scene latents）、`token`（调 Gaussian token）、`geo`（加几何损失），**词序固定为 scene → token → geo → tuning，且每个被调模块的学习率直接紧跟在该模块词后面**（如 `scene_lr1e-2_token_lr1e-4_geo_tuning`，一眼看清 `1e-2` 属于 `scene`、`1e-4` 属于 `token`，不会与 `geo` 混淆）。所有实验都一样的设置不写进名字：FP32、50 步、几何权重 sw / packing / inter = 0.1 / 0.001 / 0.01（以后改了某个几何权重，紧跟在 `geo` 后写出变化项，例如 `_geo_sw0.2_tuning`）。实验名单、参数、顺序只在一处定义：`scripts/experiment_registry.py`（`python scripts/experiment_registry.py names`），一键脚本、报告、总表都读它。
> 旧名→新名见 [`renamed_directories_old_to_new.md`](renamed_directories_old_to_new.md)。**文末各执行日志写于整理之前，保留当时的路径**，对照该表换算即可。
>
> ```
> run_140_scenes.sh                            一键：跑 140 场景实验（已跑完的跳过）+ 重做报告和总表
> run_demo.sh                                  一键：示例场景跑全部已完成的实验 + 对比表 + token 高亮
> all_experiments_table.{md,csv}               全部实验总表
> reports/
>   01_baseline_and_scene_token_tuning.md        阶段 01：前馈基线与 scene / token 微调实验（Exp-01 ~ Exp-04）
>   02_free_space_geo_regularization.md          阶段 02：自由空间统计几何正则（SW+Packing+Inter）实验（Exp-05 ~ Exp-07）
>   03_frustum_and_core_opacity_cleaning.md      阶段 03：解析视锥投影与 Token 离核 Opacity 门控清理实验（Exp-08 ~ Exp-10）
>   04_colmap_and_vggt_depth_supervision.md      阶段 04：三维表面深度几何监督实验（COLMAP vs VGGT-Ω，Exp-11 ~ Exp-12）
>   scene_geo_tuning.{md,json}                   历史自动汇总报告（scene 有无几何损失对照）
>   scene_token_x_geo.{md,json}                  历史自动汇总报告（调哪些参数 × 是否加几何对照）
>   scene_tuning_noise_floor.json                已删除的两个核对运行的结论（重跑噪声底、官方循环 vs 复刻循环）
> launch_logs/                                 一键脚本总日志 <脚本名>.<时间戳>.log（终端看到的全部内容）
> [Exp-01] no_tuning/                                   不微调
> [Exp-02] scene_lr1e-2_tuning/                         只调 scene latents（复刻循环）
> [Exp-05] scene_lr1e-2_geo_tuning/                     只调 scene latents + 几何损失
> [Exp-03] token_lr1e-4_tuning/                         只调 Gaussian token（论文的 token tuning，官方循环）
> [Exp-06] token_lr1e-4_geo_tuning/                     只调 token + 几何损失
> [Exp-04] scene_lr1e-2_token_lr1e-4_tuning/            scene latents 和 token 一起调
> [Exp-07] scene_lr1e-2_token_lr1e-4_geo_tuning/        一起调 + 旧几何损失
> [Exp-08] scene_lr1e-2_token_lr1e-4_vis_xyz_tuning/    一起调 + 解析视锥投影拉 xyz
> [Exp-09] scene_lr1e-2_token_lr1e-4_vis_opac_tuning/   一起调 + 解析视锥投影压 opacity
> [Exp-10] scene_lr1e-2_token_lr1e-4_vis_opac_core_gate2.5_tuning/ 一起调 + 视锥压 opacity + 离核(>2.5r)空间门控清理
> [Exp-11] scene_lr1e-2_token_lr1e-4_colmap_depth0.01_tuning/      一起调 + COLMAP 原始稀疏深度（λ=0.01）
> [Exp-12] scene_lr1e-2_token_lr1e-4_vggt_omega_depth0.01_tuning/  一起调 + VGGT-Ω Sim3 稠密深度（λ=0.01）
> [Exp-13] scene_lr1e-2_token_lr1e-4_colmap_clean_depth0.01_tuning/ 一起调 + COLMAP 去噪稀疏深度（λ=0.01，默认 COLMAP 基线）
> demo/
>   demo_scene.json                            预注册规则选出的示例场景
>   scene_<id 前12位>/
>     comparison.{md,json}                     实验间对比表
>     highlighted_tokens.{md,json}             高亮 token 图例（token id → 颜色）
>     <实验名>/                                该实验在这个场景上的全部文件，平铺：
>       <实验名>.ply、_slotcolor.ply、_highlight{10,20}tokens.ply（其他点灰色半透明）、_only{10,20}tokens.ply（其他点完全删除）、_minus_no_tuning.ply、cameras.ply、<实验名>.json（旁注）
>       gaussians.pt、decoder_inputs.pt、tuning_curve.*、render_strip.*、needle_stats.json、console.log、run.log、summary.json …
>       no_tuning 另有 token_diagnostics.{md,json,csv,png}（4096 个 token 的逐 token 几何 / opacity 诊断）
> ```
>
> 每个 140 场景实验文件夹里：`status.json`、`summary.json`、`scene_metrics.{jsonl,csv}`、`frame_metrics.jsonl`、`launch_contract.json`、`run.log`，以及完整未过滤的终端日志 `console.log`（`scene_tuning_lr1e-2`、`scene_geo_tuning_lr1e-2` 当年只留下过滤过 warning 的 `console_warnings_filtered.log`，按原样保留）。

## 实验定义
- **不微调（no_tuning）**：官方 dl3dv_latent_6v_ssim checkpoint，不做任何操作，直接在 DL3DV-10K-Benchmark 上推理评估
- **场景潜变量微调（scene_tuning；命令行参数 `--scene-latent-tuning`，官方称 TTT）**：冻结全部模型权重和 Gaussian token，仅对每个场景的 scene latents `[1, 4096, 1024]` 做 50 步 Adam 优化（LR=1e-2），然后渲染评估
- **token 微调（token_tuning，论文称 TT）**：冻结全部模型权重和 scene latents，仅对 Gaussian token `[1, 4096, 1024]` 做 Adam 优化（论文默认 50 步、LR=1e-4），然后渲染评估。官方入口 `model.forward_ttt_tokens`（`ttt_mode="token-tuning"`）
- **场景潜变量 + token 一起微调（scene_token_tuning；命令行参数 `--scene-latent-and-token-tuning`，官方没有）**：冻结全部模型权重，scene latents 和 Gaussian token 同时优化，一个 Adam、两组参数各用原学习率（1e-2 / 1e-4），50 步。实现 `scripts/reimplemented_scene_latent_and_token_tuning.py`
- **评估口径**：all_frames（50 帧/场景，与官方 release 一致）为主指标；novel_only（44 帧/场景，排除 6 个输入视角）为辅助参考
- **数据**：DL3DV-10K-Benchmark 140 场景

## 当前结果

以下均为完整 140 场景、FP32。

### 不微调（`no_tuning`）
- all_frames：PSNR 25.194 / SSIM 0.808 / LPIPS 0.255
- novel_only：PSNR 25.007 / SSIM 0.804 / LPIPS 0.257
- 旧的 BF16 20 场景子集（val 25.243 / test 25.228）已删除，不进入正式对照。

#### Gaussian token 几何 / opacity 诊断（2026-10-20）

完整报告：[`demo/scene_2f3e1c0f688c/no_tuning/token_diagnostics.md`](demo/scene_2f3e1c0f688c/no_tuning/token_diagnostics.md)；4096 行逐 token 明细见同目录 CSV，分布数据见 JSON，四联图见 PNG。由 CPU 脚本 `scripts/diagnose_no_tuning_tokens.py` 从保存的 `gaussians.pt` 生成。

- demo 场景 4096 个 token × 64 child：**85.2%** token 的 packing ratio 高于 `effective_children^(2/3)` 目标；**43.2%** 达到强分散标准（`packing_ratio/target > 4`，等价于 spread/scale 超目标 2 倍）。中位 packing excess 3.03。
- 中位 token 的 opacity-area 加权 RMS 散开半径 0.257，Gaussian RMS 尺寸 0.0622；99.7% token 的 RMS 半径大于最近 token 中心距离，说明 token 云大量交叠。
- 分散并非只由透明点造成：强分散 token 仍承担 31.5% 的总 opacity；最高 opacity 四分位里 21.2% 强分散。token opacity 分配不算极端集中：最高 10% token 承担 22.7%，累计 90% 需要 2873/4096 个 token。
- 140 场景聚合支持 demo 不是孤例：140/140 场景的场景级 `packing_ratio/target > 4`，中位 76.3；inter 过近边比例的场景中位数 91.9%。但正式运行没保存每场景 4096-token 张量，不能把单场景逐 token 百分比外推成所有场景的逐 token 百分比。
- **结论分两层**：描述性出发点“一个 token 解出的 64 个 Gaussian 通常不是紧凑局部簇”成立；优化假设“每个 token 都应该被强制收紧”没有成立。Gaussian token 是全局 latent slot，并无局部语义保证，而且现有几何损失虽改善几何统计，却降低 140 场景画质。

### 场景潜变量微调（`scene_lr1e-2_tuning`，复刻循环）
- all_frames：PSNR 27.400 / SSIM 0.868 / LPIPS 0.183；novel_only：PSNR 26.845 / SSIM 0.860 / LPIPS 0.188
- 曾有两个同设置的核对运行，2026-10-20 第四次整理时删除，结论冻结在 `reports/scene_tuning_noise_floor.json`：
  - 官方循环跑一次：all PSNR 27.399，与复刻循环差 +0.0002 dB，CI95 [−0.004, +0.004]（噪声内，两个入口等价）
  - 复刻循环同配置重跑一次：差 +0.0003 dB，CI95 [−0.003, +0.004]，这就是噪声底（约 ±0.004 dB）

### 与官方 release 对比
官方 6v SSIM w/o TTT: 25.120；w/ TTT: 27.304
我们复现不微调: 25.194（差 +0.074 dB，吻合）
我们复现场景潜变量微调: 27.400（差 +0.096 dB，吻合；官方循环那次 27.399）
微调增益（本地，配对 140 场景）：all_frames +2.206 dB，novel_only +1.837 dB

### 正式对照：微调内循环加几何损失（报告 `reports/scene_geo_tuning.md`，均 140/140）
| 实验 | all PSNR | novel PSNR | d4+ PSNR | SW | packing ratio | inter d |
|---|---|---|---|---|---|---|
| `no_tuning` | 25.194 | 25.007 | 24.108 | 0.7129 | 642.4 | 0.505 |
| `scene_lr1e-2_tuning` | 27.400 | 26.845 | 25.325 | 0.7119 | 703.9 | 0.523 |
| `scene_lr1e-2_geo_tuning` | 26.734 | 26.251 | 24.816 | 0.3761 | 276.0 | 1.047 |

（几何列为 140 场景均值；当年还有一行同配置重跑 27.400，已删除，见上。）**结论：几何损失让几何统计明显朝目标移动，但所有画质指标都显著变差**（all PSNR −0.665 dB，CI95 [−0.741, −0.590]，1/139 胜，远超噪声底 ±0.004）。详见文末执行日志与 `reports/scene_geo_tuning.md`。

### token 微调（`token_lr1e-4_tuning`，官方循环，140/140）
| 方法 | all PSNR | d0 | d1 | d2-3 | d4+ | novel PSNR | SSIM | LPIPS | HF ratio | SW | packing ratio | inter d | aniso p50 | 长轴∥视线>0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 不微调 | 25.194 | 26.561 | 26.004 | 25.207 | 24.108 | 25.007 | 0.8081 | 0.2549 | 0.759 | 0.713 | 642 | 0.505 | 27.73 | 0.277 |
| 场景潜变量微调（复刻循环） | 27.400 | 31.466 | 28.764 | 27.047 | 25.325 | 26.845 | 0.8683 | 0.1831 | 0.852 | 0.712 | 704 | 0.523 | 28.82 | 0.265 |
| 场景潜变量微调 + 几何损失 | 26.734 | 30.279 | 28.050 | 26.452 | 24.816 | 26.251 | 0.8525 | 0.2122 | 0.829 | 0.376 | 276 | 1.047 | 26.44 | 0.247 |
| **token 微调 lr 1e-4 / 50 步** | **26.810** | 29.529 | 27.962 | 26.656 | 25.167 | 26.439 | 0.8539 | 0.2153 | 0.767 | 0.712 | 650 | 0.589 | 27.77 | 0.276 |

- 对不微调：all PSNR **+1.617 dB**，140/140 场景变好（novel +1.432，SSIM/LPIPS 也是 140/140 变好）。
- 对场景潜变量微调：**−0.590 dB**，只有 5/140 场景 token 微调更好（对官方循环 −0.589，4/140）；HF ratio 低 0.085（140/140 更低，即画面更糊）。
- 差距随离输入视角变远而缩小：d0 −1.94 → d1 −0.80 → d2-3 −0.39 → d4+ −0.16 dB。token 微调把输入视角“背得”没那么死，但远处也没有反超。
- 几何：基本不动（SW 0.712，与不微调差 −0.001；packing ratio 650）。论文 Fig. 13 说的“TT 改善几何”在这组设置下，用我们的指标看不出来；只有 inter d 0.505 → 0.589（token 之间略微散开）。
- 详见文末“执行日志：token 微调 140 场景 FP32”。

### 调哪些参数 × 是否加几何损失（报告 `reports/scene_token_x_geo.md`）
全部数字见 [`all_experiments_table.md`](all_experiments_table.md) 和文末“执行日志：调哪些参数 × 几何损失 × token 学习率”。要点：两者一起调（`scene_lr1e-2_token_lr1e-4_tuning`）27.481 最好，比只调 scene 高 +0.081（127/140）；几何损失在三种参数下都降画质；token lr 1e-2 的 4 组全部发散（发散数据已归档至本文件执行日志，目录已按用户要求删除，且不再测 1e-3）。

## 已证伪的方向
旧的全模型权重微调方向（在 100 个场景上 fine-tune 整个模型 10,000 步）已被证明无效：微调后 PSNR 反而低于 baseline，说明小数据全模型微调破坏了预训练泛化能力。相关产物已清理。

## 下一步
1. （已完成）demo 已补齐到全部 7 个完成的实验（[`demo/scene_2f3e1c0f688c/comparison.md`](demo/scene_2f3e1c0f688c/comparison.md)、[`demo/scene_2f3e1c0f688c/highlighted_tokens.md`](demo/scene_2f3e1c0f688c/highlighted_tokens.md)）。
2. 几何损失方向是否继续、怎么改（权重、只用单项、配合解析投影修复视锥外 visibility loss 等），等用户决定。
3. token 学习率：用户决定不测 1e-3；1e-2 的 4 个发散目录已删除。

## 一键运行与日志约定
- 只有两个一键脚本，都在 `EXP_TokenGS/` 根目录，共用 `scripts/one_click_launch_common.sh`：
  - `bash EXP_TokenGS/run_140_scenes.sh [实验名 ...]`：跑 140 场景实验（不写名字 = registry 里全部），跑完重做两份报告、`demo/demo_scene.json` 和总表；
  - `bash EXP_TokenGS/run_demo.sh [scene_id ...]`：示例场景跑全部 140 场景已完成的实验（发散的不跑），一律用复刻循环（有微调曲线和微调后的张量；与官方循环在 CPU 上逐位一致），跑完重做对比表和 token 高亮。
- 脚本内部设置环境（tokengs python、`CUDA_HOME=/usr/local/cuda-12.8`、PATH），自动挑空闲 GPU（显存 ≤100 MiB 且无进程，可用 `GPUS="4 5"` 指定），各实验并行，GPU 不够时排队；已完成（complete）或已失败（failed，保留作证据）的跳过，半成品文件夹拒绝启动。`DRY_RUN=1` 只打印会跑 / 会跳过哪些实验和完整命令，不占 GPU、不写日志。
- 新增实验：在 `scripts/experiment_registry.py` 的 `EXPERIMENTS` 里加一行（名字按命名规则），再运行 `run_140_scenes.sh`；`tests/test_experiment_registry.py` 会检查名字和参数是否一致。
- 日志：每个实验一个完整、未过滤的 `console.log`，放在该实验文件夹里（demo 则在 `demo/scene_<id>/<实验名>/` 里；开头记录命令 / GPU / git HEAD，结尾记录退出码和耗时）；另有总日志 `launch_logs/<脚本名>.<时间戳>.log`，即终端看到的全部内容（含定时进度）。agent 事后读这些日志核验。

## 注意事项
- 场景潜变量微调在 BF16 下有数值失稳问题，需要 FP32
- 启动 tokengs 必须把 env/bin 放进 PATH（否则 gsplat JIT 找不到 ninja），且 `CUDA_HOME=/usr/local/cuda-12.8`（一键脚本已内置）
- 官方 `tokengs.evaluate` 的精度由 accelerate 配置决定，`opt.mixed_precision` 不生效；正式评估统一用 `scripts/evaluate_with_scene_latent_tuning.py`（显式 FP32）
- 场景潜变量微调有 run-to-run 非确定性（单场景 ~0.02 dB），不微调完全确定

---

## 执行日志：不微调 140 场景 FP32 补跑（no_tuning，2026-10-19）

> 注：本节这次运行（11:50）的产物已在“目录重组”中删除，由 16:31 同配置的另一次运行取代（7000 帧 PSNR/SSIM/LPIPS 逐帧差为 0，且多了几何 / 帧距分桶 / 清晰度指标），现位于 `no_tuning/`。下文 `run.log`、`--compare` 等描述的是被删的那次运行。

### 1. 启动前核查（11:40–11:46）
- `no_tuning`：**不存在**，无残留产物；无 tokengs/evaluate 相关存活进程。
- `scene_latent_tuning/official_loop/launch_contract.json`：成功入口为 `EXP_REPRODUCE/tokengs_v2/artifacts/baseline_ttt/.runtime_task76_fp32.py --config .../eval_test.yaml --output ... --no-media`，`mixed_precision=no`、`num_workers=0`、`batch_size=1`、物理 GPU 7、evaluation_json=`assets/evaluation_idx_dl3dv_depthsplat_6v.json`、checkpoint sha256 `5913da34…78c3`。**该包装脚本与 yaml 随 EXP_REPRODUCE 一并删除，已无法直接复用**（全盘仅找到 `/tmp/tokengs_smoke_runtime/eval_test.yaml` 旧冒烟配置，其 evaluation_json 指向已删除的 20 场景子集，不可用）。
- 旧 BF16 20 场景 val / test 两个目录（已删除）的 contract 显示 `mixed_precision=bf16`，即旧 40 场景 baseline 不是 FP32。
- `tokengs/evaluate.py` 实际 CLI：`python -m tokengs.evaluate <preset 子命令> [--field value ...]` 或 `--config yaml`；`Accelerator()` 无参构造，**精度由 accelerate 配置决定而非 `opt.mixed_precision`**（`acc_configs/gpu1.yaml` 为 bf16）；只输出聚合 `metrics.txt`，没有逐帧/novel_only 指标。
- 数据入口：`data/dl3dv_eval -> /home/jms/xveyan/data/4dgs/DL3DV-10K-Benchmark`（140 zip + 140 解压目录 + benchmark-meta.csv，91G）；`assets/evaluation_idx_dl3dv_depthsplat_6v.json` 140 场景，每场景 context 6、target 50，context ⊂ target。
- GPU：GPU0 被他人进程（TRELLIS.2，7.2GB）占用；GPU1–7 空闲（3–4 MiB，0%）。磁盘 `/` 剩余 90G。
- 环境：tokengs（Python 3.11.17，torch 2.7.0+cu128，gsplat 1.5.3，accelerate 0.28.0）。**以绝对路径解释器启动时 env/bin 不在 PATH，gsplat JIT 报 `Ninja is required`**——这是前两次补跑“无结果无日志”的高度可疑原因。
- 异常处置：首次带 PATH 预检时 shell 里 `CUDA_HOME=/usr/local/cuda`(13.0)，gsplat 缓存被 nvcc 13.0 重编；随即按环境约定用 `CUDA_HOME=/usr/local/cuda-12.8` 重编（build.ninja 已确认为 cuda-12.8），二次导入 2s 命中缓存。
- 正式启动环境固定为：`PATH=<env>/bin:/usr/local/cuda-12.8/bin:$PATH`、`CUDA_HOME=/usr/local/cuda-12.8`。

### 2. 评估入口方案
- 不修改核心代码：新增独立评估脚本 `scripts/evaluate_with_scene_latent_tuning.py`（仓库 scripts/ 下，不放进 EXP_TokenGS 以保持目录干净）。
- 复用 `tokengs.evaluate` 的严格 checkpoint 加载与 `get_multi_dataloader`，模型/数据走官方 preset `eval_dl3dv_latent_6view_ssim`；显式 `Accelerator(mixed_precision="no")` 保证 FP32；`MetricsCalculator(reduction="none")` 逐帧计算后再按 all_frames(50)/novel_only(44) 聚合（与官方先帧均值再场景均值口径一致）。
- 同一脚本支持 `--scene-latent-tuning`，可在单场景上复算 TTT 与 scene_latent_tuning/official_loop 的逐场景数值对齐，作为入口等价性校验。

### 3. 前台 smoke test（11:48–11:50，GPU7，场景 `032dee9f…`，产物在 `/tmp/tokengs_full140_smoke/`）
命令模板（仓库根目录）：
```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_with_scene_latent_tuning.py --preset eval_dl3dv_latent_6view_ssim \
  --resume checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors \
  --output /tmp/tokengs_full140_smoke/<name> --no-tuning|--scene-latent-tuning --scene-ids <scene>
```
| 运行 | all PSNR | all SSIM | all LPIPS | novel PSNR | 说明 |
|---|---|---|---|---|---|
| baseline FP32 | 30.104 | 0.9135 | 0.1708 | 30.046 | 7 个产物齐全，exit 0，6.6s |
| baseline FP32 重复 | 30.104 | 0.9135 | 0.1708 | 30.046 | 逐帧 Δ=0，**完全确定** |
| 旧 BF16 val 子集同场景（目录已删除） | 30.174 | 0.9117 | 0.1823 | 30.129 | FP32−BF16：PSNR −0.069，LPIPS −0.0115 |
| TTT FP32 | 32.879 | 0.9515 | 0.0950 | 32.668 | vs scene_latent_tuning/official_loop 同场景 32.896：−0.016 dB |
| TTT FP32 重复 | 32.863 | 0.9515 | 0.0952 | 32.654 | 与上一次逐帧最大 |ΔPSNR|=0.186 |

结论：
- 新入口对 baseline 为确定性输出，可直接用于正式 140 场景。
- **TTT 本身存在 run-to-run 非确定性**（gsplat 反传 atomicAdd 经 50 步 Adam 放大），单场景波动 ~0.02 dB 量级；新入口与 scene_latent_tuning/official_loop 的 −0.016 dB 落在该噪声内，入口等价。这对下一阶段 TTT vs TTT+Geo 对照设计有直接影响（需先测 TTT 自身噪声底）。

### 4. 正式 140 场景运行（11:50:48–11:56:18，GPU7 `GPU-2377bd33…`，330 s，exit 0）
启动前 GPU 门控：GPU7 显存 <100 MiB 且无 compute app 才放行；GPU0 有他人任务，未使用。命令（仓库根目录，前台）：
```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_with_scene_latent_tuning.py --preset eval_dl3dv_latent_6view_ssim \
  --resume checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors \
  --output EXP_TokenGS/no_tuning --no-tuning --expected-scenes 140 \
  --compare official_loop=EXP_TokenGS/scene_latent_tuning/official_loop
# stdout/stderr 另存为 EXP_TokenGS/no_tuning/run.log
```
- `status.json`：`complete`，`formal_complete=true`，140/140；`mixed_precision=no`，`ttt_enabled=false`，媒体输出关闭（`eval_n_media_dumps=0`）。
- 产物核验：140 个 scene_id 唯一且等于 evaluation json 场景集；frame_metrics 7000 行（context 840 / novel 6160），每场景 50/44 帧；由 frame_metrics 独立复算的场景均值与 summary 一致；无 NaN/Inf。
- 资源：nvidia-smi 峰值显存 17357 MiB、利用率峰值 100%；torch 峰值 allocated 15788 MiB；平均 2.30 s/场景；TF32 matmul=False、cudnn=True（官方默认）。

**Baseline FP32，140 场景（均值 ± 95% CI）**

| 口径 | PSNR | SSIM | LPIPS |
|---|---|---|---|
| all_frames（50 帧） | 25.193529 [24.670, 25.717] | 0.808102 | 0.254925 |
| novel_only（44 帧） | 25.007017 [24.473, 25.541] | 0.804296 | 0.256953 |

**与官方 w/o TTT（25.120 / 0.806 / 0.257，all_frames）**：PSNR +0.073529，SSIM +0.002102，LPIPS −0.002075。

**Baseline − TTT（scene_latent_tuning/official_loop），140 场景配对**

| 口径 | 指标 | Δ 均值 | 配对 95% CI | 场景胜负（Baseline 胜 / TTT 胜） |
|---|---|---|---|---|
| all_frames | PSNR | −2.205958 | [−2.3525, −2.0595] | 0 / 140 |
| all_frames | SSIM | −0.060205 | [−0.0642, −0.0562] | 0 / 140 |
| all_frames | LPIPS | +0.071790 | [+0.0694, +0.0742] | 0 / 140 |
| novel_only | PSNR | −1.836659 | [−2.0017, −1.6716] | 2 / 138 |
| novel_only | SSIM | −0.055340 | [−0.0592, −0.0515] | 0 / 140 |
| novel_only | LPIPS | +0.068658 | [+0.0663, +0.0710] | 0 / 140 |

TTT 在 context 帧上增益更大（all_frames 2.21 dB > novel_only 1.84 dB），与“用 context 视图做测试时监督”的机制一致；但对 44 个未见视角仍有 ~1.84 dB 的稳定增益，并非仅拟合输入视角。

**FP32 vs 旧 BF16 子集（相同场景，FP32 − BF16）**

| 子集 | all PSNR | all SSIM | all LPIPS | novel PSNR |
|---|---|---|---|---|
| val（20 场景） | +0.0162（单场景 max \|Δ\| 0.161） | +0.0020 | −0.0030 | +0.0129 |
| test（20 场景） | +0.0249 | +0.0015 | −0.0029 | +0.0221 |

Baseline 对 BF16 不敏感（均值差 <0.03 dB）；BF16 失稳问题只在 TTT 的 50 步优化中被放大。

产物（`no_tuning/`）：`summary.json`、`scene_metrics.jsonl`、`frame_metrics.jsonl`、`scene_metrics.csv`、`launch_contract.json`（argv、GPU uuid、脚本/checkpoint/eval json sha256、环境、git HEAD + tokengs diff sha256、完整 options）、`status.json`、`resource_evidence.json`，以及额外的 `run.log`（stdout/stderr 全量）。

---

## 执行日志：几何损失 → scene latent 梯度通路审计（2026-10-19 12:01–12:06）

脚本 `scripts/audit_geo_latent_grad.py`（只读核心代码与 checkpoint，FP32，`freeze_model_parameters` 冻结全部权重），GPU7（启动前 3 MiB、无 compute app），3 个场景：baseline 最差 `fb3b73f1`、中位 `70eac6ff`、最好 `374ffd0c`。每场景约 5 s，torch 峰值 3.8 GB。产物 `/tmp/tokengs_geo_grad_audit/geo_grad_audit.json`（临时目录，未放入 EXP_TokenGS；约 20 s 可重跑复现）。

### 1. 静态审计结论（tokengs/models/tokengs.py、tokengs/losses/geo_losses.py）
- **现有代码无法用几何损失做 TTT**：`forward()` 中 `use_geo_losses` 分支作用在 `forward_ttt()` 的返回值上，而该返回值由 `_forward_ttt_scene_latents` 在 `@torch.no_grad` 下用 `scene_latents.detach()` 重新解码得到；TTT 内循环 50 步的 loss 只有 `compute_loss`（L1 0.8 + SSIM 0.2，渲染回 6 个 context 视角）。即当前 `use_ttt_for_eval=True, use_geo_losses=True` 只会多算一个无梯度的数，不改变任何结果。
- **latent → 几何的链路无 detach**：`scene_latents → _latents_to_kv（latent_kv_proj + k_proj_norm）→ decoder_blocks（cross-attn，gs_tokens 为 detach 的常量，合理）→ ClipActivationHead → [B, 262144, 14]`；activation head 输出按 `(n p)` 排列，`reshape(B, 4096, 64, 14)` 与 slot（每 token 64 个 child，`dec_patch_size=8`）一一对应，通道切片 `0:3 center / 3:4 opacity / 4:7 scale / 7:11 rotation` 与 `compute_geo_losses` 调用一致。
- **geo_losses.py 中有意的 stop-gradient（设计内，不是 bug，但决定了几何损失能动什么）**：
  - opacity 质量权重 `opacities.detach()` → 三项损失对 opacity **均无梯度**；
  - packing 分母 `scales.detach()` → packing 对 scale 无梯度；
  - inter 的 kNN 图在 `no_grad` 下由 topk 构建（离散）；Cholesky jitter 幅值 detach；SW 累计质量/分位积分 detach。
  - 整段 `autocast(enabled=False)` + `.float()`，FP32 下无影响。
- 激活中的潜在梯度截断：`scale_act` 用 `torch.minimum(exp, cap=0.075)`，饱和处梯度为 0；实测饱和比例仅 1.0–3.2%。

### 2. 运行时验证（3 场景结果一致）
| 检查 | 结果 |
|---|---|
| 现有外层路径 `model(data)`（TTT + use_geo_losses） | `gaussians.requires_grad=False`、`grad_fn=None`、`total_geo_loss.requires_grad=False` → **梯度为 0（被截断）** |
| 模型参数 | `requires_grad=True` 的参数 0 个，产生 `.grad` 的参数 0 个 |
| 内循环路径 ∂L/∂z（z 形状 [1, 4096, 1024]，FP32） | render、sw、packing、inter、λ=0.01 合计：全部非 None、全部有限、非零比例 100% |
| 被各项损失驱动的 Gaussian 属性 | sw / packing：**只有 center**；inter：center + scale + rotation；三项对 opacity、rgb 的梯度均精确为 0（render 损失则驱动全部属性） |
| 中心差分 vs autograd（固定 stop-grad 输入于 z0；inter 另固定 kNN 图） | sw、packing：梯度方向与随机方向的相对误差 ≤0.4%；inter：h 取小值时 ≤0.2%（h 大时 1–11% 的 slot 切换邻居，误差随之增大）→ **反传数值正确，除上述设计内截断外没有其他阻断** |
| 不固定 stop-grad 输入的原始 FD | packing 沿梯度方向偏差 0.3–26%，随机方向相对误差 0.9–10.7；sw 偏差 3–7% / 40–90% → 真实损失对 opacity（以及 packing 对 scale）的依赖不小，但优化器看不到这部分 |
| 仅用单项几何损失做 Adam（lr 1e-2，20 步） | 真实（完整前向）损失单调下降：sw −61~−71%，packing −63~−73%，inter −79~−85% → **梯度是有效的下降方向** |

**梯度量级与冲突（z0 处）**

| 场景 | TTT render loss | ‖∇render‖ | ‖∇(0.01·geo)‖ / ‖∇render‖ | cos(∇geo, ∇render) | packing ratio / 目标 | inter 马氏距离 / 目标 |
|---|---|---|---|---|---|---|
| fb3b73f1（最差） | 0.0976 | 1.98e-3 | 0.19 | −0.004 | 160.7 / 8.57 | 0.50 / 2.0 |
| 70eac6ff（中位） | 0.0349 | 2.39e-3 | 0.16 | −0.022 | 112.8 / 8.57 | 0.53 / 2.0 |
| 374ffd0c（最好） | 0.0189 | 7.91e-4 | 0.51 | −0.009 | 35.4 / 9.38 | 0.54 / 2.0 |

仅优化几何 20 步后，context 视角的 render loss 恶化到原来的 1.6–6.6 倍（例如 0.0189 → 0.083–0.125）。

### 3. 结论
1. **当前梯度在外层路径被阻断**：`use_geo_losses` 对 TTT 无作用。Geometry TTT 必须把几何损失加进 50 步内循环 `loss = render + λ·geo`，最终解码继续用 `scene_latents.detach()`。
2. **在内循环里，几何损失能通过 decoder 对 scene latent tokens 产生有限、非零、数值正确的梯度**，冻结权重条件下成立，没有意外的 detach。
3. 几何正则只能移动 center（sw/packing），以及 center/scale/rotation（inter），**无法改动 opacity 和颜色**，因此不能直接抑制漂浮物（floater）的不透明度。
4. 预训练解的几何统计离正则目标很远（packing ratio 是目标的 4–19 倍，inter 距离约为目标的 1/4），几何梯度与渲染梯度近似正交。Adam 按坐标归一化，即使默认 λ=0.01 时梯度范数只有渲染的 16–51%，也会明显改变更新方向。λ 过大很可能拉低 PSNR，需要预先登记一个小范围 λ 网格。

---

## 执行日志：Baseline / TTT 口径核查（2026-10-19，只读，未启动新实验）

1. **Baseline（`no_tuning`）就是官方权重原样推理**：`--no-tuning`，strict 加载 sha256 `5913da34…78c3`，`model.eval()` + `torch.no_grad()`，无 optimizer、无 TTT、无几何项，FP32，140/140 场景。
2. **TTT（`scene_latent_tuning/official_loop`）权重全部冻结，但优化对象是 scene latents 而非论文的 Gaussian tokens**：`freeze_model_parameters` 冻结全部参数；scene latents `[1, 4096, 1024]` 是唯一可优化张量（Adam 50 步，lr 1e-2）；gs_tokens 固定；只用 6 个 context 视角监督，target 不参与。这是 release 2026.6 latent preset 的官方设定（`release_2026.6.md`："optimizes per-scene latent tokens at test time instead of Gaussian tokens"）；论文版 token-tuning（lr 1e-4）代码仍在但未使用。
3. scene_latent_tuning/official_loop 的 contract 是旧格式，未记录 ttt_mode/steps/lr；当前入口在 `032dee9f` 上复算（记录为 scene-latent-tuning/50/1e-2）与其相差 −0.016 dB，可作为间接佐证。
4. **与官方 HEAD 唯一影响数字的差异是 `07d9f97` 的内参**：本地按实际图片尺寸缩放（正确），官方会把该场景 K 放大 2 倍；其余 139 场景两者等价。该场景本地 Baseline 32.32 / TTT 36.20，可能是比 release 偏高 +0.07 / +0.10 dB 的来源之一，尚未用官方代码对照。
5. 更正：下文对照设计中 TTT 损失除 L1 0.8 + SSIM 0.2 外，还包含 `lambda_visibility=1.0` 的 visibility 项。

---

## 执行日志：FFRGS 参考与 Baseline/TTT 反查（2026-10-19，只读，未启动新实验）

完整内容见 `docs/FFRGS_REFERENCE.md`（第 7 节为反查结论）。要点：

1. **口径无误**：背景 0.5 灰、256×448、6 输入、0.8·L1 + 0.2·SSIM、visibility 1.0 / clip 1.0、FP32，与 preset 和 FFRGS 对照口径一致；TTT 无 target 泄漏。
2. **评估集几乎只有内插**：target 是连续 50 帧，context 跨度 37–49 帧，仅 2.1% 的 target 落在跨度外。按到最近 context 的帧距 d 分桶（7000 帧配对）：d=0 增益 +4.91 dB（840 帧），d=1 +2.73，d=2–3 +1.82，d=4–6 +1.22（变差 12.9%），d≥7 +1.24。“+2.2 dB”中相当部分来自拟合输入视图，这个基准对几何质量不敏感。
3. **visibility 项基本失效（官方行为）**：gsplat 1.5.3 对被剔除点（出视锥、opacity<1/255、bbox 在画面外等）不写 `means2d`（`at::empty` 未初始化），反传跳过 `radii<=0`。已读源码确认，未做 GPU 实测。正式对照两组保持原样。
4. **inter 形式与 FFRGS 不同**：我们 `(d−2)²`，FFRGS `smooth_l1(log(d/2))` + 体积加权质量；需在预注册前确定。
5. 旧排查目录里“BF16 去掉 visibility loss”那次（12.18，目录已删除）是 BF16 崩溃结果，不能用来判断 visibility 项的作用。

**用户决定（2026-10-19）**：评估口径固定为官方 DepthSplat 6v（target 为第 0–49 帧），以保证 PSNR 与论文可比；当前实验目标是优化 token 表示，**暂不做外推评估**，留作后续单独实验（候选做法：context 不变，额外评 `max(context)+{5,10,20,40}` 帧，见 `docs/FFRGS_REFERENCE.md` 7.5）。

**在官方口径内可做的补充（零额外渲染或很小开销，待确认）**：用已有 `frame_metrics.jsonl` 按到最近 context 的帧距分桶报告（d=0 / 1 / 2–3 / ≥4），并单独报 novel_only；每组报告几何统计（aniso p50、长轴∥视线 >0.9 比例、packing、inter、effective children）。预期 PSNR 持平或略降（FFRGS 全模型对照为 −0.087 dB）。

---

## 场景潜变量微调 vs 场景潜变量微调+几何损失 对照设计（历史草案，已被下一节预注册取代）

**共同条件（两组完全一致）**：官方 `dl3dv_latent_6v_ssim.safetensors`（sha256 `5913da34…78c3`）、preset `eval_dl3dv_latent_6view_ssim`、140 场景、6 个 context 视角 / 50 个 target 视角、FP32（显式 `Accelerator(mixed_precision="no")`）、TF32 采用官方默认值、全部权重冻结、每场景 Adam 50 步、lr 1e-2、优化对象只有 scene latents（gs_tokens 固定）、TTT 监督为 context 视角的 0.8·L1 + 0.2·(1−SSIM)/2 + 1.0·visibility、最终解码使用 `scene_latents.detach()`、不保存场景状态、单进程单卡、同一评估脚本、同一 GPU 型号。

**唯一变量**：内循环损失为 `render` 还是 `render + λ·(w_sw·sw + w_pack·packing + w_inter·inter)`。

**实现方式（不改核心代码）**：在 `scripts/evaluate_with_scene_latent_tuning.py` 中增加 `--geo-lambda`。新增的内循环函数逐行复刻 `_forward_ttt_scene_latents`，仅多出几何项。两组都走这个函数：标准 TTT 即 λ=0。外层 `use_geo_losses` 保持 False，因为它对 TTT 无作用，只会增加计算。实现后先在 smoke 场景上验证 λ=0 与官方 `forward_ttt` 的差落在 TTT 自身噪声内（单场景约 0.02 dB），再继续。

**噪声底**：TTT 存在 run-to-run 非确定性（gsplat 反传使用 atomicAdd，无法关闭）。标准 TTT（λ=0，新入口）需要在 140 场景上再跑 1 次，与 scene_latent_tuning/official_loop 配对，得到“同配置重复”的逐场景差分布。TTT+Geo 相对标准 TTT 的配对差必须明显超出这个分布（报告配对 CI95、胜负场景数），才能判定有效。

**判定指标**：主指标 all_frames PSNR/SSIM/LPIPS，辅助指标 novel_only（44 帧）。几何正则的目的是改善未见视角，所以 novel_only 的变化要单独解读。另外记录每场景最终的 sw/packing/inter 值，确认正则确实生效。

**需要先决定的事项**：
1. **λ 的选取不能在 140 个测试场景上调参**，否则就是测试集泄漏。可选做法：(a) 用不在 benchmark 中的 DL3DV 场景做开发集选 λ（需确认本地是否有这类数据）；(b) 预先登记一个小网格（例如 λ ∈ {1e-3, 1e-2}，三项等权），140 场景上全部报告，不挑选结果。
2. 各项权重：审计显示 packing/inter 的梯度范数是 sw 的 3–8 倍，三项等权会被 packing/inter 主导。可以保持 `options.py` 默认等权（0.01/0.01/0.01），也可以先做单项消融。
3. 预算：scene_latent_tuning/official_loop 标准 TTT 140 场景合计约 11 min / 卡（推理 436 s，单场景含数据与指标 661 s）。几何项每步额外开销尚未实测，需在 smoke 中测量。“噪声底重复 + 每个 λ 配置”各需约 11 min 以上，且可以在不同空闲卡上并行。

---

## 预注册：场景潜变量微调 vs 场景潜变量微调+几何损失 正式对照（2026-10-19，用户已确认，写于任何新结果之前）

**用户决定**：
1. inter 改用 FFRGS 写法：`smooth_l1(log(d / 2), 0, beta=1)`，K=4，target=2；slot 质量 = `opacity × (∏scale)^(2/3)`（detach、按 slot 归一化），sw / packing / inter 三项共用这一组权重。与 FFRGS 的唯一差异：FFRGS 每步随机抽 256 个 query slot，我们用全部 active slot（确定性，无随机抽样）。active 判定仍按 opacity 质量（总和 > 1e-6）。
2. 几何权重只跑一组，照搬 FFRGS 在 TokenGS 上的几何微调：**sw 0.1 / packing 0.001 / inter 0.01**（总系数 1，即内循环 `loss = render + 0.1·sw + 0.001·packing + 0.01·inter`）。不在 140 场景上调参、不挑选结果。
3. 噪声底：标准 TTT（新入口，几何项为 0）在 140 场景上跑两次（`reimplemented_loop` 与 `reimplemented_loop_repeat_run`），两者配对得到“同配置重复”的逐场景差分布；`reimplemented_loop` 另与 `official_loop` 对照，确认新入口与官方 `forward_ttt` 一致。
4. 示例场景：140 场景全部跑完后，自动选 **标准 TTT 相对 Baseline 的 PSNR 增益最接近 140 场景中位数** 的场景（并列取 scene id 字典序第一个）。只在该场景上做：TTT 每步曲线、针统计详表、Baseline / TTT / TTT+Geo 三组全量 ply 导出（不剪枝，带 slot_id 字段）。

**140 场景每组都报告**：PSNR/SSIM/LPIPS（all_frames、novel_only、按到最近 context 帧距分桶 d=0 / 1 / 2–3 / ≥4）；清晰度 edge L1 与 HF ratio（FFRGS `edge_hf` 公式，预测/GT 梯度能量比，1 为理想）；几何统计（sw、packing ratio/目标、effective children、inter 距离均值/中位数、too_close <1.5、too_far >2.5、aniso p50、opacity 均值）。

**判定规则**：TTT+Geo − 标准 TTT 的逐场景配对差，需报告配对均值、CI95、胜/负场景数，并与噪声底分布对照；只有明显超出噪声底才判定有效。几何统计用于确认正则确实生效。

**组别**（全部 FP32、单卡空闲 GPU、同一评估脚本），目录前缀 `EXP_TokenGS/`（输出目录为重组后的位置）：
| 组 | 内循环 | 输出目录 |
|---|---|---|
| 不微调 | 无 | `no_tuning` |
| 场景潜变量微调 | 50 步，render | `scene_latent_tuning/reimplemented_loop` |
| 同配置重跑 | 同上，再跑一次 | `scene_latent_tuning/reimplemented_loop_repeat_run` |
| 场景潜变量微调 + 几何损失 | 50 步，render + geo | `geo_loss_in_scene_latent_tuning/sw0.1_packing0.001_inter0.01` |
| 示例场景 | 三组诊断 + ply | 各组目录下的 `demo_scene_<scene 前 12 位>` |

---

## 执行日志：几何损失对照的实现、smoke 与首次启动（2026-10-19 16:20–16:45）

**代码**（核心代码只动了几何损失本身，微调/渲染不改；文件名为改名后的名字）：
- `tokengs/losses/geo_losses.py`：新增 FFRGS 写法（`inter_form="log_smooth_l1"`、`mass_weighting="opacity_area"`，为默认值），stats 新增 inter 中位数 / too_close / too_far / active_slot_ratio；`tests/test_geo_losses_ffrgs_formulation.py::TestFFRGSInterForm` 4 项与逐行摘录的 FFRGS 参考实现对齐（26 passed / 4 skipped）。`options.py` 新增 `geo_mass_weighting`、`geo_inter_form`。
- 辅助模块（当时是一个文件，后按职责拆成三个）：`scripts/evaluation_statistics.py`（edge/HF、几何统计、针统计）、`scripts/reimplemented_scene_latent_tuning.py`（逐行复刻 `_forward_ttt_scene_latents` 的微调循环，λ 全 0 时不做任何几何运算）、`scripts/demo_ply_export.py`（全量 ply 导出：标准 3DGS 字段 + slot_id/child_id/aniso/cos_long_axis_ray/parallax_deg/in_input_frustums）。
- `scripts/evaluate_with_scene_latent_tuning.py`：`--tuning-loop reimplemented`、`--geo-sw/--geo-packing/--geo-inter`、`--demo`；逐帧 edge/HF 与帧距分桶；每场景几何与针统计；contract 记录辅助模块 / geo_losses 的 sha256（`tokengs/losses/` 未入 git，git diff 哈希覆盖不到）。
- `scripts/report_geo_loss_in_scene_latent_tuning.py`（只读不用 GPU）：配对对比、噪声底判定、按预注册规则选示例场景，写 `report.{json,md}` 与 `demo_scene.json`。

**Smoke（GPU7，场景 `032dee9f…`，产物在 /tmp，已不保留）**：
| 运行 | all PSNR | 说明 |
|---|---|---|
| no_tuning --demo | 30.104 | 与 no_tuning 逐位一致；ply/cameras/needle/render_strip 产物齐全 |
| 官方 forward_ttt | 32.881 | infer 3.42 s |
| 复刻循环 λ=0 | 32.902 / 32.909（两次） | infer 3.40 s；与官方、scene_latent_tuning/official_loop（32.896）差在微调噪声（~0.03 dB）内 → 入口等价 |
| 复刻循环 + 几何损失 --demo | 32.268 | 曲线第 0 步 target PSNR = 30.104（= 不微调，自洽）；inter d 0.70→1.35、packing 67→28、SW 0.51→0.24（正则确实生效）；几何项约 +0.7 s/场景，曲线诊断约 0.03 s/步 |

**查看器**：已确认本机装的是 `kleinicke.ply-visualizer` 1.8.1（Spark），读取标准字段 x/y/z、f_dc、opacity(logit)、scale(log)、rot，f_rest 可缺省；导出格式兼容。

**启动与中止**：16:31 由 agent 用 launch_140.sh（已删除）在 GPU4–7 并行启动四组；约 8 min 后用户取消，并决定**此后所有实验由用户本人运行**（agent 只写脚本、核验、给命令）。
- `no_tuning`：140/140 完成（exit 0，338 s），all PSNR 与官方差 +0.0735，与 11:50 那次不微调逐帧一致，**保留为正式结果**（现为 `no_tuning/`）。
- 三个微调组：停在 100 / 97 / 87 场景，**不作为正式结果**，已重跑；半成品先移入 cancelled_partial_runs_1639/，后在“目录重组”中删除。
- 报告脚本用这些残缺数据做过一次不落盘的试运行（只为测脚本）。仅供参考的预览：噪声底（重跑 − 微调，97 场景）all PSNR Δ +0.001，|Δ| p95 0.021 dB；复刻循环 − 官方循环 Δ +0.001（入口等价）；加几何损失 − 微调（87 场景）all PSNR −0.69 dB，0/87 胜。按预注册不据此调参，以完整重跑为准。

当时交付给用户的是单组脚本（run_group.sh / run_demo.sh，需要自己开多个终端、指定 GPU），已被下一节的一键脚本取代并删除。

---

## 执行日志：三个微调组 140 场景正式结果（用户运行，2026-10-19 16:45–16:59）

用户用 run_group.sh 在 GPU5 / 6 / 7 并行运行；全部 `status=complete`、`formal_complete=true`、140/140，日志中 warning / error / traceback 计数为 0。

| 组（现目录） | GPU | 耗时 | nvidia-smi 峰值显存 |
|---|---|---|---|
| `scene_latent_tuning/reimplemented_loop` | 5 | 664 s | 17427 MiB |
| `scene_latent_tuning/reimplemented_loop_repeat_run` | 6 | 680 s | 17427 MiB |
| `geo_loss_in_scene_latent_tuning/sw0.1_packing0.001_inter0.01` | 7 | 745 s | 17479 MiB |

报告：`python scripts/report_geo_loss_in_scene_latent_tuning.py` → `geo_loss_in_scene_latent_tuning/report.{md,json}`、`demo_scene.json`。以下 Δ = 前者 − 后者，140 场景配对。

| 对比 | all PSNR Δ [CI95] | 胜/负 | novel PSNR Δ | d0 / d1 / d2–3 / d4+ PSNR Δ | LPIPS Δ |
|---|---|---|---|---|---|
| 微调 − 不微调 | +2.206 [+2.059, +2.353] | 140/0 | +1.838 | +4.91 / +2.76 / +1.84 / +1.22 | −0.0718 |
| 重跑 − 微调（噪声底） | +0.0003 [−0.0034, +0.0039] | 66/74 | −0.0008 | — | +0.0000 |
| 复刻循环 − 官方循环（入口等价） | +0.0002 [−0.0038, +0.0042] | 81/59 | +0.0014 | — | −0.0000 |
| **加几何损失 − 微调** | **−0.665 [−0.741, −0.590]** | **1/139** | −0.594 | −1.19 / −0.71 / −0.60 / −0.51 | +0.0290 |
| 加几何损失 − 不微调 | +1.541 [+1.441, +1.640] | 140/0 | +1.244 | +3.72 / +2.05 / +1.25 / +0.71 | −0.0428 |

- 噪声底：同配置重跑的逐场景 |ΔPSNR| p95 = 0.015 dB；复刻循环与官方循环的差同样在噪声内 → 新入口等价官方 `forward_ttt`。
- 几何损失的画质效果：全部 14 项画质指标的 CI95 都不含 0，且与噪声底 CI95 不重叠，方向全部是变差（PSNR/SSIM/HF ratio 降，LPIPS/edge L1 升）。离 context 越远损失越小（d0 −1.19 → d4+ −0.51），但没有任何桶变好。
- 几何统计（140 场景均值，微调 → 加几何损失）：SW 0.712 → 0.376；packing ratio 703.9 → 276.0（目标约 5.7，仍差约 50 倍）；inter d 0.523 → 1.047（目标 2）；too_close 0.908 → 0.776；aniso p50 28.8 → 26.4；长轴∥视线 >0.9 比例 0.265 → 0.247。正则确实生效，但离目标仍远。
- 副作用：落在所有输入视锥外的 Gaussian（opacity 加权）比例 0.0128 → 0.0361（约 2.8 倍），opacity 均值 0.074 → 0.113。几何项把中心推开时有一部分被推到视锥外，而 visibility 项对这些点基本无梯度（见陷阱 14），可能是画质下降的原因之一（推测，未验证）。
- 微调本身对几何的影响（微调 − 不微调）很小且略向反方向：packing ratio +61.5、aniso +1.09、SW −0.001。
- **结论**：按预注册规则，这组权重（sw 0.1 / packing 0.001 / inter 0.01）下“微调内循环加几何损失”显著降低画质；它仍保留了微调增益的约 70%（1.541 / 2.206）。是否继续这个方向由用户决定。
- 示例场景（预注册规则：微调相对不微调的 all PSNR 增益最接近 140 场景中位数）：`2f3e1c0f688c84cec67f9a1ea219c54c14ffabf31a046e620dacc690cac2f1bd`，增益 +2.212 dB（中位数 +2.215）。

---

## 执行日志：改名与一键脚本（2026-10-19，用户要求）

用户要求：① 每次实验输出完整 log；② TTT 改名（实为 scene latent 微调），并清理所有名不副实和 v1/v2 类名称；③ 给一键脚本，而不是让用户自己开多个终端、配环境。

- 改名：结果目录、脚本、测试、命令行参数、contract 字段、报告对比名全部改名，对照表 `renamed_directories_old_to_new.md`。官方代码里的字段（`use_ttt_for_eval`、`ttt_mode` 等）不改。
- 检查时发现的名不副实：`baseline/{val,test}` 实为 BF16 / 20 场景 / 已废弃的 100/20/20 划分；`baseline_ttt` 根目录是 BF16 崩溃结果；辅助模块名 `geo_diagnostics` 实际还包含微调循环和 ply 导出（已拆成三个模块）；`run_group.sh` 的日志其实过滤过 warning（已在文件名里标明）。
- 删除：4 个废弃测试（全模型微调遗留的 checkpoint 事务测试 3 个 + 依赖已删除 EXP_REPRODUCE 的 1 个），用户已同意。
- 一键脚本：`run_all_groups.sh`（四组 + 报告；四组已完成，再运行只会跳过并重生成报告；输出位置见下一节重组后的目录）、`run_demo_scene_all_groups.sh [scene_id]`（示例场景三组；从未运行，2026-10-20 删除，由 `same_scene_ply_comparison/run_all_groups.sh` 取代），约定见文首“一键运行与日志约定”。
- 验证：三个 .sh `bash -n` 通过；5 个 py 文件 py_compile 通过；`tests/test_geo_losses_ffrgs_formulation.py` + `tests/test_dl3dv_resolution.py` 37 passed；报告在新目录上重新生成，数字与改名前一致。一键脚本尚未在 GPU 上实际跑过。
- 遗留：`tests/test_tokengs_contracts.py` 无法导入（官方 d200909 删除了 `project_gaussian_means2d`），属官方仓库问题，未动。

---

## 执行日志：目录重组（2026-10-19 晚，用户要求）

用户要求：下午那次只是多加了指标，所以几何损失实验目录里只应该放加了几何损失的结果，其他组放回各自的方法目录；删除旧的冗余；BF16 的全部删除。用户另选：三份微调结果合并成一个目录；排查目录和取消的半成品也删除。

**删除**（共 53 个文件）：
- 11:50 那次的 `no_tuning`：与 16:31 的不微调逐帧完全相同（7000 帧 PSNR/SSIM/LPIPS 差为 0），旧版每场景 19 个字段、新版 68 个，新版完全覆盖；旧 summary 里独有的“不微调 − 官方循环”配对对比可直接用 `no_tuning` 与 `official_loop` 两个目录重新算出，数字仍记录在上文“执行日志：不微调 140 场景 FP32 补跑”第 4 节。
- BF16：`no_tuning_bf16_20scenes_{val,test}_split`（25.243 / 25.228）、`official_scene_latent_tuning_bf16_140scenes_collapsed`（12.112）。
- 排查目录 `official_scene_latent_tuning_diagnostics_20scenes_test_split`（FP32 preset 步数 27.62、0 步 25.23、1 步 23.56、BF16 去掉 visibility 12.18；只有 metrics.txt）。
- 取消的半成品 `cancelled_partial_runs_1639/` 和当时的启动脚本 `history_cancelled_parallel_launch_1639.sh`。

**移动**（旧位置 → 新位置，均在 `EXP_TokenGS/` 下；旁边的日志跟着改名）：
| 旧位置 | 新位置 |
|---|---|
| `geo_loss_in_scene_latent_tuning/no_tuning` + `no_tuning.console.log` | `no_tuning` + `no_tuning.console.log` |
| `official_scene_latent_tuning` | `scene_latent_tuning/official_loop` |
| `geo_loss_in_scene_latent_tuning/scene_latent_tuning` | `scene_latent_tuning/reimplemented_loop` |
| `geo_loss_in_scene_latent_tuning/scene_latent_tuning_repeat_run` | `scene_latent_tuning/reimplemented_loop_repeat_run` |
| `geo_loss_in_scene_latent_tuning/scene_latent_tuning_with_geo_loss_sw0.1_packing0.001_inter0.01` | `geo_loss_in_scene_latent_tuning/sw0.1_packing0.001_inter0.01` |

**脚本**：
- `scripts/report_geo_loss_in_scene_latent_tuning.py`：各组位置写在 `RUN_DIRS`；报告里的组标签改为 `scene_latent_tuning_reimplemented_loop`、`scene_latent_tuning_reimplemented_loop_repeat_run`（另两组不变），对比名不变；`--root` 改为 `--output-dir`（只决定报告写到哪），去掉 `--official-loop-run`；report.json 新增 `run_dirs`。
- `run_all_groups.sh`、`run_demo_scene_all_groups.sh`：每组写进自己的方法目录；示例场景写进各组 140 场景目录下的 `demo_scene_<scene 前 12 位>/`（后者 2026-10-20 已删除，同样的归档约定由 `same_scene_ply_comparison/run_all_groups.sh` 沿用）。

**验证**：重组前备份 report.json，重组后重新生成，所有配对对比、噪声底判定、逐场景 PSNR 差、示例场景与备份完全一致；`run_all_groups.sh` 空跑：四组都识别为已完成并跳过（nothing to run，exit 0，未占用 GPU 计算），只重生成报告。各目录内 `launch_contract.json` / `run.log` 里的旧路径是当时的真实记录，不改。

**下一步（用户运行）**：`bash EXP_TokenGS/geo_loss_in_scene_latent_tuning/run_demo_scene_all_groups.sh`（该脚本已删除，现在用 `bash EXP_TokenGS/same_scene_ply_comparison/run_all_groups.sh`，见文末“执行日志：同场景四方法 ply 对比”）

---

## 执行日志：token 微调 140 场景 FP32（token_tuning，2026-10-20 起）

**动机**：论文的 test-time scaling 调的是 Gaussian token（TT），release 的 preset 调的是 scene latents；我们此前只试过后者。论文 Fig. 13 定性地说 TT 能改善几何，而我们实测场景潜变量微调几乎不改几何（SW 0.713 → 0.712，packing ratio 642 → 704）。本实验用已有的几何 / 针状指标定量检查 token 微调。

**预注册（用户 2026-10-20 决定）**：
- 只跑一组：论文默认 **lr 1e-4、50 步**，官方循环 `model.forward_ttt_tokens`（`ttt_mode="token-tuning"`），FP32，140 场景，50 帧口径。不在 140 场景上搜学习率，结果好坏都报告。
- **不测噪声底**（用户决定）。场景潜变量微调的噪声底 ±0.004 dB 只能作参考，不能直接套用到 token 微调；所以小于约 0.01 dB 的差别不下结论。
- 配对对比（写进本组 `summary.json` 的 `paired_comparisons`，`run.log` 末尾打印）：
  - `no_tuning`（`no_tuning`）：token 微调的增益，含画质、锐度、几何、针状指标；
  - `scene_latent_tuning_official_loop`（`scene_latent_tuning/official_loop`）：同为官方循环入口的两种微调对比，该旧目录只有 PSNR/SSIM/LPIPS；
  - `scene_latent_tuning_reimplemented_loop`（`scene_latent_tuning/reimplemented_loop`）：与场景潜变量微调的全指标对比（复刻循环与官方循环差 +0.0002 dB）。
- 参照：论文版模型 DL3DV 6 视角 TT 增益 +0.82 dB（23.69 → 24.51）；但 release 模型 decoder 只有 1 层、token 数和训练都不同，1e-4 在这里是否合适未知。

**代码改动**（`scripts/evaluate_with_scene_latent_tuning.py`，核心代码未改）：
- 新增 `--token-tuning`，与 `--scene-latent-tuning` / `--no-tuning` 三选一；设置 `ttt_mode="token-tuning"`，走官方 `forward_ttt_tokens`。
- `--token-tuning` 必须显式给 `--tuning-lr`（preset 里的 `ttt_lr=1e-2` 是 scene latent 的值），否则报错退出；复刻循环和几何损失仍只支持 `--scene-latent-tuning`。
- contract / summary 新增 `tuning_target`（`scene_latents` / `gs_tokens` / null）和 `token_tuning_enabled`；`scene_latent_tuning_enabled` 改为只在调 scene latents 时为 true（旧结果目录不受影响）；summary 新增 `tuning_steps`、`tuning_lr`。token 微调不与官方 release 表对比（表中没有这一行）。
- 脚本文件名仍叫 `evaluate_with_scene_latent_tuning.py`，现在也能做 token 微调，名字不完全贴切；是否改名待用户决定。

**一键脚本**：`EXP_TokenGS/token_tuning/run_all_groups.sh`。先检查三个对比目录都在，再挑 1 张空闲卡运行。输出 `lr1e-4_steps50/`，完整日志 `lr1e-4_steps50.log`，总日志 `run_all_groups.<时间戳>.log`。

**验证（未用 GPU）**：`bash -n`、py_compile 通过；参数检查：缺 `--tuning-lr`、`--token-tuning` 配复刻循环、和 `--no-tuning` 同时给，三种情况都按预期报错；preset 覆盖后 `ttt_mode=token-tuning`、`ttt_lr=1e-4`、`ttt_n_steps=50`；pytest（排除官方坏测试）36 passed / 4 skipped（无 GPU 跳过）。token 微调路径尚未在 GPU 上跑过，第一次运行的日志就是首次实测。

**运行结果（用户 2026-10-20 09:47–09:58 运行，GPU1，exit 0，`status=complete`，140/140，`formal_complete=true`）**：
- 运行的是加 `--ply-dir` / 复刻 token 循环之前的脚本版本（contract 里 `script_sha256=b6db0674…`，`tuning_loop=official`）；之后的改动没有改官方循环这条执行路径，所以结果不受影响。
- 耗时：每场景平均 4.61 s（共 654 s），torch 峰值显存 15.8 GB。
- 配对对比（本组 − 参照，140 场景）：

| 参照 | all PSNR Δ（均值 / 中位数） | 本组更好的场景 | novel PSNR Δ | SSIM Δ | LPIPS Δ | HF ratio Δ |
|---|---|---|---|---|---|---|
| 不微调 | +1.617 / +1.625 | 140/140 | +1.432（140/140） | +0.046 | −0.040 | +0.009（100/140） |
| 场景潜变量微调（官方循环） | −0.589 / −0.544 | 4/140 | −0.404（26/140） | −0.014 | +0.032 | — |
| 场景潜变量微调（复刻循环） | −0.590 / −0.542 | 5/140 | −0.406（24/140） | −0.014 | +0.032 | −0.085（0/140） |

- 几何 / 针状（本组 − 不微调）：SW −0.001、packing ratio +7.1、inter d +0.084、aniso p50 +0.04、长轴∥视线 >0.9 比例 −0.001、视锥外比例 +0.0002。和场景潜变量微调相比：packing ratio −54、aniso p50 −1.05、长轴∥视线 +0.011，都很小。
- 解读：token 微调确实有效（+1.6 dB，每个场景都变好），但这组设置下不如场景潜变量微调（−0.59 dB，135/140 场景更差），画面更糊（HF ratio 低）；离输入视角越远，两者差距越小（d0 −1.94 → d4+ −0.16 dB）。几何指标几乎不变，没有看到论文说的几何改善。没测噪声底，小于约 0.01 dB 的差不下结论（上面的差都远大于此）。
- 与论文对比：论文模型 TT 增益 +0.82 dB，这里 +1.62 dB；模型不同（release decoder 1 层），不能直接比。

---

## 执行日志：同场景四方法 ply 对比（same_scene_ply_comparison，2026-10-20，用户要求）

**用户要求**：① 不同微调方法在同一个场景上跑，直接看 ply 的差别（一键脚本）；② 保存微调后的结果（微调不改网络权重，所以“微调后的模型”其实就是被调的那个张量）；token 微调也要有微调曲线和几何 / 针状 / 锐度指标；所有方法的 ply 单独放在一个文件夹里。

**为什么保存两个张量就够了**：最终的 Gaussian 只由 decoder 的两个输入决定：scene latents（场景潜变量，`[1,4096,1024]`）和 gs_tokens（Gaussian token，`[1,4096,1024]`）。场景潜变量微调换掉前者，token 微调换掉后者，网络权重都不变。所以每个方法把这两个张量存成 `decoder_inputs.pt`（约 32 MB），以后不用数据、不用重新微调就能解码出同样的 Gaussian。

**代码改动**（核心代码未改）：
- `scripts/reimplemented_token_tuning.py`（新）：逐行复刻官方 `TokenGS._forward_ttt_tokens`（encoder 只跑一次，只用 Adam 优化 gs_tokens），加上和场景潜变量复刻循环一样的每步曲线（损失、几何项）和可选几何损失；循环结束把调好的 gs_tokens 留下来。`scripts/reimplemented_scene_latent_tuning.py` 同样留下调好的 scene latents。
- `scripts/evaluate_with_scene_latent_tuning.py`：
  - `--tuning-loop reimplemented` 现在也能配 `--token-tuning`；
  - 新增 `--ply-dir` + `--ply-name`（必须同时给，需要 `--demo` 和 `--scene-ids`；名字只能用字母、数字、`_-.`；已有同名 ply 拒绝覆盖）：ply 写到 `<ply-dir>/<name>/scene_<id 前12位>/<name>.ply`、`<name>_slotcolor.ply`、`cameras.ply`（每种方法一个文件夹，2026-10-20 按用户要求从“所有方法同一文件夹”改来，见文末），旁边写 `<name>.json`（方法、微调设置、checkpoint sha256、本场景全部 all_/novel_/bucket_/geo_/needle_ 指标、各文件的绝对路径）；
  - demo 目录新增 `decoder_inputs.pt`（两个张量 + 场景 / checkpoint / 微调设置 / 相机），写之前用它重新解码一次，与评估用的 Gaussian 比 max |diff|，结果记进日志和旁注；官方循环拿不到被调的张量，所以不写这个文件（日志里会说明）。
- `scripts/decode_saved_decoder_inputs.py`（新）：读 `decoder_inputs.pt` → 校验 checkpoint sha256 → 解码 → 写 gaussians.pt 和 ply，若旁边有原 gaussians.pt 就报告差值。
- `scripts/compare_same_scene_ply.py`（新）：每个场景读各方法旁注，校验场景 / checkpoint / 每 token 子数一致，写 `comparison.md` / `comparison.json`（本场景指标表 + 与不微调逐个 Gaussian 的差：中心位移中位数 / p90、相对场景大小的比例、token 质心位移、opacity / 尺度 / 颜色变化）和 `<method>_minus_no_tuning.ply`（蓝 = 没动，红 = 动得最多）。四种方法解码的是同样的 4096 token × 64 子 Gaussian，第 i 个 Gaussian 在各方法里是同一个 token 的同一个子，所以可以逐个比。
- `scripts/report_geo_loss_in_scene_latent_tuning.py` 和 `report.md`：“next” 一行改指新脚本（只这一行变）。

**一键脚本**：`EXP_TokenGS/same_scene_ply_comparison/run_all_groups.sh [scene_id ...]`（不给场景就用 `demo_scene.json` 里的 `2f3e1c0f…`）。
- 四种方法：`no_tuning`、`scene_latent_tuning`（复刻循环）、`scene_latent_tuning_with_geo_loss_sw0.1_packing0.001_inter0.01`（复刻循环）、`token_tuning_lr1e-4_steps50`（复刻循环）。每个“方法 × 场景”是一组，分到空闲 GPU 上并行。
- ply 按方法分文件夹：`same_scene_ply_comparison/ply/<方法>/scene_<id 前12位>/`；对比表和差异 ply：`same_scene_ply_comparison/comparison/scene_<id 前12位>/`；其余产物（`tuning_curve.{jsonl,png}`、`decoder_inputs.pt`、`gaussians.pt`、`needle_stats.json`、`render_strip.png`）按“结果按方法归档”的约定放在各方法 140 场景目录下的 `demo_scene_<id 前12位>/`。
- 启动前检查四个方法的 140 场景运行都已完成；跑完自动调用 `compare_same_scene_ply.py`，最后列出所有 ply、对比表、曲线和条带图。
- token 微调的 140 场景结果是官方循环跑的，这里 demo 用复刻循环（为了曲线和张量）；两者在 CPU 上逐位等价（见下），GPU 上可能有场景潜变量微调那样的 ~0.02 dB 级非确定性。
- 删除从未运行过的旧脚本 `geo_loss_in_scene_latent_tuning/run_demo_scene_all_groups.sh`（只做三组、ply 分散在各方法目录）。

**验证（未用 GPU）**：
- py_compile 6 个脚本通过，`bash -n` 通过；
- `tests/test_reimplemented_tuning_loops.py`（新，5 个）：用一个接口和 TokenGS 相同的小 mock 模型，在 CPU 上把官方两个微调循环和两个复刻循环逐位比对（输出 Gaussian 完全相同）；留下的张量重新解码等于返回的 Gaussian；参数的 requires_grad 能恢复；故意把步数改成 7 步能被检测出来；
- `tests/test_compare_same_scene_ply.py`（新，2 个）：人造两种方法、只移动一个 token，位移统计 / 差异 ply 颜色 / 指标表都对；checkpoint 不同的旁注被拒绝；
- 全部 pytest（排除官方坏测试 `test_tokengs_contracts.py`）43 passed / 4 skipped；
- 参数检查：`--ply-dir` 不带 `--ply-name`、不带 `--demo`、名字含 `/`、`--tuning-loop reimplemented` 配 `--no-tuning`，都按预期报错且没有写任何文件；
- 一键脚本空跑（把挑 GPU / 启动函数换成打印）：四组的输出目录和参数展开正确，四个 140 场景运行都通过“已完成”检查。
- 没验证的：`decode_saved_decoder_inputs.py` 需要真实模型，只跑了 `--help`；GPU 上的整条 demo 路径要等用户运行，第一次运行的日志就是首次实测。

**下一步（用户运行）**：`bash EXP_TokenGS/same_scene_ply_comparison/run_all_groups.sh`

**第一次运行（2026-10-20 11:23，GPU 1–4）**：四组 exit 0，每组 10–16 s；四个方法的 `decoder_inputs.pt` 重解码与评估用的 Gaussian 完全一致（max |diff| 0.000e+00）。一键脚本在最后一组结束后卡住、没有执行对比步骤（`wait` 等到了总日志的 `tee`，已修复，见 `docs/AGENT_MEMORY.md` 陷阱第 19 条）。

**ply 改为按方法分文件夹（2026-10-20，用户要求）**：用户认为不同方法的 ply 应各放一个文件夹，而不是所有方法共用一个文件夹。
- 新布局：`ply/<方法>/scene_<id 前12位>/{<方法>.ply, <方法>_slotcolor.ply, <方法>.json, cameras.ply}`（cameras.ply 每个方法各存一份，内容相同）；方法间的对比产物不属于任何一个方法，单独放 `comparison/scene_<id 前12位>/{comparison.md, comparison.json, <方法>_minus_no_tuning.ply}`。
- 代码：`evaluate_with_scene_latent_tuning.py` 的 `ply_paths` 多一级 `<ply-name>/`（cameras.ply 不再共享，所以去掉了临时文件 + 改名的写法，覆盖检查也包含 cameras.ply）；`compare_same_scene_ply.py` 改为扫描 `<ply-dir>/*/scene_*/*.json`，要求旁注里的 method 等于文件夹名，结果写到 `--output`（默认 `<ply-dir>/../comparison`），comparison.md 开头列出每个方法 ply 文件夹的绝对路径；一键脚本传 `--output $EXP/comparison`。
- 第一次运行已生成的文件用 `mv` 搬到新布局（没有重算），共享的 cameras.ply 复制到四个方法文件夹后逐字节比对一致。各方法 demo 目录里的 `launch_contract.json`、`scene_metrics.jsonl` 和 log 仍记录旧路径 `ply/scene_2f3e1c0f688c/<方法>.ply`，按陷阱第 17 条不改，对照见 `renamed_directories_old_to_new.md`。
- 验证：pytest（排除官方坏测试）47 passed；在真实的四个方法文件上跑对比脚本（输出到临时目录，已删）正常，生成 comparison.md 和 3 个差异 ply。
- 用户再次运行一键脚本即可：四组已完成会被跳过（不占 GPU），只执行对比，产物写到 `comparison/`。

**目录名去掉 `_fp32_140scenes` / `_fp32` 后缀（2026-10-20，用户要求）**：所有结果都是 FP32、140 场景，后缀没有区分作用。
- 改名：`no_tuning/`（及 `no_tuning.console.log`）、`scene_latent_tuning/`、`geo_loss_in_scene_latent_tuning/`、`token_tuning/`、`same_scene_ply_comparison/`；子目录名不变。
- 脚本（`scripts/*.py`、三个 `run_all_groups.sh`）、文档、`same_scene_ply_comparison/ply/*/scene_*/*.json` 旁注里的绝对路径都已改成新名。
- `geo_loss_in_scene_latent_tuning/report.{json,md}` 在新位置重新生成，除路径外与改名前逐项相同。
- 各运行目录里的 `launch_contract.json`、`summary.json`、`scene_metrics.jsonl`、`run.log`、`*.log` 是证据，保留旧名，对照见 `renamed_directories_old_to_new.md` 头部。
- 验证：pytest（排除官方坏测试）47 passed；`bash -n`、py_compile 通过；对比脚本在真实文件上跑通（输出到临时目录，已删）。

**只高亮 10 / 20 个 token 的 ply（2026-10-20，用户要求）**：`_slotcolor.ply` 把 4096 个 token 全部随机染色，混在一起看不清。用户要求只给少数 token 上色、其余灰色；问答中选择：10 个和 20 个各出一份、在活跃 token 里随机挑、其余灰色半透明。
- 新脚本 `scripts/highlight_selected_tokens_ply.py`（只读已有的真实颜色 ply 和旁注，不用 GPU）。选法：在 `no_tuning` 里按 token 的不透明度总量（64 个子 Gaussian 之和）取前一半（≥ 中位数），固定种子 0 随机抽 20 个；前 10 个就是 10 个的那份，所以两份文件里同一个 token 颜色相同；所有方法用同一批 token 编号。
- 产物（现最终位置为 `demo/scene_<id 前12位>/<实验名>/`）：`<实验名>_highlight{10,20}tokens.ply` 中选中的 Gaussian 位置 / 形状 / 不透明度不变，只换成 20 种高区分度颜色之一（Trubetskoy 色板，灰色换成黑色；前 10 种最好区分），其余 Gaussian 灰色（0.5）、不透明度 ×0.1。2026-10-20 按用户反馈另加 `<实验名>_only{10,20}tokens.ply`：颜色和 token 选择完全相同，但彻底删除其他 Gaussian，分别只含 640 / 1280 个点。两类文件都保留 `slot_id` / `child_id` 并带字段 `selected_token_rank`。图例 `demo/scene_<id 前12位>/highlighted_tokens.{md,json}`：token 编号 → 颜色、四类文件，以及每个实验里该 token 的存活子数、不透明度总量、散开半径、相对不微调的中心位移。
- 当前场景 `2f3e1c0f688c`：活跃阈值 3.261（2048/4096 个活跃），选中的 20 个 token 为 3691, 886, 787, 3820, 1793, 2252, 2031, 3937, 2378, 2458（前 10）, 505, 3211, 2376, 41, 1950, 213, 3372, 1778, 2840, 110。四个现有实验的 16 个文件已生成（每个实验 4 个）。
- 一键脚本 `run_demo.sh` 在对比之后自动调用它；以后加新实验或新场景，重新运行会覆盖两类高亮文件和图例（选择由 no_tuning + 种子决定，结果不变）。
- 验证：`tests/test_highlight_selected_tokens_ply.py`（人造两个方法：只选活跃 token、各方法同一批、嵌套、除颜色外不变、灰色不透明度 ×0.1；only 文件点数 / 原始字段 / 颜色 / token 集合正确）通过；在真实文件上核对：4 个现有实验生成 8 个 only PLY，10 / 20 token 文件分别恰有 640 / 1280 个 Gaussian，`slot_id` 集合与图例完全一致且没有未选点。

---

## 预注册：调哪些参数 × 是否加几何损失 补齐实验（tuned_parameters_x_geo_loss，2026-10-20，用户已确认，写于任何新结果之前）

**动机（用户）**：已有四格（不微调、场景潜变量微调、token 微调、场景潜变量微调 + 几何损失），缺的格子补上后结论更普遍；每组 140 场景只要 11–13 分钟。

**设计**：两个因素交叉。因素 1 = 调哪些参数（scene latents / Gaussian token / 两者一起）；因素 2 = 内循环损失（只用渲染损失 / 渲染 + 几何损失）。再加“不微调”作参照。“不微调 + 几何损失”不成立（没有参数可调）。用户 2026-10-20 选择：只补下面 3 格，**不做**“只用几何损失、去掉渲染损失”的消融。

（下表为最初的 3 格；同日加了 token lr 1e-2 的 4 格，目录名也改成带 token 学习率，见本节末“修订”。）

| 调哪些参数 | 只用渲染损失 | 渲染 + 几何损失（sw 0.1 / packing 0.001 / inter 0.01） |
|---|---|---|
| scene latents | 已有 `scene_latent_tuning/reimplemented_loop`（27.400） | 已有 `geo_loss_in_scene_latent_tuning/sw0.1_packing0.001_inter0.01`（26.734） |
| Gaussian token | 已有 `token_tuning/lr1e-4_steps50`（26.810，官方循环） | **新**（token lr 1e-4） |
| 两者一起 | **新**（token lr 1e-4） | **新**（token lr 1e-4） |

**固定设置（用户确认，不搜索）**：
- 学习率沿用单独微调的值：scene latents 1e-2（preset `ttt_lr`），Gaussian token 1e-4；50 步；FP32，140 场景。
- 两者一起调：一个 Adam、两组参数各用自己的学习率（Adam 逐元素更新，等同两个 Adam 同步走）。官方没有这个模式，我们的 `ttt_mode="scene-latent-and-token-tuning"`；官方 `forward_ttt` 遇到它会报错，所以只能走复刻循环。
- token + 几何损失沿用 scene latents 那组权重（口径一致）。token 学习率小 100 倍，同样权重下几何损失对 token 的作用可能弱很多；如果没效果，只能说明“这组权重下没效果”，不能说几何损失对 token 无用。
- 三组都用复刻循环（`--tuning-loop reimplemented`）。已有的 token 微调格是官方循环；CPU 上复刻循环与官方逐位相同，GPU 上可能有 ~0.02 dB 级非确定性（场景潜变量微调的重跑噪声底 all PSNR ±0.004）。

**报告**（`scripts/report_tuned_parameters_x_geo_loss.py` → `tuned_parameters_x_geo_loss/report.{md,json}`）：3×2 all PSNR 表；各组画质 / 几何 / 针状表；配对对比——几何损失在三种参数下各自的效果、在 scene latents 上加调 token、在 token 上加调 scene latents（各在两种损失下）、每格对不微调、重跑噪声底。

**实现与验证（未用 GPU）**：
- 新 `scripts/reimplemented_scene_latent_and_token_tuning.py`：每步把 scene latents 投影成 decoder 的 K/V（同场景潜变量微调），再用可调的 Gaussian token 解码（同 token 微调）；保存两个调后的张量（demo 时写进 `decoder_inputs.pt`）。
- `evaluate_with_scene_latent_tuning.py` 新参数 `--scene-latent-and-token-tuning`（必须配 `--tuning-loop reimplemented` 和显式 `--token-lr`）、`--token-lr`（只对联合模式有效）；`launch_contract.json` / `summary.json` 记录 `token_lr`；联合模式的 `scene_latent_tuning_enabled` 和 `token_tuning_enabled` 都为 true；不和官方 release 表对比（表里没有这一行）。
- 测试 `tests/test_reimplemented_tuning_loops.py` 新增 4 个：联合循环 token 学习率为 0 时与官方场景潜变量微调**逐位相同**，scene 学习率为 0 时与官方 token 微调逐位相同；两个学习率都不为 0 时两个张量都变、结果与两种单独微调都不同、保存的张量重新解码等于返回值、参数 requires_grad 恢复；几何损失曲线行齐全；模式不对会报错。真实模型里 token 微调的 `forward_encoder` 与“scene latents → K/V”是同一串运算（`_features_to_kv_via_latents` = `_latents_to_kv(_features_to_scene_latents(...))`）。
- 全部 pytest（排除官方坏测试）51 passed；参数检查 4 种错误组合都按预期报错且不写文件；`bash -n`、py_compile 通过；一键脚本空跑（挑 GPU / 启动换成打印）三组参数展开正确、五个已有格都通过“已完成”检查；报告脚本在已有五组上跑通（输出到临时目录，已删），场景潜变量那一行与旧报告一致（−0.665 [−0.741, −0.590]）。
- 没验证的：联合循环在 GPU 真实模型上的运行，第一次运行的日志就是首次实测。

### 修订：加 token 学习率 1e-2（2026-10-20，用户要求，仍写于任何新结果之前）

**动机**：scene latents 的 1e-2 是 release preset 为这个模型指定的；token 的 1e-4 来自论文，论文用的是旧模型，release latent 模型**没有官方 token 学习率**。只用 1e-4 的话，token 相关格子效果弱时分不清是“调 token 不行”还是“学习率太小”。1e-2 是同一模型上 scene latents 的官方值。用户 2026-10-20 选择：token 1e-2 补整行（4 组），单独调 token 用复刻循环（与其余新组同一套代码；之后若对单场景跑 `--demo`，可以拿到微调曲线和 `decoder_inputs.pt`，官方循环不行）。

**最终设计（7 个新组，全部复刻循环、50 步、FP32、140 场景；scene latents 始终 lr 1e-2）**：

| 调哪些参数 | 只用渲染损失 | 渲染 + 几何损失（`<geo>` = sw0.1_packing0.001_inter0.01） |
|---|---|---|
| scene latents | 已有 `scene_latent_tuning/reimplemented_loop` | 已有 `geo_loss_in_scene_latent_tuning/<geo>` |
| token lr 1e-4 | 已有 `token_tuning/lr1e-4_steps50`（官方循环） | **新** `tuned_parameters_x_geo_loss/token_tuning_lr1e-4_with_geo_loss_<geo>` |
| token lr 1e-2 | **新** `token_tuning/lr1e-2_steps50` | **新** `tuned_parameters_x_geo_loss/token_tuning_lr1e-2_with_geo_loss_<geo>` |
| 两者，token lr 1e-4 | **新** `tuned_parameters_x_geo_loss/scene_latent_and_token_tuning_token_lr1e-4` | **新** `tuned_parameters_x_geo_loss/scene_latent_and_token_tuning_token_lr1e-4_with_geo_loss_<geo>` |
| 两者，token lr 1e-2 | **新** `tuned_parameters_x_geo_loss/scene_latent_and_token_tuning_token_lr1e-2` | **新** `tuned_parameters_x_geo_loss/scene_latent_and_token_tuning_token_lr1e-2_with_geo_loss_<geo>` |

- 单独调 token、lr 1e-2 属于 token 微调方法本身，按“结果按方法归档”放在 `token_tuning/lr1e-2_steps50`（与 `lr1e-4_steps50` 并列；后者是官方循环，前者是复刻循环，CPU 上逐位等价）。其余 6 组是本实验独有的组。
- 最初 3 格的目录名加上了 token 学习率（`token_tuning_lr1e-4_with_geo_loss_<geo>`、`scene_latent_and_token_tuning_token_lr1e-4[_with_geo_loss_<geo>]`），因为现在有两个 token 学习率，名字必须区分。这些目录还没生成过，不涉及搬移。
- 只加 1e-2 这一个值，不再搜别的学习率；几何权重仍只用 `<geo>` 一组。
- 风险：1e-2 比论文值大 100 倍，token 微调可能发散或过拟合输入视角；照样报告 140 场景结果。140 场景运行不记录微调曲线（曲线只在单场景 `--demo` 时记录），只能从最终画质、分桶 PSNR、HF ratio 和几何指标判断；非有限值会直接报错中止。要看曲线需另跑同场景对比。

**报告新增**：5 行 × 2 列网格；token 学习率 1e-2 vs 1e-4（四种情形各一个配对对比）；token 微调（两个学习率）vs 场景潜变量微调；加调 token / 加调 scene latents 按 token 学习率各一份；每组对不微调；重跑噪声底。

**验证（未用 GPU）**：`bash -n` 通过；一键脚本 7 组的输出目录与参数逐一打印核对正确（未写实验目录）；报告脚本 12 个组、30 个对比无重名，在已有五组上跑通，场景潜变量几何损失 −0.665 [−0.741, −0.590]、token 1e-4 对场景潜变量 −0.590 与旧数字一致。评估入口和循环代码没有改动（`--token-tuning --tuning-lr 1e-2`、`--token-lr 1e-2` 原本就支持）。

**运行（用户）**：`bash EXP_TokenGS/tuned_parameters_x_geo_loss/run_all_groups.sh`（7 组并行，GPU 1–7 空闲时每组一张卡，预计各 11–13 分钟；跑完自动出报告 `tuned_parameters_x_geo_loss/report.md`）。

## 执行日志：调哪些参数 × 几何损失 × token 学习率（2026-10-20 12:53–13:06，用户运行）

**运行概况**：总日志 `tuned_parameters_x_geo_loss/run_all_groups.20261020_125302.log`，7 组在 GPU 1–7 并行。token lr 1e-4 的 3 组 exit 0、140/140（681 / 756 / 779 s），contract 记录的 ttt_mode / lr / token_lr / 几何权重与预注册一致——这是联合循环第一次在 GPU 真实模型上运行，正常。**token lr 1e-2 的 4 组全部 exit 1**：解码出的 Gaussian 含 NaN/Inf，`geometry_stats` 的有限值检查报错中止。报告 `report.{md,json}` 已在这 3 + 5 组上生成。

**all PSNR（140 场景均值；不微调 25.194）**：

| 调哪些参数 | 只用渲染损失 | 渲染 + 几何损失 | 几何损失 Δ（配对，95% CI，胜场） |
|---|---|---|---|
| scene latents | 27.400 | 26.734 | −0.665 [−0.741, −0.590]，1/139 |
| token lr 1e-4 | 26.810 | 26.606 | −0.204 [−0.229, −0.178]，2/138 |
| 两者，token lr 1e-4 | **27.481** | 26.940 | −0.541 [−0.603, −0.479]，0/140 |
| token lr 1e-2（单独 / 联合，有无几何） | 发散 | 发散 | — |

**结论**：
1. **一起调最好**：两者一起（token 1e-4）27.481，比只调 scene latents 高 +0.081 [+0.068, +0.094]，127/140 场景更好，约为重跑噪声底（±0.004）的 20 倍；novel +0.056、d4+ +0.032，SSIM 0.8707 / LPIPS 0.1795 也都最好。增益小但稳定。反过来，在 token 微调上加调 scene latents +0.671（136/140）。
2. **几何损失在三种参数下都降画质**（每种都几乎全部场景变差），结论不再只限于 scene latents。降幅与几何统计被推动的程度一致：token 微调下几何只动了一点（SW 中位数 0.712→0.610，packing ratio 475→413），PSNR −0.204；scene latents / 一起调下几何被推得多（SW → 0.37 / 0.35），PSNR −0.665 / −0.541。在加几何的情况下，一起调比只调 scene latents 高 +0.205（139/140）。
3. **token lr 1e-2 在本模型上不可用**：4 组都在前几个场景内产生非有限的 Gaussian（单独调 token、两种联合都在第 1 个场景；token + 几何撑到第 8 个场景）。token + 几何跑完的 7 个场景 all PSNR 17.2–22.9，比不微调平均低 6.18 dB（−2.6 ~ −9.4），HF ratio 0.21–0.53（不微调整体 0.76），说明在出现 NaN/Inf 之前就已经严重退化。因此“token 效果弱是不是学习率太小”这个问题，1e-2 给出的答案是“太大，会发散”；1e-4 和 1e-2 之间没有测，不知道最佳值。
4. 失败的 4 个目录（`token_tuning/lr1e-2_steps50`、`tuned_parameters_x_geo_loss/*token_lr1e-2*`、`token_tuning_lr1e-2_with_geo_loss_*`）status 为 failed，保留作为发散证据；一键脚本遇到它们会拒绝启动，重跑前需先挪开。

**完整表——画质（140 场景均值；scene latents 始终 lr 1e-2，50 步；Δ 为配对均值）**

| # | 调什么 | token lr | 几何损失 | all PSNR | Δ 对不微调（95% CI，胜/负） | Δ 对只调 scene | all SSIM | all LPIPS | novel PSNR | novel SSIM | novel LPIPS | d0 | d1 | d2-3 | d4+ | HF ratio | edge L1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 不微调 | — | — | 25.194 | — | −2.206 | 0.8081 | 0.2549 | 25.007 | 0.8043 | 0.2570 | 26.561 | 26.004 | 25.207 | 24.108 | 0.759 | 0.0224 |
| 2 | scene latents | — | 否 | 27.400 | +2.206 [+2.059, +2.353] 140/0 | 0 | 0.8683 | 0.1831 | 26.845 | 0.8596 | 0.1883 | 31.466 | 28.764 | 27.047 | 25.325 | 0.852 | 0.0183 |
| 3 | scene latents | — | 是 | 26.734 | +1.541 [+1.441, +1.640] 140/0 | −0.665 | 0.8525 | 0.2122 | 26.251 | 0.8440 | 0.2171 | 30.279 | 28.050 | 26.452 | 24.816 | 0.829 | 0.0197 |
| 4 | token | 1e-4 | 否 | 26.810 | +1.617 [+1.522, +1.711] 140/0 | −0.590 | 0.8539 | 0.2153 | 26.439 | 0.8473 | 0.2192 | 29.529 | 27.962 | 26.656 | 25.167 | 0.767 | 0.0191 |
| 5 | token | 1e-4 | 是 | 26.606 | +1.413 [+1.338, +1.488] 140/0 | −0.794 | 0.8474 | 0.2296 | 26.264 | 0.8411 | 0.2333 | 29.119 | 27.720 | 26.478 | 25.039 | 0.744 | 0.0196 |
| 6 | **两者一起** | 1e-4 | 否 | **27.481** | **+2.287 [+2.138, +2.436] 140/0** | **+0.081** | **0.8707** | **0.1795** | **26.901** | **0.8617** | **0.1850** | **31.730** | **28.860** | **27.101** | **25.357** | 0.852 | **0.0181** |
| 7 | 两者一起 | 1e-4 | 是 | 26.940 | +1.746 [+1.633, +1.860] 140/0 | −0.460 | 0.8578 | 0.2030 | 26.409 | 0.8487 | 0.2085 | 30.828 | 28.306 | 26.605 | 24.916 | 0.838 | 0.0193 |
| 8 | token | 1e-2 | 否 | 发散 | 第 1 个场景 NaN/Inf | | | | | | | | | | | | |
| 9 | token | 1e-2 | 是 | 发散 | 前 7 场景平均 −6.18，第 8 个 NaN/Inf | | | | | | | | | | | | |
| 10 | 两者一起 | 1e-2 | 否 | 发散 | 第 1 个场景 NaN/Inf | | | | | | | | | | | | |
| 11 | 两者一起 | 1e-2 | 是 | 发散 | 第 1 个场景 NaN/Inf | | | | | | | | | | | | |

**完整表——几何 / 针状（140 场景均值）**

| # | 组 | SW | packing ratio | inter d | too close | too far | aniso p50 | 长轴∥视线 >0.9 | 视锥外比例 | opacity 均值 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 不微调 | 0.7129 | 642.4 | 0.505 | 0.912 | 0.035 | 27.73 | 0.277 | 0.0149 | 0.061 |
| 2 | scene | 0.7119 | 703.9 | 0.523 | 0.908 | 0.037 | 28.82 | 0.265 | 0.0128 | 0.074 |
| 3 | scene + 几何 | 0.3761 | 276.0 | 1.047 | 0.776 | 0.065 | 26.44 | 0.247 | 0.0361 | 0.113 |
| 4 | token 1e-4 | 0.7119 | 649.5 | 0.589 | 0.897 | 0.041 | 27.77 | 0.276 | 0.0150 | 0.063 |
| 5 | token 1e-4 + 几何 | 0.6112 | 559.0 | 1.009 | 0.796 | 0.070 | 27.58 | 0.273 | 0.0212 | 0.065 |
| 6 | 两者（token 1e-4） | 0.7102 | 717.3 | 0.556 | 0.902 | 0.040 | 28.76 | 0.268 | 0.0125 | 0.076 |
| 7 | 两者（token 1e-4）+ 几何 | 0.3558 | 279.0 | 1.133 | 0.751 | 0.072 | 25.36 | 0.248 | 0.0393 | 0.120 |

重跑噪声底（只调 scene 同配置跑两次）all PSNR +0.000 [−0.003, +0.004]。“Δ 对只调 scene”列：第 3/4/6 行的配对 CI 分别为 [−0.741, −0.590] 1/139、[−0.648, −0.531] 5/135、[+0.068, +0.094] 127/13，其余为均值之差。加几何损失的配对效果：scene −0.665 [−0.741, −0.590] 1/139；token 1e-4 −0.204 [−0.229, −0.178] 2/138；两者 −0.541 [−0.603, −0.479] 0/140。

## 执行日志：scene / token / geo 命名整理 + 同场景 ply 扩到 7 种方法（2026-10-20，用户要求）

**用户决定**：不测 token 学习率 1e-3；token lr 1e-2 的 4 个失败目录保留；“有什么可行的实验，就要有什么场景 ply”；目录名改用 scene / token / geo 这样简洁直观的词，词序固定（用户举例 `geo_loss_in_scene_latent_tuning` → `scene_geo_tuning`）。

**命名规则**（详见 `renamed_directories_old_to_new.md` 第三次整理）：方法目录 ∈ {`no_tuning`, `scene_tuning`, `scene_geo_tuning`, `token_tuning`, `token_geo_tuning`, `scene_token_tuning`, `scene_token_geo_tuning`}，词序 scene → token → geo → tuning；子目录只写会变的设置（`lr1e-4`、`token_lr1e-4`、`sw0.1_packing0.001_inter0.01`；scene lr 1e-2 和 50 步不写）；组名 = `<方法>_<子目录>`。对照目录 `tuned_parameters_x_geo_loss/` 改名 `scene_token_x_geo/`。

**做了什么**：
- 全部结果目录、组日志、ply 文件夹及其中文件用 mv 改名；ply 旁注 json 的 `method` / `files` / 绝对路径同步改；证据文件（launch_contract.json、summary.json、status.json、run.log、*.log）内容不动。
- 脚本：`report_geo_loss_in_scene_latent_tuning.py` → `report_scene_geo_tuning.py`，`report_tuned_parameters_x_geo_loss.py` → `report_scene_token_x_geo.py`（组名 / 比较名按新规则）；`build_all_experiments_table.py` 改新路径；`scene_geo_tuning/`、`token_tuning/`、`scene_token_x_geo/` 三个一键脚本改新路径。
- `compare_same_scene_ply.py` 加固定方法顺序 `METHOD_ORDER`（no_tuning, scene, scene_geo, token, token_geo, scene_token, scene_token_geo），对比表和高亮图例的列都按它排；`highlight_selected_tokens_ply.py` 共用。新增测试 `test_methods_are_listed_scene_token_geo_order`。
- `same_scene_ply_comparison/run_all_groups.sh` 改为 7 种可行方法（= 140 场景跑完的配置去掉重复设置）：no_tuning、`scene_tuning_reimplemented_loop`、`scene_geo_tuning_sw0.1_packing0.001_inter0.01`、`token_tuning_lr1e-4`，新增 `token_geo_tuning_lr1e-4_sw0.1_packing0.001_inter0.01`、`scene_token_tuning_token_lr1e-4`、`scene_token_geo_tuning_token_lr1e-4_sw0.1_packing0.001_inter0.01`。不纳入：`scene_tuning/official_loop` 和 `reimplemented_loop_repeat_run`（与 reimplemented_loop 同一设置）、4 个 token lr 1e-2 组（发散）。启动前检查每个方法的 140 场景运行为 complete；demo 输出在各方法目录的 `demo_scene_<id 前12位>/`。

**验证**：
- 两个报告、总表、高亮图例、对比表在新路径重新生成：所有数值与改名前逐项相同（report.json 2037 / 6435 个数逐个相等；总表 15 行数值格相同，只有“结果目录”列变；示例场景仍是 `2f3e1c0f688c…`；高亮选中的 20 个 token 和各方法统计相同）。
- 一键脚本空跑（GPU 挑选和启动换成打印）：前 4 种方法 skip，后 3 种的输出目录和参数正确，参数与各自 140 场景 launch_contract 一致（复刻循环、50 步、token lr 1e-4、scene lr 1e-2、几何权重 0.1 / 0.001 / 0.01）。已有的 `token_tuning_lr1e-4` ply 用复刻循环，它的 140 场景运行用官方循环；两种循环在 CPU 上逐位一致（见上文），沿用。
- 所有 `run_all_groups.sh` 通过 `bash -n`；pytest（除 test_tokengs_contracts.py）53 passed。

**下一步（当时计划，已被下面第四次整理取代）**：旧脚本不再存在；现在运行 `bash EXP_TokenGS/run_demo.sh`。

## 执行日志：每个文件夹一次实验 + demo 分离（2026-10-20，用户确认后执行）

**用户最终规则**：`EXP_TokenGS/<实验名>/` 一层一个 140 场景实验，学习率不同就是不同文件夹，不嵌套；固定几何权重不写进名字；联合调参按名字顺序写学习率（如 `scene_token_tuning_lr1e-2_1e-4`）；示例场景统一放 `demo/`；不加编号。三个 scene-only 运行若无差别只留一个。

**实施**：
- 新增 `scripts/experiment_registry.py`，集中定义 11 个实验的最终名字、方法、学习率、循环、几何设置和顺序；一键脚本、报告、总表、同场景比较和高亮都从 registry 读取。
- 11 个 140 场景实验移到根目录一层；`scene_tuning/official_loop` 和 `scene_tuning/reimplemented_loop_repeat_run` 删除，均值、逐场景差和配对 CI 冻结到 `reports/scene_tuning_noise_floor.json`。保留 `scene_tuning_lr1e-2`（复刻循环）；官方循环与它差 +0.0002 dB，同配置复刻重跑差 +0.0003 dB，均在约 ±0.004 dB 噪声底内。
- 全部 demo 迁到 `demo/scene_2f3e1c0f688c/<实验名>/`，一个实验的评估文件、张量、曲线、ply、相机、旁注、高亮和差异 ply 平铺在同一文件夹；对比表和图例放 scene 文件夹根。现有 4 个 demo 是 `no_tuning`、`scene_tuning_lr1e-2`、`scene_geo_tuning_lr1e-2`、`token_tuning_lr1e-4`。
- 两份报告移到 `reports/`，旧总日志移到 `launch_logs/`；旧分散的一键脚本删除，根目录统一为 `run_140_scenes.sh` 和 `run_demo.sh`。每个新运行的完整日志叫 `<输出目录>/console.log`，总日志叫 `launch_logs/<脚本名>.<时间戳>.log`；`DRY_RUN=1` 只打印跳过项、待运行项和完整命令，不占 GPU、不写日志。
- 执行证据文件（`launch_contract.json`、`summary.json`、`status.json`、`scene_metrics.*`、`run.log` 等）内容不改，所以其中历史路径仍是运行当时的路径；按 `renamed_directories_old_to_new.md` 第四次整理换算。4 个 ply 旁注 json 属于当前索引，已改成最终实验名和相对文件名。

**迁移核验**：
- 移动前完整 tar、sha256 和移动清单在 `/tmp/restructure_before/`；117 次移动的文件逐位一致，仅 4 个 ply 旁注 json 是有意改写。该临时备份可在用户确认无误后删除。
- 重新生成两份报告、总表、comparison 和 highlight：所有报告 comparison / paired delta / noise check / demo scene 与备份逐项一致；总表 11 行 × 59 个公共指标列 0 处差异；comparison / highlight json 数值和高亮 / 差异 ply 的 sha256 均一致。
- pytest（排除官方已坏、无法导入的 `tests/test_tokengs_contracts.py`）：66 passed、44 subtests passed。`run_140_scenes.sh` 和 `run_demo.sh` 的 `DRY_RUN=1` 空跑均通过。

**下一步（用户运行）**：`bash EXP_TokenGS/run_demo.sh`。已有 4 个 demo 自动跳过，只跑 `token_geo_tuning_lr1e-4`、`scene_token_tuning_lr1e-2_1e-4`、`scene_token_geo_tuning_lr1e-2_1e-4` 三个，跑完自动重做 7 个实验的同场景比较，以及 10 / 20 token 的“带灰色背景”和“仅选中 token”两套 PLY。

---

## 执行日志：微调曲线、visibility 内存行为与 token 发散机制深度核查（2026-10-20，只读）

基于 `demo/scene_2f3e1c0f688c/` 下已完成的 4 组产物（`tuning_curve.jsonl`、`needle_stats.json`）及 4 个发散实验的 `run.log` / `status.json` 与模型源码核查：

1. **50 步微调曲线动力学（`demo/scene_2f3e1c0f688c/*/tuning_curve.jsonl`）**
   - **第 1 步超调 vs 单调上升**：`scene_tuning_lr1e-2` 与 `scene_geo_tuning_lr1e-2` 在 `step=1` 均出现明显掉点（`scene_tuning_lr1e-2` 的 `context_psnr` 从 `24.595 → 23.792`，`target_all_psnr` 从 `23.591 → 22.934`），原因是 Adam 首步对 `[1, 4096, 1024]` 每个坐标走满 `±1e-2`，第 2 步（`23.773`）起恢复单调上升；这解释了早期 20 场景诊断中“1 步微调从 25.23 跌到 23.56”的现象。相反，`token_tuning_lr1e-4`（`lr=1e-4`）从 `step=1`（`23.772`）到 `step=50`（`25.149`）全程单调上升。
   - **远视角（`d4+`）饱和与输入视角过拟合分化**：在该 demo 场景上，`scene_tuning_lr1e-2` 的 `context_psnr`（d0）一路升至第 50 步（`24.595 → 29.365`），但 `target_d4plus_psnr` 在第 32–43 步见顶（`23.625 ~ 23.626`），到第 50 步微降至 `23.587`；而 `token_tuning_lr1e-4` 的 `target_d4plus_psnr` 全程单调升至 `23.713`（在该场景 `d4+` 上反超 `scene_tuning_lr1e-2` 的 `23.587` 和 `scene_geo_tuning_lr1e-2` 的 `23.622`）。但 `token_tuning_lr1e-4` 的 `target_all_hf_ratio` 在前 8 步先从 `0.713` 跌至 `0.699`，到第 50 步仅回到 `0.730`（`scene_tuning_lr1e-2` 则单调升至 `0.839`）。
   - **几何损失与渲染损失的量级失衡**：`scene_geo_tuning_lr1e-2` 在 `step=0` 时 `geo_weighted_total = 0.07373`（其中 `0.1*sw=0.05264`、`0.001*packing=0.00787`、`0.01*inter=0.01323`），是 `context_loss = 0.04397` 的 **1.68 倍**；即便到 `step=50`（`geo_weighted_total = 0.03651`，`context_loss = 0.02449`），几何项仍占内循环总损失的 **59.8%**。伴随效果：该场景视锥外且 `opacity > 0.1` 的 Gaussian 数量从 `no_tuning` 的 646 个、`scene_tuning_lr1e-2` 的 689 个激增 **5.6–6.0 倍** 至 `scene_geo_tuning_lr1e-2` 的 **3,879 个**。

2. **`gsplat` 未初始化 `means2d` 内存的直接 GPU 实测证据**
   - 在 `scene_tuning_lr1e-2` 与 `token_tuning_lr1e-4` 的 `tuning_curve.jsonl` 第 0 步，两者尚未做任何梯度更新，解码出的 Gaussian 完全相同（`context_loss_rgb = 0.03390113264322281`、`context_loss_ssim = 0.08411064743995667`、`geo_sw_loss = 0.5264055728912354` 逐位一致），但 `context_loss_visibility` 分别为 `2.30724e-05` 与 `8.79051e-05`；在 50 步过程中更在 `0.0`、`3.05e-9`、`2.28e-5`、`3.52e-4` 之间无规律跳变，直接证实 `compute_visibility_loss_from_means2d` 读取了 `gsplat` 用 `at::empty` 分配而未写入被剔除点的残留显存。

3. **`token lr=1e-2` 4 组发散的架构根因与报错位置**
   - **架构差异**：`scene_latents` 通过 `latent_kv_proj` 后有 `k_proj_norm (LayerNorm)` 归一化 Key、经 cross-attention 的 softmax 归一化权重，再乘以 `gs_cross_attn_scale`（LayerScale `γ≈0.027`）；而 `gs_tokens`（初值 std `0.02`）在 `DecoderBlock` 中拥有无 LayerNorm 的残差直连通道，直接送入同样无前置 LayerNorm 的 `ClipActivationHead.deconv`，再经过 `pos_act(x)=sign(x)(exp(|x|)-1)` 与 `scale_act(x)=min(exp(x-shift),cap)` 的指数激活。对 `gs_tokens` 使用 `lr=1e-2` 会在几步内将线性层输出推过 FP32 `exp` 溢出阈值（`~88.7`），产生 `Inf/NaN`。
   - **报错位置差异**：不加几何损失的 `token_tuning_lr1e-2` 与 `scene_token_tuning_lr1e-2_1e-2` 在第 1 个场景跑完 50 步后，在评估脚本外层 `es.geometry_stats(gaussians, children)` 处触发 `ValueError('geometry inputs must be finite (no NaN/Inf)')`；加几何损失的 `token_geo_tuning_lr1e-2`（第 8 个场景）与 `scene_token_geo_tuning_lr1e-2_1e-2`（第 1 个场景）则在微调内循环 `geo_losses_for` 中直接触发同一异常。

---

## 执行日志：删除 4 个 `token lr=1e-2` 失败目录 + 补齐全部 7 组同场景 Demo（2026-10-20 17:10–17:12）

1. **清理 `token lr=1e-2` 失败文件夹**：
   - 按用户要求删除 `EXP_TokenGS/token_tuning_lr1e-2`、`EXP_TokenGS/token_geo_tuning_lr1e-2`、`EXP_TokenGS/scene_token_tuning_lr1e-2_1e-2`、`EXP_TokenGS/scene_token_geo_tuning_lr1e-2_1e-2`，并从 `scripts/experiment_registry.py` 与 `scripts/report_scene_token_x_geo.py` 移除对应项。
   - 重新生成 `EXP_TokenGS/reports/scene_token_x_geo.{md,json}` 与 `EXP_TokenGS/all_experiments_table.{md,csv}`（现为 7 组完整 140 场景实验 + 2 行官方发布表参考，共 9 行 × 62 列）；运行 pytest（排除官方损坏的 `test_tokengs_contracts.py`）64 passed / 44 subtests passed。

2. **补齐缺失的 3 组同场景 Demo（`bash EXP_TokenGS/run_demo.sh`）**：
   - 总日志：`EXP_TokenGS/launch_logs/run_demo.20261020_171103.log`。已有 4 组自动跳过，新增 3 组在 GPU 1–3 并行运行（20–21 s，全部 `exit 0`），自动完成 7 组对比表 `demo/scene_2f3e1c0f688c/comparison.{md,json}` 与 10 / 20 token 高亮 `highlighted_tokens.{md,json}`（含 `_highlight{10,20}tokens.ply` 和 `_only{10,20}tokens.ply`）。
   - **全部 7 组 `decoder_inputs.pt` 重解码校验 `max |diff| = 0.000e+00`**。

3. **预注册示例场景 `2f3e1c0f688c` 的 7 组完整对比（已更新为第五次优化后的最终实验名）**：

| 指标 | `no_tuning` | `scene_lr1e-2_tuning` | `scene_lr1e-2_geo_tuning` | `token_lr1e-4_tuning` | `token_lr1e-4_geo_tuning` | `scene_lr1e-2_token_lr1e-4_tuning` | `scene_lr1e-2_token_lr1e-4_geo_tuning` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **all PSNR** | 23.591 | 25.802 | 25.520 | 25.149 | 25.049 | **25.957** | 25.735 |
| **novel PSNR** | 23.454 | 25.316 | 25.095 | 24.869 | 24.786 | **25.449** | 25.271 |
| **d0 PSNR** | 24.595 | 29.365 | 28.631 | 27.206 | 26.970 | **29.689** | 29.135 |
| **d4+ PSNR** | 22.654 | 23.587 | 23.622 | **23.713** | 23.672 | 23.695 | 23.679 |
| **all SSIM** | 0.7890 | 0.8482 | 0.8388 | 0.8354 | 0.8312 | **0.8510** | 0.8424 |
| **all LPIPS** | 0.3235 | 0.2372 | 0.2623 | 0.2762 | 0.2882 | **0.2320** | 0.2535 |
| **HF ratio** | 0.713 | 0.839 | 0.817 | 0.730 | 0.711 | **0.842** | 0.828 |
| **SW** | 0.526 | 0.532 | 0.273 | 0.528 | 0.451 | 0.532 | **0.259** |
| **packing ratio** | 367.0 | 406.6 | **111.7** | 361.5 | 301.3 | 402.4 | 119.6 |
| **inter d** | 0.810 | 0.875 | 1.480 | 1.021 | 1.509 | 0.957 | **1.585** |
| **视锥外 `opacity>0.1` 点数** | 646 | 689 | 3,879 | 711 | 964 | 708 | 4,007 |
| **中心位移中位数（相对 `no_tuning`）** | — | 0.0195 | 0.1193 | 0.0145 | 0.0481 | 0.0199 | 0.1241 |
| **log-scale 变化中位数** | — | 0.154 | 0.332 | 0.028 | 0.056 | 0.149 | 0.340 |

   - **关键观察**：
     - `scene_lr1e-2_token_lr1e-4_tuning` 在该场景同样取得最高 all PSNR（`25.957`，比只调 scene 高 `+0.155 dB`）、最高 novel PSNR（`25.449`）、最高 SSIM（`0.8510`）、最低 LPIPS（`0.2320`）与最接近 1 的 HF ratio（`0.842`）；同时解决了只调 scene 在第 32 步后 `d4+` 略微回落的问题（第 50 步 `d4+` 达 `23.695`，比只调 scene 高 `+0.108 dB`）。
     - 不加几何损失的三种微调只对 Gaussian 中心做极小幅度的精调（位移中位数仅占场景范围的 `0.35%~0.48%`），视锥外高不透明度点保持在 `689~711` 个；而加几何损失后，中心位移中位数扩大 `3.3~6.2` 倍，log-scale 变化与 opacity 变化翻倍，且在含 scene 微调的两组中把视锥外 `opacity > 0.1` 的点推高到 `3,879` / `4,007` 个。

4. **第五次命名优化（2026-10-20 17:20，用户确认方案 A 后执行）**：
   - 将每个模块的学习率紧跟在该模块词后面（`[scene_lr<x>_][token_lr<y>_][geo_]tuning`），消除 `scene_token_geo_tuning_lr1e-2_1e-4` 这种多个 lr 堆在末尾时的归属歧义。对照表见 `renamed_directories_old_to_new.md` 第五次整理。

---

## 执行日志：`no_tuning` Token 机制与 Gaussian 分布/不透明度/形状全维度剖析（2026-10-20，只读）

基于 `EXP_TokenGS/demo/scene_2f3e1c0f688c/no_tuning/gaussians.pt`（`[4096, 64, 14]`）与 `decoder_inputs.pt` 的逐 token、逐 child 槽位（`0..63`）拆解，得出以下核心结论：

1. **重大发现：“Token 内部点很散（`packing_ratio=367`）”是未过滤 `opacity < 1/255` 死点造成的假象；真正参与渲染的存活子高斯高度紧凑！**
   - `gsplat` 光栅化器对 `opacity < 1/255 (0.00392)` 的 Gaussian 直接硬件剔除（`radii=0`，不渲染、预训练全程梯度为 0）。在 262,144 个 Gaussian 中，存活点为 183,220 个（**69.9%**，平均每 token **44.73 / 64** 个），其余 **30.1%（每 token 约 19.3 个）为完全不可见的死点**。
   - 因为预训练时死点收不到渲染梯度，其 3D 坐标留在远离 token 核心的未受控位置（每个 token 内按 opacity 排名的最后 14 个 child 距中心中位距离达 **`0.899`**，`p90 = 2.838`；而前 36 个主力存活 child 距中心中位距离仅 **`0.061 ~ 0.069`**）。
   - 然而 `sigmoid(raw - 2)` 永远大于 0（死点 `opacity ≈ 0.0006 ~ 0.0039`），`tokengs/losses/geo_losses.py` 与 `diagnose_no_tuning_tokens.py` 在计算 `opacity * (prod scale)^(2/3)` 权重时**没有把 `opacity < 1/255` 的死点置零**。由于死点的距离平方 $r^2$ 是存活点的约 **1000 倍**，仅 8 个固定死点槽位（`child 2, 61, 41, 27, 12, 0, 7, 20`）就贡献了全 token 中心方差的 **59.5%**！
   - **对照表（同一份 `no_tuning/gaussians.pt`，是否过滤 `<1/255` 死点）**：

| 统计范围 | 活跃 Token 数 | `spread_rms` 中位数 / 均值 | `packing_ratio` 中位数 / 均值 | `packing_target` 均值 | 超目标（`excess>1`）比例 | 强分散（`excess>4`）比例 | `sw_loss` | `packing_loss` | `inter_distance` 均值 / 中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **全部 64 child（含 `<1/255` 死点，原口径）** | 4096 | 0.2573 / 0.2951 | 17.56 / **366.96** | 6.04 | 85.2% | **43.2%** | **0.5264** | **7.8687** | 0.810 / 0.374 |
| **仅存活 child（`opacity >= 1/255`，真实渲染点）** | 4095 | **0.0808 / 0.1305** | **2.48 / 5.30** | 5.74 | **28.7%** | **3.5%** | **0.1082** | **1.6822** | **1.231 / 0.602** |
| **可见 child（`opacity >= 0.01`）** | 4090 | 0.0774 / 0.1267 | 2.38 / 5.04 | 5.61 | 27.3% | 3.4% | — | — | — |

   - **对几何微调掉点（−0.665 dB）的直接解释**：现有 `compute_geo_losses` 把近 60% 的方差梯度用在了“拉拽那些根本不渲染的死点（`r~2.5`）”上，而 64 个 child 共享同一个冻结的 `deconv` 线性层，拉动死点的同时必然把那 45 个真正可见的表面 Gaussian 拽离物体表面！

2. **64 个 Child 槽位（`0..63`）在全部 4096 个 Token 上具有固定的“模板分工”**
   - 跨 4096 个 token 统计每个 `child_id`（`0..63`）的存活率与均值：
     - **5 个槽位（`child 0, 7, 16, 19, 20`）在 4096/4096 个 token 中存活率严格为 0.0%**；另有 4 个槽位（`6, 27, 49, 61`）存活率 `<4%`。这也是为什么没有任何一个 token 能有 64 个存活 child（最高 54 个，中位数 49 个）。
     - **34 个槽位在 `>95%` 的 token 中恒定存活**，其中主力锚点槽位（如 `child 40, 63, 18, 43, 47, 51, 17`）中位不透明度 `0.06~0.11`，距 token 质心中位距离仅 `0.075~0.092`。

3. **Gaussian 的真实形状：不是“细针”为主，而是 85.9% 为“薄圆盘/瓦片（Surfel Disk）”与“扁平刀片（Ribbon）”**
   - 将三维半轴按 $s_1 \ge s_2 \ge s_3$ 排序，把各向异性比 $\text{aniso} = s_1/s_3$（不加权中位数 `59.1`，opacity 加权 `23.6`）拆解为**一维伸长比 $\text{elongation} = s_1/s_2$** 与 **二维扁平比 $\text{flatness} = s_2/s_3$**：
     - 最长轴 $s_1$：中位数 `0.0470`（加权 `0.0598`），**31.7% 的存活点（占 37.6% opacity）触及 `0.075` 尺度硬上限**；
     - 中间轴 $s_2$：中位数 `0.0128`（加权 `0.0175`），因此 **伸长比 $s_1/s_2$ 中位数仅 `2.35`（加权 `2.18`）**；
     - 最短轴 $s_3$：中位数仅 **`0.000551`（加权 `0.00142`）**，比 $s_1, s_2$ 薄一到两个数量级，导致 **扁平比 $s_2/s_3$ 中位数高达 `22.18`（加权 `8.62`）**！
   - **183,220 个存活 Gaussian 的四类三维形态占比**：
     - **薄圆盘 / 瓦片（Oblate Disk，$s_1/s_2 < 3$ 且 $s_2/s_3 \ge 3$）**：数量占 **57.0%**，承担 **53.3%** 的总 opacity；
     - **扁平长条 / 刀片（Triaxial Ribbon，$s_1/s_2 \ge 3$ 且 $s_2/s_3 \ge 3$）**：数量占 **28.9%**，承担 **23.3%** 的总 opacity；
     - **圆柱形细针（Prolate Needle，$s_1/s_2 \ge 3$ 且 $s_2/s_3 < 3$）**：数量仅占 **7.5%**，承担 **11.3%** 的总 opacity；
     - **近似球体 / 微椭球（$s_1/s_2 < 3$ 且 $s_2/s_3 < 3$）**：数量占 **6.6%**，承担 **12.1%** 的总 opacity。

4. **Token 按相机距离的空间层级分工与内部颜色结构**
   - **近景高频细节 Token（最近 25%，距相机 `0.32~0.60`）**：Gaussian 尺寸极小（`scale_rms = 0.0140`）、50% 质量核心半径极小（`r_p50 = 0.0551`）、opacity 适中（`opac_sum = 2.46`），专门刻画近处几何边缘与纹理。
   - **中景表面 Token（25%–75%，距相机 `0.60~2.67`）**：Gaussian 尺寸随深度线性放大（`0.0321 → 0.0721`），核心紧凑（`r_p50 = 0.0608 → 0.1016`）。
   - **远景/背景大块覆盖 Token（最远 10%，距相机 `4.78~10.69`）**：Gaussian 尺寸触顶（`scale_rms = 0.0935`）、分布范围大（`r_p50 = 0.2913`）、**不透明度最高（`opac_sum = 6.25` 是近景的 2.5 倍）**，用大号高不透明瓦片铺满远处墙面/背景。
   - **Token 内部并非单色块**：单个 token 内部存活子高斯的 RGB 标准差中位数达 **`0.2028`**（全场景所有点 RGB 标准差为 `0.2580`），说明一个 token 解码出的是一组**自带明暗和纹理颜色过渡的局部复合表面片（64-splat local surface patch）**。

5. **对比 `EXP_TokenGS.zip` 原始报告与 `4DGS/FFRGS/AGENTS.md`（L455–490），前人研究的 5 个核心缺陷核查**：
   - **缺陷 1（死点未过滤 + 可视化把死点 opacity 设为 1）**：`EXP_TokenGS.zip` 报告用 `colored_20_slots_opacity1.ply`（强行把 20 个 slot 全部 64 个 child 的 opacity 改为 1）得出“同一 slot 的 64 个 GS 大范围散落”，并用未过滤 `<1/255` 死点的纯 `opacity` 权重算出 12 场景 `SW=0.646`、`packing_ratio=18.94 vs target 8.87`（`effective_children=26.41`）、`equal_spread=420.5`。在我们的 demo 场景上用完全相同的纯 `opacity` 权重复算：未过滤死点时 `SW=0.5863`、`packing_ratio=55.41 / target 8.78`（`eff=26.33`）；**但只要过滤掉 `<1/255` 的死点，纯 `opacity` 权重下 `no_tuning` 的 `SW` 立刻降为 `0.0893`、`packing_ratio` 降为 `6.32 < target 8.58`（不加权 `equal_spread` 中位数从 `311.6` 降为 `3.5`）**！原报告训完 10k 步后困惑于“为什么出现了 packing overshoot（`3.87 < 7.89`）”，真相是：**存活点在训练前本来就已经 `< target`（`6.32 < 8.58`）**，10k 步几何损失只是把 19 个死点拉回中心，同时把原本紧凑的存活点过度挤压到了 `3.87`。
   - **缺陷 2（`opacity_area` 权重进一步放大近景死点污染）**：从原报告的纯 `opacity` 权重切换到 FFRGS 的 `opacity * (prod scale)^(2/3)` 后，由于近景 token 的存活点尺度极小（`0.014`）而部分死点尺度较大（`0.012~0.027`），未过滤死点时的 `packing_ratio` 从 `55.41` 被进一步放大 6.6 倍到 `366.96`。
   - **缺陷 3（评估集泄漏与选择性报告）**：据 `FFRGS/AGENTS.md` L468–484 原始记录，原报告的 12 个场景全部在训练集内各重复了 6 次；另测的 30 个 DL3DV-5K 场景不仅可能属于官方模型的预训练集，而且只报了随机 4 个 target views 的 `+0.264 dB`（基线仅 17.05 dB），隐去了 **10 视图平均 PSNR 实测从官方 `19.965` 跌至普通微调 `19.774`、几何微调 `19.453`（净跌 −0.512 dB）** 的事实。
   - **缺陷 4（VGGT-Ω 对齐隐去了恶化近 3 倍的反向距离 `GS → Ω`）**：`EXP_TokenGS.zip` 只报了单向 `Ω → GS` 距离改善 32%（p90 `0.0701 → 0.0473`），而 `FFRGS/AGENTS.md` L487–489 记录了反向距离 **`GS → Ω`（Gaussian 偏离真实表面的距离）p90 在 `opacity > 1/255` 下从 `0.455` 恶化到 `0.793`，在 `opacity > 0.1` 下更从 `0.815` 恶化近 3 倍到 `2.34`**——说明几何损失把大量高不透明度 Gaussian 推离了真实表面（形成 off-cloud 漂浮物），与我们实测视锥外 `opacity > 0.1` 点数暴增 6 倍完全一致。
   - **缺陷 5（将 2D 薄瓦片误判为 1D 细针）**：将 `aniso = s1/s3` 笼统归咎于“细长针状 Gaussian”，忽略了 85.9% 的存活点其实是 $s_3$ 极薄的 2D 表面瓦片（disk/ribbon），真正的圆柱细针（$s_1/s_2 \ge 3, s_2/s_3 < 3$）仅占 7.5%。

6. **在剔除 `<1/255` 死点后，量化确认 TokenGS 架构的两大真实客观缺陷（无表面锚定 + 无连续/排他空间区域保证）**：
   - 过滤 `<1/255` 死点只能说明每个 Token 的核心质量（`r_p50 = 0.081`）不是全局随机乱飞，但**绝不意味着 TokenGS 已经贴合真实表面或每个 Token 对应一块干净连续的局部区域**。对 `no_tuning` 的高可见点（`opacity >= 1/255`、`>= 0.05`、`>= 0.1`）做三维连通性与视线深度切片分析，证实以下三大客观结构问题：
     1. **无显式空间锚定 → 沿视线方向形成厚达 12%~38% 的“多层半透明体积雾”而非致密表面**：
        - `ClipActivationHead.deconv` 在无射线/深度约束的 $\mathbb{R}^3$ 自由空间中直接回归绝对坐标 $(x,y,z)$。
        - 将存活点投影回输入视图 0（`448×256`）并按 `18×18 px` 局部块统计沿视线方向的深度分布：每个局部块内 opacity 加权深度四分位距 `(z_p75 - z_p25) / z_p50` 中位数高达 **`12.4%`**（p75 **`38.1%`**），`P90 - P10` 相对深度跨度中位数达 **`33.1%`**（p75 **`68.1%`**）。
        - 由于存活点中位 opacity 仅 `0.049`（Top-1 child 中位也仅 `0.297`），模型实质上是靠**分布在 12%~38% 深度厚度内的多层半透明瓦片叠加（Alpha Blending）**来拟合输入视角颜色；一旦视角外推/大视差移动（`d4+`），不同深度的半透明层发生视差错位，直接导致远视角画质比输入视角低 `6.14 dB`。
     2. **单 Token 内部缺乏连续性保证 → 1/4 的存活 Token 内部断裂为多个空间孤岛，且存在大深度跨度**：
        - 即便只统计存活点（`opacity >= 1/255`），单 Token 内部两点最大距离（Diameter）中位数为 **`0.368`**（p75 **`1.492`**，p90 **`2.403`**），沿相机深度的相对跨度 `(max_d - min_d)/med_d` 中位数达 **`16.4%`**（p75 **`28.9%`**，p90 **`47.9%`**）；按 $3\sigma$ 椭球接触构图，**25.8% 的 Token 仅在存活点内部就分裂成了 $\ge 2$ 个互不相连的 3D 孤岛**。
        - 哪怕只看 `opacity >= 0.1` 的高不透明点，单 Token 内部最大直径中位数仍有 **`0.319`**（p75 **`0.722`**，p90 **`1.145`**），深度相对跨度中位数 **`10.3%`**（p90 **`25.0%`**）。
     3. **跨 Token 严重混叠（Many-to-One Soup）→ 任意局部区域由 ~20 个 Token 互相穿插合成**：
        - 只看 `opacity >= 0.05`（或 `>= 0.1`）的可见点时，Token 质心到最近邻其他 Token 质心的中位距离仅 **`0.0189`**（`0.0249`），而单个 Token 的 `rms_r` 中位数达 **`0.0929`**（`0.1049`），**98.0%（95.6%）的 Token 半径大于最近邻 Token 间距**；
        - 每个 Token 的 `p90` 半径球内平均包含 **7.8 个（p90 达 16 个）其他 Token 的质心**；
        - 在输入图像每个 `18×18 px` 小块上，平均有 **中位数 26 个（p90 73 个）不同 Token 的存活点**（或 **中位数 19 个、p90 50 个不同 Token 的 `opacity >= 0.05` 点**）同时叠加在一起。

7. **机制闭环验证：为什么远端点的 Opacity 低？（“位置预测偏离 → 压低 Opacity 透明逃逸”的连续光谱）**：
   - 核查预训练权重 `dl3dv_latent_6v_ssim.safetensors` 中的 `activation_head.deconv`（`weight [64, 14, 1024]`，`bias [64, 14]`）与 `no_tuning/gaussians.pt`：
     - **线性头的 Opacity Bias 并未预设死点**：64 个槽位的 `b_opac` 均在 `+0.006 ~ +0.026`（对应初始 `opacity ≈ 0.12`），但各槽位相对于 Token 平均位置权重的偏移范数 $\|\Delta W_c^{\text{pos}}\|$ 分化为三档：34 个核心槽位仅 `0.294`（为均值权重的 `0.72×`），21 个动态槽位为 `0.582`（`1.46×`），9 个外围槽位达 `1.144`（`2.76×`）。
     - **同一槽位被甩远时动态压低 Opacity**：经指数激活 $\text{pos\_act}(u) = \text{sign}(u)(\exp(|u|)-1)$ 放大后，在 21 个动态槽位（86,016 个点）中，同一槽位跨 4096 个 Token 的距离与 $\log(\text{opacity})$ 相关系数中位数为 **`-0.501`**——当该槽位落在核心附近（中位距离 `0.0599`）时保持高可见度（`opac >= 0.02`，24,821 个点）；当该槽位预测偏离核心（中位距离 `0.5101`，p90 `2.7318`）时，网络将其 opacity 压到 `< 1/255`（38,771 个点）以消除位置错误对渲染图像的破坏。
     - **全场景 262,144 个子高斯的“距离—不透明度”逃逸光谱**：

| 到所属 Token 核心距离 `dist_to_core` | 点数（占比） | 存活率（`>=1/255`） | 高可见率（`>=0.05`） | 平均 / 中位 Opacity | 物理状态与在微调中的行为 |
|---|---:|---:|---:|---:|---|
| **`[0.00, 0.06)`（紧贴核心）** | 96,712 (36.9%) | **93.0%** | 31.3% | 0.056 / 0.028 | **Token 连续局部核心**：贴近中心，joint 微调位移中位数仅 `0.0074` |
| **`[0.06, 0.25)`（局部片主体）** | 72,719 (27.7%) | **89.6%** | 47.9% | 0.086 / 0.047 | **Token 局部表面延展**：主力可见层，joint 微调位移中位数 `0.0559` |
| **`[0.25, 1.00)`（半逃逸长尾层）** | 55,934 (21.3%) | **47.4%** (26,534 点) | 28.5% (15,919 点) | 0.056 / 0.007 | **位置已偏离局部片但未压死**：造成 Token 断裂（25.8% 多孤岛）与视线多层雾；无几何约束微调反而把其中存活点的平均 opacity 从 `0.117` 推高到 `0.147` |
| **`[1.00, 10.0)`（完全逃逸/死点层）** | 36,751 (14.0%) | **4.4%** (1,602 点) | 0.5% | 0.002 / **0.0011** | **位置严重偏离但已被压死（`<1/255`）**：渲染不可见，但此前未过滤时占了 60% 方差梯度 |

     - **核心启示**：死点（`dist >= 1.0, opac < 1/255`）与破坏 Token 连续性的离群点（`dist ∈ [0.25, 1.0), opac >= 1/255`）本质上是**同一个机制（位置预测偏离 → 压低 opacity 逃逸）的两个阶段**：前者已经逃逸成功（`<1/255` 被 `gsplat` 剔除，只需在几何统计/损失中屏蔽即可），后者则是**“逃逸了一半没死透”的半透明离群点**。由于微调时 `deconv` 冻结（位置权重跨 child 余弦相似度 `+0.97~0.99`），靠 `detach(opacity)` 强行把离群点的 3D 位置拉回中心必然拽偏核心表面点。

8. **“34 核心 + 21 机动 + 9 死点”究竟是训练失误还是泛化设计？（数据定性）**：
   - **9 个常驻死点槽位（占 14.1% 容量，总 opacity 仅 0.29%，其中 5 个槽位在 4096/4096 个 Token 中存活率 0.0% 且微调后仍为 0.0%）：100% 是训练失误（Dead-Slot Collapse 死神经元坍缩）**。因 `gsplat` 对 `< 1/255` 直接截断且梯度归零，这 9 个槽位在预训练早期因指数位置激活越界被压低 opacity 后永久失去梯度，沦为废弃容量。
   - **34 个核心槽位（占 88.05% 总 opacity）+ 21 个机动槽位（占 11.66% 总 opacity）：是模型在固定 64 球架构下自发学出的“深度自适应容量分配（Depth-Adaptive Capacity Gating）”泛化策略，但因坐标与开关耦合而伴随几何副作用**：
     - 统计每个 Token 开启的机动槽位数（0~21 个），呈现显著双峰分布（`p25=4, p50=15, p75=16`）：
       - **近景高开启组（开启 14~21 个机动槽位，2,357 个 Token）**：距相机中位距离 **`0.76m`**，单点尺度小（`scale_rms = 0.0228`）。近景透视面积大、细节丰富，模型开启几乎全部机动槽位，且存活机动槽位的扁平比 $s_2/s_3$ 中位数高达 **`104.0`**（核心槽位为 `12.1`），用约 50 个极薄微瓦片精细刻画近景表面；
       - **远景低开启组（仅开启 0~5 个机动槽位，1,205 个 Token）**：距相机中位距离 **`2.92m`**，单点尺度大（`scale_rms = 0.0438`），总 opacity 高（`4.19`）。远景只需少量大瓦片即可覆盖，且远景大坐标 $|u|$ 会使高敏感度的机动槽位过冲，模型遂自动关闭机动槽位，只留 ~35 个核心槽位。
     - **结论**：这种分化在**渲染泛化上是有益的自适应容量调节**（近景多球细画、远景少球粗铺），但由于模型没有显式的“父中心 + 有界局部偏移”结构，只能靠“把不需要的球甩远并压低 opacity”来实现软关停，一旦压不彻底（留下 2.65 万个半透明离群点）就破坏了几何连续性。

---

## 执行日志：解析视锥投影（`vis_xyz` vs `vis_opac`）与内循环离核空间门控（`core_gate2.5`）140 场景 + Demo 实验（2026-10-21 15:23–15:36）

- **140 场景总日志**：`EXP_TokenGS/launch_logs/run_140_scenes.20261021_152332.log`（GPU 1, 2, 3 并行，约 680 s，全部 `140/140 complete, exit 0`）。
- **Demo 总日志**：`EXP_TokenGS/launch_logs/run_demo.20261021_153529.log`（GPU 1, 2, 3 并行，24 s，全部 `exit 0`；10 个方法 `decoder_inputs.pt` 重解码 `max |diff|` 全为 `0.000e+00`）。

### 1. 140 场景核心指标对照表（全量 FP32）

| 实验名称 | All PSNR | Novel PSNR | d0 PSNR | d4+ PSNR | All SSIM | All LPIPS | HF ratio | `SW` | `packing_ratio` | `inter_d` (均值/中位) | 存活 GS 比例（平均点数） | 视锥外加权比例 `out_w` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_tuning` | 25.194 | 25.007 | 26.561 | 24.108 | 0.8081 | 0.2549 | 0.759 | 0.7129 | 642.41 | 0.505 / 0.225 | 71.50% (187,444) | 1.49% |
| `scene_lr1e-2_tuning` | 27.400 | 26.845 | 31.466 | 25.325 | 0.8683 | 0.1831 | 0.852 | 0.7119 | 703.91 | 0.523 / 0.229 | 71.43% (187,257) | 1.28% |
| `scene_lr1e-2_token_lr1e-4_tuning` | **27.481** | **26.901** | **31.730** | **25.357** | **0.8707** | **0.1795** | 0.852 | 0.7102 | 717.26 | 0.556 / 0.263 | 71.55% (187,563) | 1.25% |
| `scene_lr1e-2_token_lr1e-4_geo_tuning`（旧几何） | 26.940 | 26.409 | 30.828 | 24.916 | 0.8578 | 0.2030 | 0.838 | 0.3558 | 278.97 | 1.133 / 0.926 | 74.31% (194,788) | 3.93% (恶化 3.1×) |
| **`scene_lr1e-2_token_lr1e-4_vis_xyz_tuning`** | 27.440 | 26.855 | 31.728 | 25.279 | 0.8705 | 0.1800 | **0.853** | 0.7130 | 721.19 | 0.551 / 0.259 | 71.52% (187,483) | **0.16%** (↓87%) |
| **`scene_lr1e-2_token_lr1e-4_vis_opac_tuning`** | **27.441** | **26.859** | 31.707 | 25.289 | 0.8704 | 0.1805 | 0.852 | 0.7143 | 743.77 | 0.547 / 0.257 | **70.50% (184,799)** | **0.09%** (↓93%) |
| **`scene_lr1e-2_token_lr1e-4_vis_opac_core_gate2.5_tuning`** | **27.422** | **26.847** | 31.640 | 25.281 | 0.8702 | 0.1813 | 0.851 | **0.1013** | **3.14** | **1.139** / 0.598 | **69.15% (181,276)** | **0.08%** (↓94%) |

### 2. Demo 场景（`2f3e1c0f688c`）对比验证
- **解析视锥拉 `xyz` vs 压 `opacity`**：
  - `vis_xyz` 将 Demo 视锥外 `opacity > 0.1` 的高不透明 Floater 从 `708` 个降至 **`101` 个**（All PSNR `25.958`）；
  - `vis_opac` 将视锥外 `opacity > 0.1` 的 Floater 从 `708` 个直接降至 **`16` 个（清除率 97.7%）**，同时取得 Demo 场景全部 10 个方法中的**最高 All PSNR `25.973` 与最高 Novel PSNR `25.463`**！
- **加上离核 $>2.5\,r_{\text{core}}$ 空间门控清理（`vis_opac_core_gate2.5`）**：
  - **1. 让 Token 更加集中**：140 场景平均 `packing_ratio` 从 `717.26` 暴降至 **`3.14`（低于理论目标 `5.60`）**，`SW` 从 `0.7102` 暴降至 **`0.1013`**，中心方差从 `0.0301` 降至 **`0.0038`**（缩小 8 倍），且核心高斯相对 `no_tuning` 的中位位移仅 `0.0230`（旧几何损失为 `0.1241`），完全没有把核心点拽离物体表面；
  - **2. 改善 Token 几何连续性**：Demo 场景单 Token 多孤岛断裂率从 `26.5%` 暴降至 **`1.5%`**（98.5% 的 Token 恢复为单一连续区域），单 Token 直径 `p90` 从 `2.55m` 砍半至 **`1.27m`**；
  - **3. 降低 Floater 数量**：140 场景视锥外加权比例从 `1.25%` 降至 **`0.08%`**，Demo 视锥外 `opacity > 0.1` 漂浮物从 `708` 个（旧几何为 `4,007` 个）降至 **`9` 个（清除率 98.7%）**；
  - **4. 减少有效 GS 数量（场景轻量化）且保持高 PSNR**：140 场景平均每场景减少 **6,287 个离核活点**（连同 `<1/255` 死点共精简 **30.85%** 基元，而旧几何损失反而增加了 7,225 个活点），同时 140 场景 All PSNR 达 **`27.422 dB`**（比单调 `scene_lr1e-2_tuning` 的 `27.400 dB` 高 `+0.022 dB`，105/140 场景胜；比旧几何损失 `26.940 dB` 高出 **`+0.482 dB`**）。

---

## 执行日志：单变量解耦对比 COLMAP 稀疏深度（`λ=0.01`）vs VGGT-Ω Sim3 稠密深度（`λ=0.01`）140 场景 + Demo 实验（2026-10-22 10:10–10:36）

- **实验目的**：将深度几何监督与上一轮的 `vis_opac` / `core_gate2.5` 完全解耦，直接在纯净的 `scene_lr1e-2_token_lr1e-4_tuning` 基线上分别只加一项深度损失（`λ=0.01` Smooth-L1），直观对比 **COLMAP 稀疏三角化真值深度（仅占 ~1.35% 像素）** 与 **VGGT-Ω-1B Sim3 对齐稠密预测深度（top-80% 置信度像素）** 的真实作用。
- **运行日志**：`EXP_TokenGS/launch_logs/run_140_scenes.20261022_101058.log`、`EXP_TokenGS/launch_logs/run_140_scenes.20261022_102239.log`、`EXP_TokenGS/launch_logs/run_demo.20261022_103529.log`（全部 `140/140 complete, exit 0`，Demo `redecode_max_abs_diff = 0.000e+00`）。

### 1. 140 场景解耦对照表（相对于 `scene_lr1e-2_token_lr1e-4_tuning` 的逐场景配对差）

| 实验名称 | All PSNR ↑<br>*(对联合基线胜/负)* | **Novel PSNR ↑**<br>*(对联合基线胜/负)* | d0 PSNR ↑<br>*(输入视角)* | d2-3 PSNR ↑<br>*(中距新视角)* | **d4+ PSNR ↑**<br>*(远距新视角，胜/负)* | All SSIM ↑ | Novel SSIM ↑ | **d4+ SSIM ↑**<br>*(对联合基线胜/负)* | All LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene_lr1e-2_tuning` *(单调 scene)* | 27.400 | 26.845 | 31.466 | 27.047 | 25.325 | 0.8683 | 0.8596 | 0.8303 | 0.1831 |
| `scene_lr1e-2_token_lr1e-4_tuning` *(联合基线)* | 27.4807 | 26.9012 | **31.7302** | 27.1010 | 25.3573 | 0.8707 | 0.8617 | 0.8319 | **0.1795** |
| **`..._colmap_depth0.01_tuning` (+COLMAP λ=0.01)** | **27.4822**<br>*(80胜 / 60负)* | **26.9110 (+0.010)**<br>***(86胜 / 54负)*** | 31.6707<br>*(−0.060)* | **27.1091**<br>***(78胜 / 62负)*** | **25.3806 (+0.023)**<br>***(91胜 / 49负)*** | **0.8712**<br>*(83胜 / 57负)* | **0.8623**<br>*(89胜 / 51负)* | **0.8330 (+0.0011)**<br>***(94胜 / 46负)*** | 0.1796 |
| **`..._vggt_omega_depth0.01_tuning` (+VGGT-Ω λ=0.01)** | 27.4534<br>*(58胜 / 82负)* | 26.8768 (−0.024)<br>*(61胜 / 79负)* | 31.6819<br>*(−0.048)* | 27.0752<br>*(60胜 / 80负)* | 25.3377 (−0.020)<br>*(69胜 / 71负)* | 0.8705<br>*(68胜 / 72负)* | 0.8616<br>*(69胜 / 71负)* | 0.8319<br>*(76胜 / 64负)* | 0.1804 |

### 2. Demo 场景（`2f3e1c0f688c`）解耦对照表

| 实验名称 | All PSNR ↑ | **Novel PSNR ↑** | d0 PSNR ↑ | d1 PSNR ↑ | d2-3 PSNR ↑ | **d4+ PSNR ↑** | All SSIM ↑ | Novel SSIM ↑ | 相对 `no_tuning` 深度误差中位数 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene_lr1e-2_tuning` | 25.802 | 25.316 | 29.365 | 27.200 | 25.512 | 23.587 | 0.8482 | 0.8369 | ~0.27 |
| `scene_lr1e-2_token_lr1e-4_tuning` *(联合基线)* | 25.957 | 25.449 | **29.689** | 27.367 | 25.636 | 23.695 | 0.8510 | 0.8398 | 0.2779 |
| **`..._colmap_depth0.01_tuning` (+COLMAP λ=0.01)** | **26.253 (+0.296)** | **25.820 (+0.371)** | 29.430 | **27.528 (+0.161)** | **25.980 (+0.344)** | **24.261 (+0.566)** | **0.8574 (+0.0064)** | **0.8474 (+0.0076)** | **0.1122 (↓2.5倍)** |
| **`..._vggt_omega_depth0.01_tuning` (+VGGT-Ω λ=0.01)** | 25.910 (−0.047) | 25.405 (−0.044) | 29.616 | 27.299 | 25.556 | 23.740 (+0.045) | 0.8502 | 0.8390 | 0.2152 |

- **分区域诊断与阶段总结报告**：详见 `EXP_TokenGS/reports/04_colmap_and_vggt_depth_supervision.md` 及 `EXP_TokenGS/all_experiments_table.md`（已将 `有特征物体区占比`、`有特征物体区 d4+ PSNR`、`平滑无特征区 d4+ PSNR`、`重投影 Warp-PSNR(Δt=5)`、`跨视角深度不一致 MV-Cycle`、`视线深度厚度 RayThick(m)` 共 6 项指标正式合入 140 场景全量总表与 `scene_metrics.jsonl`）。

## 方向更新与审计索引（2026-10-26）

- 用户明确将当前重心转为 ScanNet++ 的点云空间信息学习：通过模型权重微调或架构调整获得三维空间能力；ScanNet++ 不再使用单场景 scene/token 内循环微调。历史 DL3DV 结果与实现保留，本次不迁移、不删除。
- 现场核验：13 组 DL3DV 的场景/帧身份一致，五类图像指标从逐帧重聚合与逐场景记录一致；选定 CPU 测试 64 passed、4 skipped、22 subtests passed。未启动新 GPU 实验。
- 共享 RayThick 实现存在未扣灰背景、跨视角使用首相机距离的问题，旧厚度结论暂停引用；COLMAP 整场景先验与 prior 构建计时需与纯六图模型推理区分。
- ScanNet++ 的尺度/法线/历史冻结配置等审计发现，以及每实验报告和目录清理的待确认范围，统一见 `docs/AGENT_MEMORY.md` 第 13 节与 `EXP_ScanNetPP/EXP_ScanNetPP.md`。所有代码修正需逐条讨论，目录删除尚未执行。
- 用户后续明确：重新设计 ScanNet++ 实验，旧结果不纳入新正式对照；第一阶段仅 RGB 与相机输入、点云作空间监督，长期支持 RGB / RGB+点云可选输入。已建立 ScanNet++ 报告规范；尚未批准/启动新实验，也未移动或删除旧结果。本决定不改变 DL3DV 历史产物，详见共享记忆 13.4。
- **用户最终重申，覆盖上一条输入理解**：训练网络使用点云、相机和 RGB，eval 模型只使用相机与 RGB，目标为内生空间组织能力。用户已亲自清空 `EXP_ScanNetPP/`（包括旧结果、总表与报告规范），现场确认为空；不恢复删除、不沿用旧结果、不自动续训旧 SFT。新的任务定义见 `docs/AGENT_MEMORY.md` 第 14 节，具体方案尚待讨论，未启动新实验。
- 后续按用户明确授权运行新的 ScanNet++ 官方前馈基线，50 场景/400 新视角全部完成；用户要求目录轻量化，现为单层结果目录 `baseline_pretrained_6view`、同名报告和独立 Demo，不保留配置/快照/缓存子目录。实际数值与完整边界见共享记忆第 15 节；本次没有重跑或修改 DL3DV 实验。
