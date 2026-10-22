# 结果目录 / 脚本改名对照表（2026-10-19）

2026-10-19 至 2026-10-20 共做了四次整理。下面前半部分的“新名 / 现位置”只用于换算早期日志；**当前最终位置以文末“第四次整理”为准**。

**2026-10-20 再次改名：去掉 `_fp32_140scenes` / `_fp32` 后缀**（用户：所有结果都是 FP32、140 场景，后缀没有区分作用）。
下面各表“现位置”列和文末各节里带 `_fp32_140scenes` 的目录，现在一律去掉后缀：

| 2026-10-19 晚 ~ 2026-10-20 的名字 | 现在 |
|---|---|
| `no_tuning_fp32_140scenes/`、`no_tuning_fp32_140scenes.console.log` | `no_tuning/`、`no_tuning.console.log` |
| `scene_latent_tuning_fp32_140scenes/` | `scene_latent_tuning/` |
| `geo_loss_in_scene_latent_tuning_fp32_140scenes/` | `geo_loss_in_scene_latent_tuning/` |
| `token_tuning_fp32_140scenes/` | `token_tuning/` |
| `same_scene_ply_comparison_fp32/` | `same_scene_ply_comparison/` |

子目录名（`official_loop`、`reimplemented_loop`、`sw0.1_packing0.001_inter0.01`、`lr1e-4_steps50`、`demo_scene_*`、`ply/<方法>/` 等）不变。
脚本（`scripts/*.py`、各 `run_all_groups.sh`）和文档（`EXP_TokenGS.md`、`docs/AGENT_MEMORY.md`、`docs/FFRGS_REFERENCE.md`）已改成新名；
`geo_loss_in_scene_latent_tuning/report.{json,md}` 在新位置重新生成，数字与改名前逐项相同（只有 `run_dirs` 和 next 一行的路径变了）；
`same_scene_ply_comparison/ply/*/scene_*/*.json` 旁注里的绝对路径（对比脚本要用）已同步改。
各运行目录里的 `launch_contract.json`、`summary.json`（含 `paired_comparisons` 的参照路径）、`scene_metrics.jsonl`、`run.log`、`*.log` 仍是旧名，按本表换算。

改名原因：旧名里的 `ttt` 名不副实（官方叫 TTT，实际只优化 scene latent），`v1/v2/formal/task52` 之类指代不清，
`baseline` 不写精度和场景数。新名一律写清楚：方法、精度、场景数、划分。

**术语**：官方代码叫 TTT（`ttt_mode="scene-latent-tuning"`）= 本项目 **scene latent tuning（场景潜变量微调）**：
冻结全部网络权重和 Gaussian token，只对 `scene_latents [1,4096,1024]` 做 Adam 50 步、lr 1e-2。
它和官方另一种模式 token-tuning（优化 `gs_tokens`）不是一回事。官方配置字段名（`use_ttt_for_eval`、`ttt_mode`、
`ttt_n_steps`、`ttt_lr`、`forward_ttt`）属于官方代码，保持不变。

**证据文件不改**：各目录里的 `launch_contract.json`、`summary.json`、`run.log`、`*.console*.log` 记录的是当时的真实
命令和路径（旧名），保留原样作为证据；按本表换算即可。

## 结果目录（均在 `EXP_TokenGS/` 下）

| 旧名 | 新名 | 现位置 | 内容 | all PSNR |
|---|---|---|---|---|
| `baseline/full_140` | `no_tuning_fp32_140scenes` | 已删除（被下面 `formal_v2/baseline` 取代，逐帧数值相同） | 不微调，FP32，140 场景，11:50 那次 | 25.194 |
| `baseline/val` | `no_tuning_bf16_20scenes_val_split` | 已删除（BF16） | 不微调，BF16，20 场景，来自已废弃的 100/20/20 划分的 val | 25.243 |
| `baseline/test` | `no_tuning_bf16_20scenes_test_split` | 已删除（BF16） | 同上，test 划分 | 25.228 |
| `baseline_ttt/fp32_formal` | `official_scene_latent_tuning_fp32_140scenes` | `scene_latent_tuning_fp32_140scenes/official_loop` | 官方微调循环，FP32，140 场景（权威官方结果） | 27.399 |
| `baseline_ttt`（根目录文件） | `official_scene_latent_tuning_bf16_140scenes_collapsed` | 已删除（BF16） | 官方微调循环，BF16，140 场景，数值崩溃 | 12.112 |
| `baseline_ttt/diagnostic_fp32` | `official_scene_latent_tuning_diagnostics_20scenes_test_split/fp32_preset_steps` | 已删除（排查目录） | FP32，preset 步数 | 27.62 |
| `baseline_ttt/diagnostic_steps0` | `.../bf16_0_tuning_steps` | 已删除（排查目录） | 0 步微调；精度为推断：与 BF16 test 划分不微调结果逐场景差 ≤5e-5 | 25.23 |
| `baseline_ttt/diagnostic_steps1` | `.../1_tuning_step_precision_unrecorded` | 已删除（排查目录） | 1 步微调；当时没记录精度 | 23.56 |
| `baseline_ttt/diagnostic_no_visibility` | `.../bf16_without_visibility_loss_collapsed` | 已删除（排查目录） | BF16，去掉 visibility loss，崩溃 | 12.18 |
| `formal_v2` | `geo_loss_in_scene_latent_tuning_fp32_140scenes` | 同新名（只放几何损失组 + 报告 + 一键脚本） | 正式对照：微调里加几何损失，FP32，140 场景 | |
| `formal_v2/baseline` | `.../no_tuning` | `no_tuning_fp32_140scenes` | 不微调，16:31 那次（指标比 11:50 那次多） | 25.194 |
| `formal_v2/ttt` | `.../scene_latent_tuning` | `scene_latent_tuning_fp32_140scenes/reimplemented_loop` | 复刻循环微调 | 27.400 |
| `formal_v2/ttt_repeat` | `.../scene_latent_tuning_repeat_run` | `scene_latent_tuning_fp32_140scenes/reimplemented_loop_repeat_run` | 同配置重跑，用来量噪声底 | 27.400 |
| `formal_v2/ttt_geo` | `.../scene_latent_tuning_with_geo_loss_sw0.1_packing0.001_inter0.01` | `geo_loss_in_scene_latent_tuning_fp32_140scenes/sw0.1_packing0.001_inter0.01` | 复刻循环微调 + 几何损失（权重写在名字里） | 26.734 |
| `formal_v2/_aborted_1639` | `.../cancelled_partial_runs_1639` | 已删除 | 16:39 取消的半成品三组 | |

官方结果目录最早写在 `EXP_REPRODUCE/tokengs_v2/artifacts/...`（launch_contract 里能看到），后移到 `EXP_TokenGS/`。

## 日志

| 旧名 | 新名 → 现位置 | 说明 |
|---|---|---|
| `formal_v2/baseline.stdout.log` | `no_tuning.console.log` → `no_tuning_fp32_140scenes.console.log` | launch_140.sh 直接重定向，未过滤 |
| 三个微调组的终端日志（run_group.sh 产出） | `<组名>.console_warnings_filtered.log` → 跟着组目录走，`<现目录>.console_warnings_filtered.log` | 当时用 grep 过滤过 warning（实际 warning 数为 0） |
| `_aborted_1639/<旧组名>.stdout.log` | `cancelled_partial_runs_1639/<组名>.console.log` → 已删除 | 未过滤 |
| — | `<组输出目录>.log` | 今后一键脚本产出：完整、未过滤，带命令/GPU/git/退出码/耗时 |

## 脚本与测试（均在仓库根）

| 旧名 | 新名 |
|---|---|
| `scripts/eval_frame_metrics.py` | `scripts/evaluate_with_scene_latent_tuning.py` |
| `scripts/geo_diagnostics.py` | 拆成 `scripts/evaluation_statistics.py`、`scripts/reimplemented_scene_latent_tuning.py`、`scripts/demo_ply_export.py` |
| `scripts/formal_v2_report.py` | `scripts/report_geo_loss_in_scene_latent_tuning.py` |
| `formal_v2/launch_140.sh` | `.../history_cancelled_parallel_launch_1639.sh` → 已删除 |
| `formal_v2/run_group.sh`、`run_demo.sh` | 删除，由 `.../run_all_groups.sh`、`.../run_demo_scene_all_groups.sh` 取代 |
| `geo_loss_in_scene_latent_tuning_fp32_140scenes/run_demo_scene_all_groups.sh`（三组示例场景，从未运行） | 2026-10-20 删除，由 `same_scene_ply_comparison_fp32/run_all_groups.sh`（四种方法）取代 |
| `same_scene_ply_comparison_fp32/ply/scene_<id 前12位>/<方法>{.ply,_slotcolor.ply,.json}` + 共享的 `cameras.ply`（所有方法同一文件夹） | 2026-10-20 按用户要求改为每种方法一个文件夹：`same_scene_ply_comparison_fp32/ply/<方法>/scene_<id 前12位>/`（各带一份 cameras.ply）；对比产物放 `same_scene_ply_comparison_fp32/comparison/scene_<id 前12位>/`。第一次运行的文件用 mv 搬过去；各方法 `demo_scene_2f3e1c0f688c/` 里的 `launch_contract.json`、`scene_metrics.jsonl` 和 `demo_scene_2f3e1c0f688c.log` 仍是旧路径 |
| `tests/test_geo_losses_v2.py` | `tests/test_geo_losses_ffrgs_formulation.py` |
| `tests/test_task52_validation_checkpoint.py`、`test_task60_checkpoint_transactions.py`、`test_task64_checkpoint_retention.py`、`test_multiscene_preparation.py` | 删除（废弃的全模型微调遗留 / 依赖已删除的 EXP_REPRODUCE） |

## 命令行参数 / 输出字段

| 旧 | 新 |
|---|---|
| `--no-ttt` / `--ttt` | `--no-tuning` / `--scene-latent-tuning` |
| `--ttt-impl {official,script}` | `--tuning-loop {official,reimplemented}` |
| 微调步数 / 学习率覆盖参数 | `--tuning-steps` / `--tuning-lr` |
| demo 微调曲线 | `tuning_curve.jsonl/png` |
| contract `ttt_enabled` / `ttt_impl` / `ttt_inner_loop_geo_weights` | `scene_latent_tuning_enabled` / `tuning_loop` / `tuning_loop_geo_weights` |
| contract `ttt_mode` / `ttt_n_steps` / `ttt_lr` | `official_option_ttt_mode` / `tuning_steps` / `tuning_lr`（值仍取自官方 opt 字段） |
| contract `geo_diagnostics_sha256` | `helper_sha256`（三个拆分后的模块各一项） |
| report 配对比较 | `scene_latent_tuning_gain`、`repeat_run_noise_floor`、`reimplemented_loop_vs_official_loop`、`geo_loss_effect`、`geo_loss_tuning_vs_no_tuning` |
| report 组标签 `scene_latent_tuning` / `scene_latent_tuning_repeat_run`（第一次改名后） | `scene_latent_tuning_reimplemented_loop` / `scene_latent_tuning_reimplemented_loop_repeat_run`（重组后） |
| report `--root`（四组都在其下） | `--output-dir`（只决定报告写到哪；各组位置写在脚本 `RUN_DIRS`，report.json 的 `run_dirs`） |
| report `--official-loop-run` | 删除（官方循环固定为 `scene_latent_tuning_fp32_140scenes/official_loop`） |

## 第二次整理（2026-10-19 晚，用户要求）

原则：同一方法的结果放在该方法自己的目录里；对照实验目录只放这个实验独有的组（这里是加几何损失的那组）、报告和一键脚本。
BF16 结果、排查目录、取消的半成品全部删除，关键数字保留在上表和 `EXP_TokenGS.md` 里。
示例场景输出：各组 140 场景目录下的 `demo_scene_<scene 前 12 位>/`。

## 第三次整理（2026-10-20，用户要求：用 scene / token / geo 简洁命名，词序固定）

命名规则（以后新实验也照此命名）：

- 方法目录 `EXP_TokenGS/<方法>/`：方法名只由 `scene`（调 scene latents）、`token`（调 Gaussian tokens）、`geo`（加几何损失）组成，
  词序永远是 **scene → token → geo → tuning**，不调就是 `no_tuning`。共 7 种：
  `no_tuning`、`scene_tuning`、`scene_geo_tuning`、`token_tuning`、`token_geo_tuning`、`scene_token_tuning`、`scene_token_geo_tuning`。
- 子目录 `<方法>/<设置>/`：只写会变的设置。scene lr 恒为 1e-2、微调步数恒为 50，都不写；
  只调 token 写 `lr1e-4`，两者一起调写 `token_lr1e-4`（避免误读成 scene 的 lr）；几何权重写 `sw0.1_packing0.001_inter0.01`。
- 组名（ply 文件夹名 / 报告列名）= `<方法>_<设置>`，例如 `token_geo_tuning_lr1e-4_sw0.1_packing0.001_inter0.01`。
- 对照实验（比较多个方法的报告和一键脚本）单独一个目录，例如 `scene_token_x_geo/`；里面不放结果。
- 各组 140 场景目录里的证据文件（launch_contract.json、summary.json、status.json、run.log、*.log）内容不改，里面仍是当时的旧路径。

| 旧 | 新 |
|---|---|
| `scene_latent_tuning/{official_loop,reimplemented_loop,reimplemented_loop_repeat_run}` | `scene_tuning/{official_loop,reimplemented_loop,reimplemented_loop_repeat_run}` |
| `geo_loss_in_scene_latent_tuning/sw0.1_packing0.001_inter0.01`（及 report、demo_scene.json、run_all_groups.sh） | `scene_geo_tuning/sw0.1_packing0.001_inter0.01`（report 等同在 `scene_geo_tuning/`） |
| `token_tuning/lr1e-4_steps50`、`token_tuning/lr1e-2_steps50`（及同名 .log） | `token_tuning/lr1e-4`、`token_tuning/lr1e-2` |
| `tuned_parameters_x_geo_loss/token_tuning_lr{1e-4,1e-2}_with_geo_loss_sw0.1_packing0.001_inter0.01` | `token_geo_tuning/lr{1e-4,1e-2}_sw0.1_packing0.001_inter0.01` |
| `tuned_parameters_x_geo_loss/scene_latent_and_token_tuning_token_lr{1e-4,1e-2}` | `scene_token_tuning/token_lr{1e-4,1e-2}` |
| `tuned_parameters_x_geo_loss/scene_latent_and_token_tuning_token_lr{1e-4,1e-2}_with_geo_loss_sw0.1_packing0.001_inter0.01` | `scene_token_geo_tuning/token_lr{1e-4,1e-2}_sw0.1_packing0.001_inter0.01` |
| `tuned_parameters_x_geo_loss/`（report、run_all_groups.sh、总 log） | `scene_token_x_geo/` |
| ply 文件夹 `scene_latent_tuning` | `scene_tuning_reimplemented_loop` |
| ply 文件夹 `scene_latent_tuning_with_geo_loss_sw0.1_packing0.001_inter0.01` | `scene_geo_tuning_sw0.1_packing0.001_inter0.01` |
| ply 文件夹 `token_tuning_lr1e-4_steps50` | `token_tuning_lr1e-4` |
| `scripts/report_geo_loss_in_scene_latent_tuning.py` | `scripts/report_scene_geo_tuning.py` |
| `scripts/report_tuned_parameters_x_geo_loss.py` | `scripts/report_scene_token_x_geo.py` |
| 两个报告里的组名 / 比较名（旧名用 `scene_latent_tuning`、`scene_latent_and_token_tuning`、`with_geo_loss`、`steps50` 等） | 一律换成上面的组名规则，例如 `scene_tuning_reimplemented_loop`、`scene_geo_tuning_sw0.1_packing0.001_inter0.01`、比较 `scene_tuning_gain`、`geo_loss_effect_in_token_tuning_lr1e-4`、`adding_tokens_lr1e-4_to_scene_tuning`；完整列表见各 report.json 的 `comparisons` |

ply 文件夹里的文件名、旁注 json 的 `method` / `files` / 绝对路径一并改了；报告、总表、高亮图例重新生成后，所有数字与改名前逐项一致（只有路径和名字变了）。

## 第四次整理（2026-10-20，用户要求：每个文件夹就是一次 140 场景实验，demo 分离）

### 最终规则

- `EXP_TokenGS/<实验名>/` 一层一个 140 场景实验；不同学习率就是不同实验文件夹，不再按方法嵌套。实验文件夹里只放 140 场景记录。
- 实验名为 `<方法>_lr<学习率>`；方法词序固定 **scene → token → geo → tuning**，学习率按被调参数在名字中的顺序跟在后面。例如 `scene_token_tuning_lr1e-2_1e-4` 表示 scene lr 1e-2、token lr 1e-4。
- 所有实验都相同的 FP32、50 步、几何权重 sw / packing / inter = 0.1 / 0.001 / 0.01 不写进名字。以后某个权重变化，只追加变化项（例如 `_sw0.2`）。
- 唯一实验清单和参数定义为 `scripts/experiment_registry.py`；140 场景一键脚本、demo 一键脚本、两份报告、总表、同场景对比和高亮都读取 registry。
- 所有示例场景产物集中到 `EXP_TokenGS/demo/scene_<id 前12位>/<实验名>/`，一个实验在该场景的 ply、旁注、调参曲线、张量、评估结果和日志全部平铺在一个文件夹。
- 报告集中到 `reports/`，总日志集中到 `launch_logs/`；只有两个一键脚本：根目录的 `run_140_scenes.sh` 和 `run_demo.sh`。

### 140 场景实验：旧位置 → 最终位置

| 第三次整理后的旧位置 | 最终位置 | 状态 |
|---|---|---|
| `no_tuning/` | `no_tuning/` | complete |
| `scene_tuning/reimplemented_loop/` | `scene_tuning_lr1e-2/` | complete；唯一保留的 scene-only 运行 |
| `scene_geo_tuning/sw0.1_packing0.001_inter0.01/` | `scene_geo_tuning_lr1e-2/` | complete |
| `token_tuning/lr1e-4/` | `token_tuning_lr1e-4/` | complete |
| `token_tuning/lr1e-2/` | `token_tuning_lr1e-2/` | failed，已于 2026-10-20 按用户要求删除 |
| `token_geo_tuning/lr1e-4_sw0.1_packing0.001_inter0.01/` | `token_geo_tuning_lr1e-4/` | complete |
| `token_geo_tuning/lr1e-2_sw0.1_packing0.001_inter0.01/` | `token_geo_tuning_lr1e-2/` | failed，已于 2026-10-20 按用户要求删除 |
| `scene_token_tuning/token_lr1e-4/` | `scene_token_tuning_lr1e-2_1e-4/` | complete |
| `scene_token_tuning/token_lr1e-2/` | `scene_token_tuning_lr1e-2_1e-2/` | failed，已于 2026-10-20 按用户要求删除 |
| `scene_token_geo_tuning/token_lr1e-4_sw0.1_packing0.001_inter0.01/` | `scene_token_geo_tuning_lr1e-2_1e-4/` | complete |
| `scene_token_geo_tuning/token_lr1e-2_sw0.1_packing0.001_inter0.01/` | `scene_token_geo_tuning_lr1e-2_1e-2/` | failed，已于 2026-10-20 按用户要求删除 |

`scene_tuning/official_loop/` 和 `scene_tuning/reimplemented_loop_repeat_run/` 与保留的 `scene_tuning_lr1e-2/` 设置相同，因此删除。两个运行的均值、逐场景差和配对 CI 完整冻结在 `reports/scene_tuning_noise_floor.json`：官方循环 vs 复刻循环差 +0.0002 dB，复刻循环重跑差 +0.0003 dB，均在约 ±0.004 dB 的噪声底内。

### 报告、脚本、日志：旧位置 → 最终位置

| 旧位置 | 最终位置 |
|---|---|
| `scene_geo_tuning/report.{md,json}` | `reports/scene_geo_tuning.{md,json}` |
| `scene_geo_tuning/demo_scene.json` | `demo/demo_scene.json` |
| `scene_token_x_geo/report.{md,json}` | `reports/scene_token_x_geo.{md,json}` |
| 三个目录内的 `run_all_groups.sh` | 删除，由根目录 `run_140_scenes.sh` 统一取代 |
| `same_scene_ply_comparison/run_all_groups.sh` | 删除，由根目录 `run_demo.sh` 取代 |
| `<旧实验目录>.log` | `<最终实验目录>/console.log` |
| 各 `run_all_groups.<时间戳>.log` | `launch_logs/<旧目录名>.run_all_groups.<时间戳>.log` |

`scene_tuning_lr1e-2/` 和 `scene_geo_tuning_lr1e-2/` 当年只生成了 `console_warnings_filtered.log`，没有完整终端日志；文件按原名保留，不伪造 `console.log`。

### demo：旧位置 → 最终位置

以实验名 `x`、场景前缀 `2f3e1c0f688c` 为例：

- 评估文件：`<旧实验目录>/demo_scene_2f3e1c0f688c/` → `demo/scene_2f3e1c0f688c/x/`；其内 `demo/<完整 scene id>/` 的 `gaussians.pt`、`decoder_inputs.pt`、`tuning_curve.*`、`render_strip.*`、`needle_stats.json` 也提升到这个最终文件夹，不再嵌套。
- ply：`same_scene_ply_comparison/ply/<旧组名>/scene_2f3e1c0f688c/` → 同一个 `demo/scene_2f3e1c0f688c/x/`；ply、slotcolor、高亮、相机、旁注均按最终实验名重命名。
- 差异 ply：`same_scene_ply_comparison/comparison/scene_2f3e1c0f688c/<旧组名>_minus_no_tuning.ply` → 对应实验的最终 demo 文件夹。
- 对比表和图例：旧 `comparison/`、`ply/highlighted_tokens_*` → `demo/scene_2f3e1c0f688c/{comparison,highlighted_tokens}.{md,json}`。

demo 的 `launch_contract.json`、`summary.json`、`scene_metrics.*`、`run.log` 等证据文件保留当时真实记录，里面的旧输出路径 `.../demo_scene_<前缀>/demo/<完整 scene id>/x` 现在对应同一最终 demo 文件夹；不要改这些证据内容。ply 旁注 json 是索引文件，不是执行证据，已改成最终实验名和相对文件名，并删除失效的 `source_run` / `demo_dir` 字段。

### 迁移与核验

- 移动前完整备份：`/tmp/restructure_before/EXP_TokenGS_before_restructure.tar`；移动清单和 sha256：同目录 `moves.tsv`、`sha_premove.txt`。该备份仅供本次确认，用户确认无误后可删除。
- 117 次移动逐文件 sha256 核对一致；只有 4 个 demo ply 旁注 json 按上述规则有意改写。
- 两份报告的全部 comparison / paired delta / noise check / demo scene 与备份逐项相同；总表 11 行 × 59 个公共指标列 0 处差异；comparison / highlight 的 json 数值和生成 ply 的 sha256 均相同。
- pytest（排除官方已坏的 `tests/test_tokengs_contracts.py`）66 passed；`run_140_scenes.sh`、`run_demo.sh` 的 `DRY_RUN=1` 空跑通过。

## 第五次整理（2026-10-20，用户要求：lr 紧跟对应模块词，消除多模块组合时的归属歧义）

### 最终命名规则（当前生效）
- 格式：`no_tuning` 或 `[scene_lr<x>_][token_lr<y>_][geo_]tuning`
- 模块词序固定为 **scene → token → geo → tuning**，且每个被调模块的学习率直接紧跟在该模块词后面（例如 `scene_lr1e-2_token_lr1e-4_geo_tuning`），不再把多个学习率堆在末尾。
- 若未来修改某个几何损失权重，紧跟在 `geo` 后追加变化项（例如 `_geo_sw0.2_tuning`）。

| 第四次整理的名字 | 第五次整理后的最终名字（当前位置） |
|---|---|
| `no_tuning` | `no_tuning` |
| `scene_tuning_lr1e-2` | `scene_lr1e-2_tuning` |
| `scene_geo_tuning_lr1e-2` | `scene_lr1e-2_geo_tuning` |
| `token_tuning_lr1e-4` | `token_lr1e-4_tuning` |
| `token_geo_tuning_lr1e-4` | `token_lr1e-4_geo_tuning` |
| `scene_token_tuning_lr1e-2_1e-4` | `scene_lr1e-2_token_lr1e-4_tuning` |
| `scene_token_geo_tuning_lr1e-2_1e-4` | `scene_lr1e-2_token_lr1e-4_geo_tuning` |

（`demo/scene_2f3e1c0f688c/<实验名>/` 下的子文件夹名、PLY 文件名和 `.json` 旁注同步改为上表右列名字。）
