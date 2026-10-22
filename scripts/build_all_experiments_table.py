"""One table of every formal 140-scene experiment in EXP_TokenGS: one row per experiment, one column per metric.

Reads only run directories (no GPU). Each metric is the 140-scene mean of the per-scene value in
scene_metrics.jsonl (same numbers as summary.json); runs that did not finish all 140 scenes because the
Gaussians became non-finite are written as "发散" in every metric cell. Rows = experiment_registry.EXPERIMENTS
(same names as the directories); the single-scene runs in EXP_TokenGS/demo/ are not listed.
Usage (repo root, tokengs env python):
    python scripts/build_all_experiments_table.py
    # writes EXP_TokenGS/all_experiments_table.{md,csv}
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiment_registry import BY_NAME, EXP, EXPERIMENTS
from report_scene_geo_tuning import BUCKETS, FRAME_METRICS, load_run, paired

EXPECTED_SCENES = 140
DIVERGED = "发散"
NOT_RECORDED = "未记录"
NO_TUNING_DIR = BY_NAME["no_tuning"].dir
SCENE_DIR = BY_NAME["scene_lr1e-2_tuning"].dir
# official release table (release_2026.6.md, all target frames, 140 scenes): reference rows, not our runs
OFFICIAL_REFERENCE = (("参考：官方发布表，不微调", "无", "—", "—",
                       {"all_psnr": 25.12, "all_ssim": 0.806, "all_lpips": 0.257}),
                      ("参考：官方发布表，调 scene latents", "scene latents", "官方", "1e-2",
                       {"all_psnr": 27.304, "all_ssim": 0.866, "all_lpips": 0.185}))
SETTING_COLUMNS = ("实验", "说明", "调哪些参数", "微调循环", "scene lr", "token lr", "几何权重 sw / packing / inter",
                   "完成场景")
DIRECTION = {"psnr": "↑", "ssim": "↑", "lpips": "↓", "edge_l1": "↓", "hf_ratio": "→1"}
FRAME_NAMES = {"psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS", "edge_l1": "edge L1", "hf_ratio": "HF ratio"}
GEO_COLUMNS = (("sw_loss", "SW"), ("packing_loss", "packing loss"), ("inter_loss", "inter loss"),
               ("packing_ratio", "packing ratio"), ("packing_target", "packing target"),
               ("effective_children", "有效子高斯数"), ("center_variance", "中心方差"),
               ("local_scale_variance", "局部尺度方差"), ("inter_distance", "平均间距"),
               ("inter_distance_median", "间距中位数"), ("inter_too_close", "过近比例"),
               ("inter_too_far", "过远比例"), ("active_slot_ratio", "活跃 slot 比例"))
NEEDLE_COLUMNS = (("alive_ratio", "存活比例"), ("opacity_mean", "平均不透明度"),
                  ("aniso_p50_weighted", "拉长程度中位数（加权）"), ("aniso_p50_unweighted", "拉长程度中位数（不加权）"),
                  ("cos_long_axis_ray_p50_weighted", "长轴与视线夹角 cos 中位数"),
                  ("long_axis_along_ray_gt0.9_weighted", "长轴沿视线 >0.9 比例"),
                  ("outside_all_input_frustums_weighted", "视锥外比例"))
SURF_COLUMNS = (("feat_pixel_ratio", "有特征物体区占比", 4),
                ("d4+_feat_psnr", "有特征物体区 d4+ PSNR ↑", 3),
                ("d4+_smooth_psnr", "平滑无特征区 d4+ PSNR ↑", 3),
                ("warp_psnr_dt5", "重投影 Warp-PSNR(Δt=5) ↑", 3),
                ("mv_cycle_err_dt5", "跨视角深度不一致 MV-Cycle ↓", 4),
                ("ray_thickness_med", "视线深度厚度 RayThick(m) ↓", 4))
TIMING_COLUMNS = (("scene_total_seconds", "每场景耗时 s"), ("torch_peak_allocated_mib", "显存峰值 MiB"))


def metric_columns() -> list[tuple[str, str, int]]:
    """(scene_metrics key, column title, decimals) for every metric column, left to right."""
    cols = []
    for part, name in (("all", "全部帧"), ("novel", "新视角")):
        cols += [(f"{part}_{m}", f"{name} {FRAME_NAMES[m]} {DIRECTION[m]}", 3 if m == "psnr" else 4)
                 for m in FRAME_METRICS]
    for bk in BUCKETS:
        cols += [(f"bucket_{bk}_{m}", f"{bk} {FRAME_NAMES[m]} {DIRECTION[m]}", 3 if m == "psnr" else 4)
                 for m in FRAME_METRICS]
    cols += [(f"surf_{k}", t, nd) for k, t, nd in SURF_COLUMNS]
    cols += [(f"geo_{k}", f"几何 {t}", 4) for k, t in GEO_COLUMNS]
    cols += [(f"needle_{k}", f"针状 {t}", 4) for k, t in NEEDLE_COLUMNS]
    cols += [(k, t, 2 if k == "scene_total_seconds" else 0) for k, t in TIMING_COLUMNS]
    return cols


def scene_mean(rows: dict, key: str) -> float | None:
    vals = [r[key] for r in rows.values() if r.get(key) is not None]
    if not vals:
        return None
    return float(np.max(vals)) if key == "torch_peak_allocated_mib" else float(np.mean(vals))


def delta_cell(this: dict, ref: dict | None) -> str:
    """Paired all-frames PSNR delta: mean [95% CI] better/worse scenes."""
    if ref is None or (r := paired(this["rows"], ref["rows"], "all_psnr", "psnr")) is None:
        return NOT_RECORDED
    lo, hi = r["ci95"]
    return f"{r['delta_mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] {r['this_better']}/{r['ref_better']}"


def diverged_note(status: dict, run: dict | None, base: dict) -> str:
    n = len(run["rows"]) if run else 0
    note = f"第 {n + 1} 个场景出现 NaN/Inf"
    if n and (r := paired(run["rows"], base["rows"], "all_psnr", "psnr")) is not None:
        note += (f"；已跑完的 {n} 个场景全部帧 PSNR {r['this_mean']:.3f}，同场景不微调 {r['ref_mean']:.3f}"
                 f"（{r['delta_mean']:+.3f}）")
    return note + f"；报错 {status.get('error', NOT_RECORDED)}"


def build(cols: list) -> tuple[list[str], list[list[str]], list[str]]:
    header = [*SETTING_COLUMNS, "Δ 全部帧 PSNR 对不微调（95% CI，胜/负场景）",
              "Δ 全部帧 PSNR 对 scene_lr1e-2_tuning", *(t for _, t, _ in cols)]
    base = load_run(Path(NO_TUNING_DIR))
    scene = load_run(Path(SCENE_DIR))
    rows, notes = [], []
    for e in EXPERIMENTS:
        d = e.dir
        status = json.loads((Path(d) / "status.json").read_text())
        run = load_run(Path(d), require_complete=False)
        n = len(run["rows"]) if run else 0
        ok = status.get("status") == "complete" and n == EXPECTED_SCENES
        settings = [f"`{e.name}`", e.title, e.tuned, e.loop, e.scene_lr, e.token_lr, e.geo_weights,
                    f"{n}/{EXPECTED_SCENES}"]
        if ok:
            vals = []
            for key, _, nd in cols:
                v = scene_mean(run["rows"], key)
                vals.append(NOT_RECORDED if v is None else f"{v:.{nd}f}")
            deltas = ["0" if d == NO_TUNING_DIR else delta_cell(run, base),
                      "0" if d == SCENE_DIR else delta_cell(run, scene)]
        else:
            vals = [DIVERGED] * len(cols)
            deltas = [DIVERGED, DIVERGED]
            notes.append(f"- `{e.name}`：{diverged_note(status, run, base)}")
        rows.append([*settings, *deltas, *vals])
    for name, tuned, loop, scene_lr, ref in OFFICIAL_REFERENCE:
        vals = [f"{ref[key]:.3f}" if key in ref else "—" for key, _, _ in cols]
        rows.append(["`release_2026.6.md`", name, tuned, loop, scene_lr, "—", "—", "140/140", "—", "—", *vals])
    return header, rows, notes


def markdown(header: list[str], rows: list[list[str]], notes: list[str]) -> str:
    lines = ["# EXP_TokenGS 全部实验总表（140 场景）", "",
             "由 `scripts/build_all_experiments_table.py` 从各实验目录的 `scene_metrics.jsonl` 生成，不要手改；"
             "新实验跑完后重新运行脚本即可更新。同一份数据另存为 `all_experiments_table.csv`，可用表格软件打开。", "",
             f"- 每行一个实验，每列一个指标；第一列就是实验文件夹名（`{EXP}/<实验>/`）；数值是 140 个场景的平均值（显存峰值取最大值）。",
             "- 没跑完 140 个场景、因为 Gaussian 出现 NaN/Inf 而中止的实验，指标格写“发散”。",
             "- ↑ 越大越好，↓ 越小越好，→1 越接近 1 越好（HF ratio：渲染图与真值图的高频能量之比）；几何和针状指标没有统一的好坏方向。",
             "- Δ 列：同一批场景逐个相减后取平均，方括号是 95% 置信区间，后面是“更好/更差”的场景数。"
             "重跑同一配置的噪声约 ±0.004 dB。",
             "- 所有实验：FP32、release latent 模型 `dl3dv_latent_6v_ssim`；微调的实验都调 50 步；“官方 / 复刻”循环在 CPU 上逐位一致，"
             "140 场景上两者相差 +0.0002 dB（见 `reports/scene_tuning_noise_floor.json`）。",
             "- “参考”两行是官方 release 文档里的数字，不是我们跑的，只有全部帧 PSNR / SSIM / LPIPS。",
             "- 单场景运行（`demo/`）不在表内。", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if notes:
        lines += ["", "## 发散实验的细节", "", *notes]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=EXP, help="where all_experiments_table.{md,csv} are written")
    args = p.parse_args()
    root = Path(args.output_dir)
    header, rows, notes = build(metric_columns())
    (root / "all_experiments_table.md").write_text(markdown(header, rows, notes))
    with open(root / "all_experiments_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[c.strip("`") for c in r] for r in rows])
    print(f"{len(rows)} rows x {len(header)} columns -> {root}/all_experiments_table.{{md,csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
