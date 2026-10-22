"""Report for "geometry loss inside scene latent tuning" (FP32, 140 scenes): paired comparisons,
repeat-run noise floor and the pre-registered demo-scene pick.

Reads only finished run directories EXP_TokenGS/<experiment> (no GPU; names from experiment_registry.py) and
the frozen noise floor EXP_TokenGS/reports/scene_tuning_noise_floor.json (two extra runs of scene_tuning_lr1e-2:
same loop again and the official loop; the runs themselves were deleted after that file was written).
Usage (repo root, tokengs env python):
    python scripts/report_scene_geo_tuning.py
    # writes EXP_TokenGS/reports/scene_geo_tuning.{json,md} and EXP_TokenGS/demo/demo_scene.json

Pre-registration: EXP_TokenGS/EXP_TokenGS.md "预注册：场景潜变量微调 vs 场景潜变量微调+几何损失 正式对照".
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from experiment_registry import BY_NAME, DEMO_SCENE_JSON, REPORT_DIR

FRAME_METRICS = ("psnr", "ssim", "lpips", "edge_l1", "hf_ratio")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "edge_l1": False, "hf_ratio": None}
BUCKETS = ("d0", "d1", "d2-3", "d4+")
GEO_KEYS = ("sw_loss", "packing_ratio", "packing_target", "effective_children", "inter_distance",
            "inter_distance_median", "inter_too_close", "inter_too_far", "active_slot_ratio")
NEEDLE_KEYS = ("aniso_p50_weighted", "cos_long_axis_ray_p50_weighted", "long_axis_along_ray_gt0.9_weighted",
               "alive_ratio", "opacity_mean", "outside_all_input_frustums_weighted")
NO_TUNING = "no_tuning"
TUNING = "scene_lr1e-2_tuning"
TUNING_GEO = "scene_lr1e-2_geo_tuning"
GROUPS = (NO_TUNING, TUNING, TUNING_GEO)
RUN_DIRS = {g: BY_NAME[g].dir for g in GROUPS}
NOISE_FLOOR_JSON = f"{REPORT_DIR}/scene_tuning_noise_floor.json"
# (label, this run, reference) ; delta = this - reference
COMPARISONS = (
    ("scene_tuning_gain", TUNING, NO_TUNING),
    ("geo_loss_effect", TUNING_GEO, TUNING),
    ("scene_geo_tuning_vs_no_tuning", TUNING_GEO, NO_TUNING),
)
# frozen in NOISE_FLOOR_JSON: (label, this run, reference)
FROZEN_COMPARISONS = (
    ("repeat_run_noise_floor", f"{TUNING} (repeat run)", TUNING),
    ("reimplemented_loop_vs_official_loop", TUNING, f"{TUNING} (official loop)"),
)


def load_noise_floor(path: str = NOISE_FLOOR_JSON) -> dict:
    """label -> compare() result and per-scene PSNR deltas of the two deleted extra runs of scene_tuning_lr1e-2."""
    frozen = json.loads(Path(path).read_text())
    return {"comparisons": frozen["comparisons"], "paired_psnr_deltas_by_scene": frozen["paired_psnr_deltas_by_scene"]}


def load_run(path: Path, require_complete: bool = True) -> dict | None:
    status = path / "status.json"
    if not status.is_file():
        return None
    st = json.loads(status.read_text())
    if require_complete and not (st.get("status") == "complete" and st.get("formal_complete")):
        return None
    rows = [json.loads(l) for l in (path / "scene_metrics.jsonl").read_text().splitlines() if l.strip()]
    summary = json.loads((path / "summary.json").read_text()) if (path / "summary.json").is_file() else {}
    return {"rows": {r["scene_id"]: r for r in rows}, "summary": summary}


def ci95(d: np.ndarray) -> tuple[float, float, float]:
    n = d.size
    mean, std = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * std / math.sqrt(n)
    return mean, mean - half, mean + half


def paired(a_rows: dict, b_rows: dict, key: str, metric: str | None) -> dict | None:
    ids = sorted(set(a_rows) & set(b_rows))
    if not ids or not all(key in a_rows[s] and key in b_rows[s] for s in ids):
        return None
    a = np.array([a_rows[s][key] for s in ids], dtype=np.float64)
    b = np.array([b_rows[s][key] for s in ids], dtype=np.float64)
    d = a - b
    mean, lo, hi = ci95(d)
    out = {"n": len(ids), "this_mean": float(a.mean()), "ref_mean": float(b.mean()), "delta_mean": mean,
           "delta_median": float(np.median(d)), "delta_std": float(d.std(ddof=1)), "ci95": [lo, hi],
           "abs_delta_p95": float(np.percentile(np.abs(d), 95))}
    if metric is not None:
        if HIGHER_IS_BETTER[metric] is None:
            better, worse = np.abs(a - 1) < np.abs(b - 1), np.abs(a - 1) > np.abs(b - 1)
        elif HIGHER_IS_BETTER[metric]:
            better, worse = d > 0, d < 0
        else:
            better, worse = d < 0, d > 0
        out.update(this_better=int(better.sum()), ref_better=int(worse.sum()))
    return out


def compare(a: dict, b: dict) -> dict:
    res = {}
    for p in ("all", "novel"):
        for m in FRAME_METRICS:
            r = paired(a["rows"], b["rows"], f"{p}_{m}", m)
            if r is not None:
                res[f"{p}_{m}"] = r
    for bk in BUCKETS:
        ids = [s for s in a["rows"] if s in b["rows"] and a["rows"][s].get(f"bucket_{bk}_psnr") is not None
               and b["rows"][s].get(f"bucket_{bk}_psnr") is not None]
        r = paired({s: a["rows"][s] for s in ids}, {s: b["rows"][s] for s in ids}, f"bucket_{bk}_psnr", "psnr")
        if r is not None:
            res[f"bucket_{bk}_psnr"] = r
    for k in GEO_KEYS:
        if (r := paired(a["rows"], b["rows"], f"geo_{k}", None)) is not None:
            res[f"geo_{k}"] = r
    for k in NEEDLE_KEYS:
        if (r := paired(a["rows"], b["rows"], f"needle_{k}", None)) is not None:
            res[f"needle_{k}"] = r
    return res


def noise_check(geo: dict, noise: dict) -> dict:
    """Per metric: is the geo effect clearly outside the same-config repeat distribution?"""
    out = {}
    for key, g in geo.items():
        n = noise.get(key)
        if n is None or key.startswith(("geo_", "needle_")):
            continue
        out[key] = {
            "geo_delta_mean": g["delta_mean"], "geo_ci95": g["ci95"],
            "noise_delta_mean": n["delta_mean"], "noise_ci95": n["ci95"], "noise_abs_delta_p95": n["abs_delta_p95"],
            "geo_ci_excludes_zero": g["ci95"][0] > 0 or g["ci95"][1] < 0,
            "geo_ci_disjoint_from_noise_ci": g["ci95"][0] > n["ci95"][1] or g["ci95"][1] < n["ci95"][0],
            "abs_geo_mean_over_noise_p95": abs(g["delta_mean"]) / n["abs_delta_p95"] if n["abs_delta_p95"] > 0 else None,
        }
    return out


def pick_demo(tuned: dict, untuned: dict) -> dict:
    ids = sorted(set(tuned["rows"]) & set(untuned["rows"]))
    gain = {s: tuned["rows"][s]["all_psnr"] - untuned["rows"][s]["all_psnr"] for s in ids}
    median = float(np.median(list(gain.values())))
    scene = min(ids, key=lambda s: (abs(gain[s] - median), s))
    return {"rule": "scene_tuning - no_tuning all_frames PSNR gain closest to the 140-scene median; ties -> "
                    "lexicographically first scene_id (pre-registered)",
            "scene_id": scene, "gain_db": gain[scene], "median_gain_db": median, "n_scenes": len(ids)}


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def markdown(runs: dict, comps: dict, noise: dict, demo: dict | None) -> str:
    lines = ["# Geometry loss inside scene latent tuning (FP32, 140 scenes): report", ""]
    lines += ["## Per group (140-scene mean)", "",
              "| group | all PSNR | all SSIM | all LPIPS | novel PSNR | edge L1 | HF ratio | d4+ PSNR |",
              "|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        if g not in runs:
            lines.append(f"| {g} | not finished | | | | | | |")
            continue
        s = runs[g]["summary"]
        am, nm = s["all_frames"]["metrics"], s["novel_only"]["metrics"]
        d4 = s.get("distance_buckets", {}).get("d4+", {}).get("psnr", {}).get("mean")
        lines.append(f"| {g} | {fmt(am['psnr']['mean'], 3)} | {fmt(am['ssim']['mean'])} | {fmt(am['lpips']['mean'])} | "
                     f"{fmt(nm['psnr']['mean'], 3)} | {fmt(am['edge_l1']['mean'])} | {fmt(am['hf_ratio']['mean'], 3)} | "
                     f"{fmt(d4, 3)} |")
    lines += ["", "## Geometry / needles (scene median)", "",
              "| group | SW | packing ratio / target | inter d | too close | too far | eff. children | aniso p50 | long axis ∥ ray >0.9 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        if g not in runs:
            continue
        geo, nd = runs[g]["summary"]["geometry"], runs[g]["summary"]["needle"]
        lines.append(f"| {g} | {fmt(geo['sw_loss']['median'])} | {fmt(geo['packing_ratio']['median'], 1)} / "
                     f"{fmt(geo['packing_target']['median'], 2)} | {fmt(geo['inter_distance']['median'], 3)} | "
                     f"{fmt(geo['inter_too_close']['median'], 3)} | {fmt(geo['inter_too_far']['median'], 3)} | "
                     f"{fmt(geo['effective_children']['median'], 1)} | {fmt(nd['aniso_p50_weighted']['median'], 1)} | "
                     f"{fmt(nd['long_axis_along_ray_gt0.9_weighted']['median'], 3)} |")
    lines += ["", "## Paired comparisons (delta = this - reference, 95% CI, scene wins this/ref)", "",
              f"The last two are frozen in `{NOISE_FLOOR_JSON}` (extra runs of {TUNING}, deleted after freezing).", ""]
    for label, this, ref in (*COMPARISONS, *FROZEN_COMPARISONS):
        c = comps.get(label)
        if c is None:
            lines.append(f"- {label} ({this} - {ref}): not available")
            continue
        lines += [f"### {label}: {this} - {ref}", "", "| key | Δ mean | 95% CI | wins | |Δ| p95 |", "|---|---|---|---|---|"]
        for key, r in c.items():
            if key.startswith(("geo_", "needle_")):
                continue
            wins = f"{r['this_better']}/{r['ref_better']}" if "this_better" in r else ""
            lines.append(f"| {key} | {r['delta_mean']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {wins} | "
                         f"{r['abs_delta_p95']:.4f} |")
        lines.append("")
    if noise:
        lines += ["## Geometry loss effect vs repeat-run noise floor", "",
                  "| key | geo Δ | geo CI95 | noise Δ | noise CI95 | CI excl. 0 | disjoint from noise CI |",
                  "|---|---|---|---|---|---|---|"]
        for key, r in noise.items():
            lines.append(f"| {key} | {r['geo_delta_mean']:+.4f} | [{r['geo_ci95'][0]:+.4f}, {r['geo_ci95'][1]:+.4f}] | "
                         f"{r['noise_delta_mean']:+.4f} | [{r['noise_ci95'][0]:+.4f}, {r['noise_ci95'][1]:+.4f}] | "
                         f"{r['geo_ci_excludes_zero']} | {r['geo_ci_disjoint_from_noise_ci']} |")
    if demo:
        lines += ["", "## Demo scene", "", f"- {demo['scene_id']}: gain {demo['gain_db']:+.3f} dB "
                  f"(median {demo['median_gain_db']:+.3f} dB) — {demo['rule']}",
                  "- next: `bash EXP_TokenGS/run_demo.sh` (every finished experiment, one folder per experiment in "
                  "`EXP_TokenGS/demo/scene_<id[:12]>/`)"]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=REPORT_DIR, help="where scene_geo_tuning.{json,md} are written")
    p.add_argument("--demo-scene-json", default=DEMO_SCENE_JSON, help="where the demo-scene pick is written")
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs = {g: r for g in GROUPS if (r := load_run(Path(RUN_DIRS[g]))) is not None}
    frozen = load_noise_floor()
    print("finished groups:", sorted(runs))
    comps = {label: compare(runs[a], runs[b]) for label, a, b in COMPARISONS if a in runs and b in runs}
    comps.update(frozen["comparisons"])
    noise = (noise_check(comps["geo_loss_effect"], comps["repeat_run_noise_floor"])
             if "geo_loss_effect" in comps else {})
    demo = pick_demo(runs[TUNING], runs[NO_TUNING]) if {TUNING, NO_TUNING} <= set(runs) else None
    report = {"groups_finished": sorted(runs), "run_dirs": RUN_DIRS, "noise_floor_file": NOISE_FLOOR_JSON,
              "comparisons": comps, "geo_vs_noise_floor": noise, "demo_scene": demo,
              "paired_psnr_deltas_by_scene": {
                  **{label: {s: runs[a]["rows"][s]["all_psnr"] - runs[b]["rows"][s]["all_psnr"]
                             for s in sorted(set(runs[a]["rows"]) & set(runs[b]["rows"]))}
                     for label, a, b in COMPARISONS if label in comps},
                  **frozen["paired_psnr_deltas_by_scene"]}}
    (root / "scene_geo_tuning.json").write_text(json.dumps(report, indent=2) + "\n")
    md = markdown(runs, comps, noise, demo)
    (root / "scene_geo_tuning.md").write_text(md)
    if demo is not None:
        Path(args.demo_scene_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.demo_scene_json).write_text(json.dumps(demo, indent=2) + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
