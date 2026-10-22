"""Report for "tuned parameters x geometry loss" (140 scenes): which tensor is tuned (scene latents /
Gaussian tokens / both, with the Gaussian token learning rate at 1e-4 or 1e-2) crossed with the tuning loss
(render only / render + geometry loss), plus no tuning.

Reads only finished run directories EXP_TokenGS/<experiment> (no GPU; group label = experiment name from
experiment_registry.py). Scene latents always lr 1e-2, Gaussian tokens lr 1e-4 (paper) or 1e-2 (= the release
scene latent value), 50 steps; geometry weights sw 0.1, packing 0.001, inter 0.01. Token tuning at lr 1e-4
without geometry loss is the earlier official-loop run; the reimplemented loop is bit-identical on CPU
(tests/test_reimplemented_tuning_loops.py). The repeat-run noise floor is frozen in
EXP_TokenGS/reports/scene_tuning_noise_floor.json.
Usage (repo root, tokengs env python):
    python scripts/report_scene_token_x_geo.py
    # writes EXP_TokenGS/reports/scene_token_x_geo.{json,md}
"""

import argparse
import json
from pathlib import Path

from experiment_registry import BY_NAME, GEO_WEIGHTS, REPORT_DIR
from report_scene_geo_tuning import compare, fmt, load_noise_floor, load_run

TOKEN_LRS = ("1e-4",)
NO_TUNING = "no_tuning"
SCENE = "scene_lr1e-2_tuning"
SCENE_GEO = "scene_lr1e-2_geo_tuning"
TOKEN = {lr: f"token_lr{lr}_tuning" for lr in TOKEN_LRS}
TOKEN_GEO = {lr: f"token_lr{lr}_geo_tuning" for lr in TOKEN_LRS}
JOINT = {lr: f"scene_lr1e-2_token_lr{lr}_tuning" for lr in TOKEN_LRS}
JOINT_GEO = {lr: f"scene_lr1e-2_token_lr{lr}_geo_tuning" for lr in TOKEN_LRS}
GROUPS = (NO_TUNING, SCENE, SCENE_GEO, *(g[lr] for g in (TOKEN, TOKEN_GEO) for lr in TOKEN_LRS),
          *(g[lr] for g in (JOINT, JOINT_GEO) for lr in TOKEN_LRS))
RUN_DIRS = {g: BY_NAME[g].dir for g in GROUPS}
# rows of the grid: (label, render loss only, render + geometry loss)
GRID = (("scene latents (lr 1e-2)", SCENE, SCENE_GEO),
        *((f"Gaussian tokens (lr {lr})", TOKEN[lr], TOKEN_GEO[lr]) for lr in TOKEN_LRS),
        *((f"scene latents (lr 1e-2) + Gaussian tokens (lr {lr})", JOINT[lr], JOINT_GEO[lr]) for lr in TOKEN_LRS))
# (label, this run, reference); delta = this - reference
COMPARISONS = (
    ("geo_loss_effect_in_scene_tuning", SCENE_GEO, SCENE),
    *((f"geo_loss_effect_in_token_tuning_lr{lr}", TOKEN_GEO[lr], TOKEN[lr]) for lr in TOKEN_LRS),
    *((f"geo_loss_effect_in_scene_token_tuning_token_lr{lr}", JOINT_GEO[lr], JOINT[lr]) for lr in TOKEN_LRS),
    *((f"token_tuning_lr{lr}_vs_scene_tuning", TOKEN[lr], SCENE) for lr in TOKEN_LRS),
    *((f"adding_tokens_lr{lr}_to_scene_tuning", JOINT[lr], SCENE) for lr in TOKEN_LRS),
    *((f"adding_tokens_lr{lr}_to_scene_geo_tuning", JOINT_GEO[lr], SCENE_GEO) for lr in TOKEN_LRS),
    *((f"adding_scene_latents_to_token_tuning_lr{lr}", JOINT[lr], TOKEN[lr]) for lr in TOKEN_LRS),
    *((f"adding_scene_latents_to_token_geo_tuning_lr{lr}", JOINT_GEO[lr], TOKEN_GEO[lr]) for lr in TOKEN_LRS),
    *((f"{g}_vs_no_tuning", g, NO_TUNING) for g in GROUPS if g != NO_TUNING),
)
# frozen in EXP_TokenGS/reports/scene_tuning_noise_floor.json: (label, this run, reference)
FROZEN_COMPARISONS = (("repeat_run_noise_floor", f"{SCENE} (repeat run)", SCENE),)
# compact table: (key in compare(), column title, with CI and wins)
SUMMARY_KEYS = (("all_psnr", "all PSNR", True), ("novel_psnr", "novel PSNR", True),
                ("bucket_d4+_psnr", "d4+ PSNR", True), ("all_lpips", "LPIPS", False),
                ("all_hf_ratio", "HF ratio", False), ("geo_sw_loss", "SW", False),
                ("geo_packing_ratio", "packing ratio", False), ("geo_inter_distance", "inter d", False),
                ("needle_aniso_p50_weighted", "aniso p50", False),
                ("needle_long_axis_along_ray_gt0.9_weighted", "∥ ray >0.9", False))


def mean_of(summary: dict, key: str) -> float | None:
    """140-scene mean of one scene_metrics key, read from summary.json."""
    if key.startswith(("all_", "novel_")):
        section, metric = key.split("_", 1)
        return summary[{"all": "all_frames", "novel": "novel_only"}[section]]["metrics"][metric]["mean"]
    if key.startswith("bucket_"):
        return summary["distance_buckets"][key[len("bucket_"):-len("_psnr")]]["psnr"]["mean"]
    return None


def markdown(runs: dict, comps: dict) -> str:
    lines = ["# Tuned parameters x geometry loss (140 scenes): report", "",
             "Scene latents lr 1e-2, Gaussian tokens lr 1e-4 or 1e-2, "
             f"50 steps; geometry loss weights sw / packing / inter = {GEO_WEIGHTS}. Token tuning lr 1e-4 without "
             "geometry loss = the earlier "
             "official-loop run.", "",
             "## all PSNR grid (140-scene mean; no tuning "
             f"{fmt(mean_of(runs[NO_TUNING]['summary'], 'all_psnr'), 3) if NO_TUNING in runs else '—'})", "",
             "| tuned | render loss only | render + geometry loss | geometry loss Δ (paired, 95% CI) |",
             "|---|---|---|---|"]
    for label, plain, geo in GRID:
        cells = [fmt(mean_of(runs[g]["summary"], "all_psnr"), 3) if g in runs else "not finished" for g in (plain, geo)]
        c = comps.get(next(lbl for lbl, a, b in COMPARISONS if (a, b) == (geo, plain)), {}).get("all_psnr")
        delta = f"{c['delta_mean']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]" if c else "—"
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {delta} |")
    lines += ["", "## Per group (140-scene mean)", "",
              "| group | all PSNR | all SSIM | all LPIPS | novel PSNR | HF ratio | d4+ PSNR |", "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        if g not in runs:
            lines.append(f"| {g} | not finished | | | | | |")
            continue
        s = runs[g]["summary"]
        am = s["all_frames"]["metrics"]
        lines.append(f"| {g} | {fmt(am['psnr']['mean'], 3)} | {fmt(am['ssim']['mean'])} | {fmt(am['lpips']['mean'])} | "
                     f"{fmt(mean_of(s, 'novel_psnr'), 3)} | {fmt(am['hf_ratio']['mean'], 3)} | "
                     f"{fmt(mean_of(s, 'bucket_d4+_psnr'), 3)} |")
    lines += ["", "## Geometry / needles (scene median)", "",
              "| group | SW | packing ratio / target | inter d | too close | too far | aniso p50 | long axis ∥ ray >0.9 |",
              "|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        if g not in runs:
            continue
        geo, nd = runs[g]["summary"]["geometry"], runs[g]["summary"]["needle"]
        lines.append(f"| {g} | {fmt(geo['sw_loss']['median'])} | {fmt(geo['packing_ratio']['median'], 1)} / "
                     f"{fmt(geo['packing_target']['median'], 2)} | {fmt(geo['inter_distance']['median'], 3)} | "
                     f"{fmt(geo['inter_too_close']['median'], 3)} | {fmt(geo['inter_too_far']['median'], 3)} | "
                     f"{fmt(nd['aniso_p50_weighted']['median'], 1)} | "
                     f"{fmt(nd['long_axis_along_ray_gt0.9_weighted']['median'], 3)} |")
    lines += ["", "## Paired comparisons (Δ = this - reference, mean over scenes; PSNR with 95% CI and scene wins "
                  "this/ref)", "",
              "| comparison | " + " | ".join(t for _, t, _ in SUMMARY_KEYS) + " |",
              "|---|" + "---|" * len(SUMMARY_KEYS)]
    for label, this, ref in (*COMPARISONS, *FROZEN_COMPARISONS):
        c = comps.get(label)
        if c is None:
            lines.append(f"| {label} | not available |" + " |" * (len(SUMMARY_KEYS) - 1))
            continue
        cells = []
        for key, _, full in SUMMARY_KEYS:
            r = c.get(key)
            if r is None:
                cells.append("—")
            elif full:
                cells.append(f"{r['delta_mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] "
                             f"{r['this_better']}/{r['ref_better']}")
            else:
                cells.append(f"{r['delta_mean']:+.4g}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Comparison definitions (this - reference): "
              + "; ".join(f"{label} = {this} - {ref}" for label, this, ref in (*COMPARISONS, *FROZEN_COMPARISONS))]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=REPORT_DIR, help="where scene_token_x_geo.{json,md} are written")
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs = {g: r for g in GROUPS if (r := load_run(Path(RUN_DIRS[g]))) is not None}
    print("finished groups:", sorted(runs))
    frozen = load_noise_floor()
    comps = {label: compare(runs[a], runs[b]) for label, a, b in COMPARISONS if a in runs and b in runs}
    comps.update({label: frozen["comparisons"][label] for label, _, _ in FROZEN_COMPARISONS})
    report = {"groups_finished": sorted(runs), "run_dirs": RUN_DIRS,
              "comparison_definitions": {label: {"this": a, "reference": b}
                                         for label, a, b in (*COMPARISONS, *FROZEN_COMPARISONS)},
              "comparisons": comps,
              "paired_psnr_deltas_by_scene": {
                  **{label: {s: runs[a]["rows"][s]["all_psnr"] - runs[b]["rows"][s]["all_psnr"]
                             for s in sorted(set(runs[a]["rows"]) & set(runs[b]["rows"]))}
                     for label, a, b in COMPARISONS if label in comps},
                  **{label: frozen["paired_psnr_deltas_by_scene"][label] for label, _, _ in FROZEN_COMPARISONS}}}
    (root / "scene_token_x_geo.json").write_text(json.dumps(report, indent=2) + "\n")
    md = markdown(runs, comps)
    (root / "scene_token_x_geo.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
