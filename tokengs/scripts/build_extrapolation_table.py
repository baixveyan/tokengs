"""Build EXP_ScanNetPP/eval_extrapolation/extrapolation_table.{md,csv}: interp (official) vs ext_near vs ext_far."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "EXP_ScanNetPP"
OUT = EXP / "eval_extrapolation"
EXPS = ["baseline_pretrained_6view", "sft_latenttail_rgb", "sft_readout_rgb", "sft_readout_rgb_depth",
        "sft_readout_rgb_depth_thick", "sft_readout_rgb_depth_thick_sh1", "sft_readout_rgb_depth_thick_posmlp", "sft_readout_rgb_depth_thick_posmlp_surfel_binar",
        "sft_readout_rgb_depth_surface3d", "sft_readout_rgb_depth_surface3d_posmlp",
        "sft_latenttail_readout_rgb_depth_surface3d_posmlp", "sft_readout_rgb_depth_surface3d_posmlp_ep20", "sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20", "sft_latenttail_readout_rgb_depth_surface3d_posmlp_attrmlp_ep20", "sft_latenttail_readout_rgb_depth_surface3d_posmlp_opamlp"]
COLS = [("psnr", "PSNR ↑", 3), ("ssim", "SSIM ↑", 4), ("lpips", "LPIPS ↓", 4), ("depth_absrel", "AbsRel ↓", 4),
        ("ray_thick_p90_m", "厚度p90(m) ↓", 4), ("surf_fscore_5cm", "F@5cm ↑", 4)]
MODES = [("interp", "窗口内插值（官方，距最近输入 1 帧）"), ("ext_near", "近外推（窗口外 3–12 帧）"),
         ("ext_far", "远外推（窗口外 18–36 帧）")]


def load(mode, exp):
    p = (EXP / exp / "summary.json") if mode == "interp" else (OUT / mode / exp / "summary.json")
    return json.loads(p.read_text()) if p.is_file() else None


def main():
    rows, md = [], ["# 外推视角评测（同 6 输入，目标视角在 18 帧窗口之外）", ""]
    for mode, title in MODES:
        md += [f"## {title}", "", "| 实验 | " + " | ".join(c[1] for c in COLS) + " |", "|---|" + "---:|" * len(COLS)]
        for exp in EXPS:
            s = load(mode, exp)
            if s is None:
                continue
            md.append(f"| {exp} | " + " | ".join(f"{s[k]:.{d}f}" if k in s else "—" for k, _, d in COLS) + " |")
            rows.append({"mode": mode, "experiment": exp, **{k: s.get(k) for k, _, _ in COLS}})
        md.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "extrapolation_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(OUT / "extrapolation_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "experiment", *[c[0] for c in COLS]])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
