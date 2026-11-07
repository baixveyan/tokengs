"""Spatial confusion counts of trained grid-occupancy probes: how many cells are right / wrong.

Pure 3D metric over every grid cell (no target views, no frustum scopes). Reads the experiment's manifest,
model.safetensors and the cached per-scene cell features; writes confusion_metrics.csv and confusion_summary.json.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from geometrygs.grid_probe import GridClassifier, dilate

GEO_DIM = 28
SHELLS = (1, 2, 3)


def shell_counts(pred, y, dims):
    """FP cells within k cells (Chebyshev) of a true surface cell; FN cells within k cells of a predicted cell."""
    py, pp = torch.as_tensor(y), torch.as_tensor(pred)
    out = {}
    for k in SHELLS:
        out[f"fp_within_{k}"] = int((pp & ~py & dilate(py, dims, k)).sum())
        out[f"fn_within_{k}"] = int((~pp & py & dilate(pp, dims, k)).sum())
    return out


def counts(pred, y):
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


def rates(c):
    n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    pos, neg = c["tp"] + c["fn"], c["fp"] + c["tn"]
    recall, specificity = c["tp"] / max(pos, 1), c["tn"] / max(neg, 1)
    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    shells = {f"{e}_within_{k}_fraction": c[f"{e}_within_{k}"] / max(c[e], 1)
              for e in ("fp", "fn") for k in SHELLS if f"{e}_within_{k}" in c}
    return dict(shells, cells=n, correct=c["tp"] + c["tn"], wrong=c["fp"] + c["fn"],
                accuracy=(c["tp"] + c["tn"]) / max(n, 1), all_empty_accuracy=neg / max(n, 1),
                recall=recall, specificity=specificity, precision=precision,
                balanced_accuracy=(recall + specificity) / 2,
                iou=c["tp"] / max(c["tp"] + c["fp"] + c["fn"], 1),
                f1=2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, default=Path("EXP/grid_occupancy_10cm"))
    ap.add_argument("--workspace", type=Path,
                    default=Path("/mnt/disk/xveyan/workspace_scannetpp/geometrygs_grid_occupancy_10cm"))
    ap.add_argument("--probability", type=float, default=.5)
    args = ap.parse_args()
    state = load_file(str(args.experiment / "model.safetensors"))
    manifest = json.loads((args.experiment / "manifest.json").read_text())
    logit = float(np.log(args.probability / (1 - args.probability)))
    probes = {}
    for name in ("camera_geometry", "projected_image"):
        s = {k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + ".")}
        probe = GridClassifier(s["mean"], s["std"], hidden=(256, 128))
        probe.load_state_dict(s, strict=True)
        probes[name] = probe.eval()
    rows, pooled = [], {}
    for entry in manifest:
        rec = torch.load(args.workspace / f"{entry['scene_id']}.pt", weights_only=False)
        y = rec["y"]
        x = torch.as_tensor(rec["x"])
        for name, probe in probes.items():
            dims = GEO_DIM if name == "camera_geometry" else x.shape[1]
            with torch.no_grad():
                score = torch.cat([probe(c[:, :dims].float()) for c in x.split(65536)]).numpy()
            c = counts(score >= logit, y)
            c.update(shell_counts(score >= logit, y, rec["dims"]))
            rows.append(dict(model=name, split=entry["split"], scene_id=entry["scene_id"], **c, **rates(c)))
            acc = pooled.setdefault(entry["split"], {}).setdefault(name, {k: 0 for k in c})
            for k in acc:
                acc[k] += c[k]
        print(entry["split"], entry["scene_id"], flush=True)
    with (args.experiment / "confusion_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(threshold_probability=args.probability, threshold_logit=logit,
                   note="probes were trained with 1/2 positive batches; probability 0.5 assumes balanced classes",
                   pooled={split: {name: dict(c, **rates(c)) for name, c in models.items()}
                           for split, models in pooled.items()},
                   scene_median={split: {name: {k: float(np.median([r[k] for r in rows
                                                                     if r["split"] == split and r["model"] == name]))
                                                for k in ("accuracy", "recall", "precision", "iou", "balanced_accuracy")}
                                         for name in probes} for split in pooled})
    (args.experiment / "confusion_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
