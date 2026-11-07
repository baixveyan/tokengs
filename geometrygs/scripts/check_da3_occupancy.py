"""Pretrained DA3 depth -> back-projected points -> 10 cm cells, compared cell by cell with the laser labels.

Re-uses the label grids of EXP/grid_occupancy_10cm (input-visible first laser hit = 1, everything else 0; laser box +
1 cell; 10 cm). For every validation window: DA3-GIANT-1.1, pose-conditioned on the six input cameras, predicts depth
from the six input images (512x896 crop of the same views). Depth scale comes from aligning DA3's cameras to the input
cameras (Umeyama, no laser). A cell is predicted occupied if at least one back-projected pixel lands in it.
Diagnostics only: the same depth with a per-scene laser-median scale. Laser is used for labels and errors only.
Main set: the 20 ScanNet++ scenes DA3 held out from training; the other 30 validation scenes are reported separately.
"""

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

import geometrygs
from geometrygs.da3_tools import DA3_HELD_OUT, da3_window, load_da3
from geometrygs.grid_probe import backproject, cell_counts, dilate, grid_index
from geometrygs.options import config_defaults as sfm_defaults
from geometrygs.data import SfmProvider

SHELLS = (1, 2, 3)
MIN_POINTS = (1, 3, 10)  # a cell is occupied if it receives at least this many back-projected pixels


def confusion(pred, y, dims):
    p, t = torch.as_tensor(pred), torch.as_tensor(y)
    out = dict(tp=int((p & t).sum()), fp=int((p & ~t).sum()), fn=int((~p & t).sum()), tn=int((~p & ~t).sum()))
    for k in SHELLS:
        out[f"fp_within_{k}"] = int((p & ~t & dilate(t, dims, k)).sum())
        out[f"fn_within_{k}"] = int((~p & t & dilate(p, dims, k)).sum())
    return out


def rates(c):
    n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    out = dict(cells=n, correct=c["tp"] + c["tn"], wrong=c["fp"] + c["fn"], accuracy=(c["tp"] + c["tn"]) / max(n, 1),
               recall=c["tp"] / max(c["tp"] + c["fn"], 1), precision=c["tp"] / max(c["tp"] + c["fp"], 1),
               iou=c["tp"] / max(c["tp"] + c["fp"] + c["fn"], 1),
               f1=2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1))
    for e in ("fp", "fn"):
        for k in SHELLS:
            out[f"{e}_within_{k}_fraction"] = c[f"{e}_within_{k}"] / max(c[e], 1)
    return out


def depth_errors(pred, laser, valid):
    d, g = pred[valid], laser[valid]
    err = (d - g).abs() / .3
    return dict(depth_pixels=int(valid.sum()), absrel=float(((d - g).abs() / g).mean()),
                abs_median_cm=float(err.median() * 100), within_5cm=float((err < .05).float().mean()),
                within_10cm=float((err < .10).float().mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("EXP/da3_occupancy_10cm"))
    ap.add_argument("--labels", type=Path,
                    default=Path("/mnt/disk/xveyan/workspace_scannetpp/geometrygs_grid_occupancy_10cm"))
    ap.add_argument("--workspace", type=Path,
                    default=Path("/mnt/disk/xveyan/workspace_scannetpp/geometrygs_da3_occupancy_10cm"))
    ap.add_argument("--da3-root", type=Path, default=geometrygs.GEOMETRYGS_ROOT.parent / "Depth-Anything-3")
    ap.add_argument("--checkpoint", type=Path, default=Path("/mnt/disk/xveyan/checkpoints/DA3-GIANT-1.1"))
    ap.add_argument("--image-size", type=int, nargs=2, default=(512, 896))
    ap.add_argument("--process-res", type=int, default=896)
    ap.add_argument("--demo-scene", default="fb5a96b1a2")
    ap.add_argument("--max-scenes", type=int, default=50)
    args = ap.parse_args()
    from tokengs.train import verify_idle_gpus
    verify_idle_gpus()
    if (args.output / "summary.json").exists():
        raise FileExistsError(f"{args.output} is complete; choose a new output instead of overwriting")
    args.output.mkdir(parents=True, exist_ok=True)
    args.workspace.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    start = time.perf_counter()
    model, da3_file = load_da3(args.da3_root, args.checkpoint, device)
    opt = sfm_defaults["eval_sfm_anchor"].evolve(
        num_workers=0, load_lidar_pc=False, val_depth_metric=False, sfm_image_size=tuple(args.image_size),
        dataset_kwargs={"split": "nvs_sem_val.txt", "window_size": 18})
    provider = SfmProvider("scannetpp_val_scaled_0.3", opt, training=False)
    rows, depth_rows, frames = [], [], []
    pooled = {}
    n = min(args.max_scenes, len(provider))
    for i in range(n):
        win = da3_window(model, provider, i, args.labels, args.process_res)
        sid, raw, rec, ctx, c2w, images, pred, secs = (win[k] for k in ("sid", "raw", "rec", "ctx", "c2w", "images",
                                                                         "pred", "secs"))
        depth, conf, intr, h, w, laser, ok = (win[k] for k in ("depth", "conf", "intr", "h", "w", "laser", "ok"))
        group = "da3_held_out" if sid in DA3_HELD_OUT else "da3_maybe_seen"
        y, lo, dims, cell = rec["y"], rec["lo"], rec["dims"], rec["cell"]
        valid_pred = depth > 0
        scale_oracle = float((laser[ok & valid_pred] / depth[ok & valid_pred]).median())
        keep_conf = conf >= torch.quantile(conf[valid_pred].flatten()[::7], .4)
        masks = dict(all_pixels=valid_pred, conf_top60=valid_pred & keep_conf)
        derr = depth_errors(depth, laser, ok & valid_pred)
        derr_oracle = depth_errors(depth * scale_oracle, laser, ok & valid_pred)
        depth_rows.append(dict(scene_id=sid, group=group, process_hw=f"{h}x{w}", seconds=secs,
                               laser_valid_fraction=float(ok.float().mean()), oracle_scale=scale_oracle,
                               **{f"cam_{k}": v for k, v in derr.items()},
                               **{f"oracle_{k}": v for k, v in derr_oracle.items() if k != "depth_pixels"}))
        result = {}
        # Reference: the laser depth itself on DA3's pixel grid (resolution ceiling), and with Gaussian depth noise.
        gen = torch.Generator().manual_seed(i)
        noise = torch.randn(laser.shape, generator=gen)
        sources = [("camera_scale", depth, masks), ("oracle_scale", depth * scale_oracle, masks)]
        sources += [(f"laser_reference_{n}", laser + noise * sigma * .3, dict(all_pixels=ok))
                    for n, sigma in (("exact", 0.), ("noise_2cm", .02), ("noise_5cm", .05))]
        for scale_name, d, variant_masks in sources:
            points = backproject(d, intr, c2w.float())
            for base, m in variant_masks.items():
                pts = points[m]
                _, inside = grid_index(pts, lo, cell, dims)
                counts = cell_counts(pts, lo, cell, dims)
                for min_points in MIN_POINTS:
                    variant = base if min_points == 1 else f"{base}_min{min_points}"
                    occupied = counts >= min_points
                    c = confusion(occupied, torch.as_tensor(y), dims)
                    result[(scale_name, variant)] = occupied.numpy()
                    rows.append(dict(scene_id=sid, group=group, scale=scale_name, variant=variant,
                                     points=int(m.sum()), points_outside_grid=float(1 - inside.float().mean()),
                                     **c, **rates(c)))
                    acc = pooled.setdefault((group, scale_name, variant), {kk: 0 for kk in c})
                    for kk in c:
                        acc[kk] += c[kk]
        np.savez_compressed(args.workspace / f"{sid}.npz", **{f"{a}__{b}": v for (a, b), v in result.items()},
                            labels=y, lo=lo.numpy(), dims=dims.numpy(), cell=cell)
        if sid == args.demo_scene:
            pts = backproject(depth, intr, c2w.float())[valid_pred]
            rgb = torch.from_numpy(np.stack([np.asarray(im) for im in pred.processed_images])).float() / 255
            col = rgb[valid_pred.numpy()] if rgb.shape[1:3] == (h, w) else torch.full_like(pts, .5)
            lp = backproject(laser, intr, c2w.float())[ok]
            np.savez_compressed(args.workspace / f"{sid}_demo.npz", da3_points=pts[::3].numpy(),
                                da3_colors=col[::3].numpy(), laser_points=lp[::3].numpy(),
                                input_frames=np.array(ctx), images=np.stack(images))
        frames += [dict(scene_id=sid, group=group, frame_id=f, process_hw=f"{h}x{w}",
                        laser_valid_fraction=float(ok[v].float().mean()),
                        **{f"cam_{kk}": vv for kk, vv in depth_errors(depth[v], laser[v], (ok & valid_pred)[v]).items()})
                   for v, f in enumerate(ctx)]
        main_row = next(r for r in rows[::-1] if r["scene_id"] == sid and r["scale"] == "camera_scale"
                        and r["variant"] == "all_pixels")
        print(f"[{i + 1}/{n}] {sid} ({group}) {h}x{w} {secs:.1f}s | depth |err|<10cm {derr['within_10cm']:.3f} "
              f"(oracle {derr_oracle['within_10cm']:.3f}, scale x{scale_oracle:.3f}) | cells TP/FP/FN "
              f"{main_row['tp']}/{main_row['fp']}/{main_row['fn']} IoU {main_row['iou']:.3f}", flush=True)
        del win
    for path, data in (("scene_metrics.csv", rows), ("depth_metrics.csv", depth_rows), ("frame_metrics.csv", frames)):
        with (args.output / path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    # The probe of grid_occupancy_10cm on the same scenes (probability 0.5), for direct comparison.
    probe = {}
    conf_csv = Path("EXP/grid_occupancy_10cm/confusion_metrics.csv")
    if conf_csv.exists():
        for r in csv.DictReader(conf_csv.open()):
            if r["split"] != "test" or r["model"] != "projected_image":
                continue
            g = "da3_held_out" if r["scene_id"] in DA3_HELD_OUT else "da3_maybe_seen"
            acc = probe.setdefault(g, dict(tp=0, fp=0, fn=0, tn=0))
            for kk in acc:
                acc[kk] += int(r[kk])
    summary = dict(
        protocol=vars(args) | {k: str(getattr(args, k)) for k in ("output", "labels", "workspace", "da3_root",
                                                               "checkpoint")},
        da3_code=str(da3_file), da3_precision="DA3 forward under bf16 autocast "
        "(library default); back-projection, labels and metrics FP32",
        labels="grid_occupancy_10cm workspace: input-view first laser hit, 10 cm cells, laser box + 1 cell",
        prediction="cell occupied iff >= k back-projected DA3 pixels inside (k = 1, 3, 10; suffix _min3/_min10); "
                   "conf_top60 drops pixels below the scene's 40th confidence percentile (DA3 export default)",
        laser_reference="the laser depth on DA3's pixel grid back-projected with the same rule; exact, and with "
                        "Gaussian depth noise of 2 / 5 cm (ceiling and noise sensitivity of the cell metric)",
        scale=dict(camera_scale="Umeyama scale between DA3 and input cameras (deployable, no laser)",
                   oracle_scale="per-scene median laser/DA3 depth ratio (diagnostic only)"),
        scenes={g: sum(1 for r in depth_rows if r["group"] == g) for g in ("da3_held_out", "da3_maybe_seen")},
        pooled={f"{g}/{s}/{v}": dict(c, **rates(c)) for (g, s, v), c in pooled.items()},
        probe_grid_occupancy_10cm_p50={g: dict(c, **{k: v for k, v in rates(dict(
            c, **{f"{e}_within_{kk}": 0 for e in ("fp", "fn") for kk in SHELLS})).items() if "within" not in k})
            for g, c in probe.items()},
        depth_scene_mean={g: {k: float(np.mean([r[k] for r in depth_rows if r["group"] == g]))
                              for k in depth_rows[0] if k not in ("scene_id", "group", "process_hw")}
                          for g in ("da3_held_out", "da3_maybe_seen") if any(r["group"] == g for r in depth_rows)},
        elapsed_seconds=time.perf_counter() - start)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k in ("pooled", "probe_grid_occupancy_10cm_p50",
                                                                "depth_scene_mean")}, indent=1))
    print(f"Completed: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
