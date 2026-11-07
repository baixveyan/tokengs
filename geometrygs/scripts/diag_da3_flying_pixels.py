"""Why do DA3 points bridge front and back surfaces along camera rays? Flying-pixel diagnosis on 10 cm cells.

Per DA3 pixel (pose-conditioned DA3-GIANT-1.1, camera scale, as in EXP/da3_occupancy_10cm):
  flying  = the predicted depth lies strictly between the nearest and farthest laser depth in the pixel's 5x5
            neighbourhood and is more than 10 cm from both (suspended between a front and a back surface);
  laser edge = that laser neighbourhood spans more than 20 cm in depth (an occlusion boundary or grazing surface).
Far false alarms = predicted-occupied cells that are empty and more than one cell (Chebyshev) from any surface cell.
Counterfactuals: drop pixels whose own DA3 5x5 depth range exceeds tau * depth (deployable, no laser), or drop the
laser-defined flying pixels (oracle). Laser is used for labels, the flying definition and the oracle only.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import geometrygs
from geometrygs.da3_tools import DA3_HELD_OUT, da3_window, load_da3
from geometrygs.data import SfmProvider
from geometrygs.grid_probe import backproject, dilate, grid_index
from geometrygs.options import config_defaults as sfm_defaults
from geometrygs.plyio import write_points_ply

CM = .003  # normalized units per centimetre (0.3 per metre)
TAUS = (.05, .1, .2)


def neighbourhood(x, valid, k=5):
    """5x5 min / max of x over valid pixels, per view."""
    big = torch.finfo(x.dtype).max
    lo = -F.max_pool2d(torch.where(valid, -x, -big)[:, None], k, 1, k // 2)[:, 0]
    hi = F.max_pool2d(torch.where(valid, x, -big)[:, None], k, 1, k // 2)[:, 0]
    return lo, hi


def cell_stats(points, keep, y, lo, cell, dims, far_mask_of):
    ids, inside = grid_index(points[keep], lo, cell, dims)
    occ = torch.bincount(ids[inside], minlength=len(y)) > 0
    yt = torch.as_tensor(y)
    tp, fp, fn = int((occ & yt).sum()), int((occ & ~yt).sum()), int((~occ & yt).sum())
    return dict(tp=tp, fp=fp, fn=fn, far_fp=int((occ & far_mask_of).sum()),
                iou=tp / max(tp + fp + fn, 1), recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                pixels_kept=float(keep.float().mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("EXP/da3_flying_pixels"))
    ap.add_argument("--demo", type=Path, default=Path("EXP/demo/scene_fb5a96b1a2/da3_flying_pixels"))
    ap.add_argument("--labels", type=Path,
                    default=Path("/mnt/disk/xveyan/workspace_scannetpp/geometrygs_grid_occupancy_10cm"))
    ap.add_argument("--da3-root", type=Path, default=geometrygs.GEOMETRYGS_ROOT.parent / "Depth-Anything-3")
    ap.add_argument("--checkpoint", type=Path, default=Path("/mnt/disk/xveyan/checkpoints/DA3-GIANT-1.1"))
    ap.add_argument("--image-size", type=int, nargs=2, default=(512, 896))
    ap.add_argument("--process-res", type=int, default=896)
    ap.add_argument("--demo-scene", default="fb5a96b1a2")
    args = ap.parse_args()
    from tokengs.train import verify_idle_gpus
    verify_idle_gpus()
    if (args.output / "summary.json").exists():
        raise FileExistsError(f"{args.output} is complete; choose a new output instead of overwriting")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    model, da3_file = load_da3(args.da3_root, args.checkpoint, torch.device("cuda"))
    opt = sfm_defaults["eval_sfm_anchor"].evolve(
        num_workers=0, load_lidar_pc=False, val_depth_metric=False, sfm_image_size=tuple(args.image_size),
        dataset_kwargs={"split": "nvs_sem_val.txt", "window_size": 18})
    provider = SfmProvider("scannetpp_val_scaled_0.3", opt, training=False)
    order = [i for i in range(len(provider)) if Path(provider.dataset.sample_list[i]).name in DA3_HELD_OUT]
    rows, pooled = [], {}
    for i in order:
        win = da3_window(model, provider, i, args.labels, args.process_res)
        sid, rec, d, laser, ok, intr = (win[k] for k in ("sid", "rec", "depth", "laser", "ok", "intr"))
        c2w = win["c2w"].float()
        y, lo, cell, dims = rec["y"], rec["lo"], rec["cell"], rec["dims"]
        valid = d > 0
        lmin, lmax = neighbourhood(laser, ok)
        has_nbr = ok & (lmax >= lmin)
        flying = valid & has_nbr & (d - lmin > 10 * CM) & (lmax - d > 10 * CM)
        laser_edge = has_nbr & (lmax - lmin > 20 * CM)
        dmin, dmax = neighbourhood(d, valid)
        rel = (dmax - dmin) / d.clamp_min(1e-6)
        points = backproject(d, intr, c2w)
        yt = torch.as_tensor(y)
        far_empty = ~dilate(yt, dims, 1)  # empty and not within one cell of a surface cell
        ids, inside = grid_index(points, lo, cell, dims)
        occ = torch.bincount(ids[valid & inside], minlength=len(y)) > 0
        in_far_fp = valid & inside & far_empty[ids] & occ[ids]
        row = dict(scene_id=sid, pixels=int(valid.sum()), flying_pixel_fraction=float(flying[valid].float().mean()),
                   laser_edge_pixel_fraction=float(laser_edge[valid].float().mean()),
                   far_fp_cells=int((occ & far_empty).sum()), fp_cells=int((occ & ~yt).sum()),
                   far_fp_points=int(in_far_fp.sum()),
                   far_fp_points_flying=float(flying[in_far_fp].float().mean()) if in_far_fp.any() else 0.,
                   far_fp_points_on_laser_edge=float(laser_edge[in_far_fp].float().mean()) if in_far_fp.any() else 0.)
        row["far_fp_points_pred_edge_0.1"] = float((rel[in_far_fp] > .1).float().mean()) if in_far_fp.any() else 0.
        # Fraction of far-false-alarm cells that would disappear if flying pixels were removed.
        occ_noflying = torch.bincount(ids[valid & inside & ~flying], minlength=len(y)) > 0
        row["far_fp_cells_only_from_flying"] = float(((occ & far_empty) & ~occ_noflying).sum()
                                                     / max(int((occ & far_empty).sum()), 1))
        variants = dict(all_pixels=valid, oracle_drop_flying=valid & ~flying)
        variants.update({f"drop_pred_edge_{t:g}": valid & (rel <= t) for t in TAUS})
        for name, keep in variants.items():
            s = cell_stats(points, keep, y, lo, cell, dims, far_empty)
            row.update({f"{name}_{k}": v for k, v in s.items()})
            acc = pooled.setdefault(name, dict(tp=0, fp=0, fn=0, far_fp=0, kept=0, total=0))
            for k in ("tp", "fp", "fn", "far_fp"):
                acc[k] += s[k]
            acc["kept"] += int(keep.sum())
            acc["total"] += int(valid.sum())
        rows.append(row)
        print(f"{sid}: flying {row['flying_pixel_fraction']:.3f} of pixels; far FP cells {row['far_fp_cells']}"
              f"/{row['fp_cells']} FP; their points: flying {row['far_fp_points_flying']:.2f}, laser edge "
              f"{row['far_fp_points_on_laser_edge']:.2f}; cells only from flying {row['far_fp_cells_only_from_flying']:.2f}"
              f" | IoU all {row['all_pixels_iou']:.3f} drop-edge-0.1 {row['drop_pred_edge_0.1_iou']:.3f} "
              f"oracle {row['oracle_drop_flying_iou']:.3f}", flush=True)
        if sid == args.demo_scene:
            write_demo(args.demo, win, flying, rel, points, valid, in_far_fp, lo, cell, dims)
        del win
    with (args.output / "scene_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    total = {k: sum(r[k] for r in rows) for k in ("pixels", "far_fp_points", "far_fp_cells", "fp_cells")}
    weighted = lambda key, w: sum(r[key] * r[w] for r in rows) / max(sum(r[w] for r in rows), 1)
    summary = dict(
        scenes=len(rows), da3_code=str(da3_file), definitions=__doc__,
        pooled_pixels=dict(flying_fraction=weighted("flying_pixel_fraction", "pixels"),
                           laser_edge_fraction=weighted("laser_edge_pixel_fraction", "pixels")),
        far_false_alarms=dict(cells=total["far_fp_cells"], of_all_fp_cells=total["far_fp_cells"] / max(total["fp_cells"], 1),
                              points=total["far_fp_points"],
                              points_flying=weighted("far_fp_points_flying", "far_fp_points"),
                              points_on_laser_edge=weighted("far_fp_points_on_laser_edge", "far_fp_points"),
                              points_pred_edge_0p1=weighted("far_fp_points_pred_edge_0.1", "far_fp_points"),
                              cells_only_from_flying=weighted("far_fp_cells_only_from_flying", "far_fp_cells")),
        variants={k: dict(v, iou=v["tp"] / max(v["tp"] + v["fp"] + v["fn"], 1), recall=v["tp"] / max(v["tp"] + v["fn"], 1),
                          precision=v["tp"] / max(v["tp"] + v["fp"], 1), pixels_kept=v["kept"] / max(v["total"], 1))
                  for k, v in pooled.items()},
        elapsed_seconds=time.perf_counter() - start)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("pooled_pixels", "far_false_alarms", "variants")}, indent=1))


def write_demo(out, win, flying, rel, points, valid, in_far_fp, lo, cell, dims):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    d, laser, ok = win["depth"], win["laser"], win["ok"]
    images = np.stack([np.asarray(im) for im in win["pred"].processed_images]) / 255.
    fig, axs = plt.subplots(3, 6, figsize=(20, 6.6))
    for v in range(6):
        axs[0, v].imshow(images[v]); axs[0, v].set_title(f"input frame {win['ctx'][v]}", fontsize=8)
        over = images[v].copy()
        over[flying[v].numpy()] = [1, 0, 0]
        axs[1, v].imshow(over); axs[1, v].set_title("red: flying pixels (laser-defined)", fontsize=8)
        over = images[v].copy()
        over[(valid[v] & (rel[v] > .1)).numpy()] = [1, .85, 0]
        axs[2, v].imshow(over); axs[2, v].set_title("yellow: DA3 5x5 depth range > 10% (no laser)", fontsize=8)
    for ax in axs.flat:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "flying_pixels_views.png", dpi=140); plt.close(fig)

    # Depth profile across the image row with the most flying pixels.
    v, r = np.unravel_index(int(flying.sum(2).argmax()), flying.shape[:2])
    cols = np.arange(d.shape[2])
    fig, axs = plt.subplots(2, 1, figsize=(13, 6.5), gridspec_kw=dict(height_ratios=[1, 2]))
    axs[0].imshow(images[v][max(0, r - 25):r + 25]); axs[0].axhline(min(r, 25), color="r", lw=.8)
    axs[0].set_title(f"view {win['ctx'][v]}, row {r}"); axs[0].axis("off")
    lz, dz = laser[v, r].numpy() / .3, d[v, r].numpy() / .3
    axs[1].plot(cols[ok[v, r].numpy()], lz[ok[v, r].numpy()], ".", ms=3, color="tab:green", label="laser depth")
    axs[1].plot(cols, dz, "-", lw=1, color="tab:blue", label="DA3 depth")
    fl = flying[v, r].numpy()
    axs[1].plot(cols[fl], dz[fl], "o", ms=4, mfc="none", color="tab:red", label="flying pixel")
    axs[1].set(xlabel="image column", ylabel="depth (m)", title="DA3 interpolates across depth jumps: "
               "pixels between the front and back surface become points in empty space")
    axs[1].legend(); axs[1].set_xlim(0, d.shape[2])
    fig.tight_layout(); fig.savefig(out / "depth_profile.png", dpi=150); plt.close(fig)

    # Top-down view of one 10 cm horizontal layer: laser, DA3 points, flying points.
    pts, fly = points[valid].numpy(), flying[valid].numpy()
    lpts = backproject(laser, win["intr"], win["c2w"].float())[ok].numpy()
    ijk = np.floor((pts - lo.numpy()) / cell).astype(int)
    layer = np.bincount(ijk[fly, 1].clip(0, int(dims[1]) - 1), minlength=int(dims[1])).argmax()
    sel = ijk[:, 1] == layer
    lsel = np.floor((lpts[:, 1] - float(lo[1])) / cell).astype(int) == layer
    cams = win["c2w"][:, :3, 3].numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(lpts[lsel, 0] / .3, lpts[lsel, 2] / .3, s=.3, c="tab:green", label="laser (input-visible)")
    ax.scatter(pts[sel & ~fly, 0] / .3, pts[sel & ~fly, 2] / .3, s=.3, c="0.55", label="DA3 points")
    ax.scatter(pts[sel & fly, 0] / .3, pts[sel & fly, 2] / .3, s=1.5, c="tab:red", label="DA3 flying points")
    for c in cams:
        ax.plot(c[0] / .3, c[2] / .3, "k^", ms=6)
    ax.set(aspect="equal", xlabel="x (m)", ylabel="z (m, forward)",
           title=f"{win['sid']}: one 10 cm layer (y = {(float(lo[1]) + (layer + .5) * cell) / .3:+.2f} m); "
                 "flying points line up along camera rays")
    ax.legend(markerscale=8, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "flying_points_layer.png", dpi=170); plt.close(fig)

    colors = np.tile([.6, .6, .6], (len(pts), 1))
    colors[fly] = [1, 0, 0]
    write_points_ply(out / "da3_points_flying_red.ply", pts[::2], colors[::2])
    far = in_far_fp[valid].numpy()
    write_points_ply(out / "da3_points_in_far_false_alarm_cells.ply", pts[far],
                     np.where(fly[far, None], [[1., 0, 0]], [[.2, .4, 1.]]))
    json.dump(dict(scene=win["sid"], profile_view=int(win["ctx"][v]), profile_row=int(r), layer_index=int(layer),
                   colours=dict(flying="red", other_da3="grey", laser="green",
                                far_fp_points="red if flying, blue otherwise")),
              (out / "demo.json").open("w"), indent=2)


if __name__ == "__main__":
    main()
