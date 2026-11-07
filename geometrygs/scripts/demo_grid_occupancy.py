"""Demo of the 10 cm grid-occupancy probe on one validation scene: which cells are right / wrong in 3D.

Colours: green = surface cell found (TP), purple = empty cell predicted occupied (FP), red = surface cell missed (FN).
Reads the trained probes and the cached cell features; the laser is used only to draw labels and the reference surface.
"""

import argparse
import json
from pathlib import Path

import dataclasses
import numpy as np
import torch
from safetensors.torch import load_file

import geometrygs  # noqa: F401
from geometrygs.grid_probe import GridClassifier
from geometrygs.plyio import write_cubes_ply, write_octahedra_ply, write_points_ply

GEO_DIM = 28
COLORS = dict(tp=(.15, .72, .3), fp=(.62, .35, .85), fn=(.92, .12, .12))


def load_probe(state, name):
    s = {k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + ".")}
    probe = GridClassifier(s["mean"], s["std"], hidden=(256, 128))
    probe.load_state_dict(s, strict=True)
    return probe.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, default=Path("EXP/grid_occupancy_10cm"))
    ap.add_argument("--workspace", type=Path,
                    default=Path("/mnt/disk/xveyan/workspace_scannetpp/geometrygs_grid_occupancy_10cm"))
    ap.add_argument("--scene", default="7b6477cb95")
    ap.add_argument("--probability", type=float, default=.5)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--mask-file", type=Path, help="npz with a precomputed occupancy mask (instead of the probe)")
    ap.add_argument("--mask-key", default="camera_scale__all_pixels")
    ap.add_argument("--demo-extra", type=Path, help="npz with laser_points and optional da3_points/da3_colors")
    ap.add_argument("--title", default="frozen TokenGS feature probe")
    args = ap.parse_args()
    out = args.output or Path(f"EXP/demo/scene_{args.scene}/{args.experiment.name}")
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = torch.load(args.workspace / f"{args.scene}.pt", weights_only=False)
    dims, lo, cell = [int(n) for n in rec["dims"]], rec["lo"].numpy(), float(rec["cell"])
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in dims], indexing="ij"), -1).reshape(-1, 3)
    centers = lo + (ijk + .5) * cell
    y = rec["y"]
    preds, stats = {}, {}
    if args.mask_file:
        masks = np.load(args.mask_file)
        if not (np.array_equal(masks["labels"], y) and np.array_equal(masks["dims"], dims)):
            raise RuntimeError("mask file was computed on a different label grid")
        preds["prediction"] = masks[args.mask_key].astype(bool)
    else:
        state = load_file(str(args.experiment / "model.safetensors"))
        logit = float(np.log(args.probability / (1 - args.probability)))
        x = torch.as_tensor(rec["x"]).float()
        for name, d in (("camera_geometry", GEO_DIM), ("projected_image", x.shape[1])):
            with torch.no_grad():
                preds[name] = torch.cat([load_probe(state, name)(c[:, :d]) for c in x.split(65536)]).numpy() >= logit
    for name, p in preds.items():
        c = dict(tp=int((p & y).sum()), fp=int((p & ~y).sum()), fn=int((~p & y).sum()), tn=int((~p & ~y).sum()))
        c.update(recall=c["tp"] / max(c["tp"] + c["fn"], 1), precision=c["tp"] / max(c["tp"] + c["fp"], 1),
                 iou=c["tp"] / max(c["tp"] + c["fp"] + c["fn"], 1), accuracy=(c["tp"] + c["tn"]) / len(y))
        stats[name] = c
    pred = preds["prediction" if args.mask_file else "projected_image"]
    groups = dict(tp=pred & y, fp=pred & ~y, fn=~pred & y)

    # 3D files (normalized coordinates, 0.3 units / metre).
    half = .45 * cell
    write_cubes_ply(out / "cells_label.ply", centers[y], np.tile([1., .72, .12], (int(y.sum()), 1)), half)
    write_cubes_ply(out / "cells_predicted.ply", centers[pred], np.tile([.25, .55, .95], (int(pred.sum()), 1)), half)
    for k, m in groups.items():
        write_cubes_ply(out / f"cells_{k}.ply", centers[m], np.tile(COLORS[k], (int(m.sum()), 1)), half)
    allm = groups["tp"] | groups["fp"] | groups["fn"]
    col = np.zeros((len(y), 3))
    for k, m in groups.items():
        col[m] = COLORS[k]
    write_cubes_ply(out / "cells_confusion.ply", centers[allm], col[allm], half)
    laser = rec["laser_points"].numpy() if "laser_points" in rec else None
    if args.demo_extra:
        extra = np.load(args.demo_extra)
        laser = extra["laser_points"]
        if "da3_points" in extra:
            write_points_ply(out / "da3_points.ply", extra["da3_points"], extra["da3_colors"])
    if laser is not None:
        write_points_ply(out / "laser_input_surface.ply", laser, np.tile([.55, .55, .55], (len(laser), 1)))

    # Input views and cameras for context.
    from tokengs.data.provider import Provider
    from tokengs.options import config_defaults
    opt = dataclasses.replace(config_defaults["eval_scannetpp_latent_6view_ssim"], num_workers=0, load_lidar_pc=False,
                              val_depth_metric=False, dataset_kwargs={"split": "nvs_sem_val.txt", "window_size": 18})
    provider = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    index = [Path(s).name for s in provider.dataset.sample_list].index(args.scene)
    raw = provider[index]
    assert raw["frame_ids"][:6].tolist() == rec["input_frames"], "input frames differ from the cached grid"
    cams = raw["cam_to_world_input"][:, :3, 3].numpy()
    write_octahedra_ply(out / "input_cameras.ply", cams, np.tile([0., 0., 0.], (len(cams), 1)), .02)

    # Original laser scan (ScanNet++ mesh_aligned_0.05 vertices, RGB) in the same normalized frame as the grid.
    import open3d as o3d
    from scannetpp_evaluation_protocol import normalize_world_transform
    mesh = o3d.t.io.read_triangle_mesh(str(provider.dataset.root_path / "data" / args.scene
                                           / "scans/mesh_aligned_0.05.ply"))
    ctx_raw = torch.from_numpy(provider.dataset._scene_meta_cache[args.scene]["c2ws"][rec["input_frames"]].copy())
    to_norm = normalize_world_transform(ctx_raw).numpy()
    scan = (mesh.vertex.positions.numpy() @ to_norm[:3, :3].T + to_norm[:3, 3]) * .3
    scan_rgb = mesh.vertex.colors.numpy()
    scan_rgb = scan_rgb / 255. if np.issubdtype(scan_rgb.dtype, np.integer) else scan_rgb
    scan_ijk = np.floor((scan - lo) / cell).astype(int)
    in_grid = ((scan_ijk >= 0) & (scan_ijk < np.array(dims))).all(1)
    write_points_ply(out / "laser_scan_points.ply", scan, scan_rgb)
    write_points_ply(out / "laser_scan_points_in_grid.ply", scan[in_grid], scan_rgb[in_grid])
    if laser is not None:  # the input-visible laser surface must lie on the transformed scan
        probe_pts = laser[:: max(1, len(laser) // 2000)]
        tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan.astype(np.float64))))
        gap = np.median([np.sqrt(tree.search_knn_vector_3d(p.astype(np.float64), 1)[2][0]) for p in probe_pts]) / .3
        if gap > .05:
            raise RuntimeError(f"laser scan misaligned with the grid frame (median gap {gap * 100:.1f} cm)")
    images = raw["images_input"].permute(0, 2, 3, 1).numpy()
    fig, axs = plt.subplots(1, 6, figsize=(18, 2.3))
    for ax, im, f in zip(axs, images, rec["input_frames"]):
        ax.imshow(im.clip(0, 1))
        ax.set_title(f"input frame {f}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "input_views.png", dpi=150)
    plt.close(fig)

    # Slices along the camera-down axis (y): label | prediction | confusion.
    cube = lambda m: m.reshape(dims)
    rows = np.nonzero(cube(y).sum((0, 2)))[0]
    picks = rows[np.linspace(0, len(rows) - 1, 5).round().astype(int)]
    conf = np.zeros(dims + [3])
    conf[...] = .08
    for k, m in groups.items():
        conf[cube(m)] = COLORS[k]
    fig, axs = plt.subplots(4, len(picks), figsize=(3.4 * len(picks), 12.2))
    for j, iy in enumerate(picks):
        h = (lo[1] + (iy + .5) * cell) / .3
        slab = in_grid & (scan_ijk[:, 1] == iy)
        sx, sz = (scan[slab, 0] - lo[0]) / cell - .5, (scan[slab, 2] - lo[2]) / cell - .5
        ax = axs[0, j]
        ax.imshow(np.zeros((dims[2], dims[0])), origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.scatter(sx, sz, c=scan_rgb[slab], s=.4, linewidths=0)
        ax.set_title(f"y = {h:+.2f} m: original laser scan (RGB)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for i, (img, title) in enumerate(((cube(y)[:, iy, :].T, "label"), (cube(pred)[:, iy, :].T, "prediction"),
                                          (conf[:, iy, :].transpose(1, 0, 2), "TP / FP / FN + laser (cyan)")), 1):
            ax = axs[i, j]
            ax.imshow(img if img.ndim == 3 else img.astype(float), origin="lower", interpolation="nearest",
                      cmap=None if img.ndim == 3 else "magma", vmin=0, vmax=1)
            if i == 3:
                ax.scatter(sx, sz, c="cyan", s=.15, linewidths=0, alpha=.7)
            ax.set_title(f"y = {h:+.2f} m: {title}", fontsize=8)
            ax.set_xlim(-.5, dims[0] - .5); ax.set_ylim(-.5, dims[2] - .5)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{args.scene} [{args.title}], horizontal 10 cm slices (x right, z forward). "
                 "green found surface, purple false alarm, red missed surface; laser scan = all scanned "
                 "surfaces, including those no input view sees", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "confusion_slices.png", dpi=160)
    plt.close(fig)

    # Top-down: number of occupied cells per vertical column.
    top = lambda m: cube(m).sum(1).T
    panels = [(top(y), "label: surface cells per column"), (top(pred), "prediction"),
              (top(groups["fp"]), "false alarms"), (top(groups["fn"]), "missed surface")]
    vmax = max(top(y).max(), top(pred).max())
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (img, title) in zip(axs, panels):
        im = ax.imshow(img, origin="lower", interpolation="nearest", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x (10 cm cells)"); ax.set_ylabel("z (10 cm cells, forward)")
    cam_ij = (cams - lo) / cell
    for ax in axs:
        ax.scatter(cam_ij[:, 0] - .5, cam_ij[:, 2] - .5, c="white", marker="^", s=25, edgecolors="black")
    fig.colorbar(im, ax=axs, shrink=.8)
    fig.suptitle(f"{args.scene} [{args.title}]: top-down column counts (triangles = input cameras)")
    fig.savefig(out / "confusion_topdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Distance (Chebyshev, in cells) of every false alarm to the nearest true surface cell.
    from geometrygs.grid_probe import dilate
    fp_ring = np.full(len(y), 4)
    for k in (3, 2, 1):
        fp_ring[dilate(torch.as_tensor(y), torch.tensor(dims), k).numpy()] = k
    fp_ring[~groups["fp"]] = 0
    ring_colors = {1: (.85, .75, .98), 2: (.68, .5, .92), 3: (.5, .28, .8), 4: (.28, .08, .45)}
    shells = {f"fp_within_{k}_cells": float(((fp_ring > 0) & (fp_ring <= k)).sum() / max(groups["fp"].sum(), 1))
              for k in (1, 2, 3)}
    fp_col = np.array([ring_colors[r] for r in fp_ring[groups["fp"]]])
    write_cubes_ply(out / "cells_fp_by_distance.ply", centers[groups["fp"]], fp_col, half)

    # 3D views: top = true surface cells (found / missed), bottom = false alarms by distance to the surface.
    fig = plt.figure(figsize=(16, 12.5))
    views = ((25, -60), (60, -90), (10, 0))
    found = np.where(groups["tp"][y], 0, 1)
    shown = np.nonzero(in_grid)[0]
    shown = np.random.default_rng(0).choice(shown, min(len(shown), 80000), replace=False)
    sets = (("original laser scan inside the grid (RGB, 80k points)", scan[shown], scan_rgb[shown], .8),
            ("true surface cells: green found, red missed", centers[y],
             np.array([COLORS["tp"], COLORS["fn"]])[found], 2.5),
            ("false alarms: light = 1 cell from surface ... dark = >3 cells", centers[groups["fp"]], fp_col, 2.5))
    for r, (title, pts, colors, size) in enumerate(sets):
        for c, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(3, 3, 3 * r + c + 1, projection="3d")
            ax.scatter(pts[:, 0] / .3, pts[:, 2] / .3, -pts[:, 1] / .3, c=colors, s=size, marker="s",
                       depthshade=False, linewidths=0)
            ax.set_xlim(lo[0] / .3, (lo[0] + dims[0] * cell) / .3)
            ax.set_ylim(lo[2] / .3, (lo[2] + dims[2] * cell) / .3)
            ax.set_zlim(-(lo[1] + dims[1] * cell) / .3, -lo[1] / .3)
            ax.scatter(cams[:, 0] / .3, cams[:, 2] / .3, -cams[:, 1] / .3, c="k", marker="^", s=30)
            ax.view_init(elev, azim)
            ax.set_title(f"{title}\n(elev {elev}, azim {azim})", fontsize=8)
            ax.set_xlabel("x m"); ax.set_ylabel("z m"); ax.set_zlabel("up m")
            ax.set_box_aspect(dims[0:1] + dims[2:3] + dims[1:2])
    fig.suptitle(f"{args.scene} [{args.title}]: 10 cm cells" + ("" if args.mask_file else f", probability threshold {args.probability}"))
    fig.tight_layout()
    fig.savefig(out / "confusion_3d.png", dpi=150)
    plt.close(fig)

    info = dict(scene=args.scene, prediction=args.title, mask=str(args.mask_file or ""),
                mask_key=args.mask_key if args.mask_file else None,
                probability_threshold=None if args.mask_file else args.probability, grid_dims=dims, cell_m=cell / .3,
                cells=len(y), input_frames=rec["input_frames"], counts=stats, fp_shells=shells,
                colours=dict(tp="green", fp="purple", fn="red", label="gold", prediction="blue"),
                coordinates="PLY in normalized mean-camera frame (0.3 units per metre); PNG axes in metres")
    (out / "confusion_demo.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
