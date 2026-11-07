"""No-training check of the input-view SfM on the 50 ScanNet++ validation windows (6 inputs, window 18).

Per scene: number of triangulated points and of selected anchors, their distance to the laser mesh (metres), and how
much of the laser surface seen by the input views lies near an anchor. Writes scene_metrics.csv, summary.json, run
metadata to --output.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import geometrygs  # noqa: F401
from geometrygs.data import SfmProvider, split_data_with_sfm
from geometrygs.model import anchor_quality_kwargs
from geometrygs.options import config_defaults
from geometrygs.sfm import KnownPoseSfM, SfmConfig, select_anchors
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals


def surface_points(rc_scene, cam_view, intr, img_size, c2w, stride=4):
    depth, _, ok = raycast_gt_depth_and_normals(rc_scene, cam_view, intr, img_size)
    H, W = img_size
    u, v = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy")
    pts = []
    for i in range(depth.shape[0]):
        d = depth[i][::stride, ::stride]
        m = ok[i][::stride, ::stride]
        fx, fy, cx, cy = intr[i].tolist()
        cam = torch.stack([(u[::stride, ::stride] - cx) / fx * d, (v[::stride, ::stride] - cy) / fy * d, d], -1)[m]
        pts.append(cam @ c2w[i, :3, :3].T + c2w[i, :3, 3])
    return torch.cat(pts)


def mesh_distance(rc_scene, pts):
    import open3d as o3d

    if pts.shape[0] == 0:
        return np.zeros(0)
    return rc_scene.compute_distance(o3d.core.Tensor(pts.cpu().numpy().astype(np.float32))).numpy()


def q(x, p):
    return float(np.quantile(x, p)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-scenes", type=int, default=50)
    ap.add_argument("--num-anchors", type=int, default=1024)
    ap.add_argument("--sfm-image-size", type=int, nargs=2)
    ap.add_argument("--num-features", type=int)
    ap.add_argument("--scene-stride", type=int, default=1)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    opt = config_defaults["eval_sfm_anchor"].evolve(num_workers=0, dataset_kwargs={"window_size": 18})
    if args.sfm_image_size:
        opt = opt.evolve(sfm_image_size=tuple(args.sfm_image_size))
    if args.num_features:
        opt = opt.evolve(sfm_num_features=args.num_features)
    provider = SfmProvider("scannetpp_val_scaled_0.3", opt, training=False)
    scale = float(provider.scene_scale)
    dev = torch.device("cuda:0")
    sfm = KnownPoseSfM(SfmConfig(num_features=opt.sfm_num_features, match_ratio=opt.sfm_match_ratio,
                                 max_epipolar_px=opt.sfm_max_epipolar_px, max_reproj_px=opt.sfm_max_reproj_px,
                                 min_tri_angle_deg=opt.sfm_min_tri_angle_deg, min_track=opt.sfm_min_track,
                                 z_near=opt.znear, z_far=opt.zfar))
    rows = []
    for idx in list(range(0, len(provider), args.scene_stride))[: args.num_scenes]:
        raw = provider[idx]
        batch = {k: v[None] if torch.is_tensor(v) else v for k, v in raw.items()}
        mi, _ = split_data_with_sfm(batch, opt)
        sid = raw["scene_id"]
        ctx = raw["frame_ids"][: opt.num_input_views].tolist()
        raw_c2w = torch.from_numpy(provider.dataset._scene_meta_cache[sid]["c2ws"][ctx].copy())
        rc = build_raycasting_scene(provider.dataset.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, scale)
        c2w = mi.encoder.cam_to_world_input[0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rec = sfm.reconstruct(mi.sfm["images"][0].to(dev), c2w.to(dev), mi.sfm["intrinsics"][0].to(dev))
        sel = select_anchors(rec, args.num_anchors, **anchor_quality_kwargs(opt))
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        pts = rec["points"].cpu()
        anchors = pts[sel.cpu()]
        d_pts = mesh_distance(rc, pts) / scale
        d_anc = mesh_distance(rc, anchors) / scale
        surf = surface_points(rc, raw["cam_view_input"], raw["intrinsics_input"], opt.img_size, raw["cam_to_world_input"])
        if len(anchors):
            nn = torch.cdist(surf, anchors).min(1).values.numpy() / scale
        else:
            nn = np.full(len(surf), np.inf)
        row = {
            "scene_id": sid, "num_points": len(pts), "num_anchors": len(anchors), "num_matches": int(rec["num_matches"]),
            "track_len_mean": float(rec["track_len"].float().mean()) if len(pts) else 0.0,
            "points_dist_med_cm": 100 * q(d_pts, 0.5), "points_dist_p90_cm": 100 * q(d_pts, 0.9),
            "points_within_2cm": float((d_pts < 0.02).mean()) if len(pts) else 0.0,
            "points_within_5cm": float((d_pts < 0.05).mean()) if len(pts) else 0.0,
            "anchors_dist_med_cm": 100 * q(d_anc, 0.5), "anchors_dist_p90_cm": 100 * q(d_anc, 0.9),
            "surface_to_anchor_med_cm": 100 * q(nn, 0.5), "surface_to_anchor_p90_cm": 100 * q(nn, 0.9),
            "surface_within_10cm_of_anchor": float((nn < 0.10).mean()),
            "surface_within_20cm_of_anchor": float((nn < 0.20).mean()),
            "sfm_seconds": secs,
        }
        rows.append(row)
        print(f"[{idx + 1:02d}] {sid}: pts={row['num_points']} anchors={row['num_anchors']} "
              f"dist med/p90={row['points_dist_med_cm']:.1f}/{row['points_dist_p90_cm']:.1f} cm "
              f"surf->anchor med={row['surface_to_anchor_med_cm']:.1f} cm cover10={row['surface_within_10cm_of_anchor']:.2f} "
              f"{secs:.2f}s", flush=True)
    with (args.output / "scene_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = [k for k in rows[0] if k != "scene_id"]
    summary = {
        "num_scenes": len(rows), "num_input_views": opt.num_input_views, "window_size": 18,
        "sfm_image_size": list(opt.sfm_image_size), "num_anchors_max": args.num_anchors,
        "config": {k: getattr(opt, k) for k in ("sfm_num_features", "sfm_match_ratio", "sfm_max_epipolar_px",
                                                 "sfm_max_reproj_px", "sfm_min_tri_angle_deg", "sfm_min_track")},
        "scene_mean": {k: statistics.mean(r[k] for r in rows if np.isfinite(r[k])) for k in keys},
        "scene_median": {k: statistics.median(r[k] for r in rows if np.isfinite(r[k])) for k in keys},
        "scenes_with_fewer_points_than_anchors": sum(r["num_points"] < args.num_anchors for r in rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
