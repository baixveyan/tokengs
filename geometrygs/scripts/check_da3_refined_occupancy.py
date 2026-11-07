"""DA3 depth -> 10 cm cells with three deployable refinements (no laser), compared cell by cell with the laser labels.

Same windows, DA3 setting and labels as EXP/da3_occupancy_10cm. Refinements, all from input RGB + cameras only:
  edge   drop pixels whose own DA3 5x5 depth range exceeds 10 % of the depth (flying pixels, EXP/da3_flying_pixels);
  mv     multi-view consistency (as in AnchorSplat): project each point into the other five input views and compare with
         their DA3 depth; |dz| < 3 cm + 2 % z is a match, a point more than that in front of the other view's surface
         is a violation (that view should have seen it), behind it is occlusion and ignored. Keep a pixel with
         >= M matches and <= 1 violation (mv1: M = 1, mv2: M = 2), or with <= 1 violation and no match
         requirement (mvv: keeps regions only one input view sees);
  sfm    rescale depth by the median ratio of SfM depth to DA3 depth at the observations of reliable input-view SfM
         points (known poses, SIFT 8192 at 1024x1792, anchor quality filter); global per scene or per view.
The per-scene laser/DA3 median ratio ("oracle" scale) is a diagnostic only. A cell is occupied if >= 1 kept point.
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
from geometrygs.grid_probe import backproject, dilate, grid_index, project
from geometrygs.model import anchor_quality_kwargs
from geometrygs.options import config_defaults as sfm_defaults
from geometrygs.plyio import write_points_ply
from geometrygs.sfm import KnownPoseSfM, SfmConfig, anchor_quality_mask

CM = .003  # normalized units per centimetre
EDGE_TAU = .1
MV_ABS, MV_REL = 3 * CM, .02
MIN_SFM_OBS = 50


def neighbourhood_range(d, valid, k=5):
    big = torch.finfo(d.dtype).max
    lo = -F.max_pool2d(torch.where(valid, -d, -big)[:, None], k, 1, k // 2)[:, 0]
    hi = F.max_pool2d(torch.where(valid, d, -big)[:, None], k, 1, k // 2)[:, 0]
    return (hi - lo) / d.clamp_min(1e-6)


def sample_nearest(maps, grid):
    """maps [V, h, w]; grid [V, N, 2] in [-1, 1] (align_corners=False); nearest lookup, out of frame -> 0."""
    return F.grid_sample(maps[:, None], grid[:, None], mode="nearest", align_corners=False)[:, 0, 0]


def multiview_counts(d, valid, intr, c2w, size, znear):
    """Per pixel: number of other views that see its point, that match it, and that it violates."""
    v_count = d.shape[0]
    pts = backproject(d, intr, c2w)
    match = torch.zeros_like(d, dtype=torch.int16)
    violate = torch.zeros_like(d, dtype=torch.int16)
    seen_by = torch.zeros_like(d, dtype=torch.int16)
    for v in range(v_count):
        others = [u for u in range(v_count) if u != v]
        grid, z, inside = project(pts[v].reshape(-1, 3), c2w[others], intr[others], size)
        ref = sample_nearest(d[others], grid)
        ref_ok = sample_nearest(valid[others].float(), grid) > .5
        seen = inside & (z > znear) & ref_ok
        tol = MV_ABS + MV_REL * z
        match[v] = ((seen & ((z - ref).abs() < tol)).sum(0)).reshape(d.shape[1:]).to(torch.int16)
        violate[v] = ((seen & (z < ref - tol)).sum(0)).reshape(d.shape[1:]).to(torch.int16)
        seen_by[v] = seen.sum(0).reshape(d.shape[1:]).to(torch.int16)
    return match, violate, seen_by


def sfm_scales(rec, d, size_sfm, c2w):
    """Median SfM/DA3 depth ratio over observations of reliable points: (global, per view [V], #observations)."""
    pts, obs_uv, obs = rec["points"], rec["obs_uv"], rec["obs_mask"]
    if pts.shape[0]:
        reliable = anchor_quality_mask(rec, **rec["quality"])
        pts, obs_uv, obs = pts[reliable], obs_uv[reliable], obs[reliable]
    v_count = d.shape[0]
    w2c = torch.linalg.inv(c2w)
    ratios = [[] for _ in range(v_count)]
    h, w = size_sfm
    for v in range(v_count):
        m = obs[:, v]
        if not m.any():
            continue
        z = (pts[m] @ w2c[v, :3, :3].T + w2c[v, :3, 3])[:, 2]
        grid = obs_uv[m, v] / obs_uv.new_tensor([w, h]) * 2 - 1  # same field of view at both resolutions
        dd = sample_nearest(d[v:v + 1], grid[None])[0]
        ok = (dd > 0) & (z > 0)
        ratios[v] = (z[ok] / dd[ok]).tolist()
    allr = [r for rs in ratios for r in rs]
    if len(allr) < MIN_SFM_OBS:
        return None, None, len(allr)
    g = float(np.median(allr))
    per_view = torch.tensor([float(np.median(rs)) if len(rs) >= 20 else g for rs in ratios])
    return g, per_view, len(allr)


def depth_errors(d, laser, ok):
    e = (d[ok] - laser[ok]).abs() / .3
    return dict(abs_median_cm=float(e.median() * 100), within_5cm=float((e < .05).float().mean()),
                within_10cm=float((e < .10).float().mean()))


def cells(points, keep, y, lo, cell, dims, far_empty):
    ids, inside = grid_index(points[keep], lo, cell, dims)
    occ = torch.bincount(ids[inside], minlength=len(y)) > 0
    yt = torch.as_tensor(y)
    c = dict(tp=int((occ & yt).sum()), fp=int((occ & ~yt).sum()), fn=int((~occ & yt).sum()),
             tn=int((~occ & ~yt).sum()), far_fp=int((occ & far_empty).sum()))
    for k in (1, 2):
        c[f"fp_within_{k}"] = int((occ & ~yt & dilate(yt, dims, k)).sum())
    return c, occ


def rates(c):
    return dict(recall=c["tp"] / max(c["tp"] + c["fn"], 1), precision=c["tp"] / max(c["tp"] + c["fp"], 1),
                iou=c["tp"] / max(c["tp"] + c["fp"] + c["fn"], 1),
                accuracy=(c["tp"] + c["tn"]) / max(c["tp"] + c["fp"] + c["fn"] + c["tn"], 1),
                fp_within_1_fraction=c["fp_within_1"] / max(c["fp"], 1),
                fp_within_2_fraction=c["fp_within_2"] / max(c["fp"], 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("EXP/da3_refined_occupancy_10cm"))
    ap.add_argument("--labels", type=Path,
                    default=Path("/mnt/disk/xveyan/workspace_scannetpp/geometrygs_grid_occupancy_10cm"))
    ap.add_argument("--workspace", type=Path,
                    default=Path("/mnt/disk/xveyan/workspace_scannetpp/geometrygs_da3_refined_occupancy_10cm"))
    ap.add_argument("--da3-root", type=Path, default=geometrygs.GEOMETRYGS_ROOT.parent / "Depth-Anything-3")
    ap.add_argument("--checkpoint", type=Path, default=Path("/mnt/disk/xveyan/checkpoints/DA3-GIANT-1.1"))
    ap.add_argument("--image-size", type=int, nargs=2, default=(512, 896))
    ap.add_argument("--sfm-image-size", type=int, nargs=2, default=(1024, 1792))
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
    dev = torch.device("cuda")
    start = time.perf_counter()
    model, da3_file = load_da3(args.da3_root, args.checkpoint, dev)
    base = sfm_defaults["eval_sfm_anchor"].evolve(num_workers=0, load_lidar_pc=False, val_depth_metric=False,
                                                  dataset_kwargs={"split": "nvs_sem_val.txt", "window_size": 18})
    provider = SfmProvider("scannetpp_val_scaled_0.3", base.evolve(sfm_image_size=tuple(args.image_size)), training=False)
    sfm_opt = base.evolve(sfm_image_size=tuple(args.sfm_image_size))
    sfm_provider = SfmProvider("scannetpp_val_scaled_0.3", sfm_opt, training=False)
    sfm = KnownPoseSfM(SfmConfig(num_features=sfm_opt.sfm_num_features, match_ratio=sfm_opt.sfm_match_ratio,
                                 max_epipolar_px=sfm_opt.sfm_max_epipolar_px, max_reproj_px=sfm_opt.sfm_max_reproj_px,
                                 min_tri_angle_deg=sfm_opt.sfm_min_tri_angle_deg, min_track=sfm_opt.sfm_min_track,
                                 z_near=sfm_opt.znear, z_far=sfm_opt.zfar))
    rows, depth_rows, pooled = [], [], {}
    n = min(args.max_scenes, len(provider))
    for i in range(n):
        win = da3_window(model, provider, i, args.labels, args.process_res)
        sid, rec, d, laser, ok, intr = (win[k] for k in ("sid", "rec", "depth", "laser", "ok", "intr"))
        group = "da3_held_out" if sid in DA3_HELD_OUT else "da3_maybe_seen"
        c2w = win["c2w"].float()
        size = (win["h"], win["w"])
        y, lo, cell, dims = rec["y"], rec["lo"], rec["cell"], rec["dims"]
        valid = d > 0
        edge = valid & (neighbourhood_range(d, valid) <= EDGE_TAU)
        t0 = time.perf_counter()
        match, violate, seen_by = (t.cpu() for t in multiview_counts(d.to(dev), valid.to(dev), intr.to(dev),
                                                                      c2w.to(dev), size, base.znear))
        mv_secs = time.perf_counter() - t0
        mv1 = edge & (match >= 1) & (violate <= 1)
        mv2 = edge & (match >= 2) & (violate <= 1)
        mvv = edge & (violate <= 1)  # no match required: keeps regions that only one input view sees
        # Input-view SfM on the same six views at 1024x1792 (no laser).
        raw_s = sfm_provider[i]
        if raw_s["scene_id"] != sid or raw_s["frame_ids"][:6].tolist() != win["ctx"]:
            raise RuntimeError(f"{sid}: SfM window differs from the DA3 window")
        t0 = time.perf_counter()
        srec = sfm.reconstruct(raw_s["images_sfm"].to(dev), raw_s["cam_to_world_input"].to(dev),
                               raw_s["intrinsics_sfm"].to(dev))
        srec = {k: v.cpu() if torch.is_tensor(v) else v for k, v in srec.items()}
        srec["quality"] = anchor_quality_kwargs(sfm_opt)
        sfm_secs = time.perf_counter() - t0
        g, per_view, n_obs = sfm_scales(srec, d, tuple(args.sfm_image_size), c2w)
        oracle = float((laser[ok & valid] / d[ok & valid]).median())
        scales = dict(camera=torch.ones(6), oracle=torch.full((6,), oracle))
        if g is not None:
            scales["sfm_global"] = torch.full((6,), g)
            scales["sfm_view"] = per_view
        depth_rows.append(dict(scene_id=sid, group=group, sfm_points=int(srec["points"].shape[0]),
                               sfm_observations_used=n_obs, sfm_global_scale=g, oracle_scale=oracle,
                               sfm_over_oracle=(g / oracle) if g is not None else None,
                               mv_seconds=mv_secs, sfm_seconds=sfm_secs,
                               pixels_seen_by_no_other_view=float((seen_by[valid] == 0).float().mean()),
                               pixels_with_violation_ge2=float((violate[valid] >= 2).float().mean()),
                               edge_pixels_failing_mv1=float((~mv1[edge]).float().mean()),
                               **{f"{s}_{k}": v for s, sc in scales.items()
                                  for k, v in depth_errors(d * sc[:, None, None], laser, ok & valid).items()}))
        far_empty = ~dilate(torch.as_tensor(y), dims, 1)
        variants = [("camera_all", "camera", valid), ("camera_edge", "camera", edge), ("camera_edge_mv1", "camera", mv1),
                    ("camera_edge_mv2", "camera", mv2), ("camera_edge_mvv", "camera", mvv)]
        sfm_scale = "sfm_global" if g is not None else "camera"  # too few SfM observations: camera scale
        variants += [("sfm_global_edge", sfm_scale, edge), ("sfm_view_edge", "sfm_view" if g is not None else "camera", edge),
                     ("sfm_global_edge_mv1", sfm_scale, mv1), ("sfm_global_edge_mv2", sfm_scale, mv2),
                     ("sfm_global_edge_mvv", sfm_scale, mvv)]
        variants += [("oracle_edge", "oracle", edge), ("oracle_edge_mv1", "oracle", mv1)]
        masks, line = {}, []
        for name, scale_name, keep in variants:
            points = backproject(d * scales[scale_name][:, None, None], intr, c2w)
            c, occ = cells(points, keep, y, lo, cell, dims, far_empty)
            masks[name] = occ.numpy()
            rows.append(dict(scene_id=sid, group=group, variant=name, scale=scale_name,
                             pixels_kept=float(keep.float().mean() / valid.float().mean()), **c, **rates(c)))
            acc = pooled.setdefault((group, name), {k: 0 for k in c} | {"kept": 0, "total": 0})
            for k in c:
                acc[k] += c[k]
            acc["kept"] += int(keep.sum())
            acc["total"] += int(valid.sum())
            if name in ("camera_all", "camera_edge", "camera_edge_mvv", "sfm_global_edge", "sfm_global_edge_mvv"):
                line.append(f"{name} {rates(c)['iou']:.3f}")
        np.savez_compressed(args.workspace / f"{sid}.npz", labels=y, dims=dims.numpy(), lo=lo.numpy(), cell=cell,
                            **masks)
        print(f"[{i + 1}/{n}] {sid} ({group}) SfM pts {srec['points'].shape[0]} obs {n_obs} scale sfm/oracle "
              f"{(g / oracle) if g else float('nan'):.3f} (camera/oracle {1 / oracle:.3f}) | kept mv1 "
              f"{float(mv1.sum() / valid.sum()):.3f} mvv {float(mvv.sum() / valid.sum()):.3f} | IoU " + ", ".join(line), flush=True)
        if sid == args.demo_scene:
            final = backproject(d * scales["sfm_global" if g is not None else "camera"][:, None, None], intr, c2w)
            rgb = torch.from_numpy(np.stack([np.asarray(im) for im in win["pred"].processed_images])).float() / 255
            lp = backproject(laser, intr, c2w)[ok]
            np.savez_compressed(args.workspace / f"{sid}_demo.npz", da3_points=final[edge].numpy(),
                                da3_colors=rgb[edge.numpy()].numpy(), laser_points=lp[::3].numpy())
            removed = valid & ~edge  # what the edge filter drops (shown red)
            col = np.tile([1., .1, .1], (int(removed.sum()), 1))
            demo_dir = Path(f"EXP/demo/scene_{sid}/{args.output.name}")
            demo_dir.mkdir(parents=True, exist_ok=True)
            write_points_ply(demo_dir / "da3_points_removed.ply", final[removed].numpy(), col)
            if srec["points"].shape[0]:
                rel = anchor_quality_mask(srec, **srec["quality"])
                write_points_ply(demo_dir / "sfm_points_reliable.ply", srec["points"][rel].numpy(),
                                 np.tile([1., .55, 0.], (int(rel.sum()), 1)))
        del win
    for path, data in (("scene_metrics.csv", rows), ("depth_metrics.csv", depth_rows)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with (args.output / path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    summary = dict(
        protocol=vars(args) | {k: str(getattr(args, k)) for k in ("output", "labels", "workspace", "da3_root",
                                                               "checkpoint")},
        da3_code=str(da3_file), definitions=__doc__,
        constants=dict(edge_tau=EDGE_TAU, mv_tolerance=f"{MV_ABS / CM:.0f} cm + {MV_REL:.0%} z",
                       min_sfm_observations=MIN_SFM_OBS, sfm_view_min_observations=20),
        scenes={g: sum(1 for r in depth_rows if r["group"] == g) for g in ("da3_held_out", "da3_maybe_seen")},
        sfm_fallback_scenes=[r["scene_id"] for r in depth_rows if r["sfm_global_scale"] is None],
        pooled={f"{g}/{v}": dict(c, **rates(c), pixels_kept=c["kept"] / max(c["total"], 1))
                for (g, v), c in pooled.items()},
        depth_scene_mean={g: {k: float(np.mean([r[k] for r in depth_rows if r["group"] == g and r.get(k) is not None]))
                              for k in depth_rows[0] if k not in ("scene_id", "group")
                              and any(r["group"] == g and r.get(k) is not None for r in depth_rows)}
                          for g in ("da3_held_out", "da3_maybe_seen")},
        elapsed_seconds=time.perf_counter() - start)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("pooled", "depth_scene_mean", "sfm_fallback_scenes")}, indent=1))
    print(f"Completed: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
