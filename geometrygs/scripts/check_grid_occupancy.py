"""Can a small probe tell, from input RGB + cameras only, which 10 cm cells should hold Gaussians?

Label (laser, labels only): a cell is 1 iff the first laser-mesh hit of at least one input-view ray falls inside it;
every other cell (free, occluded, outside every input view, inside walls) is 0. No unknown class.
Grid: bounding box of those input-visible hits + one cell margin, fixed 10 cm cells.
Probe inputs: frozen official TokenGS image features (PCA to 128-D, fitted on training scenes) sampled at the cell's
3D points in every input view, plus cell/camera geometry in the mean-camera frame. No laser, SfM, Gaussians, or
bounding-box-relative coordinates.
Run via EXP/run_grid_occupancy.sh.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import geometrygs
from geometrygs.grid_probe import (GridClassifier, backproject, bbox_grid, cell_counts, dilate, grid_index,
                                   occupancy_features, project, ray_crossed_cells)
from geometrygs.plyio import write_points_ply
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data
from tokengs.models.tokengs import TokenGS
from tokengs.options import config_defaults

GEO_DIM = 28
SCOPES = ("all", "in_frustum", "near_surface")
RECALLS = (.95, .99)
DEMO_SCENE = "7b6477cb95"


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recall_threshold(logits, recall):
    positive = np.sort(logits)
    return float(positive[int(np.floor((1 - recall) * len(positive)))])


def scope_metrics(logits, y, scope, threshold, ranking=True):
    from sklearn.metrics import average_precision_score, roc_auc_score
    l, y = logits[scope], y[scope]
    keep = l >= threshold
    pos, neg = y.sum(), (~y).sum()
    out = dict(cells=int(len(y)), positive_cells=int(pos), negative_cells=int(neg),
               surface_recall=float((keep & y).sum() / max(pos, 1)),
               negative_closure=float((~keep & ~y).sum() / max(neg, 1)),
               precision=float((keep & y).sum() / max(keep.sum(), 1)),
               kept_fraction=float(keep.mean()) if len(y) else 0.)
    if ranking:
        both = pos > 0 and neg > 0
        out.update(auc=float(roc_auc_score(y, l)) if both else float("nan"),
                   average_precision=float(average_precision_score(y, l)) if both else float("nan"))
    return out


def scopes_of(rec, exclude_holes=False):
    masks = dict(all=np.ones_like(rec["y"]), in_frustum=rec["in_frustum"], near_surface=rec["near_surface"])
    if exclude_holes:
        clean = ~(rec["hole"] & ~rec["y"])
        masks = {k: v & clean for k, v in masks.items()}
    return masks


def raycast(rc, cam_view, intr, size, factor=1):
    h, w = size
    depth, _, ok = raycast_gt_depth_and_normals(rc, cam_view, intr * factor, (h * factor, w * factor))
    c2w = torch.linalg.inv(cam_view.transpose(-1, -2))
    return backproject(depth, intr * factor, c2w), ok, c2w


def extract(provider, index, role, model, opt, pca, args, device):
    raw = provider[index]
    sid = raw["scene_id"]
    batch = {k: v[None].to(device) for k, v in raw.items() if torch.is_tensor(v)}
    batch["has_mask"] = torch.as_tensor(raw["has_mask"], device=device).reshape(1)
    mi, _ = split_data(batch, opt)
    h, w = opt.img_size
    with torch.no_grad():
        bb = model.enc_dec_backbone
        feats = bb._encode_features(model._embed_encoder_input(mi.encoder))[0]
        maps = ((feats - pca["mean"]) @ pca["components"]).reshape(6, h // 8, w // 8, -1).permute(0, 3, 1, 2)
    del feats, batch, mi
    ctx = raw["frame_ids"][:6].tolist()
    original = torch.from_numpy(provider.dataset._scene_meta_cache[sid]["c2ws"][ctx].copy())
    rc = build_raycasting_scene(provider.dataset.root_path / "data" / sid / "scans/mesh_aligned_0.05.ply", original, .3)
    cell = args.cell_m * .3
    points, ok, c2w = raycast(rc, raw["cam_view_input"], raw["intrinsics_input"], opt.img_size, args.ray_factor)
    hits = points[ok]
    if not len(hits):
        raise RuntimeError(f"{sid}: no laser hit in the input views")
    centers, lo, dims = bbox_grid(hits, cell)
    y = cell_counts(hits, lo, cell, dims) > 0
    # Rays without any laser hit (glass, screens, scan holes): cells they cross are flagged for a sensitivity check.
    _, ok1, _ = raycast(rc, raw["cam_view_input"], raw["intrinsics_input"], opt.img_size)
    unit = backproject(torch.ones(6, h, w), raw["intrinsics_input"], c2w) - c2w[:, None, None, :3, 3]
    origins = c2w[:, None, None, :3, 3].expand_as(unit)
    hole = ray_crossed_cells(origins[~ok1].to(device), unit[~ok1].to(device), lo.to(device), cell,
                             dims.to(device), opt.znear).cpu()
    uv, z, valid = project(centers, c2w, raw["intrinsics_input"], opt.img_size)
    in_frustum = (valid & (z > opt.znear)).any(0)
    near = dilate(y, dims, args.near_radius)
    tpoints, tok, _ = raycast(rc, raw["cam_view"], raw["intrinsics"], opt.img_size)
    tids, tinside = grid_index(tpoints[tok], lo, cell, dims)
    target_counts = torch.bincount(tids[tinside], minlength=len(centers))
    target_total = int(tok.sum())
    x = occupancy_features(maps, centers.to(device), cell, c2w.to(device), raw["intrinsics_input"].to(device),
                           opt.img_size, opt.znear, .3).half().cpu()
    del maps
    n = len(centers)
    nz = target_counts.nonzero()[:, 0]
    row = dict(scene_id=sid, split=role, grid_x=int(dims[0]), grid_y=int(dims[1]), grid_z=int(dims[2]),
               bbox_x_m=float(dims[0] * args.cell_m), bbox_y_m=float(dims[1] * args.cell_m),
               bbox_z_m=float(dims[2] * args.cell_m), cells=n, positive_cells=int(y.sum()),
               positive_fraction=float(y.float().mean()), in_frustum_fraction=float(in_frustum.float().mean()),
               near_surface_cells=int(near.sum()), hole_crossed_negative_fraction=float((hole & ~y).float().mean()),
               input_rays_without_laser=float(1 - ok1.float().mean()), target_laser_hits=target_total,
               target_hits_outside_grid=float(1 - target_counts.sum() / max(target_total, 1)),
               target_hits_not_in_positive_cells=float(1 - target_counts[y].sum() / max(target_total, 1)))
    frames = [dict(scene_id=sid, split=role, frame_id=f, laser_valid_rays=float(ok[v].float().mean()),
                   laser_valid_rays_1x=float(ok1[v].float().mean())) for v, f in enumerate(ctx)]
    rec = dict(scene_id=sid, role=role, x=x, y=y.numpy(), in_frustum=in_frustum.numpy(),
               near_surface=near.numpy(), hole=hole.numpy(), lo=lo, dims=dims, cell=cell,
               target_ids=nz.numpy(), target_counts=target_counts[nz].numpy(), target_total=target_total,
               row=row, frames=frames,
               input_frames=ctx, target_frames=raw["frame_ids"][6:].tolist())
    if sid == DEMO_SCENE:
        rec["laser_points"] = hits[::max(1, len(hits) // 300000)].clone()
    del rc
    return rec


def train_probe(name, dims, shard, args, device, world, rank, local):
    xs = [torch.as_tensor(r["x"][:, :dims]) for r in shard]
    ys = [torch.as_tensor(r["y"]) for r in shard]
    hard = [torch.as_tensor(r["near_surface"] & ~r["y"]) for r in shard]
    x = torch.cat(xs).to(device)
    y = torch.cat(ys).to(device)
    h = torch.cat(hard).to(device)
    stats = torch.zeros(2 * dims + 1, dtype=torch.float64, device=device)
    for chunk in x.split(65536):
        c = chunk.double()
        stats[:dims] += c.sum(0)
        stats[dims:-1] += c.square().sum(0)
    stats[-1] = len(x)
    if world > 1:
        dist.all_reduce(stats)
    mean = stats[:dims] / stats[-1]
    std = (stats[dims:-1] / stats[-1] - mean.square()).clamp_min(0).sqrt()
    torch.manual_seed(args.seed)
    probe = GridClassifier(mean.float(), std.float(), hidden=(256, 128)).to(device)
    ddp = DDP(probe, device_ids=[local]) if world > 1 else probe
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    pos, neg, hard_neg = y.nonzero()[:, 0], (~y).nonzero()[:, 0], h.nonzero()[:, 0]
    if not len(hard_neg):
        hard_neg = neg
    gen = torch.Generator(device=device).manual_seed(args.seed + rank)
    b = args.batch_per_rank
    draw = lambda idx, k: idx[torch.randint(len(idx), (k,), generator=gen, device=device)]
    for step in range(args.steps):
        ii = torch.cat((draw(pos, b // 2), draw(neg, b // 4), draw(hard_neg, b - b // 2 - b // 4)))
        loss = F.binary_cross_entropy_with_logits(ddp(x[ii].float()), y[ii].float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        if rank == 0 and (step + 1) % 250 == 0:
            print(f"[{name}] step {step + 1}/{args.steps} sampled BCE={float(loss):.4f}", flush=True)
    del x, y, h
    torch.cuda.empty_cache()
    return probe.eval()


@torch.no_grad()
def score(probe, rec, dims, device):
    x = torch.as_tensor(rec["x"][:, :dims])
    return torch.cat([probe(c.to(device).float()) for c in x.split(65536)]).cpu().numpy()


def draw_summary(results, records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    width = .2
    for k, (name, res) in enumerate(results.items()):
        for j, recall in enumerate(RECALLS):
            m = res["test"][f"recall_{recall:g}"]
            vals = [m[s]["negative_closure"] for s in SCOPES]
            axs[0].bar(np.arange(3) + (2 * k + j - 1.5) * width, vals, width,
                       label=f"{name}, calib. {int(recall * 100)}% recall")
    axs[0].set(xticks=range(3), xticklabels=["All cells", "In an input frustum", "Within 2 cells of surface"],
               ylim=(0, 1), ylabel="Negative cells closed", title="Validation (thresholds from calibration scenes)")
    axs[0].legend(fontsize=7)
    test = [r for r in records if r["role"] == "test"]
    order = np.argsort([r["row"]["positive_fraction"] for r in test])
    for name, res in results.items():
        vals = [res["test_scenes"][test[i]["scene_id"]]["in_frustum"]["negative_closure"] for i in order]
        axs[1].plot(vals, marker=".", label=name)
    axs[1].set(xlabel="Validation scenes, sorted by positive-cell fraction", ylabel="In-frustum negatives closed",
               title="Per scene at calibrated 99% recall", ylim=(0, 1))
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "grid_occupancy.png", dpi=180)
    plt.close(fig)


def write_demo(rec, logits, threshold, demo, workspace):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    demo.mkdir(parents=True, exist_ok=True)
    saved = torch.load(workspace / f"{rec['scene_id']}.pt", weights_only=False)
    dims, lo, cell = saved["dims"], saved["lo"], saved["cell"]
    axes = [torch.arange(int(n)) for n in dims]
    centers = (lo + (torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) + .5) * cell).numpy()
    y, keep = rec["y"], logits >= threshold
    write_points_ply(demo / "grid_labels.ply", centers[y], np.tile([[1., .7, .1]], (int(y.sum()), 1)))
    shown = y | keep
    colors = np.zeros((len(y), 3))
    colors[y & keep], colors[y & ~keep], colors[~y & keep] = [.1, .7, .3], [1., .1, .1], [.65, .35, .85]
    write_points_ply(demo / "grid_predictions.ply", centers[shown], colors[shown])
    holes = rec["hole"] & ~y
    write_points_ply(demo / "laser_hole_cells.ply", centers[holes], np.tile([[.95, .55, .2]], (int(holes.sum()), 1)))
    laser = saved["laser_points"].numpy()
    write_points_ply(demo / "laser_input_surface.ply", laser, np.tile([[.15, .75, .35]], (len(laser), 1)))
    shape = tuple(int(n) for n in dims)
    cube = y.reshape(shape)
    iy = int(cube.sum((0, 2)).argmax())
    prob = 1 / (1 + np.exp(-logits.reshape(shape)))
    panels = [(cube, "Label: input-visible laser surface"), (prob, "Predicted probability"),
              (keep.reshape(shape) & ~cube, "Kept negatives"), (y.reshape(shape) & ~keep.reshape(shape), "Missed positives"),
              (rec["in_frustum"].reshape(shape), "Inside an input frustum"), (holes.reshape(shape), "Crossed by no-laser rays")]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (data, title) in zip(axs.flat, panels):
        ax.imshow(data[:, iy, :].T.astype(float), origin="lower", interpolation="nearest", vmin=0, vmax=1)
        ax.set(title=title, xlabel="x cell (10 cm)", ylabel="z cell (10 cm, camera forward)")
    fig.suptitle(f"{rec['scene_id']}: slice y={iy} (camera-down axis); threshold = calibrated 99% recall")
    fig.tight_layout()
    fig.savefig(demo / "local_section.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("EXP/grid_occupancy_10cm"))
    ap.add_argument("--demo", type=Path, default=Path(f"EXP/demo/scene_{DEMO_SCENE}/grid_occupancy_10cm"))
    ap.add_argument("--workspace", type=Path,
                    default=Path("/mnt/disk/xveyan/workspace_scannetpp/geometrygs_grid_occupancy_10cm"))
    ap.add_argument("--train-scenes", type=int, default=128)
    ap.add_argument("--calibration-scenes", type=int, default=32)
    ap.add_argument("--test-scenes", type=int, default=50)
    ap.add_argument("--cell-m", type=float, default=.1)
    ap.add_argument("--ray-factor", type=int, default=2)
    ap.add_argument("--near-radius", type=int, default=2)
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--pca-scenes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-per-rank", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    from tokengs.train import verify_idle_gpus
    verify_idle_gpus()
    world, rank, local = (int(os.environ.get(k, d)) for k, d in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if (args.output / "summary.json").exists():
        raise FileExistsError(f"{args.output} is complete; choose a new output instead of overwriting")
    args.output.mkdir(parents=True, exist_ok=True)
    args.workspace.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    opt = dataclasses.replace(config_defaults["eval_scannetpp_latent_6view_ssim"], lambda_lpips=0., num_workers=0,
                              load_lidar_pc=False, val_depth_metric=False, use_render_geo_supervision=False)
    providers, tasks = {}, []
    for split in ("train", "val"):
        providers[split] = Provider("scannetpp_val_scaled_0.3", dataclasses.replace(opt, dataset_kwargs={
            "split": f"nvs_sem_{split}.txt", "window_size": 18}), training=False)
    train_p = providers["train"]
    order = sorted(range(len(train_p)), key=lambda i: hashlib.sha256(
        f"{args.seed}:{Path(train_p.dataset.sample_list[i]).name}".encode()).hexdigest())
    for j, i in enumerate(order[:args.train_scenes + args.calibration_scenes]):
        tasks.append(("train", i, "train" if j < args.train_scenes else "calibration"))
    if len(providers["val"]) < args.test_scenes:
        raise ValueError("not enough validation scenes")
    tasks += [("val", i, "test") for i in range(args.test_scenes)]
    ids = [Path(providers[s].dataset.sample_list[i]).name for s, i, _ in tasks]
    assert len(set(ids)) == len(ids), "scene splits overlap"
    checkpoint = geometrygs.TOKENGS_ROOT / "checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors"
    model = TokenGS(opt).eval().requires_grad_(False)
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model.to(device)

    # PCA of frozen multi-scale patch features, fitted on the first training scenes only (rank 0, then broadcast).
    pca = dict(mean=torch.zeros(4096, device=device), components=torch.zeros(4096, args.pca_dim, device=device),
               explained=torch.zeros(1, device=device, dtype=torch.float64))
    if rank == 0:
        feats = []
        for split, i, role in tasks[:args.pca_scenes]:
            assert role == "train"
            raw = train_p[i]
            batch = {k: v[None].to(device) for k, v in raw.items() if torch.is_tensor(v)}
            batch["has_mask"] = torch.as_tensor(raw["has_mask"], device=device).reshape(1)
            mi, _ = split_data(batch, opt)
            with torch.no_grad():
                bb = model.enc_dec_backbone
                feats.append(bb._encode_features(model._embed_encoder_input(mi.encoder))[0].double())
        feats = torch.cat(feats)
        mean = feats.mean(0)
        cov = (feats - mean).T @ (feats - mean) / len(feats)
        values, vectors = torch.linalg.eigh(cov)
        comps = vectors[:, -args.pca_dim:].flip(-1)
        comps *= torch.sign(comps[comps.abs().argmax(0), torch.arange(args.pca_dim, device=device)])
        pca["mean"], pca["components"] = mean.float().contiguous(), comps.float().contiguous()
        pca["explained"][0] = values[-args.pca_dim:].sum() / values.sum()
        del feats, cov, vectors
    if world > 1:
        for t in pca.values():
            dist.broadcast(t, 0)
    pca_hash = hashlib.sha256(pca["components"].cpu().numpy().tobytes()).hexdigest()[:16]
    meta = dict(version=1, cell_m=args.cell_m, ray_factor=args.ray_factor, near_radius=args.near_radius,
                pca_dim=args.pca_dim, pca_scenes=args.pca_scenes, pca_hash=pca_hash, checkpoint=str(checkpoint),
                checkpoint_bytes=checkpoint.stat().st_size, seed=args.seed)
    records = []
    for job, (split, i, role) in enumerate(tasks):
        if job % world != rank:
            continue
        path = args.workspace / f"{ids[job]}.pt"
        rec = torch.load(path, weights_only=False) if path.exists() else None
        if rec is None or rec.get("meta") != meta or rec["role"] != role:
            rec = extract(providers[split], i, role, model, opt, pca, args, device)
            rec["meta"] = meta
            torch.save(rec, path)
            print(f"[rank {rank}] {role} {rec['scene_id']}: grid {tuple(rec['dims'].tolist())}, "
                  f"positive {rec['row']['positive_cells']}/{rec['row']['cells']}", flush=True)
        rec.pop("laser_points", None)
        records.append(rec)
    del model, providers
    torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
    if rank == 0:
        print(f"Extraction done in {time.perf_counter() - start:.0f}s", flush=True)

    feature_dim = records[0]["x"].shape[1]
    probes = {}
    shard = [r for r in records if r["role"] == "train"]
    for name, dims in (("camera_geometry", GEO_DIM), ("projected_image", feature_dim)):
        probes[name] = (train_probe(name, dims, shard, args, device, world, rank, local), dims)
    del shard
    compact = []
    for r in records:
        if r["role"] == "train":
            continue
        c = {k: r[k] for k in ("scene_id", "role", "y", "in_frustum", "near_surface", "hole", "target_ids",
                               "target_counts", "target_total")}
        c["logits"] = {name: score(p, r, d, device) for name, (p, d) in probes.items()}
        compact.append(c)
    rows = [dict(r["row"], input_frames=" ".join(map(str, r["input_frames"])),
                 target_frames=" ".join(map(str, r["target_frames"]))) for r in records]
    frames = [f for r in records for f in r["frames"]]
    # Exchange through files: NCCL object collectives failed with large score arrays.
    part = args.workspace / f"_scores_rank{rank}.pt"
    torch.save((compact, rows, frames), part)
    if world > 1:
        dist.barrier()
    if rank == 0:
        parts = [args.workspace / f"_scores_rank{r}.pt" for r in range(world)]
        gathered = [torch.load(p, weights_only=False) for p in parts]
        for p in parts:
            p.unlink()
        compact = sorted([c for g in gathered for c in g[0]], key=lambda c: ids.index(c["scene_id"]))
        rows = sorted([r for g in gathered for r in g[1]], key=lambda r: ids.index(r["scene_id"]))
        frames = [f for g in gathered for f in g[2]]
        write_csv(args.output / "occupancy_scene_metrics.csv", rows)
        write_csv(args.output / "frame_metrics.csv", frames)
        json.dump([dict(scene_id=r["scene_id"], split=r["split"], input_frames=r["input_frames"],
                        target_frames=r["target_frames"]) for r in rows],
                  (args.output / "manifest.json").open("w"), indent=2)
        results, scene_rows, oracle = {}, [], {}
        cat = lambda cs, key: np.concatenate([c[key] for c in cs])
        for name in probes:
            cal = [c for c in compact if c["role"] == "calibration"]
            cal_pos = np.concatenate([c["logits"][name][c["y"]] for c in cal])
            thresholds = {f"recall_{r:g}": recall_threshold(cal_pos, r) for r in RECALLS}
            res = dict(thresholds_logit=thresholds, test_scenes={})
            for role in ("calibration", "test"):
                cs = [c for c in compact if c["role"] == role]
                pooled = dict(y=cat(cs, "y"), in_frustum=cat(cs, "in_frustum"), near_surface=cat(cs, "near_surface"),
                              hole=cat(cs, "hole"))
                logits = np.concatenate([c["logits"][name] for c in cs])
                for variant, exclude in (("", False), ("_excluding_laser_hole_cells", True)):
                    if exclude and role != "test":
                        continue
                    masks = scopes_of(pooled, exclude)
                    res[role + variant] = {key: {s: scope_metrics(logits, pooled["y"], m, t)
                                                 for s, m in masks.items()} for key, t in thresholds.items()}
                if role == "test":
                    oracle[name] = {f"recall_{r:g}": scope_metrics(
                        logits, pooled["y"], scopes_of(pooled)["in_frustum"],
                        recall_threshold(logits[pooled["y"]], r), ranking=False) for r in RECALLS}
            for c in [c for c in compact if c["role"] == "test"]:
                for key, t in thresholds.items():
                    l = c["logits"][name]
                    keep = l >= t
                    per = {s: scope_metrics(l, c["y"], m, t, ranking=False) for s, m in scopes_of(c).items()}
                    target = float(c["target_counts"][keep[c["target_ids"]]].sum() / max(c["target_total"], 1))
                    if key == "recall_0.99":
                        res["test_scenes"][c["scene_id"]] = per
                    scene_rows.append(dict(model=name, threshold=key, scene_id=c["scene_id"],
                                           target_surface_hits_in_kept_cells=target,
                                           **{f"{s}_{k}": v for s, m in per.items() for k, v in m.items()}))
            results[name] = res
            print(json.dumps({name: {k: v for k, v in res.items() if k != "test_scenes"}}, indent=1), flush=True)
        write_csv(args.output / "scene_metrics.csv", scene_rows)
        draw_summary(results, [dict(role=c["role"], scene_id=c["scene_id"], row=next(
            r for r in rows if r["scene_id"] == c["scene_id"])) for c in compact], args.output)
        demo = next((c for c in compact if c["scene_id"] == DEMO_SCENE), None)
        if demo is not None:
            write_demo(demo, demo["logits"]["projected_image"],
                       results["projected_image"]["thresholds_logit"]["recall_0.99"], args.demo, args.workspace)
        tensors = {"pca.mean": pca["mean"].cpu(), "pca.components": pca["components"].cpu()}
        for name, (p, _) in probes.items():
            tensors.update({f"{name}.{k}": v.cpu().contiguous() for k, v in p.state_dict().items()})
        save_file(tensors, str(args.output / "model.safetensors"))
        test_rows = [r for r in rows if r["split"] == "test"]
        numeric = [k for k, v in test_rows[0].items() if isinstance(v, (int, float))]
        summary = dict(
            protocol=vars(args) | {k: str(getattr(args, k)) for k in ("output", "demo", "workspace")},
            feature_cache=str(args.workspace), meta=meta, world_size=world,
            pca_explained_variance=float(pca["explained"][0]), feature_dim=feature_dim, geometry_dim=GEO_DIM,
            label="1 iff an input-view first laser hit (2x ray density) lies in the cell; all other cells 0",
            grid="input-visible laser hit bounding box + 1 cell margin; fixed metric cell",
            probe_inputs="frozen TokenGS image features (PCA) at 9 points/cell + camera/cell geometry; "
                         "no laser, SfM, Gaussians or box-relative coordinates",
            scopes=dict(all="all grid cells", in_frustum="cell centre inside >=1 input view",
                        near_surface=f"within {args.near_radius} cells (Chebyshev) of a positive cell"),
            training="1/2 positives, 1/4 uniform negatives, 1/4 near-surface negatives per batch",
            test_label_statistics_mean={k: float(np.mean([r[k] for r in test_rows])) for k in numeric},
            models=results,
            test_label_oracle_in_frustum_diagnostic=dict(
                warning="thresholds chosen with VALIDATION labels; ranking diagnostic only, not deployable", **oracle),
            elapsed_seconds=time.perf_counter() - start)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Completed: {args.output / 'summary.json'}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
