"""Input-camera grid statistics + frozen-feature surface-activation probe.

Run via EXP/run_grid_probe.sh. One fixed window/scene; disjoint train/calibration/test.
All laser access is confined to target construction and diagnostics.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

import geometrygs
from geometrygs.grid_probe import (make_grid, bin_points, grid_labels, sample_grid_features,
                                   GridClassifier, recall_threshold, classification_metrics)
from geometrygs.plyio import write_points_ply
from tokengs.data.provider import Provider
from tokengs.models.tokengs import TokenGS
from tokengs.models.input_types import split_data, EncoderLatent
from tokengs.options import config_defaults
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals, sha256_file


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def matrices(intr):
    k = intr.new_zeros(len(intr), 3, 3)
    k[:, 0, 0], k[:, 1, 1] = intr[:, 0], intr[:, 1]
    k[:, 0, 2], k[:, 1, 2], k[:, 2, 2] = intr[:, 2], intr[:, 3], 1
    return k


def raster(g, raw, opt, device, contribution=False, targets=False):
    from gsplat import rasterization
    intr = raw["intrinsics" if targets else "intrinsics_input"].to(device)
    views = raw["cam_view" if targets else "cam_view_input"].to(device).transpose(-1, -2)
    colors = torch.zeros_like(g[:, 11:14], requires_grad=True) if contribution else g[:, 11:14]
    image, _, _ = rasterization(
        g[:, :3], g[:, 7:11], g[:, 4:7], g[:, 3], colors,
        views, matrices(intr), opt.img_size[1], opt.img_size[0],
        near_plane=opt.znear, far_plane=opt.zfar, render_mode="RGB")
    if contribution:
        # Exact compositing mass in these input renders, including renderer culling/occlusion.
        return torch.autograd.grad(image[..., 0].sum(), colors)[0][:, 0].detach()
    return image.detach().clamp(0, 1).cpu()


def scene_extract(provider, index, role, model, opt, projector, args, device):
    import open3d as o3d
    raw = provider[index]
    sid = raw["scene_id"]
    batch = {k: v[None].to(device) for k, v in raw.items() if torch.is_tensor(v)}
    batch["has_mask"] = torch.as_tensor(raw["has_mask"], device=device).reshape(1)
    mi, _ = split_data(batch, opt)
    centers, lower, step = make_grid(args.resolution, args.extent_m)
    with torch.no_grad():
        bb = model.enc_dec_backbone
        feats = bb._encode_features(model._embed_encoder_input(mi.encoder))
        z = bb._features_to_scene_latents(feats)
        keys, vals = bb._latents_to_kv(z)
        g = model.forward_decoder(EncoderLatent(keys=keys, values=vals), mi.decoder)[0].detach()
        compressed = (feats @ projector).reshape(6, opt.img_size[0] // 8, opt.img_size[1] // 8, -1)
        compressed = compressed.permute(0, 3, 1, 2)
        x, geo_dim = sample_grid_features(
            compressed, centers.to(device), lower.to(device), step,
            raw["cam_to_world_input"].to(device), raw["intrinsics_input"].to(device), opt.img_size)
        x = x.cpu()
    del feats, z, keys, vals, compressed, batch, mi
    mass = raster(g, raw, opt, device, contribution=True).cpu()
    gcpu = g.cpu()
    ctx = raw["frame_ids"][:6].tolist()
    original = torch.from_numpy(provider.dataset._scene_meta_cache[sid]["c2ws"][ctx].copy())
    rc = build_raycasting_scene(provider.dataset.root_path / "data" / sid / "scans/mesh_aligned_0.05.ply",
                               original, .3)
    depth, normal, ok = raycast_gt_depth_and_normals(rc, raw["cam_view_input"], raw["intrinsics_input"], opt.img_size)
    distance = torch.from_numpy(rc.compute_distance(o3d.core.Tensor(centers.numpy())).numpy())
    labels, hits, dispersion, points, inside_hits, hit_ids = grid_labels(
        centers, lower, step, args.resolution, depth, ok, normal,
        raw["intrinsics_input"], raw["cam_to_world_input"], distance)
    # Labels/diagnostics above never enter x or any model input.
    ids, inside = bin_points(gcpu[:, :3], lower, step, args.resolution)
    live = mass > 1e-6  # pixel-equivalents, not an arbitrary opacity threshold
    count = torch.bincount(ids[inside & live], minlength=len(centers))
    cell_mass = torch.bincount(ids[inside], weights=mass[inside], minlength=len(centers))
    gray = raw["images_input"].mean(1)
    dx = F.pad((gray[:, :, 2:] - gray[:, :, :-2]).abs() / 2, (1, 1))
    dy = F.pad((gray[:, 2:, :] - gray[:, :-2, :]).abs() / 2, (0, 0, 1, 1))
    texture = torch.zeros(len(centers))
    texture.index_add_(0, hit_ids[inside_hits], (dx + dy)[inside_hits])
    texture /= hits.clamp_min(1)
    positive, free, unknown = labels == 1, labels == 0, labels < 0
    def mean_count(mask):
        return float(count[mask].float().mean()) if mask.any() else 0.
    row = dict(scene_id=sid, split=role, grid_cells=len(centers),
               surface_cells=int(positive.sum()), free_cells=int(free.sum()), unknown_cells=int(unknown.sum()),
               surface_fraction=float(positive.float().mean()), free_fraction=float(free.float().mean()),
               unknown_fraction=float(unknown.float().mean()),
               input_surface_hit_coverage=float(inside_hits.sum() / ok.sum().clamp_min(1)),
               contributing_gaussians=int(live.sum()), inside_contributing_gaussians=int((inside & live).sum()),
               compositing_mass_inside=float(mass[inside].sum() / mass.sum().clamp_min(1e-8)),
               surface_cell_gs_mean=mean_count(positive),
               surface_cell_gs_median=float(count[positive].float().median()) if positive.any() else 0.,
               surface_cell_gs_p90=float(torch.quantile(count[positive].float(), .9)) if positive.any() else 0.,
               surface_cells_gt64=float((count[positive] > 64).float().mean()) if positive.any() else 0.,
               free_mass_fraction=float(cell_mass[free].sum() / mass.sum().clamp_min(1e-8)))
    if positive.any():
        tex_cut = torch.quantile(texture[positive], .75)
        row.update(high_texture_gs_mean=mean_count(positive & (texture > tex_cut)),
                   lower_texture_gs_mean=mean_count(positive & (texture <= tex_cut)),
                   varying_normal_gs_mean=mean_count(positive & (dispersion > .05)),
                   planar_gs_mean=mean_count(positive & (dispersion <= .05)))
    else:
        row.update(high_texture_gs_mean=0., lower_texture_gs_mean=0., varying_normal_gs_mean=0., planar_gs_mean=0.)
    # Independent novel-view observed surfaces: monitor regions unknown in the input labels.
    from geometrygs.grid_probe import backproject
    td, _, tok = raycast_gt_depth_and_normals(rc, raw["cam_view"], raw["intrinsics"], opt.img_size)
    tp = backproject(td, raw["intrinsics"], torch.linalg.inv(raw["cam_view"].transpose(-1, -2)))
    ti, tin = bin_points(tp, lower, step, args.resolution)
    target_count = torch.bincount(ti[tok & tin], minlength=len(centers))
    frames = []
    for v, fid in enumerate(raw["frame_ids"][:6].tolist()):
        frames.append(dict(scene_id=sid, split=role, frame_id=fid, valid_laser_pixels=int(ok[v].sum()),
                           surface_pixels_inside=int(inside_hits[v].sum())))
    record = dict(scene_id=sid, role=role, x=x, y=labels, target_hits=target_count,
                  row=row, frames=frames, count=count, mass=cell_mass, texture=texture,
                  normal_dispersion=dispersion, hits=hits, geo_dim=geo_dim,
                  input_frames=ctx, target_frames=raw["frame_ids"][6:].tolist())
    if sid == "7b6477cb95":
        demo = args.demo
        demo.mkdir(parents=True, exist_ok=True)
        palette = np.array([[.65, .68, .72], [.2, .55, .9], [1., .65, .1]])
        write_points_ply(demo / "grid_labels.ply", centers.numpy(), palette[labels.numpy() + 1])
        ps = points[ok][::8].numpy()
        write_points_ply(demo / "laser_input_surface.ply", ps, np.tile([[.15, .75, .35]], (len(ps), 1)))
        keep = live.numpy()
        write_points_ply(demo / "contributing_gaussian_centers.ply", gcpu[keep, :3].numpy(), gcpu[keep, 11:14].numpy())
        cell_colors = np.random.default_rng(args.seed).uniform(.15, .95, (len(centers), 3))
        ownership_colors = cell_colors[ids.clamp(0, len(centers)-1).numpy()]
        ownership_colors[~inside.numpy()] = .65
        write_points_ply(demo / "gaussian_centers_by_grid.ply", gcpu[keep, :3].numpy(), ownership_colors[keep])
        with torch.no_grad():
            render = raster(g, raw, opt, device, targets=True)
        from PIL import Image
        truth = raw["images_output"].permute(0, 2, 3, 1)
        strip = torch.cat((torch.cat(list(truth), 1), torch.cat(list(render), 1)), 0)
        Image.fromarray((strip.numpy().clip(0, 1) * 255).astype("uint8")).save(demo / "render_strip.png")
        json_write(demo / "render_strip.json", dict(model="official TokenGS, not a trained grid generator",
                                                    layout="GT above official render; eight novel views",
                                                    input_frames=ctx, target_frames=record["target_frames"]))
    del rc, g
    return record


def draw_results(records, scores, threshold, output, demo, resolution, extent_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    test = [r for r in records if r["role"] == "test"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    fractions = np.array([[r["row"][k] for k in ("surface_fraction", "free_fraction", "unknown_fraction")] for r in test])
    bottom = np.zeros(len(test))
    for k, (label, color) in enumerate(zip(("Observed surface", "Confirmed free", "Unknown"),
                                         ("#e4a044", "#568fbd", "#bac0c8"))):
        axs[0].bar(np.arange(len(test)), fractions[:, k], bottom=bottom, color=color, label=label)
        bottom += fractions[:, k]
    axs[0].set(xticks=np.arange(len(test)), xticklabels=[r["scene_id"][:6] for r in test], ylim=(0, 1),
               title="Held-out grid labels", ylabel="Fraction of all cells")
    axs[0].legend(fontsize=8)
    vals = np.concatenate([r["count"].numpy()[r["y"].numpy() == 1] for r in test])
    axs[1].hist(vals, bins=np.geomspace(1, max(2, vals.max()), 32), color="#74a58e")
    axs[1].axvline(64, color="#ba6756", label="64 Gaussians")
    axs[1].set(xscale="log", title="Official contributing Gaussian centres / surface cell",
               xlabel="Count (zero-count cells omitted from log plot)", ylabel="Cells")
    axs[1].legend()
    fig.tight_layout(); fig.savefig(output / "grid_statistics.png", dpi=180); plt.close(fig)
    offset = 0
    for r in test:
        n = len(r["y"]); p = scores[offset:offset+n]; offset += n
        if r["scene_id"] != "7b6477cb95":
            continue
        y = r["y"].numpy()
        centers, _, _ = make_grid(resolution, extent_m)
        colors = np.full((n, 3), .7)
        colors[(y == 0) & (p < threshold)] = [.2, .55, .9]
        colors[(y == 1) & (p >= threshold)] = [.1, .7, .3]
        colors[(y == 1) & (p < threshold)] = [1., .1, .1]
        colors[(y == 0) & (p >= threshold)] = [.7, .3, .8]
        write_points_ply(demo / "grid_predictions.ply", centers.numpy(), colors)
        # Select a visible-surface-rich slice for inspection, not for model input.
        cube = y.reshape((resolution,) * 3)
        iy = int((cube == 1).sum((0, 2)).argmax())
        fig, axs = plt.subplots(1, 4, figsize=(15, 4))
        images = [cube[:, iy, :].T, p.reshape(cube.shape)[:, iy, :].T,
                  r["count"].reshape(cube.shape)[:, iy, :].T,
                  r["texture"].reshape(cube.shape)[:, iy, :].T]
        for ax, data, title in zip(axs, images, ("Laser labels: grey/blue/gold", "Predicted surface probability",
                                                "Contributing Gaussian centres", "Observed texture gradient")):
            ax.imshow(data, origin="lower", interpolation="nearest", aspect="equal")
            ax.set(title=title, xlabel="Grid x", ylabel="Grid z")
        axs[0].images[0].set_cmap(ListedColormap(["#bac0c8", "#568fbd", "#e4a044"]))
        axs[0].images[0].set_clim(-1, 1)
        axs[1].images[0].set_clim(0, 1)
        fig.suptitle(f"{r['scene_id']}  y slice {iy}; unknown cells are not negative labels")
        fig.tight_layout(); fig.savefig(demo / "local_section.png", dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("EXP/grid_activation"))
    ap.add_argument("--demo", type=Path, default=Path("EXP/demo/scene_7b6477cb95/grid_activation"))
    ap.add_argument("--train-scenes", type=int, default=16)
    ap.add_argument("--calibration-scenes", type=int, default=4)
    ap.add_argument("--test-scenes", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=16)
    ap.add_argument("--extent-m", type=float, default=6.)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-per-rank", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    from tokengs.train import verify_idle_gpus
    verify_idle_gpus()
    world, rank, local = (int(os.environ.get(k, d)) for k, d in
                          (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("Completed experiment exists; select a new output, do not overwrite")
    opt = dataclasses.replace(config_defaults["eval_scannetpp_latent_6view_ssim"],
                              lambda_lpips=0., num_workers=0, load_lidar_pc=False,
                              val_depth_metric=False, use_render_geo_supervision=False)
    providers = {}
    tasks = []
    for split, count in (("train", args.train_scenes + args.calibration_scenes), ("test", args.test_scenes)):
        p = Provider("scannetpp_val_scaled_0.3", dataclasses.replace(opt, dataset_kwargs={
            "split": "nvs_sem_train.txt" if split == "train" else "nvs_sem_val.txt", "window_size": 18}), training=False)
        providers[split] = p
        order = list(range(len(p)))
        if split == "train":
            order.sort(key=lambda i: hashlib.sha256(f'{args.seed}:{Path(p.dataset.sample_list[i]).name}'.encode()).hexdigest())
        for j, i in enumerate(order[:count]):
            tasks.append((split, i, "train" if split == "train" and j < args.train_scenes else
                          "calibration" if split == "train" else "test"))
    assert len(tasks) == args.train_scenes + args.calibration_scenes + args.test_scenes
    ids = [Path(providers[s].dataset.sample_list[i]).name for s, i, _ in tasks]
    assert len(set(ids)) == len(ids), "Scene splits overlap"
    checkpoint = geometrygs.TOKENGS_ROOT / "checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors"
    sha = sha256_file(checkpoint) if rank == 0 else None
    model = TokenGS(opt).eval().requires_grad_(False)
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model.to(device)
    gen = torch.Generator().manual_seed(173)
    projector = (torch.randn(4096, 64, generator=gen) / 4096 ** .5).to(device)
    local_records = []
    start = time.perf_counter()
    for job, (split, i, role) in enumerate(tasks):
        if job % world != rank:
            continue
        rec = scene_extract(providers[split], i, role, model, opt, projector, args, device)
        local_records.append(rec)
        print(f"[rank {rank}] {role} {rec['scene_id']}: surface/free/unknown="
              f"{rec['row']['surface_cells']}/{rec['row']['free_cells']}/{rec['row']['unknown_cells']}; "
              f"live GS={rec['row']['contributing_gaussians']}", flush=True)
    del model, projector, providers
    torch.cuda.empty_cache()
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local_records)
    else:
        gathered = [local_records]
    records = sorted([r for group in gathered for r in group], key=lambda r: ids.index(r["scene_id"]))
    manifest = [dict(scene_id=r["scene_id"], split=r["role"], input_frames=r["input_frames"],
                     target_frames=r["target_frames"]) for r in records]
    if rank == 0:
        json_write(args.output / "manifest.json", manifest)
        write_csv(args.output / "occupancy_scene_metrics.csv", [r["row"] for r in records])
        write_csv(args.output / "frame_metrics.csv", [f for r in records for f in r["frames"]])
    data = {}
    for role in ("train", "calibration", "test"):
        rs = [r for r in records if r["role"] == role]
        data[role] = (torch.cat([r["x"] for r in rs]).to(device), torch.cat([r["y"] for r in rs]).to(device))
    summary = dict(protocol=vars(args) | {"output": str(args.output), "demo": str(args.demo)},
                   checkpoint=str(checkpoint), checkpoint_sha256=sha, world_size=world,
                   grid_lower_m=[-args.extent_m/2, -args.extent_m/2, 0],
                   cell_size_m=args.extent_m/args.resolution,
                   label_definition="observed input surface / conservative confirmed free / unknown",
                   feature_definition="64D fixed random projection of frozen multi-scale image features; 9 samples per cell",
                   predictor_inputs="RGB image features and input camera/grid geometry only; no depth, SfM, GS or mesh",
                   gaussian_count_definition="centres of GS with exact summed input-render compositing weight >1e-6",
                   checkpoints_are_probe_only=True, models={})
    metrics_rows, saved = [], {}
    for name, dims in (("camera_geometry", records[0]["geo_dim"]), ("projected_image", data["train"][0].shape[1])):
        train_x, train_y = data["train"]
        known = train_y >= 0
        tx = train_x[:, :dims]
        torch.manual_seed(args.seed)
        predictor = GridClassifier(tx[known].mean(0), tx[known].std(0)).to(device)
        ddp = DDP(predictor, device_ids=[local]) if world > 1 else predictor
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3, weight_decay=1e-4)
        positives, negatives = (train_y == 1).nonzero()[:, 0], (train_y == 0).nonzero()[:, 0]
        generator = torch.Generator(device=device).manual_seed(args.seed + rank)
        for step in range(args.steps):
            npos = args.batch_per_rank // 2
            ii = torch.cat((positives[torch.randint(len(positives), (npos,), generator=generator, device=device)],
                            negatives[torch.randint(len(negatives), (args.batch_per_rank-npos,), generator=generator, device=device)]))
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(ddp(tx[ii]), train_y[ii].float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 5.)
            optimizer.step()
            if rank == 0 and (step + 1) % 200 == 0:
                print(f"[{name}] step {step+1}/{args.steps} train balanced BCE={float(loss):.5f}", flush=True)
        if rank == 0:
            from sklearn.metrics import roc_auc_score, average_precision_score
            predictor.eval()
            scored = {}
            with torch.no_grad():
                for role, (x, y) in data.items():
                    p = torch.cat([predictor(xb[:, :dims]).sigmoid() for xb in x.split(4096)]).cpu().numpy()
                    scored[role] = (p, y.cpu().numpy())
            results = dict(parameters=sum(p.numel() for p in predictor.parameters()), thresholds={}, splits={})
            for recall in (.95, .99):
                threshold = recall_threshold(*scored["calibration"], recall)
                key = f"calibration_recall_{recall:g}"
                results["thresholds"][key] = threshold
                for role, (p, y) in scored.items():
                    result = classification_metrics(p, y, threshold)
                    k = y >= 0
                    result.update(auc=float(roc_auc_score(y[k], p[k])),
                                  average_precision=float(average_precision_score(y[k], p[k])))
                    results["splits"].setdefault(role, {})[key] = result
                    offset = 0
                    for r in [r for r in records if r["role"] == role]:
                        q = p[offset:offset+len(r["y"])]; offset += len(r["y"])
                        row = classification_metrics(q, r["y"].numpy(), threshold)
                        target = r["target_hits"].numpy()
                        row.update(target_surface_cell_recall=float(((q >= threshold) & (target > 0)).sum() / max(1, (target > 0).sum())),
                                   target_surface_hit_recall=float(target[q >= threshold].sum() / max(1, target.sum())))
                        metrics_rows.append(dict(model=name, split=role, scene_id=r["scene_id"],
                                                 calibration_recall=recall, threshold=threshold, **row))
            summary["models"][name] = results
            for key, value in predictor.state_dict().items():
                saved[f"{name}.{key}"] = value.cpu().contiguous()
            print(json.dumps({name: results}, indent=2), flush=True)
            if name == "projected_image":
                draw_results(records, scored["test"][0], results["thresholds"]["calibration_recall_0.99"],
                             args.output, args.demo, args.resolution, args.extent_m)
                np.savez_compressed(args.output / "test_predictions.npz", scores=scored["test"][0],
                                    labels=scored["test"][1], scene_ids=np.array([r["scene_id"] for r in records if r["role"] == "test"]))
                oracle = {"warning": "Thresholds chosen using TEST LABELS: diagnostic ranking only, not deployable/test performance at a calibrated threshold."}
                for recall in (.95, .99):
                    t = recall_threshold(*scored["test"], recall)
                    oracle[str(recall)] = {"threshold": t, **classification_metrics(*scored["test"], t)}
                json_write(args.output / "test_oracle_thresholds.json", oracle)
        if world > 1:
            dist.barrier()
        del ddp, predictor, optimizer
    if rank == 0:
        save_file(saved, str(args.output / "model.safetensors"))
        write_csv(args.output / "scene_metrics.csv", metrics_rows)
        summary["elapsed_seconds"] = time.perf_counter() - start
        summary["test_occupancy_scene_mean"] = {
            k: float(np.mean([r["row"][k] for r in records if r["role"] == "test"]))
            for k in records[0]["row"] if k not in ("scene_id", "split")}
        json_write(args.output / "summary.json", summary)
        print(f"Completed: {args.output / 'summary.json'}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
