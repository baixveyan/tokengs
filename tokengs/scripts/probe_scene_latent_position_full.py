"""Full-data scene-latent position probe: fit on ALL ScanNet++ training scenes, test on ALL 50 validation scenes.

Same probe definition as scripts/probe_scene_latent_position.py (input = 1024-d scene latent of the frozen
official model, target = layer-7 local-head GT attention surface point), but the probe is trained on the
853-scene training split (deterministic 18-frame window per scene, the same rule as validation) so that a
weak result can no longer be blamed on too little probe data.

Two stages:
  extract  one process per idle GPU, shards the (split, scene) list round-robin, writes
           <cache>/shard_<k>.pt. Training scenes keep --train-latents-per-scene random valid latents
           (fixed seed per scene); validation scenes keep all valid latents.
  fit      loads every shard, fits ridge + MLP probes and controls, reports held-out validation errors
           (total / along-ray depth / lateral), and re-exports the demo-scene PLYs.

    PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/probe_scene_latent_position_full.py extract --shard k --num-shards N
    PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/probe_scene_latent_position_full.py fit
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from diagnose_latent_attention_geometry import capture_cross_attention, patch_gt_points
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, idle_gpu
from probe_scene_latent_position import err_colour, fit_linear, fit_mlp, split_error, stats, write_points
from scannetpp_evaluation_protocol import build_raycasting_scene, sha256_file
from tokengs.data.provider import Provider
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults

CACHE = Path("/mnt/disk/xveyan/workspace_scannetpp/probe_scene_latent_full")
SPLITS = {"train": "nvs_sem_train.txt", "val": "nvs_sem_val.txt"}
DEMO_SCENE = "7b6477cb95"


def make_opt(split: str):
    return dataclasses.replace(
        config_defaults["eval_scannetpp_latent_6view_ssim"],
        evaluating=True, use_ttt_for_eval=False, ttt_n_steps=0, use_geo_losses=False, load_lidar_pc=False,
        use_input_supervision=False, mixed_precision="no", num_workers=0,
        dataset_kwargs={"window_size": 18, "split": SPLITS[split]},
    )


def extract_one(model, provider, idx, opt, device, layer):
    ds = provider.dataset
    scene_id = Path(ds.sample_list[idx]).name
    context, _target = ds.get_context_target_frames(idx)
    raw = provider[idx]
    data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
    model_input, _ = split_data(data, opt)
    s = float(provider.scene_scale)
    with torch.inference_mode():
        latents, attn = capture_cross_attention(model, model_input, [layer])
    a = attn[layer].cpu()
    cam_in, intr_in = data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu()
    raw_ctx_c2w = torch.from_numpy(ds._scene_meta_cache[scene_id]["c2ws"][context].copy())
    mesh_ply = ds.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    rc_scene = build_raycasting_scene(mesh_ply, raw_ctx_c2w, s)
    pts, valid, _ = patch_gt_points(rc_scene, cam_in, intr_in, opt.img_size, opt.patch_size)
    pts_flat, valid_flat = pts.reshape(-1, 3), valid.reshape(-1)
    eff = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
    local = [h for h in range(a.shape[0]) if float(eff[h].median()) < 30]
    pv = a[local] * valid_flat.float()
    mass = pv.sum(-1)
    cent = (pv / mass.unsqueeze(-1).clamp_min(1e-12)) @ pts_flat
    target = cent.mean(0)
    spread_m = cent.std(0).norm(dim=-1) / s
    valid_lat = (mass.min(0).values > 0.5) & (spread_m < 0.30)
    V = cam_in.shape[0]
    cam_centres = torch.stack([torch.linalg.inv(cam_in[v].float().T)[:3, 3] for v in range(V)])
    view_mass = a[local].mean(0).reshape(a.shape[1], V, -1).sum(-1)
    dom_cam = cam_centres[view_mass.argmax(-1)]
    return scene_id, latents[0].float().cpu(), target, valid_lat, dom_cam, len(local)


def cmd_extract(args):
    gpu = idle_gpu()
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected checkpoint SHA256")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    providers = {sp: Provider("scannetpp_val_scaled_0.3", make_opt(sp), training=False) for sp in SPLITS}
    tasks = [(sp, i) for sp in ("val", "train") for i in range(len(providers[sp]))]
    mine = tasks[args.shard::args.num_shards]
    opt = make_opt("val")
    model = model_registry[opt.model_type](opt).eval().requires_grad_(False)
    model.load_state_dict(load_file(str(CHECKPOINT), device="cpu"), strict=True)
    model = model.to(device=device, dtype=torch.float32)
    print(f"GPU {gpu['index']}: shard {args.shard}/{args.num_shards}, {len(mine)} windows "
          f"(train {len(providers['train'])}, val {len(providers['val'])} total)", flush=True)
    out = []
    t0 = time.time()
    for n, (sp, i) in enumerate(mine):
        sid, lat, tgt, val, cam, n_local = extract_one(model, providers[sp], i, make_opt(sp), device, args.layer)
        keep = torch.nonzero(val).squeeze(1)
        if sp == "train" and len(keep) > args.train_latents_per_scene:
            g = torch.Generator().manual_seed(int(sid, 16) % (2 ** 31))
            keep = keep[torch.randperm(len(keep), generator=g)[:args.train_latents_per_scene]]
        out.append({"split": sp, "scene_id": sid, "index": keep.to(torch.int16), "latent": lat[keep].half(),
                    "target": tgt[keep], "dom_cam": cam[keep], "n_valid": int(val.sum()), "n_local_heads": n_local})
        torch.cuda.empty_cache()
        if (n + 1) % 10 == 0 or n + 1 == len(mine):
            print(f"[shard {args.shard}] {n + 1}/{len(mine)} windows, {time.time() - t0:.0f}s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"shard_{args.shard}.tmp"
    torch.save(out, tmp)
    tmp.rename(CACHE / f"shard_{args.shard}.pt")
    print(f"wrote {CACHE / f'shard_{args.shard}.pt'}")


def cmd_fit(args):
    idle_gpu()
    device = torch.device("cuda:0")
    recs = []
    for p in sorted(CACHE.glob("shard_*.pt")):
        recs.extend(torch.load(p, map_location="cpu", weights_only=False))
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "val"]
    if len({r["scene_id"] for r in te}) != 50:
        raise RuntimeError(f"expected 50 validation scenes, got {len({r['scene_id'] for r in te})}")
    s = 0.3
    cat = lambda rs, k: torch.cat([r[k] for r in rs])
    Xtr, Ytr = cat(tr, "latent").float(), cat(tr, "target")
    Xte, Yte, Cte = cat(te, "latent").float(), cat(te, "target"), cat(te, "dom_cam")
    print(f"train: {len(tr)} scenes, {len(Xtr)} latents | test: {len(te)} scenes, {len(Xte)} latents", flush=True)

    idx_sum, idx_cnt = torch.zeros(4096, 3), torch.zeros(4096)
    idx_tr = cat(tr, "index").long()
    idx_sum.index_add_(0, idx_tr, Ytr)
    idx_cnt.index_add_(0, idx_tr, torch.ones(len(idx_tr)))
    idx_mean = idx_sum / idx_cnt.clamp_min(1).unsqueeze(-1)

    lin = fit_linear(Xtr, Ytr)
    mlp = fit_mlp(Xtr, Ytr, device, hidden=args.hidden, epochs=args.epochs)
    preds = {
        "linear_probe": lin(Xte),
        "mlp_probe": mlp(Xte),
        "index_prior_control": idx_mean[cat(te, "index").long()],
        "scene_centroid_control": torch.cat([r["target"].mean(0, keepdim=True).expand(len(r["target"]), 3) for r in te]),
    }
    res = {"checkpoint_sha256": CHECKPOINT_SHA, "n_train_scenes": len(tr), "n_train_latents": int(len(Xtr)),
           "n_test_scenes": len(te), "n_test_latents": int(len(Xte)),
           "train_latents_per_scene": args.train_latents_per_scene, "mlp_hidden": args.hidden, "mlp_epochs": args.epochs}
    for name, P in preds.items():
        tot, along, lat = split_error(P, Yte, Cte, s)
        res[name] = {"total": stats(tot), "depth_along_ray": stats(along), "lateral": stats(lat)}
    sub = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(0))[:200_000]
    tot_tr, along_tr, lat_tr = split_error(mlp(Xtr[sub]), Ytr[sub], cat(tr, "dom_cam")[sub], s)
    res["mlp_probe_train"] = {"total": stats(tot_tr), "depth_along_ray": stats(along_tr), "lateral": stats(lat_tr)}

    # per-scene medians (MLP) for spread across scenes
    off = 0
    per_scene = {}
    for r in te:
        n = len(r["target"])
        tot, along, lat = split_error(preds["mlp_probe"][off:off + n], r["target"], r["dom_cam"], s)
        per_scene[r["scene_id"]] = {"total_median_cm": float(tot.median() * 100), "depth_median_cm": float(along.median() * 100),
                                    "lateral_median_cm": float(lat.median() * 100), "n": n}
        off += n
    res["per_val_scene_mlp"] = per_scene

    demo = next(r for r in te if r["scene_id"] == DEMO_SCENE)
    P0 = mlp(demo["latent"].float())
    T0 = demo["target"]
    e0 = (P0 - T0).norm(dim=-1) / s
    rgb = np.array([err_colour(float(e)) for e in e0])
    d = ROOT / "EXP_ScanNetPP" / "demo" / "scene_7b6477cb95" / "baseline_pretrained_6view"
    write_points(d / "baseline_pretrained_6view_scene_latent_probe_full_pred.ply", P0.numpy(), rgb, (e0 * 100).numpy())
    write_points(d / "baseline_pretrained_6view_scene_latent_gt_attention.ply", T0.numpy(), np.ones((len(T0), 3)),
                 np.zeros(len(T0)), radius=0.003)
    ts = torch.linspace(0, 1, 7).view(1, -1, 1)
    seg = (P0.unsqueeze(1) * (1 - ts) + T0.unsqueeze(1) * ts).reshape(-1, 3)
    write_points(d / "baseline_pretrained_6view_scene_latent_probe_full_links.ply", seg.numpy(),
                 np.repeat(rgb, 7, axis=0), np.repeat((e0 * 100).numpy(), 7), radius=0.0015)
    _, a0, l0 = split_error(P0, T0, demo["dom_cam"], s)
    res["demo_scene_mlp_probe"] = {"total": stats(e0), "depth_along_ray": stats(a0), "lateral": stats(l0)}

    out = ROOT / "docs" / "DIAG_scene_latent_position_probe_full.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for name in ("linear_probe", "mlp_probe", "index_prior_control", "scene_centroid_control", "mlp_probe_train"):
        r = res[name]
        print(f"{name:24s} total med {r['total']['median_cm']:6.1f} p90 {r['total']['p90_cm']:6.1f} cm | "
              f"depth med {r['depth_along_ray']['median_cm']:6.1f} p90 {r['depth_along_ray']['p90_cm']:6.1f} | "
              f"lateral med {r['lateral']['median_cm']:6.1f} p90 {r['lateral']['p90_cm']:6.1f} | <5cm {r['total']['frac_lt_5cm']:.2f}")
    med = sorted(v["total_median_cm"] for v in per_scene.values())
    print(f"per-val-scene MLP total median: min {med[0]:.1f} / median {med[len(med) // 2]:.1f} / max {med[-1]:.1f} cm")
    print("demo:", json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in res["demo_scene_mlp_probe"].items()}))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["extract", "fit"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--train-latents-per-scene", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    cmd_extract(args) if args.mode == "extract" else cmd_fit(args)


if __name__ == "__main__":
    main()
