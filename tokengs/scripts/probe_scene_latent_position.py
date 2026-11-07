"""Probe: does each scene latent EXPLICITLY encode the 3D position of the surface it is responsible for?

Decoupled from Gaussian tokens and the deconv head: input = the 1024-d scene latent after Stage 4 (frozen
official model), target = the latent's GT attention surface point (local heads of latent block --layer;
patch centres raycast onto the GT mesh, attention-weighted mean, exactly as in D0/D1).

Probes (fit on --train-scenes validation windows, tested on held-out --test-scenes, incl. demo scene 0):
  * linear ridge regression latent -> xyz
  * 2-layer MLP latent -> xyz
  * control "index prior": per-latent-index mean target over training scenes (latents start from fixed
    learnable queries, so the index alone may already predict location)
Errors are reported in metres, split into the component along the viewing ray (depth) and lateral.

Demo exports (scene 0, EXP_ScanNetPP/demo/scene_7b6477cb95/baseline_pretrained_6view/):
  baseline_pretrained_6view_scene_latent_probe_pred.ply     probe-predicted latent positions (error-coloured)
  baseline_pretrained_6view_scene_latent_gt_attention.ply   GT attention surface points (white)
  baseline_pretrained_6view_scene_latent_probe_links.ply    7 samples on each pred->GT segment (error-coloured)

    CUDA_VISIBLE_DEVICES=<idle> PYTHONPATH=scripts:. python scripts/probe_scene_latent_position.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from safetensors.torch import load_file

from diagnose_latent_attention_geometry import capture_cross_attention, patch_gt_points
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, frame_splits, idle_gpu
from scannetpp_evaluation_protocol import build_raycasting_scene, sha256_file
from tokengs.data.provider import Provider
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults

SH_C0 = 0.28209479177387814


def extract(model, provider, splits, idx, opt, device, layer):
    spec = splits[idx]
    raw = provider[idx]
    data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
    model_input, _ = split_data(data, opt)
    s = float(provider.scene_scale)
    with torch.inference_mode():
        latents, attn = capture_cross_attention(model, model_input, [layer])
    a = attn[layer].cpu()  # [H, N_lat, N_patch]
    cam_in, intr_in = data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu()
    scene_id = spec["scene_id"]
    raw_ctx_c2w = torch.from_numpy(provider.dataset._scene_meta_cache[scene_id]["c2ws"][spec["context"]].copy())
    mesh_ply = provider.dataset.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    rc_scene = build_raycasting_scene(mesh_ply, raw_ctx_c2w, s)
    pts, valid, _ = patch_gt_points(rc_scene, cam_in, intr_in, opt.img_size, opt.patch_size)
    pts_flat, valid_flat = pts.reshape(-1, 3), valid.reshape(-1)
    eff = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
    local = [h for h in range(a.shape[0]) if float(eff[h].median()) < 30]
    pv = a[local] * valid_flat.float()
    mass = pv.sum(-1)  # [H_loc, N_lat]
    p = pv / mass.unsqueeze(-1).clamp_min(1e-12)
    cent = p @ pts_flat  # [H_loc, N_lat, 3]
    target = cent.mean(0)
    spread_m = cent.std(0).norm(dim=-1) / s  # disagreement among local heads
    valid_lat = (mass.min(0).values > 0.5) & (spread_m < 0.30)
    cam_centres = torch.stack([torch.linalg.inv(cam_in[v].float().T)[:3, 3] for v in range(cam_in.shape[0])])
    # dominant camera per latent = view receiving the most local-head attention mass
    V = cam_in.shape[0]
    view_mass = a[local].mean(0).reshape(a.shape[1], V, -1).sum(-1)
    dom_cam = cam_centres[view_mass.argmax(-1)]
    return {"scene_id": scene_id, "latent": latents[0].float().cpu(), "target": target, "valid": valid_lat,
            "dom_cam": dom_cam, "s": s}


def split_error(pred, target, cam, s):
    ray = torch.nn.functional.normalize(target - cam, dim=-1)
    err = pred - target
    along = (err * ray).sum(-1)
    lateral = (err - along.unsqueeze(-1) * ray).norm(dim=-1)
    return err.norm(dim=-1) / s, along.abs() / s, lateral / s


def stats(x):
    x = x.numpy()
    return {"median_cm": float(np.median(x) * 100), "p75_cm": float(np.percentile(x, 75) * 100),
            "p90_cm": float(np.percentile(x, 90) * 100), "frac_lt_5cm": float((x < 0.05).mean()),
            "frac_lt_15cm": float((x < 0.15).mean())}


def fit_linear(X, Y, lam=10.0):
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = torch.cat([(X - mu) / sd, torch.ones(len(X), 1)], 1).double()
    W = torch.linalg.solve(Xn.T @ Xn + lam * torch.eye(Xn.shape[1], dtype=torch.float64), Xn.T @ Y.double())
    return lambda Z: (torch.cat([(Z - mu) / sd, torch.ones(len(Z), 1)], 1).double() @ W).float()


def fit_mlp(X, Y, device, hidden=512, epochs=80):
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    ym, ys = Y.mean(0, keepdim=True), Y.std(0, keepdim=True) + 1e-6
    Xn, Yn = ((X - mu) / sd).to(device), ((Y - ym) / ys).to(device)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 3)).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    for _ in range(epochs):
        perm = torch.randperm(len(Xn), device=device)
        for i in range(0, len(Xn), 2048):
            b = perm[i:i + 2048]
            optim.zero_grad()
            torch.nn.functional.smooth_l1_loss(net(Xn[b]), Yn[b], beta=0.1).backward()
            optim.step()
        sched.step()
    net.eval().cpu()

    def pred(Z):
        with torch.no_grad():
            return net((Z - mu) / sd) * ys + ym
    return pred


def err_colour(e_m):
    if e_m < 0.05:
        return (0.24, 0.71, 0.29)
    if e_m < 0.15:
        return (1.0, 0.88, 0.1)
    return (0.9, 0.1, 0.29)


def write_points(path, xyz, rgb, err_cm, radius=0.004):
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
             ("opacity", "<f4"), ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
             ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"), ("error_cm", "<f4")]
    out = np.zeros(len(xyz), dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for c in range(3):
        out[f"f_dc_{c}"] = (rgb[:, c] - 0.5) / SH_C0
    out["opacity"] = float(np.log(0.99 / 0.01))
    out["scale_0"] = out["scale_1"] = out["scale_2"] = float(np.log(radius))
    out["rot_0"] = 1.0
    out["error_cm"] = err_cm
    PlyData([PlyElement.describe(out, "vertex")]).write(str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-scenes", type=int, nargs="+", default=list(range(1, 36)))
    ap.add_argument("--test-scenes", type=int, nargs="+", default=[0] + list(range(36, 50)))
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "DIAG_scene_latent_position_probe.json")
    args = ap.parse_args()

    gpu = idle_gpu()
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected checkpoint SHA256")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    opt = dataclasses.replace(
        config_defaults["eval_scannetpp_latent_6view_ssim"],
        evaluating=True, use_ttt_for_eval=False, ttt_n_steps=0, use_geo_losses=False, load_lidar_pc=False,
        use_input_supervision=False, mixed_precision="no", num_workers=0, dataset_kwargs={"window_size": 18},
    )
    provider = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    splits = frame_splits(provider)
    device = torch.device("cuda:0")
    model = model_registry[opt.model_type](opt).eval().requires_grad_(False)
    model.load_state_dict(load_file(str(CHECKPOINT), device="cpu"), strict=True)
    model = model.to(device=device, dtype=torch.float32)
    print(f"GPU {gpu['index']}: extracting {len(args.train_scenes) + len(args.test_scenes)} scenes")

    sc = {}
    for idx in sorted(set(args.train_scenes + args.test_scenes)):
        sc[idx] = extract(model, provider, splits, idx, opt, device, args.layer)
        print(f"[{idx}] {sc[idx]['scene_id']}: valid latents {int(sc[idx]['valid'].sum())}/4096", flush=True)
        torch.cuda.empty_cache()
    s = sc[0]["s"]

    def stack(ids, key):
        return torch.cat([sc[i][key][sc[i]["valid"]] for i in ids])

    Xtr, Ytr = stack(args.train_scenes, "latent"), stack(args.train_scenes, "target")
    Xte, Yte, Cte = stack(args.test_scenes, "latent"), stack(args.test_scenes, "target"), stack(args.test_scenes, "dom_cam")

    # index-prior control
    idx_sum, idx_cnt = torch.zeros(4096, 3), torch.zeros(4096)
    for i in args.train_scenes:
        v = sc[i]["valid"]
        idx_sum[v] += sc[i]["target"][v]
        idx_cnt[v] += 1
    idx_mean = idx_sum / idx_cnt.clamp_min(1).unsqueeze(-1)
    idx_te = torch.cat([torch.arange(4096)[sc[i]["valid"]] for i in args.test_scenes])

    lin = fit_linear(Xtr, Ytr)
    mlp = fit_mlp(Xtr, Ytr, device)
    preds = {"linear_probe": lin(Xte), "mlp_probe": mlp(Xte), "index_prior_control": idx_mean[idx_te],
             "scene_centroid_control": torch.cat([sc[i]["target"][sc[i]["valid"]].mean(0, keepdim=True).expand(int(sc[i]["valid"].sum()), 3)
                                                  for i in args.test_scenes])}
    res = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "checkpoint_sha256": CHECKPOINT_SHA, "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    for name, P in preds.items():
        tot, along, lat = split_error(P, Yte, Cte, s)
        res[name] = {"total": stats(tot), "depth_along_ray": stats(along), "lateral": stats(lat)}
    tot_tr, _, _ = split_error(mlp(Xtr), Ytr, stack(args.train_scenes, "dom_cam"), s)
    res["mlp_probe_train_total"] = stats(tot_tr)

    # demo exports (scene 0)
    d = sc[0]
    v = d["valid"]
    P0 = mlp(d["latent"][v])
    T0 = d["target"][v]
    e0 = (P0 - T0).norm(dim=-1) / s
    rgb = np.array([err_colour(float(e)) for e in e0])
    demo = ROOT / "EXP_ScanNetPP" / "demo" / "scene_7b6477cb95" / "baseline_pretrained_6view"
    write_points(demo / "baseline_pretrained_6view_scene_latent_probe_pred.ply", P0.numpy(), rgb, (e0 * 100).numpy())
    write_points(demo / "baseline_pretrained_6view_scene_latent_gt_attention.ply", T0.numpy(),
                 np.ones((len(T0), 3)), np.zeros(len(T0)), radius=0.003)
    ts = torch.linspace(0, 1, 7).view(1, -1, 1)
    seg = (P0.unsqueeze(1) * (1 - ts) + T0.unsqueeze(1) * ts).reshape(-1, 3)
    write_points(demo / "baseline_pretrained_6view_scene_latent_probe_links.ply", seg.numpy(),
                 np.repeat(rgb, 7, axis=0), np.repeat((e0 * 100).numpy(), 7), radius=0.0015)
    _, along0, lat0 = split_error(P0, T0, d["dom_cam"][v], s)
    res["demo_scene_mlp_probe"] = {"n_valid_latents": int(v.sum()), "total": stats(e0), "depth_along_ray": stats(along0),
                                   "lateral": stats(lat0)}

    args.output.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for name in ("linear_probe", "mlp_probe", "index_prior_control", "scene_centroid_control"):
        r = res[name]
        print(f"{name:24s} total med {r['total']['median_cm']:6.1f} p90 {r['total']['p90_cm']:6.1f} cm | "
              f"depth med {r['depth_along_ray']['median_cm']:6.1f} p90 {r['depth_along_ray']['p90_cm']:6.1f} | "
              f"lateral med {r['lateral']['median_cm']:6.1f} p90 {r['lateral']['p90_cm']:6.1f} | <5cm {r['total']['frac_lt_5cm']:.2f}")
    print("mlp train total median cm:", round(res["mlp_probe_train_total"]["median_cm"], 1))
    print("demo:", json.dumps(res["demo_scene_mlp_probe"]))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
