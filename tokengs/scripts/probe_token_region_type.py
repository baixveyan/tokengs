"""Pre-test: do the pretrained TokenGS token features already know whether a token sits on a PLANE,
a CREASE (two planes meeting) or a COMPLEX region?

Per scene (validation split, fixed frame_splits):
  * GT region type per token from the mesh: take the token's attention surface point (D1 read position),
    gather full-resolution raycast surface points (+ face normals) within --radius metres, then
      plane   : one plane fits with RMS residual < --plane-rms and normal spread < --plane-ang
      crease  : two normal clusters (k=2), each tight, angle between them > --crease-ang, both planar
      complex : otherwise
      unknown : too few points or read position far from any surface (excluded from scoring)
  * Features per token: decoder-output token feature h (1024-d), and scene latent (1024-d).
  * Probe: multinomial logistic regression fit on --train-scenes validation scenes, tested on the rest
    (including scene 0 = demo). Reported against chance / majority and a no-learning heuristic that
    classifies from the OLD head's 64 children (PCA of alive children positions).
  * Demo exports (scene 0): token anchors coloured by GT type, by predicted type, and mismatches.

    CUDA_VISIBLE_DEVICES=<idle> PYTHONPATH=scripts:. python scripts/probe_token_region_type.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from plyfile import PlyData, PlyElement
from safetensors.torch import load_file
from scipy.spatial import cKDTree

from diagnose_latent_attention_geometry import ALIVE_OPACITY, capture_cross_attention
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, frame_splits, idle_gpu
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals, sha256_file
from tokengs.data.provider import Provider
from tokengs.models import model_registry
from tokengs.models.input_types import EncoderLatent, split_data
from tokengs.options import config_defaults

CLASSES = ("plane", "crease", "complex")
UNKNOWN = -1
CLASS_RGB = {0: (0.24, 0.71, 0.29), 1: (1.0, 0.88, 0.1), 2: (0.9, 0.1, 0.29), UNKNOWN: (0.55, 0.55, 0.55)}
MISMATCH_RGB = (0.94, 0.2, 0.9)
SH_C0 = 0.28209479177387814


def unproject_full(d_gt, n_gt, ok, cam_view, intr):
    """Full-resolution GT surface points and normals (normalized world coords) for every input view."""
    V, H, W = d_gt.shape
    pts, nrm = [], []
    for v in range(V):
        w2c = cam_view[v].float().T
        c2w = torch.linalg.inv(w2c)
        fx, fy, cx, cy = intr[v].float().tolist()
        ys, xs = torch.meshgrid(torch.arange(H).float() + 0.5, torch.arange(W).float() + 0.5, indexing="ij")
        dirs = torch.stack([(xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)], -1)
        p_cam = dirs * d_gt[v].unsqueeze(-1)
        p_w = p_cam @ c2w[:3, :3].T + c2w[:3, 3]
        n_w = n_gt[v] @ c2w[:3, :3].T  # normals given in camera frame by the protocol helper
        m = ok[v]
        pts.append(p_w[m])
        nrm.append(torch.nn.functional.normalize(n_w[m], dim=-1))
    return torch.cat(pts), torch.cat(nrm)


def fit_plane_rms(p):
    c = p.mean(0)
    q = p - c
    cov = q.T @ q / max(len(p), 1)
    ev, evec = torch.linalg.eigh(cov)
    return float(ev[0].clamp_min(0).sqrt()), evec[:, 0]


def normal_spread_deg(n):
    m = torch.nn.functional.normalize(n.mean(0), dim=0)
    return float(torch.rad2deg(torch.acos((n @ m).clamp(-1, 1))).mean())


def classify_region(p, n, s, plane_rms_m, plane_ang, crease_ang, min_pts):
    """Return class id for a local surface sample (positions p [k,3] normalized, normals n [k,3])."""
    if len(p) < min_pts:
        return UNKNOWN
    rms_m, _ = fit_plane_rms(p)
    rms_m /= s
    if rms_m < plane_rms_m and normal_spread_deg(n) < plane_ang:
        return 0
    # two-cluster split on normals (k-means, 5 iters)
    c0, c1 = n[0], n[torch.argmin(n @ n[0])]
    for _ in range(5):
        a = (n @ c0) >= (n @ c1)
        if a.sum() < min_pts // 2 or (~a).sum() < min_pts // 2:
            break
        c0 = torch.nn.functional.normalize(n[a].mean(0), dim=0)
        c1 = torch.nn.functional.normalize(n[~a].mean(0), dim=0)
    a = (n @ c0) >= (n @ c1)
    if a.sum() >= min_pts // 2 and (~a).sum() >= min_pts // 2:
        ang = float(torch.rad2deg(torch.acos((c0 @ c1).clamp(-1, 1))))
        r0, _ = fit_plane_rms(p[a])
        r1, _ = fit_plane_rms(p[~a])
        if (ang > crease_ang and r0 / s < plane_rms_m * 1.5 and r1 / s < plane_rms_m * 1.5
                and normal_spread_deg(n[a]) < plane_ang * 1.5 and normal_spread_deg(n[~a]) < plane_ang * 1.5):
            return 1
    return 2


def children_heuristic(g, s, plane_rms_m, crease_ang):
    """No-learning baseline: classify from the OLD head's alive children (PCA thickness)."""
    out = torch.full((g.shape[0],), 2, dtype=torch.long)
    for t in range(g.shape[0]):
        alive = g[t, :, 3] >= ALIVE_OPACITY
        p = g[t, alive, :3]
        if len(p) < 8:
            out[t] = UNKNOWN
            continue
        rms_m, _ = fit_plane_rms(p)
        out[t] = 0 if rms_m / s < plane_rms_m * 3 else 2
    return out


def write_type_ply(path, xyz, cls, mismatch=None, radius_norm=0.006):
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
             ("opacity", "<f4"), ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
             ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"), ("region_type", "<f4")]
    out = np.zeros(len(xyz), dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for i in range(len(xyz)):
        rgb = MISMATCH_RGB if (mismatch is not None and mismatch[i]) else CLASS_RGB[int(cls[i])]
        for c in range(3):
            out[f"f_dc_{c}"][i] = (rgb[c] - 0.5) / SH_C0
    out["opacity"] = float(np.log(0.99 / 0.01))
    out["scale_0"] = out["scale_1"] = out["scale_2"] = float(np.log(radius_norm))
    out["rot_0"] = 1.0
    out["region_type"] = cls
    PlyData([PlyElement.describe(out, "vertex")]).write(str(path))


def extract_scene(model, provider, splits, idx, opt, device, args):
    spec = splits[idx]
    raw = provider[idx]
    data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
    model_input, _ = split_data(data, opt)
    s = float(provider.scene_scale)
    cap = {}
    blk = model.enc_dec_backbone.decoder_blocks[0]

    def hook_attn(_m, inputs, _o):
        gs_tokens, keys, _values = inputs
        ca = blk.gs_cross_attn
        q = rearrange(ca.q_proj(ca.gs_token_norm(gs_tokens)), "b n (h d) -> b h n d", h=ca.num_heads)
        q = ca.q_norm(q)
        logits = torch.einsum("bhnd,bhmd->bhnm", q.float(), keys.float()) / math.sqrt(q.shape[-1])
        cap["dec_attn"] = torch.softmax(logits, dim=-1)[0].cpu()

    def hook_out(_m, _i, output):
        cap["h"] = output[0].detach().float().cpu()

    h1 = blk.gs_cross_attn.register_forward_hook(hook_attn)
    h2 = blk.register_forward_hook(hook_out)
    with torch.inference_mode():
        latents, lat_attn = capture_cross_attention(model, model_input, [7])
        keys, values = model.enc_dec_backbone._latents_to_kv(latents)
        gaussians = model.forward_decoder(EncoderLatent(keys=keys, values=values), model_input.decoder).cpu().float()
    h1.remove()
    h2.remove()

    cam_in = data["cam_view_input"][0].cpu()
    intr_in = data["intrinsics_input"][0].cpu()
    scene_id = spec["scene_id"]
    raw_ctx_c2w = torch.from_numpy(provider.dataset._scene_meta_cache[scene_id]["c2ws"][spec["context"]].copy())
    mesh_ply = provider.dataset.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    rc_scene = build_raycasting_scene(mesh_ply, raw_ctx_c2w, s)
    d_gt, n_gt, ok = raycast_gt_depth_and_normals(rc_scene, cam_in, intr_in, opt.img_size)
    surf_p, surf_n = unproject_full(d_gt, n_gt, ok, cam_in, intr_in)

    # token read position (GT) via local heads + decoder-head consensus (as in D1 / triangulation export)
    a_lat = lat_attn[7].cpu()
    eff = (-(a_lat.clamp_min(1e-12).log() * a_lat).sum(-1)).exp()
    local_heads = [hh for hh in range(16) if float(eff[hh].median()) < 30]
    H, W = int(opt.img_size[0]), int(opt.img_size[1])
    patch = opt.patch_size
    hp, wp = H // patch, W // patch
    ys = torch.arange(hp) * patch + patch // 2
    xs = torch.arange(wp) * patch + patch // 2
    V = cam_in.shape[0]
    pts_patch, valid_patch = [], []
    for v in range(V):
        w2c = cam_in[v].float().T
        c2w = torch.linalg.inv(w2c)
        fx, fy, cx, cy = intr_in[v].float().tolist()
        d = d_gt[v][ys][:, xs]
        m = ok[v][ys][:, xs]
        u = (xs.float() + 0.5).view(1, wp).expand(hp, wp)
        vv = (ys.float() + 0.5).view(hp, 1).expand(hp, wp)
        dirs = torch.stack([(u - cx) / fx, (vv - cy) / fy, torch.ones_like(u)], -1)
        p_w = (dirs * d.unsqueeze(-1)) @ c2w[:3, :3].T + c2w[:3, 3]
        pts_patch.append(p_w.reshape(-1, 3))
        valid_patch.append(m.reshape(-1))
    pts_flat = torch.cat(pts_patch)
    valid_flat = torch.cat(valid_patch)
    pv = a_lat[local_heads] * valid_flat.float()
    p = pv / pv.sum(-1, keepdim=True).clamp_min(1e-12)
    lat_gt_pos = (p @ pts_flat).mean(0)
    A = cap["dec_attn"]
    head_pts = torch.einsum("htl,ld->thd", A, lat_gt_pos)
    d_pair = torch.cdist(head_pts, head_pts) / s
    best_h = (d_pair < 0.08).float().sum(-1).argmax(dim=1)
    best_pt = head_pts[torch.arange(4096), best_h]
    inl = ((head_pts - best_pt.unsqueeze(1)).norm(dim=-1) / s < 0.08).float()
    read_pos = (inl.unsqueeze(-1) * head_pts).sum(1) / inl.sum(1, keepdim=True).clamp_min(1.0)

    # GT region labels
    tree = cKDTree(surf_p.numpy())
    labels = torch.full((4096,), UNKNOWN, dtype=torch.long)
    dist_surf, _ = tree.query(read_pos.numpy())
    r_norm = args.radius * s
    for t in range(4096):
        if dist_surf[t] / s > args.max_anchor_dist:
            continue
        idx_pts = tree.query_ball_point(read_pos[t].numpy(), r_norm)
        if len(idx_pts) > 400:
            idx_pts = list(np.random.default_rng(t).choice(idx_pts, 400, replace=False))
        if len(idx_pts) < args.min_pts:
            continue
        labels[t] = classify_region(surf_p[idx_pts], surf_n[idx_pts], s, args.plane_rms, args.plane_ang,
                                    args.crease_ang, args.min_pts)

    g = gaussians[0].reshape(-1, 64, 14)
    heur = children_heuristic(g, s, args.plane_rms, args.crease_ang)
    return {"scene_id": scene_id, "h": cap["h"], "latents": latents[0].detach().float().cpu(), "labels": labels,
            "heuristic": heur, "read_pos": read_pos, "gaussians": g}


def fit_probe(X, y, num_classes=3, iters=300, device="cpu"):
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = ((X - mu) / sd).to(device)
    y = y.to(device)
    Wt = torch.zeros(X.shape[1], num_classes, device=device, requires_grad=True)
    b = torch.zeros(num_classes, device=device, requires_grad=True)
    counts = torch.bincount(y, minlength=num_classes).float().clamp_min(1)
    cw = (counts.sum() / counts) / num_classes
    opt = torch.optim.LBFGS([Wt, b], lr=0.5, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xn @ Wt + b, y, weight=cw) + 1e-3 * (Wt ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (mu, sd, Wt.detach().cpu(), b.detach().cpu())


def fit_mlp_probe(X, y, num_classes=3, hidden=256, epochs=60, device="cpu"):
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = ((X - mu) / sd).to(device)
    y = y.to(device)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(hidden, num_classes)).to(device)
    counts = torch.bincount(y, minlength=num_classes).float().clamp_min(1)
    cw = (counts.sum() / counts) / num_classes
    optim = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            b = perm[i:i + 1024]
            optim.zero_grad()
            torch.nn.functional.cross_entropy(net(Xn[b]), y[b], weight=cw).backward()
            optim.step()
    net.eval()
    return (mu, sd, net.cpu())


def predict_mlp(probe, X):
    mu, sd, net = probe
    with torch.no_grad():
        return net((X - mu) / sd).argmax(-1)


def predict_probe(probe, X):
    mu, sd, Wt, b = probe
    return (((X - mu) / sd) @ Wt + b).argmax(-1)


def confusion(y_true, y_pred, k=3):
    m = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t >= 0:
            m[t, p] += 1
    return m


def summarize(m):
    tot = m.sum()
    acc = float(np.trace(m) / max(tot, 1))
    recall = [float(m[i, i] / max(m[i].sum(), 1)) for i in range(m.shape[0])]
    return {"accuracy": acc, "balanced_accuracy": float(np.mean(recall)), "recall": recall,
            "support": [int(x) for x in m.sum(1)], "confusion": m.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-scenes", type=int, nargs="+", default=list(range(1, 25)))
    ap.add_argument("--test-scenes", type=int, nargs="+", default=[0] + list(range(25, 35)))
    ap.add_argument("--radius", type=float, default=0.15, help="metres around the token read position")
    ap.add_argument("--plane-rms", type=float, default=0.012, help="metres")
    ap.add_argument("--plane-ang", type=float, default=12.0, help="degrees")
    ap.add_argument("--crease-ang", type=float, default=30.0, help="degrees")
    ap.add_argument("--min-pts", type=int, default=30)
    ap.add_argument("--max-anchor-dist", type=float, default=0.10, help="metres; else unknown")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "DIAG_token_region_type_probe.json")
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

    scenes = {}
    for idx in sorted(set(args.train_scenes + args.test_scenes)):
        scenes[idx] = extract_scene(model, provider, splits, idx, opt, device, args)
        lab = scenes[idx]["labels"]
        cnt = [int((lab == c).sum()) for c in range(3)] + [int((lab == UNKNOWN).sum())]
        print(f"[{idx}] {scenes[idx]['scene_id']}: GT plane/crease/complex/unknown = {cnt}", flush=True)
        torch.cuda.empty_cache()

    def stack(ids, key):
        X = torch.cat([scenes[i][key] for i in ids])
        y = torch.cat([scenes[i]["labels"] for i in ids])
        keep = y >= 0
        return X[keep], y[keep]

    results = {"args": vars(args) | {"output": str(args.output)}, "checkpoint_sha256": CHECKPOINT_SHA}
    probes = {}
    for key in ("h", "latents"):
        Xtr, ytr = stack(args.train_scenes, key)
        probes[key] = fit_probe(Xtr, ytr, device=device)
        Xte, yte = stack(args.test_scenes, key)
        m = confusion(yte.numpy(), predict_probe(probes[key], Xte).numpy())
        results[f"probe_{key}_test"] = summarize(m)
        m_tr = confusion(ytr.numpy(), predict_probe(probes[key], Xtr).numpy())
        results[f"probe_{key}_train"] = summarize(m_tr)
    # baselines on test
    Xtr_h, ytr_h = stack(args.train_scenes, "h")
    Xte_h, yte = stack(args.test_scenes, "h")
    mlp = fit_mlp_probe(Xtr_h, ytr_h, device=device)
    results["mlp_probe_h_test"] = summarize(confusion(yte.numpy(), predict_mlp(mlp, Xte_h).numpy()))
    results["mlp_probe_h_train"] = summarize(confusion(ytr_h.numpy(), predict_mlp(mlp, Xtr_h).numpy()))
    # NOTE: scene latents are NOT index-aligned with Gaussian tokens (labels are per Gaussian token), so the
    # "latents" probe is only a sanity control and is expected to be near chance.
    maj = int(torch.bincount(yte).argmax())
    results["majority_class_test"] = summarize(confusion(yte.numpy(), np.full(len(yte), maj)))
    heur = torch.cat([scenes[i]["heuristic"] for i in args.test_scenes])
    lab_all = torch.cat([scenes[i]["labels"] for i in args.test_scenes])
    keep = (lab_all >= 0) & (heur >= 0)
    results["old_children_heuristic_test"] = summarize(confusion(lab_all[keep].numpy(), heur[keep].numpy()))

    # per test scene accuracy with the h-probe
    per_scene = {}
    for i in args.test_scenes:
        sc = scenes[i]
        keep = sc["labels"] >= 0
        pred = predict_probe(probes["h"], sc["h"])
        per_scene[sc["scene_id"]] = summarize(confusion(sc["labels"][keep].numpy(), pred[keep].numpy()))
    results["per_test_scene_probe_h"] = per_scene

    # demo exports for scene 0
    sc = scenes[0]
    pred0 = predict_probe(probes["h"], sc["h"]).numpy()
    lab0 = sc["labels"].numpy()
    xyz = sc["read_pos"].numpy()
    demo_dir = ROOT / "EXP_ScanNetPP" / "demo" / "scene_7b6477cb95" / "baseline_pretrained_6view"
    write_type_ply(demo_dir / "baseline_pretrained_6view_all4096_region_type_gt.ply", xyz, lab0)
    pred_vis = np.where(lab0 == UNKNOWN, UNKNOWN, pred0)
    write_type_ply(demo_dir / "baseline_pretrained_6view_all4096_region_type_pred.ply", xyz, pred_vis)
    write_type_ply(demo_dir / "baseline_pretrained_6view_all4096_region_type_mismatch.ply", xyz, pred_vis,
                   mismatch=(lab0 >= 0) & (pred0 != lab0))
    results["demo_scene_label_counts"] = {c: int((lab0 == i).sum()) for i, c in enumerate(CLASSES)} | {
        "unknown": int((lab0 == UNKNOWN).sum())}
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for k in ("probe_h_test", "mlp_probe_h_test", "mlp_probe_h_train", "probe_latents_test", "majority_class_test", "old_children_heuristic_test"):
        r = results[k]
        print(f"{k:32s} acc={r['accuracy']:.3f} bal_acc={r['balanced_accuracy']:.3f} recall={np.round(r['recall'], 3).tolist()} support={r['support']}")
    print("demo scene 0 probe_h:", json.dumps(per_scene[sc["scene_id"]]))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
