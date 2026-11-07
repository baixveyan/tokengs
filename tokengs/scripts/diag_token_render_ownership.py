"""Does a GS token render where it attends? (no training; frozen model, 50 validation windows, input views)

Per GS token:
  attention region : token -> input-patch attention T = A_dec (decoder cross-attn, head mean) @ A_lat (layer-7
                     local-head mean) -- exactly the definition used by probe_gs_token_*.py
  render region    : per input-view pixel, the alpha-compositing weight (w_i = alpha_i * T_i) summed over the
                     token's 64 Gaussians, obtained exactly by rasterising one-hot token-membership channels with the
                     model's own renderer settings (gsplat, packed=False, same near/far), background 0
Compared per token: share of the render weight inside the attention-90% patch set (and the reverse), dominant
view agreement, 3D distance between the laser-surface centroids of the two regions.
Per Gaussian (laser surface only): distance to the full GT mesh, and the share of rendered image weight that comes
from Gaussians at <3 / 3-10 / 10-30 / >30 cm from the mesh (exact, by rasterising one-hot distance-bucket channels).

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/diag_token_render_ownership.py extract --shard k --num-shards N
  PYTHONPATH=scripts:. python scripts/diag_token_render_ownership.py summarize
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from gsplat.rendering import rasterization

from diagnose_latent_attention_geometry import capture_cross_attention
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, idle_gpu
from probe_gs_token_geometry import decode, load_model
from probe_scene_latent_limit import S, make_opt, patch_geometry
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals, sha256_file
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data

CACHE = Path("/mnt/disk/xveyan/workspace_scannetpp/diag_token_render_ownership")
OUT_JSON = ROOT / "docs" / "DIAG_token_render_ownership.json"
CHILDREN, CHUNK = 64, 512
DIST_EDGES_M = (0.03, 0.10, 0.30)
DIST_NAMES = ("<3cm", "3-10cm", "10-30cm", ">30cm")


def rasterize_channels(g, colors, viewmat, K, H, W, opt):
    """Alpha-composited per-pixel sum of `colors` [N, C] for one view (background 0), model renderer settings."""
    img, alpha, _ = rasterization(
        means=g[:, 0:3].contiguous(), quats=g[:, 7:11].contiguous(), scales=g[:, 4:7].contiguous(),
        opacities=g[:, 3].contiguous(), colors=colors, viewmats=viewmat[None], Ks=K[None], width=W, height=H,
        near_plane=float(opt.znear), far_plane=float(opt.zfar), packed=False,
        backgrounds=torch.zeros((1, colors.shape[-1]), device=g.device), render_mode="RGB",
    )
    return img[0], alpha[0, ..., 0]


def top_mass_set(w, frac=0.9):
    """Boolean mask of the smallest set of entries holding `frac` of each row's mass (row-wise)."""
    order = w.argsort(-1, descending=True)
    sw = w.gather(-1, order)
    cum = sw.cumsum(-1) / sw.sum(-1, keepdim=True).clamp_min(1e-12)
    keep_sorted = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool), cum[:, :-1] < frac], 1)
    return torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)


def cmd_extract(args):
    import open3d as o3d

    gpu = idle_gpu()
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected checkpoint SHA256")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    opt = make_opt("val")
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov.dataset._offset_frac = None  # official validation windows
    model = load_model(CHECKPOINT, opt, device)
    H, W = map(int, opt.img_size)
    patch = int(opt.patch_size)
    hp, wp = H // patch, W // patch
    mine = list(range(len(prov)))[args.shard::args.num_shards][: args.limit or None]
    print(f"GPU {gpu['index']}: shard {args.shard}/{args.num_shards}: {len(mine)} windows", flush=True)
    out, t0 = [], time.time()
    for n, idx in enumerate(mine):
        ds = prov.dataset
        sid = Path(ds.sample_list[idx]).name
        context, _ = ds.get_context_target_frames(idx)
        raw = prov[idx]
        data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
        mi, _ = split_data(data, opt)
        with torch.inference_mode():
            latents, attn = capture_cross_attention(model, mi, [7])
        a = attn[7]
        eff = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
        a_lat = a[[h for h in range(a.shape[0]) if float(eff[h].median()) < 30]].mean(0)
        A_dec, _h, g64 = decode(model, latents, mi)
        T = (A_dec @ a_lat).float()  # [Ntok, V*P]
        g = g64.reshape(-1, 14).contiguous()
        ntok = g64.shape[0]
        cam_in, intr_in = data["cam_view_input"][0], data["intrinsics_input"][0]
        V = cam_in.shape[0]
        raw_c2w = torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][context].copy())
        rc = build_raycasting_scene(ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
        pts, _nrm, pval, _dirs, _cams = patch_geometry(rc, cam_in.cpu(), intr_in.cpu(), opt.img_size, patch)
        _d, _n, ok_full = raycast_gt_depth_and_normals(rc, cam_in.cpu(), intr_in.cpu(), opt.img_size)
        dist_m = torch.from_numpy(rc.compute_distance(o3d.core.Tensor(g[:, :3].cpu().numpy().astype(np.float32))).numpy()) / S
        bucket = torch.bucketize(dist_m, torch.tensor(DIST_EDGES_M))
        token_id = torch.arange(g.shape[0], device=device) // CHILDREN

        R = torch.zeros((ntok, V * hp * wp), device=device)
        bucket_mass = torch.zeros((V, 4))
        bucket_mass_gt = torch.zeros((V, 4))
        alpha_check = []
        with torch.inference_mode():
            onehot_b = torch.nn.functional.one_hot(bucket.to(device), 4).float()
            for v in range(V):
                viewmat = cam_in[v].float().T.contiguous()
                fx, fy, cx, cy = intr_in[v].float().tolist()
                K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=device)
                bimg, alpha = rasterize_channels(g, onehot_b, viewmat, K, H, W, opt)
                gt_ok = ok_full[v].to(device)
                bucket_mass[v] = bimg.reshape(-1, 4).sum(0).cpu()
                bucket_mass_gt[v] = bimg[gt_ok].sum(0).cpu()
                tot = torch.zeros((H, W), device=device)
                for c0 in range(0, ntok, CHUNK):
                    k = min(CHUNK, ntok - c0)
                    col = torch.zeros((g.shape[0], k), device=device)
                    sel = (token_id >= c0) & (token_id < c0 + k)
                    col[sel, token_id[sel] - c0] = 1.0
                    img, _ = rasterize_channels(g, col, viewmat, K, H, W, opt)
                    tot += img.sum(-1)
                    pm = img.reshape(hp, patch, wp, patch, k).sum((1, 3)).reshape(hp * wp, k)
                    R[c0:c0 + k, v * hp * wp:(v + 1) * hp * wp] = pm.T
                alpha_check.append(float((tot - alpha).abs().max()))
        Tc, Rc = T.cpu(), R.cpu()
        out.append({
            "scene_id": sid, "T": Tc.half(), "R": Rc.half(), "pts": pts.half(), "pval": pval,
            "opacity": g[:, 3].cpu().half(), "dist_m": dist_m.half(),
            "bucket_mass": bucket_mass, "bucket_mass_gt": bucket_mass_gt, "alpha_check_max": max(alpha_check),
            "V": V, "P": hp * wp,
        })
        torch.cuda.empty_cache()
        print(f"[shard {args.shard}] {n + 1}/{len(mine)} {sid} {time.time() - t0:.0f}s "
              f"(sum of token weights vs rendered alpha: max |diff| {max(alpha_check):.2e})", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    name = f"shard_{args.shard}" if not args.limit else f"smoke_{args.shard}"
    tmp = CACHE / f"{name}.tmp"
    torch.save(out, tmp)
    tmp.rename(CACHE / f"{name}.pt")
    print("wrote", CACHE / f"{name}.pt")


def q(x, ps=(0.1, 0.5, 0.9)):
    x = x[torch.isfinite(x)].float()
    return {f"p{int(p * 100)}": float(x.quantile(p)) for p in ps} | {"mean": float(x.mean())} if x.numel() else {}


def cmd_summarize(args):
    recs = []
    for p in sorted(CACHE.glob("shard_*.pt")):
        recs.extend(torch.load(p, map_location="cpu", weights_only=False))
    if len(recs) != 50 or len({r["scene_id"] for r in recs}) != 50:
        raise RuntimeError(f"expected 50 validation windows, got {len(recs)}")
    acc = {k: [] for k in ("mass", "r_in_a90", "a_in_r90", "same_view", "centroid_cm", "n_att_views")}
    bm, bmg, child = torch.zeros(4), torch.zeros(4), {"dist_all": [], "dist_op05": [], "op": []}
    for r in recs:
        T, R = r["T"].float(), r["R"].float()
        V, P = r["V"], r["P"]
        mass = R.sum(-1)
        a90, r90 = top_mass_set(T), top_mass_set(R)
        acc["mass"].append(mass)
        acc["r_in_a90"].append((R * a90).sum(-1) / mass.clamp_min(1e-12))
        acc["a_in_r90"].append((T * r90).sum(-1) / T.sum(-1).clamp_min(1e-12))
        acc["same_view"].append((T.reshape(-1, V, P).sum(-1).argmax(-1) == R.reshape(-1, V, P).sum(-1).argmax(-1)).float())
        acc["n_att_views"].append((T.reshape(-1, V, P).sum(-1) > 0.1).sum(-1).float())
        pts, ok = r["pts"].float(), r["pval"].float()
        ca = (T * ok) @ pts / (T * ok).sum(-1, keepdim=True).clamp_min(1e-12)
        cr = (R * ok) @ pts / (R * ok).sum(-1, keepdim=True).clamp_min(1e-12)
        valid_c = ((T * ok).sum(-1) > 0.5 * T.sum(-1)) & ((R * ok).sum(-1) > 0.5 * mass) & (mass > 0)
        acc["centroid_cm"].append(torch.where(valid_c, (ca - cr).norm(dim=-1) / S * 100, torch.nan))
        bm += r["bucket_mass"].sum(0)
        bmg += r["bucket_mass_gt"].sum(0)
        op, d = r["opacity"].float(), r["dist_m"].float()
        child["dist_all"].append(d)
        child["dist_op05"].append(d[op >= 0.05])
        child["op"].append(op)
    cat = {k: torch.cat(v) for k, v in acc.items()}
    mass = cat["mass"]
    live = mass >= 1.0  # tokens whose Gaussians contribute at least one pixel-equivalent of weight
    w = mass[live]
    res = {
        "definitions": {
            "render weight": "per input-view pixel alpha-compositing weight of the token's 64 Gaussians, summed over pixels "
                             "(1.0 = one fully owned pixel); 6 input views, official validation windows",
            "attention": "A_dec @ A_lat (layer 7 local heads), as in probe_gs_token_*.py",
            "r_in_a90": "share of a token's render weight that falls in its attention-90% patch set",
            "a_in_r90": "share of a token's attention that falls in its render-90% patch set",
            "centroid_cm": "distance between the laser-surface centroids of the attention and render regions (patch "
                           "centres with a GT mesh hit; tokens with >50% of both masses on valid patches)",
            "live token": "render weight >= 1 pixel",
        },
        "tokens_total": int(mass.numel()),
        "live_token_frac": float(live.float().mean()),
        "render_weight_per_token": q(mass),
        "render_weight_share_top10pct_tokens": float(mass.sort(descending=True).values[: mass.numel() // 10].sum() / mass.sum()),
        "r_in_a90": q(cat["r_in_a90"][live]),
        "r_in_a90_weighted_by_render": float((cat["r_in_a90"][live] * w).sum() / w.sum()),
        "a_in_r90": q(cat["a_in_r90"][live]),
        "same_dominant_view_frac": float(cat["same_view"][live].mean()),
        "same_dominant_view_frac_weighted": float((cat["same_view"][live] * w).sum() / w.sum()),
        "centroid_cm": q(cat["centroid_cm"][live]),
        "centroid_cm_by_attention_views": {
            nv: q(cat["centroid_cm"][live & (cat["n_att_views"] == nv)]) for nv in (1.0, 2.0, 3.0)
        } | {"4+": q(cat["centroid_cm"][live & (cat["n_att_views"] >= 4)])},
        "rendered_weight_share_by_gaussian_distance_to_mesh": dict(zip(DIST_NAMES, (bm / bm.sum()).tolist())),
        "rendered_weight_share_by_gaussian_distance_to_mesh_gt_pixels": dict(zip(DIST_NAMES, (bmg / bmg.sum()).tolist())),
        "gaussian_distance_to_mesh_cm_all": {k: v * 100 for k, v in q(torch.cat(child["dist_all"]), (0.5, 0.9)).items()},
        "gaussian_distance_to_mesh_cm_opacity_ge_0.05": {k: v * 100 for k, v in q(torch.cat(child["dist_op05"]), (0.5, 0.9)).items()},
        "alpha_check_max_abs_diff": max(r["alpha_check_max"] for r in recs),
    }
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1)
    e.add_argument("--limit", type=int, default=0, help="smoke test: only the first N windows of the shard")
    sub.add_parser("summarize")
    a = ap.parse_args()
    cmd_extract(a) if a.cmd == "extract" else cmd_summarize(a)


if __name__ == "__main__":
    main()
