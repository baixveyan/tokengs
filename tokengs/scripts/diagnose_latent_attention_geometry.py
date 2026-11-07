"""D0 diagnosis: does Stage-4 latent->image cross-attention of the pretrained TokenGS "look scattered" in 3D?

For each ScanNet++ validation scene (6 input views), we capture the cross-attention probabilities of the
latent bottleneck blocks (4096 scene latents x 10752 image patches, per head), raycast the GT mesh at every
8x8 patch centre to obtain the true 3D point of each patch, and measure per latent:

  * attention-weighted 3D dispersion (RMS distance to the attention centroid, metres);
  * attention mass within 10 cm / 30 cm of the centroid;
  * cross-view correspondence hit@k: top-k attended patches of view i are reprojected (via GT 3D) to view j;
    how many land in the top-k attended patches of view j (random baseline k / patches_per_view);
  * distance between the attention centroid and the decoded Gaussian cluster centre of the same token.

Only the pretrained model is read; nothing is trained. GPU is checked for idleness first.

    CUDA_VISIBLE_DEVICES=<idle> python scripts/diagnose_latent_attention_geometry.py --num-scenes 5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file

from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, frame_splits, idle_gpu
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals, sha256_file
from tokengs.data.provider import Provider
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults

ALIVE_OPACITY = 1.0 / 255.0


def capture_cross_attention(model, model_input, layers: list[int]):
    """Run encoder->latents while recording per-layer cross-attention probabilities [H, N_lat, N_patch]."""
    backbone = model.enc_dec_backbone
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for li in layers:
        block = backbone.latent_blocks[li].gs_cross_attn

        def make_hook(idx, blk):
            def hook(_module, inputs, _output):
                gs_tokens, keys, _values = inputs
                q = rearrange(blk.q_proj(blk.gs_token_norm(gs_tokens)), "b n (h d) -> b h n d", h=blk.num_heads)
                q = blk.q_norm(q)
                scale = 1.0 / math.sqrt(q.shape[-1])
                logits = torch.einsum("bhnd,bhmd->bhnm", q.float(), keys.float()) * scale
                captured[idx] = torch.softmax(logits, dim=-1)[0]  # [H, N_lat, N_patch]
            return hook

        hooks.append(block.register_forward_hook(make_hook(li, block)))
    try:
        latents = model.forward_encoder_to_scene_latents(model_input.encoder)
    finally:
        for h in hooks:
            h.remove()
    return latents, captured


def patch_gt_points(rc_scene, cam_view_input, intrinsics_input, img_size, patch):
    """GT 3D point (normalized coords) and validity for every patch centre of every input view.

    Returns pts [V, hp*wp, 3], valid [V, hp*wp], depth_gt_full [V, H, W], and per-view w2c/intrinsics.
    """
    d_gt, _n_gt, ok = raycast_gt_depth_and_normals(rc_scene, cam_view_input, intrinsics_input, img_size)
    H, W = int(img_size[0]), int(img_size[1])
    hp, wp = H // patch, W // patch
    ys = torch.arange(hp) * patch + patch // 2
    xs = torch.arange(wp) * patch + patch // 2
    V = cam_view_input.shape[0]
    pts, valid = [], []
    for v in range(V):
        w2c = cam_view_input[v].float().T
        c2w = torch.linalg.inv(w2c)
        fx, fy, cx, cy = intrinsics_input[v].float().tolist()
        d = d_gt[v][ys][:, xs]              # [hp, wp]
        m = ok[v][ys][:, xs]
        u = (xs.float() + 0.5).view(1, wp).expand(hp, wp)
        vv = (ys.float() + 0.5).view(hp, 1).expand(hp, wp)
        dirs = torch.stack([(u - cx) / fx, (vv - cy) / fy, torch.ones_like(u)], -1)
        p_cam = dirs * d.unsqueeze(-1)
        p_w = p_cam @ c2w[:3, :3].T + c2w[:3, 3]
        pts.append(p_w.reshape(-1, 3))
        valid.append(m.reshape(-1))
    return torch.stack(pts), torch.stack(valid), d_gt


def project_to_patch(pts_w, w2c, intr, img_size, patch):
    """Project normalized-world points into a view; return patch index [-1 if outside] and camera z."""
    H, W = int(img_size[0]), int(img_size[1])
    p_cam = pts_w @ w2c[:3, :3].T + w2c[:3, 3]
    z = p_cam[:, 2]
    fx, fy, cx, cy = intr.tolist()
    u = fx * p_cam[:, 0] / z.clamp_min(1e-6) + cx
    v = fy * p_cam[:, 1] / z.clamp_min(1e-6) + cy
    inside = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pi = (v // patch).long().clamp(0, H // patch - 1) * (W // patch) + (u // patch).long().clamp(0, W // patch - 1)
    pi[~inside] = -1
    return pi, z


def analyse_scene(attn_by_layer, pts, valid, gaussians, cam_view_input, intrinsics_input, d_gt, img_size,
                  patch, scene_scale, topk, heads_mode):
    V, P = pts.shape[:2]
    N_patch = V * P
    pts_flat = pts.reshape(N_patch, 3)
    valid_flat = valid.reshape(N_patch)
    view_of = torch.arange(V).repeat_interleave(P)
    hp, wp = int(img_size[0]) // patch, int(img_size[1]) // patch

    # decoded token centres (opacity-weighted over alive children)
    g = gaussians[0].reshape(-1, 64, 14)
    w = g[..., 3].clamp_min(0) * (g[..., 3] >= ALIVE_OPACITY).float()
    tok_centre = (g[..., :3] * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-8)

    # precompute GT correspondences: patch (view i) -> patch index in view j (visible) else -1
    corr = torch.full((V, N_patch), -1, dtype=torch.long)
    for j in range(V):
        w2c_j = cam_view_input[j].float().T
        pj, zj = project_to_patch(pts_flat, w2c_j, intrinsics_input[j].float(), img_size, patch)
        ys = (pj // wp).clamp_min(0) * patch + patch // 2
        xs = (pj % wp).clamp_min(0) * patch + patch // 2
        d_at = d_gt[j][ys, xs]
        vis = (pj >= 0) & valid_flat & (d_at > 0) & ((zj - d_at).abs() / d_at.clamp_min(1e-6) < 0.05)
        corr[j] = torch.where(vis & (view_of != j), pj + j * P, torch.full_like(pj, -1))

    out = {}
    for li, attn in attn_by_layer.items():
        attn = attn.cpu()
        head_sets = {"head_mean": attn.mean(0, keepdim=True)}
        if heads_mode == "all":
            head_sets.update({f"head_{h}": attn[h:h + 1] for h in range(attn.shape[0])})
        layer_out = {}
        for hname, a in head_sets.items():
            a = a[0]  # [N_lat, N_patch]
            pv = a * valid_flat.float()
            mass_valid = pv.sum(-1)
            p = pv / mass_valid.clamp_min(1e-12).unsqueeze(-1)
            centroid = p @ pts_flat
            d2 = ((pts_flat.unsqueeze(0) - centroid.unsqueeze(1)) ** 2).sum(-1)
            rms_m = torch.sqrt((p * d2).sum(-1)) / scene_scale
            dist_m = torch.sqrt(d2) / scene_scale
            within10 = (p * (dist_m < 0.10).float()).sum(-1)
            within30 = (p * (dist_m < 0.30).float()).sum(-1)
            ent = -(p.clamp_min(1e-12).log() * p).sum(-1)
            eff_patches = ent.exp()
            tok_dist_m = (centroid - tok_centre).norm(dim=-1) / scene_scale
            valid_pts = pts_flat[valid_flat]
            cen_to_surf_m = torch.cdist(centroid, valid_pts).min(dim=1).values / scene_scale
            tok_to_surf_m = torch.cdist(tok_centre, valid_pts).min(dim=1).values / scene_scale

            # cross-view correspondence hit@k
            hits, totals = 0, 0
            topk_idx_by_view = []
            for v in range(V):
                av = a[:, v * P:(v + 1) * P]
                topk_idx_by_view.append(av.topk(topk, dim=-1).indices + v * P)  # [N_lat, k]
            topk_sets = torch.cat(topk_idx_by_view, dim=1)  # [N_lat, V*k]
            for i in range(V):
                src = topk_idx_by_view[i]  # [N_lat, k]
                for j in range(V):
                    if i == j:
                        continue
                    tgt = corr[j][src]  # [N_lat, k] -> patch idx in view j or -1
                    has = tgt >= 0
                    if not has.any():
                        continue
                    in_topk = (tgt.unsqueeze(-1) == topk_idx_by_view[j].unsqueeze(1)).any(-1)
                    hits += int((in_topk & has).sum())
                    totals += int(has.sum())
            hit_rate = hits / max(totals, 1)
            layer_out[hname] = {
                "rms_dispersion_m_median": float(rms_m.median()),
                "rms_dispersion_m_p90": float(rms_m.quantile(0.9)),
                "mass_within_10cm_median": float(within10.median()),
                "mass_within_30cm_median": float(within30.median()),
                "effective_patches_median": float(eff_patches.median()),
                "centroid_to_token_gaussians_m_median": float(tok_dist_m.median()),
                "attn_centroid_to_nearest_surface_m_median": float(cen_to_surf_m.median()),
                "token_gaussians_to_nearest_surface_m_median": float(tok_to_surf_m.median()),
                "attn_mass_on_valid_patches_mean": float(mass_valid.mean()),
                f"corr_hit_at_{topk}": hit_rate,
                f"corr_hit_random_baseline": topk / P,
                "corr_pairs_evaluated": totals,
            }
        out[f"layer_{li}"] = layer_out
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-scenes", type=int, default=5)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 3, 7, 9, 10, 11])
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--heads", choices=["mean", "all"], default="mean")
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "DIAG_latent_attention_geometry.json")
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
    print(f"GPU {gpu['index']} ({gpu['uuid']}): diagnosing {args.num_scenes} scene(s), layers {args.layers}")

    per_scene = {}
    for idx in range(args.num_scenes):
        spec = splits[idx]
        raw = provider[idx]
        data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
        if torch.is_tensor(raw["frame_ids"]) and raw["frame_ids"].tolist() != spec["context"] + spec["target"]:
            raise RuntimeError("frame identity mismatch")
        model_input, _ = split_data(data, opt)
        with torch.inference_mode():
            latents, attn = capture_cross_attention(model, model_input, args.layers)
            keys, values = model.enc_dec_backbone._latents_to_kv(latents)
            from tokengs.models.input_types import EncoderLatent
            gaussians = model.forward_decoder(EncoderLatent(keys=keys, values=values), model_input.decoder)
        scene_id = spec["scene_id"]
        raw_ctx_c2w = torch.from_numpy(provider.dataset._scene_meta_cache[scene_id]["c2ws"][spec["context"]].copy())
        mesh_ply = provider.dataset.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
        rc_scene = build_raycasting_scene(mesh_ply, raw_ctx_c2w, float(provider.scene_scale))
        cam_in = data["cam_view_input"][0].cpu()
        intr_in = data["intrinsics_input"][0].cpu()
        pts, valid, d_gt = patch_gt_points(rc_scene, cam_in, intr_in, opt.img_size, opt.patch_size)
        res = analyse_scene(attn, pts, valid, gaussians.cpu().float(), cam_in, intr_in, d_gt, opt.img_size,
                            opt.patch_size, float(provider.scene_scale), args.topk, args.heads)
        res["valid_patch_fraction"] = float(valid.float().mean())
        per_scene[scene_id] = res
        hm = res[f"layer_{args.layers[-1]}"]["head_mean"]
        print(f"[{idx + 1}/{args.num_scenes}] {scene_id}: last-layer rms_disp={hm['rms_dispersion_m_median']:.3f}m "
              f"within30cm={hm['mass_within_30cm_median']:.3f} eff_patches={hm['effective_patches_median']:.1f} "
              f"hit@{args.topk}={hm[f'corr_hit_at_{args.topk}']:.3f} (rand {hm['corr_hit_random_baseline']:.4f}) "
              f"centroid->token={hm['centroid_to_token_gaussians_m_median']:.3f}m", flush=True)
        del attn
        torch.cuda.empty_cache()

    # aggregate over scenes (mean of per-scene medians)
    agg = {}
    for layer_key in per_scene[next(iter(per_scene))]:
        if not layer_key.startswith("layer_"):
            continue
        agg[layer_key] = {}
        for hname in per_scene[next(iter(per_scene))][layer_key]:
            agg[layer_key][hname] = {
                m: statistics.mean(per_scene[s][layer_key][hname][m] for s in per_scene)
                for m in per_scene[next(iter(per_scene))][layer_key][hname]
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "checkpoint_sha256": CHECKPOINT_SHA, "num_scenes": args.num_scenes, "layers": args.layers,
        "topk": args.topk, "scene_scale": float(provider.scene_scale), "aggregate": agg, "per_scene": per_scene,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(agg, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
