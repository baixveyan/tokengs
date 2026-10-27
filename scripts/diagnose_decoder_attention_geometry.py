"""D1 diagnosis: where do Gaussian tokens READ from (decoder cross-attention over scene latents) vs where
their decoded Gaussians LAND?

Each scene latent gets a "surface position" = mean 3D attention centroid of its local Stage-4 heads (layer 7).
Each Gaussian token's "read position" = decoder cross-attention weights @ latent surface positions.
We compare read position vs the token's decoded child-Gaussian centre and vs the GT surface.

    CUDA_VISIBLE_DEVICES=<idle> python scripts/diagnose_decoder_attention_geometry.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
from pathlib import Path

import torch
from einops import rearrange
from safetensors.torch import load_file

from diagnose_latent_attention_geometry import ALIVE_OPACITY, capture_cross_attention, patch_gt_points
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, frame_splits, idle_gpu
from scannetpp_evaluation_protocol import build_raycasting_scene, sha256_file
from tokengs.data.provider import Provider
from tokengs.models import model_registry
from tokengs.models.input_types import EncoderLatent, split_data
from tokengs.options import config_defaults


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-index", type=int, default=0)
    ap.add_argument("--num-scenes", type=int, default=1)
    ap.add_argument("--latent-layer", type=int, default=7)
    ap.add_argument("--output", type=Path, default=ROOT / "docs" / "DIAG_decoder_attention_geometry.json")
    args = ap.parse_args()
    run(args)


def run(args):

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
    s = float(provider.scene_scale)
    all_out = {}
    for scene_index in range(args.scene_index, args.scene_index + args.num_scenes):
        all_out[splits[scene_index]["scene_id"]] = analyse_one(
            scene_index, splits, provider, model, opt, device, s, args
        )
        torch.cuda.empty_cache()
    keys_num = [k for k, v in next(iter(all_out.values())).items() if isinstance(v, (int, float))]
    agg = {k: statistics.median(o[k] for o in all_out.values()) for k in keys_num}
    agg_heads = {
        k: statistics.median(o["heads"]["head_mean"][k] for o in all_out.values())
        for k in next(iter(all_out.values()))["heads"]["head_mean"]
    }
    args.output.write_text(json.dumps({"num_scenes": len(all_out), "median_over_scenes": agg,
                                       "median_over_scenes_head_mean": agg_heads, "per_scene": all_out},
                                      indent=2) + "\n", encoding="utf-8")
    print("median over scenes:", json.dumps(agg, indent=2))
    print("median over scenes (decoder head_mean):", json.dumps(agg_heads, indent=2))
    print(f"wrote {args.output}")


def analyse_one(scene_index, splits, provider, model, opt, device, s, args):
    spec = splits[scene_index]
    raw = provider[scene_index]
    data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
    model_input, _ = split_data(data, opt)

    dec_attn = {}
    blk = model.enc_dec_backbone.decoder_blocks[0].gs_cross_attn

    def hook(_m, inputs, _o):
        gs_tokens, keys, _values = inputs
        q = rearrange(blk.q_proj(blk.gs_token_norm(gs_tokens)), "b n (h d) -> b h n d", h=blk.num_heads)
        q = blk.q_norm(q)
        logits = torch.einsum("bhnd,bhmd->bhnm", q.float(), keys.float()) / math.sqrt(q.shape[-1])
        dec_attn["a"] = torch.softmax(logits, dim=-1)[0].cpu()  # [H, N_tok, N_lat]

    h = blk.register_forward_hook(hook)
    with torch.inference_mode():
        latents, lat_attn = capture_cross_attention(model, model_input, [args.latent_layer])
        keys, values = model.enc_dec_backbone._latents_to_kv(latents)
        gaussians = model.forward_decoder(EncoderLatent(keys=keys, values=values), model_input.decoder).cpu().float()
    h.remove()
    print(f"scene {spec['scene_id']}: decoder cross-attn captured {tuple(dec_attn['a'].shape)}")

    scene_id = spec["scene_id"]
    raw_ctx_c2w = torch.from_numpy(provider.dataset._scene_meta_cache[scene_id]["c2ws"][spec["context"]].copy())
    mesh_ply = provider.dataset.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    rc_scene = build_raycasting_scene(mesh_ply, raw_ctx_c2w, s)
    cam_in, intr_in = data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu()
    pts, valid, _ = patch_gt_points(rc_scene, cam_in, intr_in, opt.img_size, opt.patch_size)
    pts_flat, valid_flat = pts.reshape(-1, 3), valid.reshape(-1)
    valid_pts = pts_flat[valid_flat]

    # latent surface position from local heads of the chosen stage-4 layer
    a_lat = lat_attn[args.latent_layer].cpu()  # [H, N_lat, N_patch]
    pv = a_lat * valid_flat.float()
    p = pv / pv.sum(-1, keepdim=True).clamp_min(1e-12)
    cent = p @ pts_flat  # [H, N_lat, 3]
    ent = -(p.clamp_min(1e-12).log() * p).sum(-1)
    eff = ent.exp()  # [H, N_lat]
    local_heads = [hh for hh in range(eff.shape[0]) if float(eff[hh].median()) < 30]
    lat_pos = cent[local_heads].mean(0)  # [N_lat, 3]
    lat_pos_spread_m = cent[local_heads].std(0).norm(dim=-1) / s  # disagreement among local heads
    lat_to_surf_m = torch.cdist(lat_pos, valid_pts).min(1).values / s

    # decoded token centres
    g = gaussians[0].reshape(-1, 64, 14)
    w = g[..., 3].clamp_min(0) * (g[..., 3] >= ALIVE_OPACITY).float()
    tok_centre = (g[..., :3] * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-8)
    tok_to_surf_m = torch.cdist(tok_centre, valid_pts).min(1).values / s
    child_alive = g[..., 3] >= ALIVE_OPACITY
    child_to_surf_m = torch.cdist(g[..., :3].reshape(-1, 3), valid_pts).min(1).values.reshape(-1, 64) / s

    # along-ray vs lateral spread of alive children around the token centre (w.r.t. first input camera)
    cam0 = torch.linalg.inv(cam_in[0].float().T)[:3, 3]
    ray_dir = torch.nn.functional.normalize(tok_centre - cam0, dim=-1)  # [N_tok, 3]
    rel = g[..., :3] - tok_centre.unsqueeze(1)  # [N_tok, 64, 3]
    along = (rel * ray_dir.unsqueeze(1)).sum(-1)
    lateral = (rel - along.unsqueeze(-1) * ray_dir.unsqueeze(1)).norm(dim=-1)
    wa = w / w.sum(1, keepdim=True).clamp_min(1e-8)
    along_rms_m = torch.sqrt((wa * along ** 2).sum(1)) / s
    lateral_rms_m = torch.sqrt((wa * lateral ** 2).sum(1)) / s
    ratio_along_lateral = along_rms_m / lateral_rms_m.clamp_min(1e-6)

    # decoder read position
    A = dec_attn["a"]  # [H, N_tok, N_lat]
    out = {"scene_id": scene_id, "latent_layer": args.latent_layer, "local_heads_used": local_heads,
           "latent_pos_local_head_disagreement_m_median": float(lat_pos_spread_m.median()),
           "latent_pos_to_surface_m_median": float(lat_to_surf_m.median()),
           "token_centre_to_surface_m_median": float(tok_to_surf_m.median()),
           "token_centre_to_surface_m_p90": float(tok_to_surf_m.quantile(0.9)),
           "alive_child_to_surface_m_median": float(child_to_surf_m[child_alive].median()),
           "alive_child_to_surface_m_p90": float(child_to_surf_m[child_alive].quantile(0.9)),
           "children_along_ray_rms_m_median": float(along_rms_m.median()),
           "children_lateral_rms_m_median": float(lateral_rms_m.median()),
           "children_along_over_lateral_median": float(ratio_along_lateral.median()),
           "frac_tokens_along_gt_lateral": float((ratio_along_lateral > 1).float().mean()),
           "heads": {}}
    for name, a in [("head_mean", A.mean(0))] + [(f"head_{hh}", A[hh]) for hh in range(A.shape[0])]:
        read_pos = a @ lat_pos
        eff_lat = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
        read_spread = torch.sqrt((a * ((lat_pos.unsqueeze(0) - read_pos.unsqueeze(1)) ** 2).sum(-1)).sum(-1)) / s
        d_read_tok = (read_pos - tok_centre).norm(dim=-1) / s
        d_read_surf = torch.cdist(read_pos, valid_pts).min(1).values / s
        out["heads"][name] = {
            "effective_latents_median": float(eff_lat.median()),
            "read_spread_m_median": float(read_spread.median()),
            "read_pos_to_token_centre_m_median": float(d_read_tok.median()),
            "read_pos_to_token_centre_m_p90": float(d_read_tok.quantile(0.9)),
            "read_pos_to_surface_m_median": float(d_read_surf.median()),
        }
    hm = out["heads"]["head_mean"]
    print(f"[{scene_index}] {scene_id}: lat->surf={out['latent_pos_to_surface_m_median']:.3f}m "
          f"read->surf={hm['read_pos_to_surface_m_median']:.3f}m read->tok={hm['read_pos_to_token_centre_m_median']:.3f}m "
          f"eff_lat={hm['effective_latents_median']:.1f} child->surf p90={out['alive_child_to_surface_m_p90']:.3f}m "
          f"along/lateral={out['children_along_over_lateral_median']:.2f}", flush=True)
    return out


if __name__ == "__main__":
    main()
