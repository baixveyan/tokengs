"""Export pure-RGB attention-ray triangulated points and GT-surface attention points for the demo scene
`scene_7b6477cb95` (both the 20 highlighted tokens and all 4,096 tokens) in the exact same coordinate frame
and 20-colour palette as `baseline_pretrained_6view_only20tokens.ply` and `gt_lidar_mesh_frustum.ply`.

Outputs in `EXP_ScanNetPP/demo/scene_7b6477cb95/baseline_pretrained_6view/`:
  * `baseline_pretrained_6view_only20tokens_triangulated_anchors.ply`
      20 pure-RGB ray-triangulated anchor Gaussians (1 per selected token, same 20 colours, 0 GT used).
  * `baseline_pretrained_6view_only20tokens_gt_surface_anchors.ply`
      20 GT-mesh-projected attention surface points (where the token's attended patches actually lie on the mesh).
  * `baseline_pretrained_6view_only20tokens_with_triangulated_and_gt.ply`
      Combined PLY: original 20-token child Gaussians (`point_role=0`) + 20 triangulated anchors (`point_role=1`,
      rendered as a compact 7-point 3D cross Marker per token so it is immediately visible in Points/Splat viewers)
      + 20 GT surface anchors (`point_role=2`).
  * `baseline_pretrained_6view_all4096_triangulated_anchors.ply`
      All 4,096 pure-RGB triangulated anchors across the scene.
  * `EXP_ScanNetPP/demo/scene_7b6477cb95/triangulated_20tokens_report.md` and `.json`
      Per-token table comparing (a) pure-RGB triangulated point, (b) decoded 64-child centre, and (c) GT surface point.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from plyfile import PlyData, PlyElement
from safetensors.torch import load_file

from diagnose_latent_attention_geometry import ALIVE_OPACITY, capture_cross_attention, patch_gt_points
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, frame_splits, idle_gpu
from highlight_selected_tokens_ply import PALETTE, SH_C0, hex_rgb
from scannetpp_evaluation_protocol import build_raycasting_scene, sha256_file
from tokengs.data.provider import Provider
from tokengs.models import model_registry
from tokengs.models.input_types import EncoderLatent, split_data
from tokengs.options import config_defaults


def triangulate_rays_per_latent_head(
    a_lat_loc: torch.Tensor,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    wp: int,
    scene_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-RGB multi-view ray triangulation for each (head, latent) -> then consensus over local heads.

    Args:
        a_lat_loc: [H_loc, N_lat, V*P] attention probabilities of local Stage-4 heads.
        origins:   [V, 3] camera centres in normalized scene coordinates.
        dirs:      [V, P, 3] unit ray directions for every patch centre in normalized scene coordinates.
        wp:        number of patches along image width.
        scene_scale: 0.3 (normalized / metres).

    Returns:
        lat_tri_pt: [N_lat, 3] triangulated 3D position per scene latent (normalized coords).
        lat_skew_m: [N_lat] median ray-pair skew distance in metres.
        lat_eff_v:  [N_lat] effective number of views attended.
    """
    H_loc, N_lat, VP = a_lat_loc.shape
    V, P, _ = dirs.shape
    assert VP == V * P
    a = a_lat_loc.reshape(H_loc * N_lat, V, P)
    view_mass = a.sum(-1)  # [H_loc*N_lat, V]
    top_p = a.argmax(-1)   # [H_loc*N_lat, V]

    p_coords = torch.stack([torch.arange(P) // wp, torch.arange(P) % wp], dim=-1).float()  # [P, 2]
    top_coords = p_coords[top_p]  # [M, V, 2]
    dist2d = (p_coords.view(1, 1, P, 2) - top_coords.unsqueeze(2)).norm(dim=-1)
    win_w = a * (dist2d <= 2.0).float()
    win_w = win_w / win_w.sum(-1, keepdim=True).clamp_min(1e-12)
    ray_dirs = torch.nn.functional.normalize((win_w.unsqueeze(-1) * dirs.unsqueeze(0)).sum(2), dim=-1)  # [M, V, 3]

    # Collect all 15 cross-view ray pairs per head
    pts_list, w_list, skew_list = [], [], []
    for v1 in range(V):
        for v2 in range(v1 + 1, V):
            d1 = ray_dirs[:, v1]
            d2 = ray_dirs[:, v2]
            o1 = origins[v1].unsqueeze(0)
            o2 = origins[v2].unsqueeze(0)
            w0 = o1 - o2
            b = (d1 * d2).sum(-1).clamp(-0.9999, 0.9999)
            denom = (1.0 - b * b).clamp_min(1e-6)
            c1 = (d1 * w0).sum(-1)
            c2 = (d2 * w0).sum(-1)
            t1 = (b * c2 - c1) / denom
            t2 = (c2 - b * c1) / denom
            p1 = o1 + t1.unsqueeze(-1) * d1
            p2 = o2 + t2.unsqueeze(-1) * d2
            mid = 0.5 * (p1 + p2)
            skew_m = (p1 - p2).norm(dim=-1) / scene_scale
            angle_deg = torch.acos(b.abs()) * (180.0 / math.pi)
            valid_pair = (t1 > 0.05) & (t2 > 0.05) & (t1 < 4.0) & (t2 < 4.0) & (angle_deg > 2.5)
            w_pair = (
                (view_mass[:, v1] * view_mass[:, v2])
                * torch.exp(-skew_m / 0.06)
                * torch.sin(angle_deg * math.pi / 180.0)
            )
            w_pair = torch.where(valid_pair, w_pair, torch.zeros_like(w_pair))
            pts_list.append(mid)
            w_list.append(w_pair)
            skew_list.append(torch.where(valid_pair, skew_m, torch.full_like(skew_m, 999.0)))

    # Across all H_loc heads * 15 pairs = 180 candidate intersections per latent
    all_pts = torch.stack(pts_list, dim=1).reshape(H_loc, N_lat, -1, 3).permute(1, 0, 2, 3).reshape(N_lat, -1, 3)
    all_w = torch.stack(w_list, dim=1).reshape(H_loc, N_lat, -1).permute(1, 0, 2).reshape(N_lat, -1)
    all_skew = torch.stack(skew_list, dim=1).reshape(H_loc, N_lat, -1).permute(1, 0, 2).reshape(N_lat, -1)

    best_idx = all_w.argmax(dim=1)
    best_pt = all_pts[torch.arange(N_lat), best_idx]
    inlier_w = ((all_pts - best_pt.unsqueeze(1)).norm(dim=-1) / scene_scale < 0.12).float() * all_w
    lat_tri_pt = (inlier_w.unsqueeze(-1) * all_pts).sum(1) / inlier_w.sum(1, keepdim=True).clamp_min(1e-12)
    lat_skew_m = all_skew.min(dim=1).values
    vm_mean = view_mass.reshape(H_loc, N_lat, V).mean(0)
    lat_eff_v = (-(vm_mean.clamp_min(1e-12).log() * vm_mean).sum(-1)).exp()
    return lat_tri_pt, lat_skew_m, lat_eff_v


def make_anchor_vertices(
    positions: np.ndarray,
    tokens: np.ndarray,
    ranks: np.ndarray,
    point_role: int,
    radius_norm: float = 0.0045,
    cross_arms: bool = False,
    colour_override: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Build PLY vertex records compatible with `baseline_pretrained_6view_only20tokens.ply` plus `point_role`."""
    offsets = [np.zeros(3, dtype=np.float32)]
    if cross_arms:
        step = radius_norm * 2.2
        for axis in range(3):
            for sign in (-1.0, 1.0):
                d = np.zeros(3, dtype=np.float32)
                d[axis] = sign * step
                offsets.append(d)
    offsets_arr = np.stack(offsets, axis=0)  # [K, 3]
    K = offsets_arr.shape[0]
    N = positions.shape[0]

    dtype = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
        ("slot_id", "<f4"), ("child_id", "<f4"), ("opacity_activated", "<f4"),
        ("aniso", "<f4"), ("cos_long_axis_ray", "<f4"), ("parallax_deg", "<f4"),
        ("in_input_frustums", "<f4"), ("selected_token_rank", "<f4"),
        ("point_role", "<f4"),
    ]
    out = np.zeros(N * K, dtype=dtype)
    pos_rep = (positions[:, None, :] + offsets_arr[None, :, :]).reshape(-1, 3)
    out["x"], out["y"], out["z"] = pos_rep[:, 0], pos_rep[:, 1], pos_rep[:, 2]

    for i in range(N):
        sl = slice(i * K, (i + 1) * K)
        if colour_override is not None:
            rgb = np.array(colour_override, dtype=np.float32)
        else:
            rgb = hex_rgb(PALETTE[int(ranks[i]) % len(PALETTE)][1])
        for c in range(3):
            out[f"f_dc_{c}"][sl] = (rgb[c] - 0.5) / SH_C0
        out["slot_id"][sl] = float(tokens[i])
        out["child_id"][sl] = float(-point_role)
        out["selected_token_rank"][sl] = float(ranks[i])
        out["point_role"][sl] = float(point_role)

    out["opacity_activated"] = 0.99
    out["opacity"] = float(np.log(0.99 / 0.01))
    log_s = float(np.log(radius_norm))
    out["scale_0"] = log_s
    out["scale_1"] = log_s
    out["scale_2"] = log_s
    out["rot_0"] = 1.0
    out["aniso"] = 1.0
    out["in_input_frustums"] = 1.0
    return out


def main() -> None:
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

    spec = splits[0]
    raw = provider[0]
    data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
    model_input, _ = split_data(data, opt)

    dec_attn = {}
    blk = model.enc_dec_backbone.decoder_blocks[0].gs_cross_attn

    def hook(_m, inputs, _o):
        gs_tokens, keys, _values = inputs
        q = rearrange(blk.q_proj(blk.gs_token_norm(gs_tokens)), "b n (h d) -> b h n d", h=blk.num_heads)
        q = blk.q_norm(q)
        logits = torch.einsum("bhnd,bhmd->bhnm", q.float(), keys.float()) / math.sqrt(q.shape[-1])
        dec_attn["a"] = torch.softmax(logits, dim=-1)[0].cpu()

    h = blk.register_forward_hook(hook)
    with torch.inference_mode():
        latents, lat_attn = capture_cross_attention(model, model_input, [7])
        keys, values = model.enc_dec_backbone._latents_to_kv(latents)
        gaussians = model.forward_decoder(EncoderLatent(keys=keys, values=values), model_input.decoder).cpu().float()
    h.remove()

    cam_in = data["cam_view_input"][0].cpu().float()
    intr_in = data["intrinsics_input"][0].cpu().float()
    H, W = int(opt.img_size[0]), int(opt.img_size[1])
    patch = opt.patch_size
    hp, wp = H // patch, W // patch
    P = hp * wp
    V = cam_in.shape[0]
    ys = (torch.arange(hp) * patch + patch // 2).float() + 0.5
    xs = (torch.arange(wp) * patch + patch // 2).float() + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    origins, dirs = [], []
    for v in range(V):
        w2c = cam_in[v].T
        c2w = torch.linalg.inv(w2c)
        fx, fy, cx, cy = intr_in[v].tolist()
        d_cam = torch.stack([(xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)], dim=-1).reshape(P, 3)
        d_w = torch.nn.functional.normalize(d_cam @ c2w[:3, :3].T, dim=-1)
        origins.append(c2w[:3, 3])
        dirs.append(d_w)
    origins = torch.stack(origins)
    dirs = torch.stack(dirs)

    a_lat = lat_attn[7].cpu()
    ent = (-(a_lat.clamp_min(1e-12).log() * a_lat).sum(-1)).exp()
    local_heads = [hh for hh in range(16) if float(ent[hh].median()) < 30]
    a_lat_loc = a_lat[local_heads]  # [H_loc, 4096, V*P]

    lat_tri_pt, lat_skew_m, lat_eff_v = triangulate_rays_per_latent_head(a_lat_loc, origins, dirs, wp, s)
    A_dec = dec_attn["a"]  # [16, 4096, 4096]
    # Per decoder head read position, then robust consensus across the 16 decoder heads
    head_tri_pts = torch.einsum("htl,ld->thd", A_dec, lat_tri_pt)  # [4096, 16, 3]
    d_pair = torch.cdist(head_tri_pts, head_tri_pts) / s           # [4096, 16, 16] metres
    inlier_counts = (d_pair < 0.08).float().sum(-1)                # [4096, 16]
    best_h = inlier_counts.argmax(dim=1)                           # [4096]
    best_h_pt = head_tri_pts[torch.arange(4096), best_h]           # [4096, 3]
    h_inlier_w = ((head_tri_pts - best_h_pt.unsqueeze(1)).norm(dim=-1) / s < 0.08).float()
    tok_tri_pt = (h_inlier_w.unsqueeze(-1) * head_tri_pts).sum(1) / h_inlier_w.sum(1, keepdim=True).clamp_min(1.0)

    # GT surface points
    scene_id = spec["scene_id"]
    raw_ctx_c2w = torch.from_numpy(provider.dataset._scene_meta_cache[scene_id]["c2ws"][spec["context"]].copy())
    mesh_ply = provider.dataset.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    rc_scene = build_raycasting_scene(mesh_ply, raw_ctx_c2w, s)
    pts, valid, _ = patch_gt_points(rc_scene, cam_in, intr_in, opt.img_size, opt.patch_size)
    pts_flat, valid_flat = pts.reshape(-1, 3), valid.reshape(-1)
    valid_pts = pts_flat[valid_flat]

    pv = a_lat_loc * valid_flat.float()
    p = pv / pv.sum(-1, keepdim=True).clamp_min(1e-12)
    lat_gt_pos = (p @ pts_flat).mean(0)
    head_gt_pts = torch.einsum("htl,ld->thd", A_dec, lat_gt_pos)
    tok_gt_read_pos = (h_inlier_w.unsqueeze(-1) * head_gt_pts).sum(1) / h_inlier_w.sum(1, keepdim=True).clamp_min(1.0)

    g = gaussians[0].reshape(-1, 64, 14)
    w = g[..., 3].clamp_min(0) * (g[..., 3] >= ALIVE_OPACITY).float()
    tok_centre = (g[..., :3] * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-8)

    tri_to_surf_m = torch.cdist(tok_tri_pt, valid_pts).min(1).values / s
    tri_to_gt_read_m = (tok_tri_pt - tok_gt_read_pos).norm(dim=-1) / s
    gt_read_to_surf_m = torch.cdist(tok_gt_read_pos, valid_pts).min(1).values / s
    tok_centre_to_surf_m = torch.cdist(tok_centre, valid_pts).min(1).values / s

    demo_scene_dir = ROOT / "EXP_ScanNetPP" / "demo" / "scene_7b6477cb95"
    method_dir = demo_scene_dir / "baseline_pretrained_6view"
    hi_meta = json.loads((demo_scene_dir / "highlighted_tokens.json").read_text(encoding="utf-8"))
    tokens_20 = np.array([int(item["token"]) for item in hi_meta["tokens"]], dtype=np.int64)
    ranks_20 = np.arange(len(tokens_20), dtype=np.int64)

    tri_20_np = tok_tri_pt[tokens_20].numpy()
    gt_20_np = tok_gt_read_pos[tokens_20].numpy()

    # 1. Only 20 triangulated anchors
    v_tri_20 = make_anchor_vertices(tri_20_np, tokens_20, ranks_20, point_role=1, radius_norm=0.006, cross_arms=False)
    PlyData([PlyElement.describe(v_tri_20, "vertex")]).write(
        str(method_dir / "baseline_pretrained_6view_only20tokens_triangulated_anchors.ply")
    )

    # 2. Only 20 GT surface anchors
    v_gt_20 = make_anchor_vertices(gt_20_np, tokens_20, ranks_20, point_role=2, radius_norm=0.006, cross_arms=False)
    PlyData([PlyElement.describe(v_gt_20, "vertex")]).write(
        str(method_dir / "baseline_pretrained_6view_only20tokens_gt_surface_anchors.ply")
    )

    # 3. Combined: original 20 tokens' child Gaussians + 20 triangulated cross-markers + 20 GT surface anchors
    orig_v = PlyData.read(str(method_dir / "baseline_pretrained_6view_only20tokens.ply"))["vertex"].data
    combined_orig = np.zeros(orig_v.shape[0], dtype=v_tri_20.dtype)
    for name in orig_v.dtype.names:
        combined_orig[name] = orig_v[name]
    combined_orig["point_role"] = 0.0
    v_tri_cross = make_anchor_vertices(tri_20_np, tokens_20, ranks_20, point_role=1, radius_norm=0.0045, cross_arms=True)
    v_gt_cross = make_anchor_vertices(
        gt_20_np, tokens_20, ranks_20, point_role=2, radius_norm=0.0045, cross_arms=False, colour_override=(1.0, 1.0, 1.0)
    )
    combined_all = np.concatenate([combined_orig, v_tri_cross, v_gt_cross], axis=0)
    PlyData([PlyElement.describe(combined_all, "vertex")]).write(
        str(method_dir / "baseline_pretrained_6view_only20tokens_with_triangulated_and_gt.ply")
    )

    # 4. All 4096 triangulated anchors
    all_ids = np.arange(4096, dtype=np.int64)
    v_tri_all = make_anchor_vertices(
        tok_tri_pt.numpy(), all_ids, all_ids % 20, point_role=1, radius_norm=0.003, cross_arms=False
    )
    PlyData([PlyElement.describe(v_tri_all, "vertex")]).write(
        str(method_dir / "baseline_pretrained_6view_all4096_triangulated_anchors.ply")
    )

    # Build report & JSON
    rows = []
    lines = [
        "# `scene_7b6477cb95` 20 个高亮 Token 的纯相机射线三角化点 vs 真实表面点对比",
        "",
        "## 生成的 PLY 文件（均位于 `EXP_ScanNetPP/demo/scene_7b6477cb95/baseline_pretrained_6view/`，与 `gt_lidar_mesh_frustum.ply` 及 `baseline_pretrained_6view_only20tokens.ply` 坐标系、颜色 100% 对齐）",
        "",
        "1. `baseline_pretrained_6view_only20tokens_triangulated_anchors.ply`：仅包含 **20 个纯 RGB 注意力射线三角化锚点**（0 点云、0 真值深度，与 20 个 token 一一同色）。",
        "2. `baseline_pretrained_6view_only20tokens_gt_surface_anchors.ply`：仅包含 **20 个注意力盯的真实网格表面点**（同色，代表该 token 眼睛看到的真墙面位置）。",
        "3. `baseline_pretrained_6view_only20tokens_with_triangulated_and_gt.ply`：**三合一叠加文件**（包含原 20 个 token 的 1,280 个子球 `point_role=0` + 同色十字标记的纯 RGB 三角化点 `point_role=1` + 纯白色标记的真表面点 `point_role=2`；在点云软件中可按 `point_role` 或颜色直接对比）。",
        "4. `baseline_pretrained_6view_all4096_triangulated_anchors.ply`：全场景 **4,096 个 token 的纯 RGB 三角化点**。",
        "",
        "## 全场景 4,096 个 Token 汇总统计",
        "",
        f"- **注意力真表面读取点 → 真墙面中位距离**：`{float(gt_read_to_surf_m.median()) * 100:.2f} cm`（p90 `{float(gt_read_to_surf_m.quantile(0.9)) * 100:.2f} cm`）",
        f"- **纯 RGB 射线三角化点（0 真值）→ 真墙面中位距离**：`{float(tri_to_surf_m.median()) * 100:.2f} cm`（p75 `{float(tri_to_surf_m.quantile(0.75)) * 100:.2f} cm`，p90 `{float(tri_to_surf_m.quantile(0.9)) * 100:.2f} cm`）",
        f"- **纯 RGB 射线三角化点 → 注意力真表面读取点中位距离**：`{float(tri_to_gt_read_m.median()) * 100:.2f} cm`",
        f"- **原 `deconv` 64 子球中心 → 真墙面中位距离**：`{float(tok_centre_to_surf_m.median()) * 100:.2f} cm`（p90 `{float(tok_centre_to_surf_m.quantile(0.9)) * 100:.2f} cm`）",
        "",
        "## 20 个高亮 Token 逐一对比（单位：厘米 cm）",
        "",
        "| 序号 | token | 颜色 | 注意力真表面点→真墙面 (cm) | 纯RGB三角化点→真墙面 (cm) | 纯RGB三角化点→注意力真表面点 (cm) | 原64子球中心→真墙面 (cm) |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for rank, tid in enumerate(tokens_20.tolist()):
        cname = hi_meta["tokens"][rank]["colour"]
        hex_c = hi_meta["tokens"][rank]["hex"]
        r = {
            "rank": rank,
            "token": tid,
            "colour": cname,
            "hex": hex_c,
            "gt_read_to_surface_cm": float(gt_read_to_surf_m[tid]) * 100.0,
            "triangulated_to_surface_cm": float(tri_to_surf_m[tid]) * 100.0,
            "triangulated_to_gt_read_cm": float(tri_to_gt_read_m[tid]) * 100.0,
            "old_token_centre_to_surface_cm": float(tok_centre_to_surf_m[tid]) * 100.0,
            "triangulated_xyz_norm": tok_tri_pt[tid].tolist(),
            "gt_read_xyz_norm": tok_gt_read_pos[tid].tolist(),
            "old_token_centre_xyz_norm": tok_centre[tid].tolist(),
        }
        rows.append(r)
        lines.append(
            f"| {rank} | {tid} | {cname} `{hex_c}` | {r['gt_read_to_surface_cm']:.1f} | "
            f"**{r['triangulated_to_surface_cm']:.1f}** | {r['triangulated_to_gt_read_cm']:.1f} | "
            f"{r['old_token_centre_to_surface_cm']:.1f} |"
        )

    (demo_scene_dir / "triangulated_20tokens_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (demo_scene_dir / "triangulated_20tokens_report.json").write_text(
        json.dumps(
            {
                "scene": "scene_7b6477cb95",
                "all_4096_summary_cm": {
                    "gt_read_to_surface_median_cm": float(gt_read_to_surf_m.median()) * 100.0,
                    "triangulated_to_surface_median_cm": float(tri_to_surf_m.median()) * 100.0,
                    "triangulated_to_surface_p75_cm": float(tri_to_surf_m.quantile(0.75)) * 100.0,
                    "triangulated_to_surface_p90_cm": float(tri_to_surf_m.quantile(0.90)) * 100.0,
                    "triangulated_to_gt_read_median_cm": float(tri_to_gt_read_m.median()) * 100.0,
                    "old_token_centre_to_surface_median_cm": float(tok_centre_to_surf_m.median()) * 100.0,
                    "old_token_centre_to_surface_p90_cm": float(tok_centre_to_surf_m.quantile(0.90)) * 100.0,
                },
                "tokens_20": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
