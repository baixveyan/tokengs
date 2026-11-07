"""Per-frame and per-scene statistics used by scripts/evaluate_with_scene_latent_tuning.py (no core-code changes).

- edge_hf_per_frame: image sharpness, FFRGS scripts/train.py::edge_hf, evaluated per target frame.
- geometry_stats: slot geometry of the final Gaussians with the FFRGS formulation
  (mass = opacity * (prod scale)^(2/3), inter = smooth_l1(log(d / 2))), same for every group.
- needle_stats: anisotropy / long axis vs input-camera ray / parallax buckets / frustum coverage,
  following FFRGS design_notes_needles.md (opacity weighted, alive = opacity > 1/255).
- frame_distance_bucket: distance (in frames) from a target frame to its nearest input frame.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from tokengs.losses.geo_losses import _quaternion_wxyz_to_rotation_matrix, compute_geo_losses

ALIVE_OPACITY = 1.0 / 255.0
PARALLAX_BUCKETS_DEG = ((0.0, 4.0), (4.0, 9.0), (9.0, 20.0), (20.0, 180.0))
GEO_STAT_CONFIG = dict(
    lambda_sw=0.0, lambda_packing=0.0, lambda_inter=0.0, k_neighbors=4, target_distance=2.0,
    num_sw_directions=8, mass_weighting="opacity_area", inter_form="log_smooth_l1",
)


def edge_hf_per_frame(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pred, gt [V,3,H,W]. Returns per-frame (edge L1, HF ratio); pred is clamped to [0,1] as in FFRGS."""
    pred = pred.float().clamp(0, 1)
    gt = gt.float()
    pdx, gdx = pred[..., :, 1:] - pred[..., :, :-1], gt[..., :, 1:] - gt[..., :, :-1]
    pdy, gdy = pred[..., 1:, :] - pred[..., :-1, :], gt[..., 1:, :] - gt[..., :-1, :]
    dims = (1, 2, 3)
    edge = 0.5 * ((pdx - gdx).abs().mean(dims) + (pdy - gdy).abs().mean(dims))
    hf = (pdx.abs().mean(dims) + pdy.abs().mean(dims)) / (gdx.abs().mean(dims) + gdy.abs().mean(dims)).clamp_min(1e-8)
    return edge, hf


def slot_view(gaussians: torch.Tensor, children: int) -> torch.Tensor:
    batch, count, channels = gaussians.shape
    if count % children:
        raise ValueError(f"Gaussian count {count} not divisible by {children} children per slot")
    return gaussians.reshape(batch, count // children, children, channels)


def analytic_frustum_oob(gaussians: torch.Tensor, cam_view: torch.Tensor, intrinsics: torch.Tensor,
                         img_size: tuple[int, int] | list[int], znear: float) -> torch.Tensor:
    """Analytic pinhole out-of-bounds violation [B, N] across all V input views (clamped to [0, 1])."""
    H, W = int(img_size[0]), int(img_size[1])
    xyz = gaussians[..., 0:3].float()
    w2c = cam_view.float().transpose(-1, -2)  # [B, V, 4, 4]
    intr = intrinsics.float()                 # [B, V, 4]
    R, t = w2c[..., :3, :3], w2c[..., :3, 3]
    xc = torch.einsum("bvij,bnj->bvni", R, xyz) + t[:, :, None, :]
    z = xc[..., 2]
    z_safe = z.clamp_min(1e-4)
    fx, fy, cx, cy = (intr[..., i, None] for i in range(4))
    u = fx * xc[..., 0] / z_safe + cx
    v = fy * xc[..., 1] / z_safe + cy
    uv_norm_u = (u / W) * 2.0 - 1.0
    uv_norm_v = (v / H) * 2.0 - 1.0
    oob_uv = F.relu(uv_norm_u.abs() - 1.0) + F.relu(uv_norm_v.abs() - 1.0)
    oob_z = F.relu(float(znear) - z) * 10.0
    return (oob_uv + oob_z).min(dim=1).values.clamp(max=1.0)


def apply_core_opacity_gate(gaussians: torch.Tensor, children: int, tau: float, beta: float = 2.0,
                            top_k: int = 24) -> torch.Tensor:
    """Attenuate opacity of children whose distance to their token's robust core center exceeds tau * r_core."""
    if tau <= 0:
        return gaussians
    slots = slot_view(gaussians, children)
    xyz = slots[..., 0:3]
    opac = slots[..., 3]
    k = min(top_k, children)
    with torch.no_grad():
        _, top_idx = opac.topk(k, dim=-1)
        top_xyz = torch.gather(xyz, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        core_center = top_xyz.median(dim=2, keepdim=True).values
        dist_to_core = (xyz - core_center).norm(dim=-1)
        top_dist = (top_xyz - core_center).norm(dim=-1)
        r_core = top_dist.median(dim=2, keepdim=True).values.clamp_min(0.01)
        excess = F.relu(dist_to_core / r_core - float(tau))
        gate = torch.exp(-float(beta) * excess.pow(2)).reshape(gaussians.shape[0], -1, 1)
    return torch.cat([gaussians[..., 0:3], gaussians[..., 3:4] * gate, gaussians[..., 4:]], dim=-1)


def _read_colmap_points_full(zf, clip_name: str):
    """Read xyz [N,3], reproj_error [N], track_len [N], and image_id_sets from sparse/0/points3D.bin."""
    import struct

    raw = zf.open(f"{clip_name}/sparse/0/points3D.bin").read()
    off = 0
    (n_pts,) = struct.unpack_from("<Q", raw, off)
    off += 8
    xyz = np.empty((n_pts, 3), dtype=np.float64)
    err = np.empty(n_pts, dtype=np.float32)
    t_lens = np.empty(n_pts, dtype=np.int32)
    tracks: list[frozenset[int]] = []
    for i in range(n_pts):
        off += 8
        xyz[i] = struct.unpack_from("<ddd", raw, off)
        off += 24 + 3
        (err[i],) = struct.unpack_from("<d", raw, off)
        off += 8
        (t_len,) = struct.unpack_from("<Q", raw, off)
        t_lens[i] = t_len
        off += 8
        tr = struct.unpack_from("<" + "II" * t_len, raw, off)
        off += 8 * t_len
        tracks.append(frozenset(tr[::2]))
    return xyz, err, t_lens, tracks


@torch.no_grad()
def build_colmap_sparse_depth(test_dataset, scene_id: str, context_ids: list[int], intrinsics_input: torch.Tensor,
                              img_size: tuple[int, int] | list[int], scene_scale: float,
                              znear: float, device: torch.device,
                              clean: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Project visible COLMAP 3D points of context_ids into [1, V, 1, H, W] depth target and binary mask.
    By default (clean=True), filters out COLMAP noise points:
    1) track_len >= 3 and reproj_error <= 2.2 px
    2) 3D kNN-5 statistical outlier removal among context-visible points
    3) 2D 9x9 local Z-buffer see-through occlusion filter (depth <= 1.25 * local_min_z)
    """
    import json
    import zipfile
    from pathlib import Path

    H, W = int(img_size[0]), int(img_size[1])
    V = len(context_ids)
    depth_gt = torch.zeros((1, V, 1, H, W), dtype=torch.float32, device=device)
    mask_gt = torch.zeros((1, V, 1, H, W), dtype=torch.float32, device=device)
    zip_path = Path(test_dataset.root_path) / f"{scene_id}.zip"
    clip_name = f"{scene_id}/gaussian_splat"
    with zipfile.ZipFile(zip_path, "r") as zf:
        tj = json.loads(zf.open(f"{clip_name}/transforms.json").read())
        xyz_colmap, err, t_lens, tracks = _read_colmap_points_full(zf, clip_name)
        w_trans = test_dataset._colmap_world_to_final_world_transform(tj)
    c2ws_all = test_dataset.load_cameras(tj)
    n_pts = len(xyz_colmap)
    xyz_h = np.concatenate([xyz_colmap, np.ones((n_pts, 1), dtype=np.float64)], axis=1)
    xyz_w = (xyz_h @ w_trans.T)[:, :3]

    if clean:
        ctx_cids = {tj["frames"][int(f)].get("colmap_im_id") for f in context_ids}
        vis_any = np.array([bool(s & ctx_cids) for s in tracks])
        cand_idx = np.flatnonzero(vis_any)
        if cand_idx.size > 6:
            pts_cand = torch.from_numpy(xyz_w[cand_idx] * float(scene_scale)).float().to(device)
            dmat = torch.cdist(pts_cand, pts_cand)
            knn5 = dmat.topk(6, largest=False, dim=-1).values[:, 1:].mean(dim=-1).cpu().numpy()
            knn_all = np.zeros(n_pts, dtype=np.float32)
            knn_all[cand_idx] = knn5
            q90_knn = float(np.quantile(knn5, 0.90))
            pt_ok = (t_lens >= 3) & (err <= 2.2) & (knn_all <= 2.5 * q90_knn)
        else:
            pt_ok = (t_lens >= 3) & (err <= 2.2)
    else:
        pt_ok = np.ones(n_pts, dtype=bool)

    intr = intrinsics_input[0].float().to(device)
    for v_i, f_idx in enumerate(context_ids):
        cid = tj["frames"][int(f_idx)].get("colmap_im_id")
        if cid is None:
            continue
        vis_idx = np.flatnonzero([(cid in tracks[i]) and pt_ok[i] for i in range(n_pts)])
        if vis_idx.size == 0:
            continue
        w2c_raw = np.linalg.inv(c2ws_all[int(f_idx)])
        pts_cam = (xyz_w[vis_idx] @ w2c_raw[:3, :3].T + w2c_raw[:3, 3]) * float(scene_scale)
        pts_cam = torch.from_numpy(pts_cam).float().to(device)
        z = pts_cam[:, 2]
        ok = z > float(znear)
        if not bool(ok.any()):
            continue
        pts_cam, z = pts_cam[ok], z[ok]
        fx, fy, cx, cy = intr[v_i]
        u = (fx * pts_cam[:, 0] / z + cx).round().long()
        v = (fy * pts_cam[:, 1] / z + cy).round().long()
        in_b = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not bool(in_b.any()):
            continue
        u, v, z = u[in_b], v[in_b], z[in_b]
        order = torch.argsort(-z)
        depth_gt[0, v_i, 0, v[order], u[order]] = z[order]
        mask_gt[0, v_i, 0, v[order], u[order]] = 1.0

    if clean and bool((mask_gt > 0).any()):
        d_large = torch.where(mask_gt > 0, depth_gt, torch.full_like(depth_gt, 1e6))
        z_local_min = -F.max_pool2d(-d_large.view(V, 1, H, W), kernel_size=9, stride=1, padding=4).view(1, V, 1, H, W)
        mask_gt = mask_gt * (depth_gt <= 1.25 * z_local_min).float()
    return depth_gt, mask_gt


def load_vggt_omega_model(device: torch.device,
                          vggt_root: str = "/mnt/disk/ethan/code/vggt-omega",
                          ckpt_path: str = "/mnt/disk/ethan/code/vggt-omega/checkpoints/vggt_omega_1b_416_reproduce.pt"):
    """Load frozen VGGT-Omega-1B (aggregator in bf16) for dense depth prior extraction."""
    import sys
    from pathlib import Path

    root = str(Path(vggt_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from vggt_omega.models import VGGTOmega

    omega = VGGTOmega(enable_camera=True, enable_depth=True)
    omega.aggregator.cached_layer_indices = {4, 11, 17, 23}
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state.get("model"), dict):
        state = state["model"]
    for name, module in (("aggregator", omega.aggregator), ("camera_head", omega.camera_head),
                         ("dense_head", omega.dense_head)):
        p = f"{name}."
        module.load_state_dict({k[len(p):]: v for k, v in state.items() if k.startswith(p)}, strict=True)
    del state
    omega.requires_grad_(False).eval().to(device)
    omega.aggregator.to(torch.bfloat16)
    return omega


@torch.no_grad()
def build_vggt_omega_sim3_depth(omega, images_input: torch.Tensor, cam_to_world_input: torch.Tensor,
                                conf_quantile: float = 0.20) -> tuple[torch.Tensor, torch.Tensor]:
    """Run VGGT-Omega on images_input [B, V, 3, H, W] and Sim3-scale depth to cam_to_world_input [B, V, 4, 4]."""
    from vggt_omega.models.aggregator import slice_expand_and_flatten
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = images_input.device
    B, V, C, H, W = images_input.shape
    agg = omega.aggregator
    x = ((images_input - agg._resnet_mean.to(device)) / agg._resnet_std.to(device)).view(B * V, C, H, W).to(torch.bfloat16)
    cam_t = slice_expand_and_flatten(agg.camera_token, B, V)
    reg_t = slice_expand_and_flatten(agg.register_token, B, V)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images_input.is_cuda):
        pt = agg.patch_embed(x)
        if isinstance(pt, dict):
            pt = pt["x_norm_patchtokens"]
        tokens = torch.cat([cam_t, reg_t, pt], dim=1)
        N, D = tokens.shape[1:]
        rs, rc = agg.rope_embed(H=H // agg.patch_size, W=W // agg.patch_size)
        rope = (rs.to(device=device, dtype=torch.float32), rc.to(device=device, dtype=torch.float32))
        outs = []
        for i in range(agg.depth):
            tokens, ft = agg._run_frame_block(tokens, B, V, N, D, i, rope)
            tokens = agg._run_inter_frame_attention_block(tokens, B, V, N, D, i, agg.inter_frame_attention_types[i])
            outs.append(torch.cat([ft, tokens], dim=-1) if i in agg.cached_layer_indices else None)
    start = agg.patch_token_start
    depth_omega, conf_omega = omega.dense_head(outs, images=images_input.float(), patch_token_start=start)
    depth_omega = depth_omega.squeeze(-1).float()  # [B, V, H, W]
    pose_enc = omega.camera_head(outs, patch_token_start=start)
    extr_omega, _ = encoding_to_camera(pose_enc.float(), image_size_hw=None, build_intrinsics=False)
    w2c_om = torch.eye(4, device=device, dtype=torch.float32).repeat(B, V, 1, 1)
    w2c_om[..., :3, :4] = extr_omega
    c2w_omega = torch.linalg.inv(w2c_om)

    Rs, Rd = c2w_omega[..., :3, :3].float(), cam_to_world_input[..., :3, :3].float().to(device)
    U, _, Vt = torch.linalg.svd((Rd @ Rs.transpose(-1, -2)).sum(dim=1))
    sign = torch.sign(torch.det(U @ Vt))
    D_mat = torch.diag_embed(torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], dim=-1))
    R_sim3 = U @ D_mat @ Vt
    cs, cd = c2w_omega[..., :3, 3].float(), cam_to_world_input[..., :3, 3].float().to(device)
    mu_s, mu_d = cs.mean(dim=1), cd.mean(dim=1)
    xs = (cs - mu_s[:, None]) @ R_sim3.transpose(-1, -2)
    xd = cd - mu_d[:, None]
    s_sim3 = ((xs * xd).sum(dim=(1, 2)) / xs.square().sum(dim=(1, 2)).clamp_min(1e-12)).clamp(1e-3, 1e3)
    depth_scaled = (depth_omega * s_sim3[:, None, None, None]).unsqueeze(2)  # [B, V, 1, H, W]
    conf_2d = conf_omega.float().unsqueeze(2)
    thresh = torch.quantile(conf_2d.reshape(B, -1), conf_quantile, dim=-1)[:, None, None, None, None]
    mask = (conf_2d >= thresh).float()
    return depth_scaled, mask


@torch.no_grad()
def build_target_colmap_feature_mask(test_dataset, scene_id: str, target_ids: list[int],
                                     intrinsics_target: torch.Tensor, img_size: tuple[int, int] | list[int],
                                     scene_scale: float, znear: float, device: torch.device,
                                     dilate_kernel: int = 9) -> torch.Tensor:
    """Return boolean mask [V_tgt, H, W] of 9x9 dilated COLMAP-visible feature regions on target views."""
    import json
    import zipfile
    from pathlib import Path

    H, W = int(img_size[0]), int(img_size[1])
    V = len(target_ids)
    has_px = torch.zeros((V, 1, H, W), dtype=torch.float32, device=device)
    zip_path = Path(test_dataset.root_path) / f"{scene_id}.zip"
    clip_name = f"{scene_id}/gaussian_splat"
    with zipfile.ZipFile(zip_path, "r") as zf:
        tj = json.loads(zf.open(f"{clip_name}/transforms.json").read())
        xyz_colmap, tracks = test_dataset._get_colmap_points(zf, clip_name)
        w_trans = test_dataset._colmap_world_to_final_world_transform(tj)
    c2ws_all = test_dataset.load_cameras(tj)
    xyz_h = np.concatenate([xyz_colmap, np.ones((len(xyz_colmap), 1), dtype=np.float64)], axis=1)
    xyz_w = (xyz_h @ w_trans.T)[:, :3]
    intr = intrinsics_target.detach().float().to(device)
    if intr.ndim == 3:
        intr = intr[0]
    for t_i, f_idx in enumerate(target_ids):
        cid = tj["frames"][int(f_idx)].get("colmap_im_id")
        if cid is None:
            continue
        vis_m = np.fromiter((cid in s for s in tracks), dtype=bool, count=len(tracks))
        if not vis_m.any():
            continue
        w2c_raw = np.linalg.inv(c2ws_all[int(f_idx)])
        pts_cam = (xyz_w[vis_m] @ w2c_raw[:3, :3].T + w2c_raw[:3, 3]) * float(scene_scale)
        pts_cam_t = torch.from_numpy(pts_cam).float().to(device)
        z = pts_cam_t[:, 2]
        ok = z > float(znear)
        if not bool(ok.any()):
            continue
        pts_cam_t, z = pts_cam_t[ok], z[ok]
        fx, fy, cx, cy = intr[t_i]
        u = (fx * pts_cam_t[:, 0] / z + cx).round().long()
        v = (fy * pts_cam_t[:, 1] / z + cy).round().long()
        in_b = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if bool(in_b.any()):
            has_px[t_i, 0, v[in_b], u[in_b]] = 1.0
    pad = dilate_kernel // 2
    return (F.max_pool2d(has_px, kernel_size=dilate_kernel, stride=1, padding=pad) > 0).squeeze(1)


@torch.no_grad()
def surface_and_region_stats(model, gaussians: torch.Tensor, pred_images: torch.Tensor, gt_images: torch.Tensor,
                             depths_pred: torch.Tensor, cam_view_target: torch.Tensor,
                             intrinsics_target: torch.Tensor, cam_view_input: torch.Tensor,
                             intrinsics_input: torch.Tensor, far_mask: torch.Tensor,
                             feat_region_mask: torch.Tensor, img_size: tuple[int, int] | list[int],
                             step_gap: int = 5) -> dict[str, float]:
    """Compute 140-scene regional PSNR and GT-free physical geometry metrics for one scene:
    - feat_pixel_ratio: fraction of target pixels covered by 9x9 dilated COLMAP feature region
    - d4+_feat_psnr: d4+ PSNR on feature-anchored object/edge regions
    - d4+_smooth_psnr: d4+ PSNR on unanchored smooth/textureless regions
    - warp_psnr_dt5: cross-view depth-warping photometric PSNR (warping real GT photo t+5 -> t using rendered depth)
    - mv_cycle_err_dt5: multi-view depth cycle relative discrepancy (t -> t+5)
    - ray_thickness_med: median transmittance-weighted ray depth standard deviation (meters in TokenGS frame)
    """
    device = gaussians.device
    H, W = int(img_size[0]), int(img_size[1])
    preds = pred_images[0].float().clamp(0, 1)  # [V_tgt, 3, H, W]
    gts = gt_images[0].float()                  # [V_tgt, 3, H, W]
    d_all = depths_pred[0, :, 0].float()        # [V_tgt, H, W]
    err2 = (preds - gts).pow(2).mean(dim=1)     # [V_tgt, H, W]

    def _masked_d4_psnr(reg_mask: torch.Tensor) -> float:
        m_d4 = reg_mask[far_mask]
        e_d4 = err2[far_mask]
        vals = [float(-10.0 * torch.log10(e_d4[k][m_d4[k]].mean().clamp_min(1e-10)))
                for k in range(e_d4.shape[0]) if bool(m_d4[k].any())]
        return float(np.mean(vals)) if vals else float("nan")

    feat_ratio = float(feat_region_mask.float().mean())
    d4_feat = _masked_d4_psnr(feat_region_mask)
    d4_smooth = _masked_d4_psnr(~feat_region_mask)

    # Cross-view depth warping PSNR & multi-view cycle consistency error (step_gap = 5)
    intr_t = intrinsics_target[0].float() if intrinsics_target.ndim == 3 else intrinsics_target.float()
    w2c_t = (cam_view_target[0] if cam_view_target.ndim == 4 else cam_view_target).float().transpose(-1, -2)
    c2w_t = torch.linalg.inv(w2c_t)
    vy, vx = torch.meshgrid(torch.arange(H, device=device).float(),
                            torch.arange(W, device=device).float(), indexing="ij")
    V_tgt = d_all.shape[0]
    warp_psnrs, cycle_errs = [], []
    for t in range(V_tgt - step_gap):
        t2 = t + step_gap
        dt0, dt1 = d_all[t], d_all[t2]
        fx0, fy0, cx0, cy0 = intr_t[t]
        xc0 = torch.stack([(vx - cx0) / fx0 * dt0, (vy - cy0) / fy0 * dt0, dt0], dim=-1)
        xw = xc0 @ c2w_t[t, :3, :3].T + c2w_t[t, :3, 3]
        xc1 = xw @ w2c_t[t2, :3, :3].T + w2c_t[t2, :3, 3]
        z1 = xc1[..., 2].clamp_min(1e-4)
        fx1, fy1, cx1, cy1 = intr_t[t2]
        u1 = (fx1 * xc1[..., 0] / z1 + cx1) / (W - 1) * 2.0 - 1.0
        v1 = (fy1 * xc1[..., 1] / z1 + cy1) / (H - 1) * 2.0 - 1.0
        grid = torch.stack([u1, v1], dim=-1).unsqueeze(0)
        dt1_s = F.grid_sample(dt1[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        in_fov = (u1.abs() < 0.95) & (v1.abs() < 0.95) & (dt0 > 0.05) & (z1 > 0.05)
        if bool(in_fov.any()):
            rel_diff = (z1 - dt1_s).abs() / z1
            cycle_errs.append(float(rel_diff[in_fov].median()))
            img_w = F.grid_sample(gts[t2:t2 + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
            mse = (img_w - gts[t]).pow(2).mean(dim=0)[in_fov].mean().clamp_min(1e-10)
            warp_psnrs.append(float(-10.0 * torch.log10(mse)))

    # Ray depth thickness (transmittance-weighted depth std along input-view rays)
    w2c_in = (cam_view_input[0] if cam_view_input.ndim == 4 else cam_view_input).float().transpose(-1, -2)
    c2w_in0 = torch.linalg.inv(w2c_in[0])
    r_dist = (gaussians[0, :, 0:3].float() - c2w_in0[:3, 3].unsqueeze(0)).norm(dim=-1) / 5.0
    r_clamped = r_dist.clamp(0.0, 1.0)
    gs_r = gaussians.float().clone()
    gs_r[0, :, 11] = r_clamped
    gs_r[0, :, 12] = r_clamped.pow(2)
    gs_r[0, :, 13] = 1.0
    cam_in_b = cam_view_input if cam_view_input.ndim == 4 else cam_view_input.unsqueeze(0)
    intr_in_b = intrinsics_input if intrinsics_input.ndim == 3 else intrinsics_input.unsqueeze(0)
    ren_r = model.render_gaussians(gs_r, cam_in_b.float(), intr_in_b.float())
    rgb_r = ren_r["images_pred"][0]
    alp_r = ren_r["alphas_pred"][0, :, 0].clamp_min(1e-4)
    er1 = (rgb_r[:, 0] / alp_r) * 5.0
    er2 = (rgb_r[:, 1] / alp_r) * 25.0
    ray_std = (er2 - er1.pow(2)).clamp_min(0.0).sqrt()

    return {
        "feat_pixel_ratio": feat_ratio,
        "d4+_feat_psnr": d4_feat,
        "d4+_smooth_psnr": d4_smooth,
        "warp_psnr_dt5": float(np.mean(warp_psnrs)) if warp_psnrs else float("nan"),
        "mv_cycle_err_dt5": float(np.median(cycle_errs)) if cycle_errs else float("nan"),
        "ray_thickness_med": float(ray_std.median()),
    }


def geo_losses_for(gaussians: torch.Tensor, children: int, **weights) -> dict:
    slots = slot_view(gaussians, children)
    cfg = {**GEO_STAT_CONFIG, **weights}
    return compute_geo_losses(
        gs_centers=slots[..., 0:3], gs_scales=slots[..., 4:7], gs_opacities=slots[..., 3:4],
        gs_rotations=slots[..., 7:11], **cfg,
    )


@torch.no_grad()
def geometry_stats(gaussians: torch.Tensor, children: int) -> dict:
    out = geo_losses_for(gaussians.float(), children)
    stats = {k: float(out[k]) for k in ("sw_loss", "packing_loss", "inter_loss")}
    stats.update({k: float(v) for k, v in out["stats"].items()})
    return stats


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> float:
    if values.numel() == 0 or float(weights.sum()) <= 0:
        return float("nan")
    order = values.argsort()
    v, w = values[order].double(), weights[order].double()
    cdf = w.cumsum(0)
    idx = int(torch.searchsorted(cdf, q * cdf[-1]).clamp(max=v.numel() - 1))
    return float(v[idx])


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    total = float(weights.sum())
    return float((values.double() * weights.double()).sum() / total) if total > 0 else float("nan")


@torch.no_grad()
def needle_per_gaussian(gaussians: torch.Tensor, cam_to_world: torch.Tensor, intrinsics: torch.Tensor,
                        img_size: tuple[int, int], znear: float) -> dict[str, torch.Tensor]:
    """gaussians [N,14]; cam_to_world [V,4,4] (OpenCV); intrinsics [V,4] (fx,fy,cx,cy in pixels)."""
    g = gaussians.float()
    mu, opacity, scales, quat = g[:, 0:3], g[:, 3], g[:, 4:7].clamp_min(1e-12), g[:, 7:11]
    rot = _quaternion_wxyz_to_rotation_matrix(quat)                     # columns are local axes
    long_idx = scales.argmax(dim=-1)
    long_axis = rot.gather(2, long_idx.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
    aniso = scales.max(dim=-1).values / scales.min(dim=-1).values

    c2w = cam_to_world.float()
    centers = c2w[:, :3, 3]                                              # [V,3]
    rays = mu.unsqueeze(1) - centers.unsqueeze(0)                        # [N,V,3]
    dist = rays.norm(dim=-1).clamp_min(1e-12)
    unit = rays / dist.unsqueeze(-1)
    nearest = dist.argmin(dim=1)
    ray_nearest = unit.gather(1, nearest.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    cos_ray = (long_axis * ray_nearest).sum(-1).abs()
    gram = torch.einsum("nvi,nwi->nvw", unit, unit).clamp(-1.0, 1.0)
    parallax = torch.rad2deg(torch.arccos(gram.amin(dim=(1, 2))))

    w2c = torch.linalg.inv(c2w)
    cam = torch.einsum("vij,nj->nvi", w2c[:, :3, :3], mu) + w2c[:, :3, 3].unsqueeze(0)
    z = cam[..., 2]
    fx, fy, cx, cy = intrinsics.float().unbind(-1)
    u = fx * cam[..., 0] / z.clamp_min(1e-12) + cx
    v = fy * cam[..., 1] / z.clamp_min(1e-12) + cy
    H, W = img_size
    inside = (z > znear) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return {
        "opacity": opacity, "aniso": aniso, "cos_ray": cos_ray, "parallax_deg": parallax,
        "in_frustum_count": inside.sum(dim=1).float(), "nearest_camera": nearest.float(),
    }


def _needle_summary(per: dict[str, torch.Tensor], mask: torch.Tensor) -> dict:
    w = per["opacity"][mask]
    aniso, cos = per["aniso"][mask], per["cos_ray"][mask]
    return {
        "children": int(mask.sum()),
        "opacity_mass_fraction": float(w.sum() / per["opacity"].clamp_min(0).sum().clamp_min(1e-12)),
        "aniso_p50_weighted": weighted_quantile(aniso, w, 0.5),
        "aniso_p50_unweighted": float(aniso.median()) if aniso.numel() else float("nan"),
        "cos_long_axis_ray_p50_weighted": weighted_quantile(cos, w, 0.5),
        "long_axis_along_ray_gt0.9_weighted": _weighted_mean((cos > 0.9).float(), w),
    }


@torch.no_grad()
def needle_stats(gaussians: torch.Tensor, cam_to_world: torch.Tensor, intrinsics: torch.Tensor,
                 img_size: tuple[int, int], znear: float, detailed: bool = False) -> tuple[dict, dict]:
    per = needle_per_gaussian(gaussians, cam_to_world, intrinsics, img_size, znear)
    alive = per["opacity"] > ALIVE_OPACITY
    out = {"alive_ratio": float(alive.float().mean()), "opacity_mean": float(per["opacity"].mean()),
           **_needle_summary(per, alive)}
    outside = per["in_frustum_count"] == 0
    out["outside_all_input_frustums_weighted"] = _weighted_mean(outside[alive].float(), per["opacity"][alive])
    out["outside_all_input_frustums_count_opacity_gt0.1"] = int((outside & (per["opacity"] > 0.1)).sum())
    if detailed:
        out["parallax_buckets"] = []
        for lo, hi in PARALLAX_BUCKETS_DEG:
            m = alive & (per["parallax_deg"] >= lo) & (per["parallax_deg"] < hi)
            out["parallax_buckets"].append({"parallax_deg": [lo, hi], **_needle_summary(per, m)})
        out["opacity_quantiles"] = {str(q): float(per["opacity"].quantile(q)) for q in (0.1, 0.5, 0.9, 0.99)}
        out["opacity_low_ratio_lt0.1"] = float((per["opacity"] < 0.1).float().mean())
        out["opacity_high_ratio_gt0.9"] = float((per["opacity"] > 0.9).float().mean())
    return out, per


def frame_distance_bucket(distance: int) -> str:
    if distance == 0:
        return "d0"
    if distance == 1:
        return "d1"
    if distance <= 3:
        return "d2-3"
    return "d4+"


DISTANCE_BUCKETS = ("d0", "d1", "d2-3", "d4+")


def nan_safe(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
