"""Small, CPU-testable contracts, experiment registry, and table builder for ScanNet++ evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


METRICS = ("psnr", "ssim", "lpips")
GEOMETRY_FRAME_METRICS = (
    "depth_cov",
    "depth_scale_ratio_med",
    "depth_absrel",
    "depth_rmse_m",
    "depth_delta_105",
    "depth_delta_110",
    "ray_thick_med_m",
    "ray_thick_p90_m",
    "normal_mean_deg",
    "normal_med_deg",
    "normal_lt11_25",
    "normal_lt30",
)
GEOMETRY_SCENE_METRICS = (
    *GEOMETRY_FRAME_METRICS,
    "surf_acc_cm",
    "surf_comp_cm",
    "surf_prec_3cm",
    "surf_rec_3cm",
    "surf_fscore_3cm",
    "surf_prec_5cm",
    "surf_rec_5cm",
    "surf_fscore_5cm",
)
ALPHA_VALID_THRESHOLD = 0.5
VOXEL_SIZE_METERS = 0.01
NORMAL_MAX_REL_JUMP = 0.05
GEOMETRY_COMPUTED_STATUS = "computed (Open3D raycast + gsplat ray thickness, physical scale)"
EXP_SCANNETPP = "EXP_ScanNetPP"
EXPECTED_SCENES = 50
DEFAULT_DEMO_SCENE = "7b6477cb95"
REFERENCE_EXPERIMENT = "baseline_pretrained_6view"


@dataclass(frozen=True)
class ScanNetPPExperiment:
    name: str
    title: str
    train_modalities: str
    eval_modalities: str
    checkpoint: str
    model_type: str = "tokengs"
    train_preset: str | None = None
    extra_args: tuple[str, ...] = ()

    @property
    def dir(self) -> str:
        return f"{EXP_SCANNETPP}/{self.name}"

    def run_args(self) -> list[str]:
        return [
            "--output", self.dir,
            "--checkpoint", self.checkpoint,
            "--model-type", self.model_type,
            *self.extra_args,
        ]

    def demo_args(self, scene_id: str) -> list[str]:
        return [
            "--demo-only",
            "--demo-scene", scene_id,
            "--demo-output", f"{EXP_SCANNETPP}/demo/scene_{scene_id[:12]}/{self.name}",
            "--ply-name", self.name,
            "--checkpoint", self.checkpoint,
            "--model-type", self.model_type,
            *self.extra_args,
        ]


EXPERIMENTS = (
    ScanNetPPExperiment(
        name="baseline_pretrained_6view",
        title="官方 6 输入预训练模型直接前馈基线（不微调）",
        train_modalities="RGB + 相机（DL3DV 预训练）",
        eval_modalities="RGB + 相机（0 点云、0 优化步）",
        checkpoint="checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors",
        train_preset=None,
    ),
    ScanNetPPExperiment(
        name="sft_rgb_only",
        title="对照 C0：ScanNet++ 纯 RGB 光度后训练（解冻 latent_blocks.8..11 + KV 投影）",
        train_modalities="RGB + 相机（无点云）",
        eval_modalities="RGB + 相机（0 点云、0 优化步）",
        checkpoint="EXP_ScanNetPP/sft_rgb_only/model.safetensors",
        train_preset="sft_scannetpp_rgb_only",
    ),
)
BY_NAME = {exp.name: exp for exp in EXPERIMENTS}


def validate_frame_split(context, target):
    if len(context) != 6 or len(target) != 8:
        raise ValueError("Expected six input views and eight target views")
    if len(set(context)) != 6 or len(set(target)) != 8:
        raise ValueError("Repeated context or target frame")
    if set(context) & set(target):
        raise ValueError("Evaluation targets must exclude every input view")


def mean_metrics(rows):
    if not rows:
        raise ValueError("Cannot aggregate empty metrics")
    result = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite {metric}")
        result[metric] = statistics.mean(values)
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv_atomic(path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    if fieldnames is None:
        if not rows:
            raise ValueError("write_csv_atomic requires fieldnames when rows is empty")
        fieldnames = list(rows[0])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def normalize_world_transform(raw_context_c2w: torch.Tensor) -> torch.Tensor:
    """Return 4x4 inverse of the mean-camera pose used by Provider('mean_cam')."""
    in_c2ws = raw_context_c2w.detach().cpu().float()
    pos_avg = in_c2ws[:, :3, 3].mean(0)
    fwd_avg = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=0)
    down_avg = in_c2ws[:, :3, 1].mean(0)
    down_avg = F.normalize(down_avg - down_avg.dot(fwd_avg) * fwd_avg, dim=0)
    right_avg = torch.linalg.cross(down_avg, fwd_avg)
    mat = torch.eye(4, dtype=torch.float32)
    mat[:3, 0], mat[:3, 1], mat[:3, 2], mat[:3, 3] = right_avg, down_avg, fwd_avg, pos_avg
    return torch.linalg.inv(mat)


def build_raycasting_scene(mesh_ply: Path, raw_context_c2w: torch.Tensor, scene_scale: float = 0.3):
    """Load ScanNet++ triangle mesh, transform vertices to normalized scene space, and build Open3D RaycastingScene."""
    import open3d as o3d

    mesh = o3d.t.io.read_triangle_mesh(str(mesh_ply))
    mat_inv = normalize_world_transform(raw_context_c2w)
    v_w = torch.from_numpy(mesh.vertex.positions.numpy()).float()
    v_n = (v_w @ mat_inv[:3, :3].T + mat_inv[:3, 3].unsqueeze(0)) * float(scene_scale)
    mesh.vertex.positions = o3d.core.Tensor.from_numpy(v_n.numpy())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def raycast_gt_depth_and_normals(
    scene,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    img_size: tuple[int, int] | list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raycast true triangle mesh depth D* (in normalized units) and camera-space face normals N* for V views."""
    import open3d as o3d

    H, W = int(img_size[0]), int(img_size[1])
    u, v = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + 0.5,
        torch.arange(H, dtype=torch.float32) + 0.5,
        indexing="xy",
    )
    V = cam_view.shape[0]
    depths_gt = torch.zeros((V, H, W), dtype=torch.float32)
    normals_gt = torch.zeros((V, H, W, 3), dtype=torch.float32)
    valid_gt = torch.zeros((V, H, W), dtype=torch.bool)

    for i in range(V):
        w2c = cam_view[i].detach().cpu().float().T
        c2w = torch.linalg.inv(w2c)
        fx, fy, cx, cy = intrinsics[i].detach().cpu().float().tolist()
        dirs_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
        dirs_world = dirs_cam @ c2w[:3, :3].T
        orig_world = c2w[:3, 3].expand_as(dirs_world)
        rays = o3d.core.Tensor.from_numpy(torch.cat([orig_world, dirs_world], dim=-1).numpy())
        ans = scene.cast_rays(rays)
        t_hit = torch.from_numpy(ans["t_hit"].numpy()).float()
        ok = torch.isfinite(t_hit) & (t_hit > 0)
        depths_gt[i] = torch.where(ok, t_hit, torch.zeros_like(t_hit))
        valid_gt[i] = ok
        n_world = torch.from_numpy(ans["primitive_normals"].numpy()).float()
        n_cam = F.normalize(n_world @ w2c[:3, :3].T, dim=-1, eps=1e-6)
        n_cam = torch.where(n_cam[..., 2:3] > 0, -n_cam, n_cam)
        normals_gt[i] = torch.where(ok.unsqueeze(-1), n_cam, torch.zeros_like(n_cam))

    return depths_gt, normals_gt, valid_gt


def depth_map_to_camera_normals(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    max_rel_jump: float = NORMAL_MAX_REL_JUMP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute camera-space surface normals from depth map [V, H, W] via central differences and mask >5% depth jumps."""
    V, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    u, v = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype) + 0.5,
        torch.arange(H, device=device, dtype=dtype) + 0.5,
        indexing="xy",
    )
    fx = intrinsics[:, 0].view(V, 1, 1)
    fy = intrinsics[:, 1].view(V, 1, 1)
    cx = intrinsics[:, 2].view(V, 1, 1)
    cy = intrinsics[:, 3].view(V, 1, 1)

    x = (u.unsqueeze(0) - cx) / fx * depth
    y = (v.unsqueeze(0) - cy) / fy * depth
    pts_cam = torch.stack([x, y, depth], dim=-1)  # [V, H, W, 3]

    dx_du = 0.5 * (pts_cam[:, 1:-1, 2:, :] - pts_cam[:, 1:-1, :-2, :])
    dx_dv = 0.5 * (pts_cam[:, 2:, 1:-1, :] - pts_cam[:, :-2, 1:-1, :])
    tu = F.normalize(dx_du, dim=-1, eps=1e-12)
    tv = F.normalize(dx_dv, dim=-1, eps=1e-12)
    cross_uv = torch.linalg.cross(tu, tv, dim=-1)
    cross_norm = cross_uv.norm(dim=-1)
    n_inner = F.normalize(cross_uv, dim=-1, eps=1e-12)
    n_inner = torch.where(n_inner[..., 2:3] > 0, -n_inner, n_inner)

    d_c = depth[:, 1:-1, 1:-1]
    d_r = depth[:, 1:-1, 2:]
    d_l = depth[:, 1:-1, :-2]
    d_d = depth[:, 2:, 1:-1]
    d_u = depth[:, :-2, 1:-1]
    pos_ok = (d_c > 0) & (d_r > 0) & (d_l > 0) & (d_d > 0) & (d_u > 0) & (cross_norm > 1e-6)
    denom = d_c.clamp_min(1e-6)
    jump_u = torch.maximum((d_r - d_c).abs(), (d_l - d_c).abs()) / denom
    jump_v = torch.maximum((d_d - d_c).abs(), (d_u - d_c).abs()) / denom
    smooth_inner = pos_ok & (jump_u <= max_rel_jump) & (jump_v <= max_rel_jump)

    normals = torch.zeros((V, H, W, 3), device=device, dtype=dtype)
    smooth_mask = torch.zeros((V, H, W), device=device, dtype=torch.bool)
    normals[:, 1:-1, 1:-1, :] = n_inner
    smooth_mask[:, 1:-1, 1:-1] = smooth_inner
    return normals, smooth_mask


def ray_thickness_from_moments(
    z1_accum: torch.Tensor,
    z2_accum: torch.Tensor,
    alpha: torch.Tensor,
    scene_scale: float = 0.3,
) -> torch.Tensor:
    """Compute physical ray thickness sigma_z (meters) from background-free first and second depth moments."""
    a_safe = alpha.clamp_min(1e-6)
    ez = z1_accum / a_safe
    ez2 = z2_accum / a_safe
    var_z = (ez2 - ez.square()).clamp_min(0.0)
    sigma_m = torch.sqrt(var_z) / float(scene_scale)
    return torch.where(alpha > 1e-4, sigma_m, torch.zeros_like(sigma_m))


def render_ray_thickness_map(
    gaussians: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    img_size: tuple[int, int] | list[int],
    near_plane: float,
    far_plane: float,
    scene_scale: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render per-view [z_cam, z_cam^2] with zero background via gsplat and return (sigma_m, var_m2) [B, V, 1, H, W]."""
    from gsplat.rendering import rasterization

    if gaussians.ndim == 2:
        gaussians = gaussians.unsqueeze(0)
    if cam_view.ndim == 3:
        cam_view = cam_view.unsqueeze(0)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)

    B, V = cam_view.shape[:2]
    H, W = int(img_size[0]), int(img_size[1])
    means3d = gaussians[..., 0:3].contiguous().float()
    opacity = gaussians[..., 3].contiguous().float()
    scales = gaussians[..., 4:7].contiguous().float()
    rotations = gaussians[..., 7:11].contiguous().float()
    viewmat = cam_view.float().transpose(3, 2).contiguous()

    Ks = torch.zeros((B, V, 3, 3), dtype=means3d.dtype, device=means3d.device)
    Ks[:, :, 0, 0] = intrinsics[:, :, 0]
    Ks[:, :, 1, 1] = intrinsics[:, :, 1]
    Ks[:, :, 0, 2] = intrinsics[:, :, 2]
    Ks[:, :, 1, 2] = intrinsics[:, :, 3]
    Ks[:, :, 2, 2] = 1.0

    sigmas, vars_m2 = [], []
    for b in range(B):
        r_w2c = viewmat[b, :, :3, :3]  # [V, 3, 3]
        t_w2c = viewmat[b, :, :3, 3]   # [V, 3]
        pts_cam = means3d[b].unsqueeze(0) @ r_w2c.transpose(-1, -2) + t_w2c.unsqueeze(1)  # [V, N, 3]
        z_cam = pts_cam[..., 2].clamp_min(0.0)
        z_colors = torch.stack([z_cam, z_cam.square()], dim=-1).contiguous()  # [V, N, 2]
        bg_zero = torch.zeros((V, 2), dtype=means3d.dtype, device=means3d.device)
        z12_render, alpha_render, _ = rasterization(
            means=means3d[b],
            quats=rotations[b],
            scales=scales[b],
            opacities=opacity[b],
            colors=z_colors,
            viewmats=viewmat[b],
            Ks=Ks[b],
            width=W,
            height=H,
            near_plane=float(near_plane),
            far_plane=float(far_plane),
            packed=False,
            backgrounds=bg_zero,
            render_mode="RGB",
        )
        z1_acc = z12_render[..., 0:1].permute(0, 3, 1, 2)  # [V, 1, H, W]
        z2_acc = z12_render[..., 1:2].permute(0, 3, 1, 2)  # [V, 1, H, W]
        a_acc = alpha_render.permute(0, 3, 1, 2)           # [V, 1, H, W]
        a_safe = a_acc.clamp_min(1e-6)
        ez = z1_acc / a_safe
        ez2 = z2_acc / a_safe
        var_z = (ez2 - ez.square()).clamp_min(0.0)
        var_m = var_z / (float(scene_scale) ** 2)
        sig_m = torch.sqrt(var_m + 1e-12) - 1e-6
        sig_m = torch.where(a_acc > 1e-4, sig_m.clamp_min(0.0), torch.zeros_like(sig_m))
        sigmas.append(sig_m)
        vars_m2.append(var_m)

    return torch.stack(sigmas, dim=0), torch.stack(vars_m2, dim=0)


def voxel_downsample_points(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Deterministic voxel-grid centroid downsampling of [N, 3] point cloud."""
    if points.numel() == 0:
        return points.reshape(0, 3)
    coords = torch.floor(points / float(voxel_size)).to(torch.int64)
    _, inv = torch.unique(coords, dim=0, return_inverse=True)
    num_voxels = int(inv.max().item()) + 1
    sums = torch.zeros((num_voxels, 3), dtype=points.dtype, device=points.device)
    counts = torch.zeros((num_voxels, 1), dtype=points.dtype, device=points.device)
    sums.index_add_(0, inv, points)
    counts.index_add_(0, inv, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device))
    return sums / counts.clamp_min(1.0)


def _chunked_nn_dist(src: torch.Tensor, dst: torch.Tensor, chunk: int = 4096, dst_chunk: int = 32768) -> torch.Tensor:
    from scipy.spatial import cKDTree

    if src.shape[0] == 0 or dst.shape[0] == 0:
        return torch.empty(0, dtype=src.dtype, device=src.device)
    tree = cKDTree(dst.detach().cpu().float().numpy())
    dists, _ = tree.query(src.detach().cpu().float().numpy(), k=1, workers=4)
    return torch.from_numpy(np.asarray(dists, dtype=np.float32)).to(device=src.device, dtype=src.dtype)


def compute_surface_fscore(
    pred_points_norm: torch.Tensor,
    gt_points_norm: torch.Tensor,
    scene_scale: float = 0.3,
    voxel_size_m: float = VOXEL_SIZE_METERS,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Compute bidirectional surface Accuracy (cm), Completeness (cm), and F-score @ 3cm / 5cm."""
    voxel_norm = float(voxel_size_m) * float(scene_scale)
    p_ds = voxel_downsample_points(pred_points_norm.float().cpu(), voxel_norm).to(device)
    g_ds = voxel_downsample_points(gt_points_norm.float().cpu(), voxel_norm).to(device)
    if p_ds.shape[0] == 0 or g_ds.shape[0] == 0:
        return {
            "surf_acc_cm": float("nan"),
            "surf_comp_cm": float("nan"),
            "surf_prec_3cm": 0.0,
            "surf_rec_3cm": 0.0,
            "surf_fscore_3cm": 0.0,
            "surf_prec_5cm": 0.0,
            "surf_rec_5cm": 0.0,
            "surf_fscore_5cm": 0.0,
        }
    d_p2g_m = _chunked_nn_dist(p_ds, g_ds) / float(scene_scale)
    d_g2p_m = _chunked_nn_dist(g_ds, p_ds) / float(scene_scale)
    acc_cm = float(d_p2g_m.mean().item() * 100.0)
    comp_cm = float(d_g2p_m.mean().item() * 100.0)
    out = {"surf_acc_cm": acc_cm, "surf_comp_cm": comp_cm}
    for cm, tau_m in ((3, 0.03), (5, 0.05)):
        prec = float((d_p2g_m < tau_m).float().mean().item())
        rec = float((d_g2p_m < tau_m).float().mean().item())
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out[f"surf_prec_{cm}cm"] = prec
        out[f"surf_rec_{cm}cm"] = rec
        out[f"surf_fscore_{cm}cm"] = f1
    return out


def compute_scene_geometry_metrics(
    depths_pred: torch.Tensor,
    alphas_pred: torch.Tensor,
    sigma_m_pred: torch.Tensor,
    depths_gt: torch.Tensor,
    normals_gt_cam: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    scene_scale: float = 0.3,
    alpha_thresh: float = ALPHA_VALID_THRESHOLD,
    nn_device: torch.device | str = "cpu",
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Compute per-frame (G1, G2, G4) and scene-level (G1-G4 including G3 3D surface F-score) geometry metrics."""
    d_pred = depths_pred.detach().cpu().float()
    if d_pred.ndim == 4:
        d_pred = d_pred.squeeze(1)
    a_pred = alphas_pred.detach().cpu().float()
    if a_pred.ndim == 4:
        a_pred = a_pred.squeeze(1)
    sig_m = sigma_m_pred.detach().cpu().float()
    if sig_m.ndim == 4:
        sig_m = sig_m.squeeze(1)

    d_gt = depths_gt.detach().cpu().float()
    n_gt = normals_gt_cam.detach().cpu().float()
    gt_ok = gt_valid_mask.detach().cpu().bool()
    intr_cpu = intrinsics.detach().cpu().float()
    n_pred, smooth_ok = depth_map_to_camera_normals(d_pred, intr_cpu)

    V, H, W = d_pred.shape
    u, v = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + 0.5,
        torch.arange(H, dtype=torch.float32) + 0.5,
        indexing="xy",
    )

    frame_geo: list[dict[str, float]] = []
    pred_pts_list: list[torch.Tensor] = []
    gt_pts_list: list[torch.Tensor] = []

    for i in range(V):
        gt_i = gt_ok[i]
        m_i = gt_i & (a_pred[i] > float(alpha_thresh)) & (d_pred[i] > 0)
        gt_count = int(gt_i.sum().item())
        m_count = int(m_i.sum().item())
        cov = float(m_count / max(gt_count, 1))

        if m_count > 0:
            dp = d_pred[i][m_i]
            dg = d_gt[i][m_i]
            ratio = torch.maximum(dp / dg, dg / dp)
            scale_ratio_med = float((dp / dg).median().item())
            absrel = float(((dp - dg).abs() / dg).mean().item())
            rmse_m = float(torch.sqrt((dp - dg).square().mean()).item() / float(scene_scale))
            d105 = float((ratio < 1.05).float().mean().item())
            d110 = float((ratio < 1.10).float().mean().item())
            sig_v = sig_m[i][m_i]
            rt_med = float(sig_v.median().item())
            rt_p90 = float(sig_v.quantile(0.90).item())
        else:
            scale_ratio_med = float("nan")
            absrel = float("nan")
            rmse_m = float("nan")
            d105 = 0.0
            d110 = 0.0
            rt_med = float("nan")
            rt_p90 = float("nan")

        n_mask = m_i & smooth_ok[i] & (n_gt[i].norm(dim=-1) > 0.5)
        if int(n_mask.sum().item()) > 0:
            cos_val = (n_pred[i][n_mask] * n_gt[i][n_mask]).sum(dim=-1).abs().clamp(0.0, 1.0)
            ang_deg = torch.acos(cos_val) * (180.0 / math.pi)
            n_mean = float(ang_deg.mean().item())
            n_med = float(ang_deg.median().item())
            n_11 = float((ang_deg < 11.25).float().mean().item())
            n_30 = float((ang_deg < 30.0).float().mean().item())
        else:
            n_mean = float("nan")
            n_med = float("nan")
            n_11 = 0.0
            n_30 = 0.0

        frame_geo.append({
            "depth_cov": cov,
            "depth_scale_ratio_med": scale_ratio_med,
            "depth_absrel": absrel,
            "depth_rmse_m": rmse_m,
            "depth_delta_105": d105,
            "depth_delta_110": d110,
            "ray_thick_med_m": rt_med,
            "ray_thick_p90_m": rt_p90,
            "normal_mean_deg": n_mean,
            "normal_med_deg": n_med,
            "normal_lt11_25": n_11,
            "normal_lt30": n_30,
        })

        w2c = cam_view[i].detach().cpu().float().T
        c2w = torch.linalg.inv(w2c)
        fx, fy, cx, cy = intr_cpu[i].tolist()
        dirs_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
        dirs_world = dirs_cam @ c2w[:3, :3].T
        orig_world = c2w[:3, 3].expand_as(dirs_world)
        if m_count > 0:
            pred_pts_list.append(orig_world[m_i] + d_pred[i][m_i].unsqueeze(-1) * dirs_world[m_i])
        if gt_count > 0:
            gt_pts_list.append(orig_world[gt_i] + d_gt[i][gt_i].unsqueeze(-1) * dirs_world[gt_i])

    pred_pts = torch.cat(pred_pts_list, dim=0) if pred_pts_list else torch.empty((0, 3))
    gt_pts = torch.cat(gt_pts_list, dim=0) if gt_pts_list else torch.empty((0, 3))
    surf_dict = compute_surface_fscore(pred_pts, gt_pts, scene_scale=scene_scale, device=nn_device)

    scene_geo: dict[str, float] = {}
    for k in GEOMETRY_FRAME_METRICS:
        finite_vals = [f[k] for f in frame_geo if math.isfinite(f[k])]
        scene_geo[k] = statistics.mean(finite_vals) if finite_vals else float("nan")
    scene_geo.update(surf_dict)
    return frame_geo, scene_geo


TABLE_GEOMETRY_COLUMNS = (
    ("depth_cov", "深度覆盖率 ↑", 4),
    ("depth_absrel", "Depth AbsRel ↓", 4),
    ("depth_rmse_m", "Depth RMSE (m) ↓", 4),
    ("depth_delta_105", "δ<1.05 ↑", 4),
    ("depth_delta_110", "δ<1.10 ↑", 4),
    ("ray_thick_med_m", "视线厚度中位 (m) ↓", 4),
    ("ray_thick_p90_m", "视线厚度 p90 (m) ↓", 4),
    ("surf_acc_cm", "Accuracy (cm) ↓", 2),
    ("surf_comp_cm", "Completeness (cm) ↓", 2),
    ("surf_fscore_3cm", "F-score@3cm ↑", 4),
    ("surf_fscore_5cm", "F-score@5cm ↑", 4),
    ("normal_mean_deg", "法线均值 (°) ↓", 2),
    ("normal_med_deg", "法线中位 (°) ↓", 2),
    ("normal_lt11_25", "法线<11.25° ↑", 4),
    ("normal_lt30", "法线<30° ↑", 4),
)


def load_scene_rows(exp_dir: Path) -> dict[str, dict[str, float]]:
    csv_path = exp_dir / "scene_metrics.csv"
    if not csv_path.is_file():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            sid = row["scene_id"]
            parsed: dict[str, float] = {
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "lpips": float(row["lpips"]),
                "forward_seconds": float(row.get("forward_seconds", 0.0)),
            }
            for k in GEOMETRY_SCENE_METRICS:
                if k in row and row[k] not in ("", None):
                    parsed[k] = float(row[k])
            rows[sid] = parsed
    return rows


def paired_delta_psnr(rows: dict[str, dict[str, float]], ref_rows: dict[str, dict[str, float]]) -> str:
    common = sorted(set(rows) & set(ref_rows))
    if len(common) != EXPECTED_SCENES:
        return "—"
    diffs = [rows[s]["psnr"] - ref_rows[s]["psnr"] for s in common]
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    half_ci = 1.96 * sd / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    better = sum(1 for d in diffs if d > 0)
    worse = sum(1 for d in diffs if d < 0)
    return f"{mean:+.4f} [{mean - half_ci:+.4f}, {mean + half_ci:+.4f}] {better}/{worse}"


def is_experiment_complete(exp_dir: Path) -> bool:
    summary_path = exp_dir / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = load_scene_rows(exp_dir)
    return summary.get("num_scenes") == EXPECTED_SCENES and len(rows) == EXPECTED_SCENES


def has_geometry_metrics(exp_dir: Path) -> bool:
    if not is_experiment_complete(exp_dir):
        return False
    summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("geometry_metrics_status") != GEOMETRY_COMPUTED_STATUS:
        return False
    ratio = summary.get("geometry_alignment_median_ratio")
    if not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)) or not (0.90 <= float(ratio) <= 1.10):
        return False
    if not all(isinstance(summary.get(k), (int, float)) and math.isfinite(float(summary[k])) for k in GEOMETRY_SCENE_METRICS):
        return False
    rows = load_scene_rows(exp_dir)
    return all(
        all(k in r and math.isfinite(r[k]) for k in GEOMETRY_SCENE_METRICS)
        for r in rows.values()
    )


def demo_names(exp_root: Path = Path(EXP_SCANNETPP)) -> list[str]:
    out = []
    for exp in EXPERIMENTS:
        if is_experiment_complete(exp_root / exp.name):
            out.append(exp.name)
    return out


def build_scannetpp_table(exp_root: Path = Path(EXP_SCANNETPP)) -> tuple[Path, Path]:
    header = [
        "实验",
        "说明",
        "训练输入",
        "推理输入",
        "完成场景",
        "Δ 新视角 PSNR 对 baseline_pretrained_6view（95% CI，胜/负）",
        "新视角 PSNR ↑",
        "新视角 SSIM ↑",
        "新视角 LPIPS ↓",
        *(label for _, label, _ in TABLE_GEOMETRY_COLUMNS),
        "每场景前馈耗时 s",
        "显存峰值 MiB",
        "报告",
    ]
    ref_rows = load_scene_rows(exp_root / REFERENCE_EXPERIMENT)
    table_rows: list[list[str]] = []
    for exp in EXPERIMENTS:
        d = exp_root / exp.name
        summary_path = d / "summary.json"
        s_rows = load_scene_rows(d)
        if summary_path.is_file() and len(s_rows) == EXPECTED_SCENES:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            means = mean_metrics(list(s_rows.values()))
            delta = "0" if exp.name == REFERENCE_EXPERIMENT else paired_delta_psnr(s_rows, ref_rows)
            fwd = float(summary.get("forward_seconds_mean", statistics.mean(r["forward_seconds"] for r in s_rows.values())))
            mem = float(summary.get("peak_allocated_mib_including_lpips", 0.0))
            geo_cells: list[str] = []
            for key, _, nd in TABLE_GEOMETRY_COLUMNS:
                vals = [r[key] for r in s_rows.values() if key in r and math.isfinite(r[key])]
                geo_cells.append(f"{statistics.mean(vals):.{nd}f}" if len(vals) == EXPECTED_SCENES else "待补算")
            table_rows.append([
                f"`{exp.name}`",
                exp.title,
                exp.train_modalities,
                exp.eval_modalities,
                f"{len(s_rows)}/{EXPECTED_SCENES}",
                delta,
                f"{means['psnr']:.4f}",
                f"{means['ssim']:.4f}",
                f"{means['lpips']:.4f}",
                *geo_cells,
                f"{fwd:.2f}",
                f"{mem:.0f}",
                f"`report/{exp.name}.md`",
            ])
        else:
            table_rows.append([
                f"`{exp.name}`",
                exp.title,
                exp.train_modalities,
                exp.eval_modalities,
                f"{len(s_rows)}/{EXPECTED_SCENES}",
                "未完成",
                *(["未完成"] * (3 + len(TABLE_GEOMETRY_COLUMNS) + 2)),
                f"`report/{exp.name}.md`",
            ])

    md_lines = [
        "# EXP_ScanNetPP 全部实验总表（50 验证场景）",
        "",
        "由 `python scripts/scannetpp_evaluation_protocol.py build-table` 从各实验目录的 `scene_metrics.csv` 与 `summary.json` 自动生成，不要手改；"
        "新实验跑完后重新运行即可更新。同一份数据另存为 `all_experiments_table.csv`。",
        "",
        "- 每行一个实验，第一列对应实验文件夹 `EXP_ScanNetPP/<实验名>/`；详细配置与分析见 `EXP_ScanNetPP/report/<实验名>.md`。",
        "- 评估固定协议：ScanNet++ v2 官方 50 个验证场景（`nvs_sem_val.txt`），`window_size=18`（6 输入视角 → 8 个不重叠新视角，共 400 帧），"
        "分辨率 `256×448`，严格 FP32（TF32 关闭），推理仅输入 RGB 与相机，0 步单场景优化。",
        "- 同场景三维高斯点云与可视化对比位于 `EXP_ScanNetPP/demo/scene_7b6477cb95/<实验名>/`。",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in table_rows),
        "",
    ]
    md_path = exp_root / "all_experiments_table.md"
    csv_path = exp_root / "all_experiments_table.csv"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for r in table_rows:
            writer.writerow([cell.strip("`") for cell in r])
    return md_path, csv_path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd, rest = (argv[0], argv[1:]) if argv else ("names", [])
    if cmd == "names":
        print("\n".join(BY_NAME))
    elif cmd == "demo-names":
        print("\n".join(demo_names()))
    elif cmd == "complete-with-geo-names":
        exp_root = Path(rest[0]) if rest else Path(EXP_SCANNETPP)
        print("\n".join(exp.name for exp in EXPERIMENTS if has_geometry_metrics(exp_root / exp.name)))
    elif cmd == "train-preset" and len(rest) == 1 and rest[0] in BY_NAME:
        print(BY_NAME[rest[0]].train_preset or "")
    elif cmd == "checkpoint" and len(rest) == 1 and rest[0] in BY_NAME:
        print(BY_NAME[rest[0]].checkpoint)
    elif cmd == "args" and len(rest) == 1 and rest[0] in BY_NAME:
        print(" ".join(BY_NAME[rest[0]].run_args()))
    elif cmd == "demo-args" and len(rest) == 2 and rest[0] in BY_NAME:
        print(" ".join(BY_NAME[rest[0]].demo_args(rest[1])))
    elif cmd == "build-table":
        out_dir = Path(rest[0]) if rest else Path(EXP_SCANNETPP)
        md_path, csv_path = build_scannetpp_table(out_dir)
        print(f"wrote {md_path} and {csv_path}")
    else:
        print(f"unknown command or experiment: {argv}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
