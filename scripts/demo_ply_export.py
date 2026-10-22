"""PLY export for the demo scene, readable by the kleinicke.ply-visualizer extension.

- export_gaussians_ply: all Gaussians (no opacity pruning, unlike tokengs.rendering.gs.save_ply) in the
  standard 3DGS layout + slot_id/child_id and needle diagnostic scalar fields for Points-mode coloring.
- export_camera_ply: input cameras red, evaluation cameras green.
"""

from __future__ import annotations

import numpy as np
import torch


def export_gaussians_ply(path, gaussians: torch.Tensor, children: int, per: dict[str, torch.Tensor],
                         color_by_slot: bool = False) -> None:
    """All Gaussians (no opacity pruning) in standard 3DGS layout + slot_id/child_id/diagnostic fields."""
    from plyfile import PlyData, PlyElement

    g = gaussians[0].detach().float().cpu()
    n = g.shape[0]
    slot_id = torch.arange(n) // children
    xyz, opacity, scales, rot, rgb = g[:, 0:3], g[:, 3:4], g[:, 4:7], g[:, 7:11], g[:, 11:14]
    if color_by_slot:
        gen = torch.Generator().manual_seed(0)
        palette = torch.rand(int(slot_id.max()) + 1, 3, generator=gen) * 0.8 + 0.1
        rgb = palette[slot_id]
    f_dc = (rgb - 0.5) / 0.28209479177387814
    columns = {
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "f_dc_0": f_dc[:, 0], "f_dc_1": f_dc[:, 1], "f_dc_2": f_dc[:, 2],
        "opacity": torch.logit(opacity[:, 0], eps=1e-6),
        "scale_0": scales[:, 0].clamp_min(1e-8).log(), "scale_1": scales[:, 1].clamp_min(1e-8).log(),
        "scale_2": scales[:, 2].clamp_min(1e-8).log(),
        "rot_0": rot[:, 0], "rot_1": rot[:, 1], "rot_2": rot[:, 2], "rot_3": rot[:, 3],
        "slot_id": slot_id.float(), "child_id": (torch.arange(n) % children).float(),
        "opacity_activated": opacity[:, 0],
        "aniso": per["aniso"].cpu(), "cos_long_axis_ray": per["cos_ray"].cpu(),
        "parallax_deg": per["parallax_deg"].cpu(), "in_input_frustums": per["in_frustum_count"].cpu(),
    }
    arr = np.empty(n, dtype=[(k, "f4") for k in columns])
    for k, v in columns.items():
        arr[k] = v.numpy().astype(np.float32)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def export_camera_ply(path, context_c2w: torch.Tensor, target_c2w: torch.Tensor | None) -> None:
    """Camera centers as colored points: context red, target green."""
    from plyfile import PlyData, PlyElement

    pts, cols = [context_c2w[:, :3, 3].cpu().numpy()], [np.tile([255, 0, 0], (context_c2w.shape[0], 1))]
    if target_c2w is not None:
        pts.append(target_c2w[:, :3, 3].cpu().numpy())
        cols.append(np.tile([0, 255, 0], (target_c2w.shape[0], 1)))
    p, c = np.concatenate(pts), np.concatenate(cols)
    arr = np.empty(p.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, k in enumerate("xyz"):
        arr[k] = p[:, i]
    for i, k in enumerate(("red", "green", "blue")):
        arr[k] = c[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def export_colmap_points_ply(demo_dir, zip_path, scene_id: str, context_ids: list[int],
                             intrinsics_input: torch.Tensor, img_size: tuple[int, int] | list[int],
                             scene_scale: float, znear: float) -> dict[str, str]:
    """Export COLMAP 3D point clouds aligned to TokenGS normalized scene coordinates:
    - colmap_supervised_points.ply: only the COLMAP 3D points that project into the 6 input views (true RGB + metadata)
    - colmap_supervised_points_highlight.ply: supervised points highlighted in bright cyan/red by view count
    - colmap_all_points.ply: all COLMAP 3D points of the scene (with is_supervised=0/1 field)
    """
    import json
    import struct
    import zipfile
    from pathlib import Path
    from plyfile import PlyData, PlyElement

    demo_dir = Path(demo_dir)
    H, W = int(img_size[0]), int(img_size[1])
    clip_name = f"{scene_id}/gaussian_splat"
    with zipfile.ZipFile(zip_path, "r") as zf:
        tj = json.loads(zf.open(f"{clip_name}/transforms.json").read())
        raw = zf.open(f"{clip_name}/sparse/0/points3D.bin").read()

    off = 0
    (n_pts,) = struct.unpack_from("<Q", raw, off)
    off += 8
    xyz = np.empty((n_pts, 3), dtype=np.float64)
    rgb = np.empty((n_pts, 3), dtype=np.uint8)
    err = np.empty(n_pts, dtype=np.float32)
    tracks: list[frozenset[int]] = []
    for i in range(n_pts):
        off += 8
        xyz[i] = struct.unpack_from("<ddd", raw, off)
        off += 24
        rgb[i] = struct.unpack_from("<BBB", raw, off)
        off += 3
        (err[i],) = struct.unpack_from("<d", raw, off)
        off += 8
        (t_len,) = struct.unpack_from("<Q", raw, off)
        off += 8
        tr = struct.unpack_from("<" + "II" * t_len, raw, off)
        off += 8 * t_len
        tracks.append(frozenset(tr[::2]))

    from tokengs.data.static.dl3dv import DL3DVEval
    w_trans = DL3DVEval._colmap_world_to_final_world_transform(tj)
    c2ws_all = DL3DVEval.load_cameras(None, tj)
    c2ws_ctx = torch.from_numpy(c2ws_all[context_ids]).float()
    pos_avg = c2ws_ctx[:, :3, 3].mean(0)
    fwd_avg = torch.nn.functional.normalize(c2ws_ctx[:, :3, 2].mean(0), dim=0)
    down_avg = c2ws_ctx[:, :3, 1].mean(0)
    down_avg = torch.nn.functional.normalize(down_avg - down_avg.dot(fwd_avg) * fwd_avg, dim=0)
    right_avg = torch.linalg.cross(down_avg, fwd_avg)
    pose_avg = torch.eye(4)
    pose_avg[:3, 0], pose_avg[:3, 1], pose_avg[:3, 2], pose_avg[:3, 3] = right_avg, down_avg, fwd_avg, pos_avg
    pose_avg_inv = torch.linalg.inv(pose_avg).double().numpy()

    xyz_h = np.concatenate([xyz, np.ones((n_pts, 1), dtype=np.float64)], axis=1)
    xyz_w = (xyz_h @ w_trans.T)[:, :3]
    pts_norm = ((xyz_w @ pose_avg_inv[:3, :3].T + pose_avg_inv[:3, 3]) * float(scene_scale)).astype(np.float32)

    # Determine which points actually project inside the 6 input views and act as depth supervision
    intr = intrinsics_input.detach().cpu().float().numpy()
    if intr.ndim == 3:
        intr = intr[0]
    sup_view_count = np.zeros(n_pts, dtype=np.int32)
    for v_i, f_idx in enumerate(context_ids):
        cid = tj["frames"][int(f_idx)].get("colmap_im_id")
        if cid is None:
            continue
        vis_idx = np.flatnonzero([cid in s for s in tracks])
        if vis_idx.size == 0:
            continue
        w2c_raw = np.linalg.inv(c2ws_all[int(f_idx)])
        pts_cam = (xyz_w[vis_idx] @ w2c_raw[:3, :3].T + w2c_raw[:3, 3]) * float(scene_scale)
        z = pts_cam[:, 2]
        fx, fy, cx, cy = intr[v_i]
        u = np.round(fx * pts_cam[:, 0] / np.maximum(z, 1e-6) + cx).astype(np.int64)
        v = np.round(fy * pts_cam[:, 1] / np.maximum(z, 1e-6) + cy).astype(np.int64)
        valid = (z > float(znear)) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        sup_view_count[vis_idx[valid]] += 1

    sup_mask = sup_view_count > 0

    def _write_ply(path: Path, p: np.ndarray, c: np.ndarray, views: np.ndarray, reproj: np.ndarray) -> None:
        arr = np.empty(p.shape[0], dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("num_input_views", "f4"), ("reproj_error", "f4"),
        ])
        arr["x"], arr["y"], arr["z"] = p[:, 0], p[:, 1], p[:, 2]
        arr["red"], arr["green"], arr["blue"] = c[:, 0], c[:, 1], c[:, 2]
        arr["num_input_views"] = views.astype(np.float32)
        arr["reproj_error"] = reproj.astype(np.float32)
        PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))

    # 1. Supervised points (true RGB)
    p_sup = demo_dir / "colmap_supervised_points.ply"
    _write_ply(p_sup, pts_norm[sup_mask], rgb[sup_mask], sup_view_count[sup_mask], err[sup_mask])

    # 2. Supervised points (highlighted: 1 input view = cyan [0, 220, 255], >=2 input views = magenta [255, 0, 200])
    c_hi = np.where((sup_view_count[sup_mask, None] >= 2),
                    np.array([255, 0, 200], dtype=np.uint8),
                    np.array([0, 220, 255], dtype=np.uint8))
    p_hi = demo_dir / "colmap_supervised_points_highlight.ply"
    _write_ply(p_hi, pts_norm[sup_mask], c_hi, sup_view_count[sup_mask], err[sup_mask])

    # 3. All scene COLMAP points (true RGB, with num_input_views > 0 indicating supervised subset)
    p_all = demo_dir / "colmap_all_points.ply"
    _write_ply(p_all, pts_norm, rgb, sup_view_count, err)

    return {
        "colmap_supervised_ply": p_sup.name,
        "colmap_supervised_highlight_ply": p_hi.name,
        "colmap_all_ply": p_all.name,
    }
