"""Demo: where the input-view SfM points are, which become token anchors, and which surface they do not cover.

For each requested ScanNet++ validation scene (6 inputs, window 18, the evaluation views) it writes, in TokenGS
normalized coordinates (same frame as EXP_ScanNetPP/demo/*/ Gaussians and gt_lidar_mesh_frustum.ply):

  sfm_points_status.ply        every triangulated point, coloured by what the anchor filter did with it
  sfm_points_laser_error.ply   every triangulated point, coloured by distance to the laser mesh (0 -> 10 cm)
  sfm_anchors.ply              the selected anchors as small octahedra (visible at any point size)
  anchor_coverage_laser.ply    laser surface seen by the inputs, coloured by distance to the nearest anchor
  gt_lidar_mesh_frustum.ply, cameras.ply
  input_views_sfm.png          the input views with every observation drawn in the status colour
  input_views_coverage.png     the input views with laser pixels coloured by distance to the nearest anchor
  stats.json, README.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import geometrygs  # noqa: F401
import demo_ply_export
from geometrygs.data import SfmProvider, split_data_with_sfm
from geometrygs.model import anchor_quality_kwargs
from geometrygs.options import config_defaults
from geometrygs.plyio import error_colors, write_octahedra_ply, write_points_ply
from geometrygs.sfm import KnownPoseSfM, SfmConfig, anchor_quality_components, select_anchors
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals

STATUS = {
    0: ("anchor (used as token position)", (0.1, 0.9, 0.1)),
    1: ("reliable, not selected (beyond 1024 or FPS redundant)", (0.2, 0.6, 1.0)),
    2: ("rejected: weak (2 views and angle < 3 deg)", (1.0, 0.6, 0.0)),
    3: ("rejected: isolated (5-NN mean > 10 cm)", (1.0, 0.0, 0.0)),
}
COVER = [(0.10, (0.1, 0.85, 0.1), "< 10 cm"), (0.20, (1.0, 0.85, 0.0), "10-20 cm"), (np.inf, (0.9, 0.1, 0.1), "> 20 cm")]


def cover_colors(d: np.ndarray) -> np.ndarray:
    out = np.zeros((len(d), 3))
    lo = -np.inf
    for hi, col, _ in COVER:
        out[(d >= lo) & (d < hi)] = col
        lo = hi
    return out


def nearest_dist(query: torch.Tensor, ref: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    if ref.shape[0] == 0:
        return torch.full((query.shape[0],), float("inf"))
    return torch.cat([torch.cdist(query[i:i + chunk], ref).min(1).values for i in range(0, len(query), chunk)])


def grid(tiles: list[Image.Image], cols: int = 3) -> Image.Image:
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), "white")
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % cols) * w, (i // cols) * h))
    return canvas


def run_scene(provider, opt, sfm, scene_id: str, out_dir: Path, device: torch.device) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = [Path(p).name for p in provider.dataset.sample_list].index(scene_id)
    raw = provider[idx]
    batch = {k: v[None] if torch.is_tensor(v) else v for k, v in raw.items()}
    mi, _ = split_data_with_sfm(batch, opt)
    V = opt.num_input_views
    scale = float(provider.scene_scale)
    ctx = raw["frame_ids"][:V].tolist()
    raw_c2w = torch.from_numpy(provider.dataset._scene_meta_cache[scene_id]["c2ws"][ctx].copy())
    mesh = provider.dataset.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    rc = build_raycasting_scene(mesh, raw_c2w, scale)

    images = mi.sfm["images"][0]
    c2w = mi.encoder.cam_to_world_input[0]
    intr_sfm = mi.sfm["intrinsics"][0]
    rec = sfm.reconstruct(images.to(device), c2w.to(device), intr_sfm.to(device))
    rec = {k: v.cpu() if torch.is_tensor(v) else v for k, v in rec.items()}
    quality = anchor_quality_kwargs(opt)
    strong, dense = anchor_quality_components(rec, **quality)
    sel = select_anchors(rec, opt.sfm_num_anchors, **quality)
    status = np.where(~dense.numpy(), 3, np.where(~strong.numpy(), 2, 1))
    status[sel.numpy()] = 0
    pts = rec["points"].numpy()
    anchors = pts[sel.numpy()]

    import open3d as o3d

    d_laser = rc.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy() / scale if len(pts) else np.zeros(0)
    status_col = np.array([STATUS[s][1] for s in status]).reshape(-1, 3)
    write_points_ply(out_dir / "sfm_points_status.ply", pts, status_col)
    write_points_ply(out_dir / "sfm_points_laser_error.ply", pts, error_colors(d_laser, 0.10))
    write_octahedra_ply(out_dir / "sfm_anchors.ply", anchors, np.tile([[0.1, 0.9, 0.1]], (len(anchors), 1)), 0.004)
    demo_ply_export.export_camera_ply(out_dir / "cameras.ply", c2w.float(), None)
    demo_ply_export.export_scannetpp_lidar_ply(out_dir / "gt_lidar_mesh_frustum.ply", mesh, raw_c2w,
                                               raw["cam_view_input"], raw["intrinsics_input"], opt.img_size, scale)

    # laser surface seen by the inputs (model resolution) and its distance to the nearest anchor
    H, W = opt.img_size
    depth, _, ok = raycast_gt_depth_and_normals(rc, raw["cam_view_input"], raw["intrinsics_input"], opt.img_size)
    u, v = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy")
    surf_maps = []
    for i in range(V):
        fx, fy, cx, cy = raw["intrinsics_input"][i].tolist()
        cam = torch.stack([(u - cx) / fx * depth[i], (v - cy) / fy * depth[i], depth[i]], -1)
        surf_maps.append(cam @ c2w[i, :3, :3].T + c2w[i, :3, 3])
    surf = torch.stack(surf_maps)  # [V, H, W, 3]
    d_cov = torch.full((V, H, W), float("inf"))
    d_cov[ok] = nearest_dist(surf[ok], torch.from_numpy(anchors)) / scale
    sub = ok.clone()
    sub[:, 1::2] = False
    sub[:, :, 1::2] = False
    write_points_ply(out_dir / "anchor_coverage_laser.ply", surf[sub].numpy(), cover_colors(d_cov[sub].numpy()))

    # 2D: observations in the status colour, coverage overlay
    half = torch.nn.functional.interpolate(images.float(), scale_factor=0.5, mode="area").clamp(0, 1)
    obs_tiles, cov_tiles = [], []
    for i in range(V):
        img = Image.fromarray((half[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        seen = rec["obs_mask"][:, i].numpy()
        order = np.argsort(-status[seen])  # draw anchors last (on top)
        uv = rec["obs_uv"][seen, i].numpy()[order] * 0.5
        for (x, y), s in zip(uv, status[seen][order]):
            c = tuple(int(255 * k) for k in STATUS[s][1])
            r = 4 if s == 0 else 3
            draw.ellipse([x - r, y - r, x + r, y + r], fill=c, outline=(0, 0, 0) if s == 0 else None)
        draw.text((8, 6), f"input view {i}  frame {ctx[i]}", fill=(255, 255, 255))
        obs_tiles.append(img)

        base = raw["images_input"][i].permute(1, 2, 0).numpy()
        col = np.full((H, W, 3), 0.35)
        okm = ok[i].numpy()
        col[okm] = cover_colors(d_cov[i].numpy()[okm])
        blend = Image.fromarray((np.clip(0.45 * base + 0.55 * col, 0, 1) * 255).astype(np.uint8)).resize((W * 2, H * 2), Image.NEAREST)
        ImageDraw.Draw(blend).text((8, 6), f"input view {i}: laser pixel -> nearest anchor", fill=(255, 255, 255))
        cov_tiles.append(blend)
    grid(obs_tiles).save(out_dir / "input_views_sfm.png")
    grid(cov_tiles).save(out_dir / "input_views_coverage.png")

    finite = d_cov[ok].numpy()
    stats = {
        "scene_id": scene_id, "context_frames": ctx, "num_points": int(len(pts)), "num_anchors": int(len(anchors)),
        "status_counts": {STATUS[k][0]: int((status == k).sum()) for k in STATUS},
        "laser_distance_cm_by_status": {
            STATUS[k][0]: {"median": float(np.median(d_laser[status == k]) * 100), "p90": float(np.quantile(d_laser[status == k], 0.9) * 100)}
            for k in STATUS if (status == k).any()
        },
        "laser_pixels_nearest_anchor": {lab: float(((finite >= lo) & (finite < hi)).mean())
                                        for (hi, _, lab), lo in zip(COVER, [-np.inf, 0.10, 0.20])},
        "units": "metres / centimetres in the real scene (normalized / 0.3)",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    write_readme(out_dir, stats)
    return stats


def write_readme(out_dir: Path, s: dict) -> None:
    status_lines = "\n".join(
        f"| {name} | {s['status_counts'][name]} | "
        + (f"{s['laser_distance_cm_by_status'][name]['median']:.1f} / {s['laser_distance_cm_by_status'][name]['p90']:.1f}"
           if name in s["laser_distance_cm_by_status"] else "-") + " |"
        for name in s["status_counts"])
    colors = {"anchor": "绿", "reliable": "蓝", "rejected: weak": "橙", "rejected: isolated": "红"}
    legend = "，".join(f"{v} = {k}" for k, v in colors.items())
    cov = "，".join(f"{k}: {v * 100:.0f}%" for k, v in s["laser_pixels_nearest_anchor"].items())
    (out_dir / "README.md").write_text(f"""# SfM 点 demo：{s['scene_id']}

输入帧 {s['context_frames']}（验证协议的 6 个输入视角）。所有 PLY 在 TokenGS 归一化坐标系，可与
`EXP_ScanNetPP/demo/scene_7b6477cb95/*/` 的高斯 PLY 直接叠加（仅 7b6477cb95 有这些高斯 demo）。

| 文件 | 看什么 |
|---|---|
| `sfm_points_status.ply` + `gt_lidar_mesh_frustum.ply` | 所有三角化点。{legend} |
| `sfm_points_laser_error.ply` | 同一批点按离激光网格距离着色：绿 0 → 黄 5 cm → 红 ≥ 10 cm |
| `sfm_anchors.ply` | 最终作为 token 固定位置的点（小八面体） |
| `anchor_coverage_laser.ply` | 输入视角看到的激光表面，按离最近锚点距离着色：绿 < 10 cm，黄 10–20 cm，红 > 20 cm |
| `cameras.ply` | 6 个输入相机中心（红） |
| `input_views_sfm.png` | 输入图上每个观测点，颜色同上（锚点带黑边） |
| `input_views_coverage.png` | 输入图上激光像素离最近锚点的距离（灰 = 无激光） |

| 点的类别 | 数量 | 离激光面 中位 / p90 (cm) |
|---|---:|---:|
{status_lines}

输入可见激光表面离最近锚点：{cov}。
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="+", default=["7b6477cb95", "e7af285f7d", "27dd4da69e"])
    ap.add_argument("--output-root", type=Path, default=Path("EXP/demo"))
    ap.add_argument("--name", default="sfm_quality_val6v")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    opt = config_defaults["eval_sfm_anchor"].evolve(num_workers=0, dataset_kwargs={"window_size": 18})
    provider = SfmProvider("scannetpp_val_scaled_0.3", opt, training=False)
    sfm = KnownPoseSfM(SfmConfig(num_features=opt.sfm_num_features, match_ratio=opt.sfm_match_ratio,
                                 max_epipolar_px=opt.sfm_max_epipolar_px, max_reproj_px=opt.sfm_max_reproj_px,
                                 min_tri_angle_deg=opt.sfm_min_tri_angle_deg, min_track=opt.sfm_min_track,
                                 z_near=opt.znear, z_far=opt.zfar))
    for sid in args.scenes:
        s = run_scene(provider, opt, sfm, sid, args.output_root / f"scene_{sid[:12]}" / args.name, torch.device(args.device))
        print(json.dumps({k: s[k] for k in ("scene_id", "num_points", "num_anchors", "status_counts", "laser_pixels_nearest_anchor")}), flush=True)


if __name__ == "__main__":
    main()
