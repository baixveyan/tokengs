"""Small binary PLY writers for demo files (viewable with VS Code kleinicke.ply-visualizer)."""

from pathlib import Path

import numpy as np


def _rgb8(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float64)
    return (np.clip(colors, 0, 1) * 255).round().astype(np.uint8)


def write_points_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rec = np.empty(len(points), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = points, _rgb8(colors).reshape(-1, 3)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(rec.tobytes())


def write_octahedra_ply(path: Path, centers: np.ndarray, colors: np.ndarray, radius: float) -> None:
    """One small octahedron per centre, so sparse points stay visible regardless of the viewer's point size."""
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    offs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32) * radius
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], np.int32)
    n = len(centers)
    verts = (centers[:, None] + offs[None]).reshape(-1, 3)
    cols = np.repeat(_rgb8(colors).reshape(-1, 3), 6, axis=0)
    vrec = np.empty(len(verts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    vrec["p"], vrec["c"] = verts, cols
    frec = np.empty(n * 8, dtype=[("k", "u1"), ("i", "<i4", 3)])
    frec["k"] = 3
    frec["i"] = (faces[None] + 6 * np.arange(n, dtype=np.int32)[:, None, None]).reshape(-1, 3)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 f"element face {n * 8}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        f.write(vrec.tobytes())
        f.write(frec.tobytes())


def write_cubes_ply(path: Path, centers: np.ndarray, colors: np.ndarray, half: float) -> None:
    """One axis-aligned cube (12 triangles) per grid cell centre."""
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32) * half
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int32)
    n = len(centers)
    vrec = np.empty(n * 8, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    vrec["p"] = (centers[:, None] + corners[None]).reshape(-1, 3)
    vrec["c"] = np.repeat(_rgb8(colors).reshape(-1, 3), 8, axis=0)
    frec = np.empty(n * 12, dtype=[("k", "u1"), ("i", "<i4", 3)])
    frec["k"] = 3
    frec["i"] = (faces[None] + 8 * np.arange(n, dtype=np.int32)[:, None, None]).reshape(-1, 3)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {n * 8}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 f"element face {n * 12}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        f.write(vrec.tobytes())
        f.write(frec.tobytes())


def error_colors(dist: np.ndarray, full_scale: float) -> np.ndarray:
    """green (0) -> yellow (full_scale / 2) -> red (>= full_scale)."""
    t = np.clip(np.asarray(dist, dtype=np.float64) / full_scale, 0, 1)
    return np.stack([np.clip(2 * t, 0, 1), np.clip(2 - 2 * t, 0, 1), np.zeros_like(t)], -1)


def write_section_preview(demo_dir: Path, gaussians: np.ndarray, anchored: np.ndarray,
                          anchors: np.ndarray, c2w: np.ndarray, scene_scale: float) -> None:
    """Camera-horizontal 6 cm slab: laser / anchored / free centres, in physical metres.

    This is a diagnostic plot, not a thickness metric; Gaussian covariance and alpha
    compositing are not represented. GT is read only after reconstruction is exported.
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plyfile import PlyData

    mesh = demo_dir.parent / "gt_lidar_mesh_frustum.ply"
    if not mesh.is_file():
        return
    vertices = PlyData.read(str(mesh))["vertex"]
    laser = np.stack([vertices[k] for k in ("x", "y", "z")], -1)
    w2c = np.linalg.inv(c2w)

    def camera(points):
        return (points @ w2c[:3, :3].T + w2c[:3, 3]) / scene_scale

    laser, centers, anchor_cam = camera(laser), camera(gaussians[:, :3]), camera(anchors)
    visible = laser[:, 2] > 0
    if not visible.any():
        return
    plane_y = float(np.median(laser[visible, 1]))

    def slab(points):
        return (np.abs(points[:, 1] - plane_y) < .03) & (points[:, 2] > 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    layers = [
        (centers, slab(centers) & (gaussians[:, 3] >= .05) & ~anchored, "#999999", "Free Gaussian centres", 4),
        (centers, slab(centers) & (gaussians[:, 3] >= .05) & anchored, "#e45858", "Anchored Gaussian centres", 5),
        (laser, slab(laser), "#16804a", "Laser surface vertices", 5),
        (anchor_cam, slab(anchor_cam), "#245bce", "SfM anchors", 20),
    ]
    for points, keep, color, label, size in layers:
        selected = points[keep]
        stride = max(1, int(np.ceil(len(selected) / 20000)))
        ax.scatter(selected[::stride, 0], selected[::stride, 2], s=size, c=color, label=label, alpha=.7)
    ax.set(xlabel="Camera x (m)", ylabel="Camera z (m)",
           title=f"6 cm horizontal slab at camera y={plane_y:.2f} m; opacity >= 0.05")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(demo_dir / "local_section.png", dpi=160)
    plt.close(fig)
    (demo_dir / "local_section.json").write_text(json.dumps({
        "camera": "first input camera", "camera_y_m": plane_y, "slab_width_m": .06,
        "opacity_threshold": .05, "note": "Centre plot only, not a compositing thickness metric.",
    }, indent=2))
