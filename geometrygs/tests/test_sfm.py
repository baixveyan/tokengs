import math

import torch

import geometrygs  # noqa: F401
from geometrygs.sfm import (
    KnownPoseSfM, SfmConfig, farthest_point_sample, fundamental_matrix, intrinsics_matrix, project,
    sampson_distance, select_anchors, triangulate_dlt,
)


def look_at(eye, target=(0.0, 0.0, 0.0)):
    eye, target = torch.tensor(eye, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    fwd = torch.nn.functional.normalize(target - eye, dim=0)
    right = torch.nn.functional.normalize(torch.linalg.cross(torch.tensor([0.0, -1.0, 0.0]), fwd), dim=0)
    down = torch.linalg.cross(fwd, right)
    c2w = torch.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
    return c2w


def scene(num_views=4, num_points=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    c2w = torch.stack([look_at((0.3 * math.cos(a), 0.05 * i, -2.0 + 0.3 * math.sin(a)))
                       for i, a in enumerate(torch.linspace(0, 1.2, num_views).tolist())])
    intr = torch.tensor([[400.0, 400.0, 448.0, 256.0]]).repeat(num_views, 1)
    pts = (torch.rand(num_points, 3, generator=g) - 0.5) * torch.tensor([1.5, 1.0, 0.6])
    return c2w, intr, pts


def test_epipolar_and_triangulation_are_exact_on_noise_free_data():
    c2w, intr, pts = scene()
    w2c = torch.linalg.inv(c2w)
    uv, z = project(pts, w2c, intr)
    assert (z > 0).all()
    K = intrinsics_matrix(intr)
    Fm = fundamental_matrix(w2c[0].double(), w2c[2].double(), K[0].double(), K[2].double()).float()
    assert sampson_distance(Fm, uv[:, 0], uv[:, 2]).max() < 1e-2
    shifted = uv[:, 2] + torch.tensor([0.0, 5.0])
    assert sampson_distance(Fm, uv[:, 0], shifted).median() > 1.0
    mask = torch.ones(len(pts), 4, dtype=torch.bool)
    mask[::3, 1] = False
    rec = triangulate_dlt(uv, mask, w2c, intr)
    assert (rec - pts).norm(dim=-1).max() < 1e-3


def test_tracks_and_filters_reject_outliers():
    c2w, intr, pts = scene(num_views=3, num_points=50)
    uv, _ = project(pts, torch.linalg.inv(c2w), intr)
    feats = [(uv[:, v] + 0.0, torch.eye(50)) for v in range(3)]
    sfm = KnownPoseSfM(SfmConfig(min_tri_angle_deg=0.5))
    ids = torch.arange(50)
    pairs = [(0, 1, torch.stack([ids, ids], 1)), (1, 2, torch.stack([ids, ids], 1))]
    bad = torch.stack([ids[:5], (ids[:5] + 7) % 50], 1)  # wrong correspondences bypassing the epipolar filter
    pairs.append((0, 2, bad))
    obs_uv, obs_mask = sfm.build_tracks(feats, pairs, 3)
    rec = sfm.triangulate(obs_uv, obs_mask, c2w, intr)
    assert rec["points"].shape[0] >= 35
    dist = torch.cdist(rec["points"], pts).min(1).values
    assert dist.max() < 1e-3
    assert (rec["reproj_px"] < 2.0).all() and (rec["angle_deg"] > 0.5).all()


def test_fps_spreads_points_and_respects_k():
    pts = torch.cat([torch.zeros(100, 3), torch.ones(3, 3) * torch.tensor([[1.0], [2.0], [3.0]])])
    idx = farthest_point_sample(pts, 4)
    assert len(set(idx.tolist())) == 4 and {100, 101, 102}.issubset(set(idx.tolist()))
    cluster = torch.rand(200, 3) * 0.05
    pts = torch.cat([cluster, torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]])])
    sfm = {"points": pts, "track_len": torch.full((202,), 3), "reproj_px": torch.zeros(202), "angle_deg": torch.zeros(202)}
    sfm["track_len"][:20] = 2  # weak: two views and a small angle
    sel = select_anchors(sfm, 1000)
    assert set(sel.tolist()) == set(range(20, 200))  # isolated outliers and weak points never become anchors
    assert select_anchors(sfm, 16).numel() == 16
    assert select_anchors({"points": torch.zeros(0, 3)}, 10).numel() == 0
