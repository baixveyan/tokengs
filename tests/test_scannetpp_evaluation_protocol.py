import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scannetpp_evaluation_protocol import (  # noqa: E402
    BY_NAME, GEOMETRY_COMPUTED_STATUS, GEOMETRY_SCENE_METRICS,
    build_scannetpp_table, compute_scene_geometry_metrics,
    compute_surface_fscore, demo_names, depth_map_to_camera_normals,
    has_geometry_metrics, is_experiment_complete, mean_metrics,
    paired_delta_psnr, ray_thickness_from_moments, sha256_file,
    validate_frame_split, voxel_downsample_points, write_csv_atomic,
    write_json,
)
import demo_ply_export  # noqa: E402
import numpy as np
import torch


def test_novel_frame_split():
    validate_frame_split(list(range(6)), list(range(6, 14)))


@pytest.mark.parametrize("context,target", [
    (list(range(5)), list(range(6, 14))),
    (list(range(6)), list(range(5, 13))),
    ([0, 0, 2, 3, 4, 5], list(range(6, 14))),
    (list(range(6)), [6, 6, 8, 9, 10, 11, 12, 13]),
])
def test_invalid_or_leaking_frames_are_rejected(context, target):
    with pytest.raises(ValueError):
        validate_frame_split(context, target)


def test_scene_aggregation_and_invalid_numbers():
    a = mean_metrics([{"psnr": 20, "ssim": .8, "lpips": .2}] * 8)
    b = mean_metrics([{"psnr": 30, "ssim": 1., "lpips": 0.}] * 8)
    result = mean_metrics([a, b])
    assert result == {"psnr": 25., "ssim": .9, "lpips": .1}
    with pytest.raises(ValueError):
        mean_metrics([])
    with pytest.raises(ValueError):
        mean_metrics([{"psnr": float("nan"), "ssim": 1., "lpips": 0.}])


def test_json_and_hash(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"status": "running"})
    before = sha256_file(path)
    write_json(path, {"status": "complete"})
    assert json.loads(path.read_text())["status"] == "complete"
    assert sha256_file(path) != before
    assert not path.with_name(path.name + ".tmp").exists()


def test_scannetpp_registry_and_table(tmp_path):
    assert "baseline_pretrained_6view" in BY_NAME
    repo_root = Path(__file__).resolve().parents[1]
    assert "baseline_pretrained_6view" in demo_names(repo_root / "EXP_ScanNetPP")
    assert is_experiment_complete(repo_root / "EXP_ScanNetPP/baseline_pretrained_6view")

    ref_rows = {f"s{i:02d}": {"psnr": 30.0, "ssim": 0.9, "lpips": 0.2, "forward_seconds": 0.7} for i in range(50)}
    cur_rows = {f"s{i:02d}": {"psnr": 30.5, "ssim": 0.92, "lpips": 0.18, "forward_seconds": 0.7} for i in range(50)}
    delta = paired_delta_psnr(cur_rows, ref_rows)
    assert delta.startswith("+0.5000") and delta.endswith("50/0")

    md_path, csv_path = build_scannetpp_table(tmp_path)
    assert "未完成" in md_path.read_text(encoding="utf-8")
    assert csv_path.is_file()


def test_export_pruned_ply(tmp_path):
    g = torch.zeros(1, 16, 14)
    g[0, :8, 3] = 0.002
    g[0, 8:12, 3] = 0.02
    g[0, 12:, 3] = 0.2
    g[0, :, 4:7] = 0.05
    g[0, :, 7] = 1.0
    per = {
        "aniso": torch.ones(16),
        "cos_ray": torch.zeros(16),
        "parallax_deg": torch.zeros(16),
        "in_frustum_count": torch.ones(16),
    }
    full = tmp_path / "full.ply"
    p01 = tmp_path / "p01.ply"
    p05 = tmp_path / "p05.ply"
    demo_ply_export.export_gaussians_ply(full, g, 8, per)
    assert demo_ply_export.export_pruned_ply(full, p01, 0.01) == 8
    assert demo_ply_export.export_pruned_ply(full, p05, 0.05) == 4


def test_export_demo_from_artifact_rejects_missing_render_strip_or_mismatched_sha(tmp_path):
    import argparse
    import evaluate_scannetpp as eval_sc

    with pytest.raises(ValueError, match="render-strip-png"):
        eval_sc.export_demo_from_artifact(argparse.Namespace(
            from_artifact=tmp_path / "missing.pt",
            render_strip_png=None,
            output=tmp_path,
            ply_name="m",
            demo_output=tmp_path / "demo",
            checkpoint=tmp_path / "ckpt.bin",
        ))

    strip = tmp_path / "strip.png"
    strip.write_bytes(b"png")
    ckpt = tmp_path / "ckpt.bin"
    ckpt.write_bytes(b"weights")
    art = tmp_path / "art.pt"
    torch.save({"scene_id": "7b6477cb95"}, art)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    write_json(exp_dir / "summary.json", {"checkpoint_sha256": "wrong_sha"})
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        eval_sc.export_demo_from_artifact(argparse.Namespace(
            from_artifact=art,
            render_strip_png=strip,
            output=exp_dir,
            ply_name="m",
            demo_output=tmp_path / "demo",
            checkpoint=ckpt,
        ))

    real_sha = sha256_file(ckpt)
    write_json(exp_dir / "summary.json", {"checkpoint_sha256": real_sha})
    with pytest.raises(ValueError, match="missing verifiable checkpoint_sha256 provenance"):
        eval_sc.export_demo_from_artifact(argparse.Namespace(
            from_artifact=art,
            render_strip_png=strip,
            output=exp_dir,
            ply_name="m",
            demo_output=tmp_path / "demo",
            checkpoint=ckpt,
        ))

    torch.save({"scene_id": "7b6477cb95", "checkpoint_sha256": "other_sha"}, art)
    with pytest.raises(ValueError, match="provenance does not match"):
        eval_sc.export_demo_from_artifact(argparse.Namespace(
            from_artifact=art,
            render_strip_png=strip,
            output=exp_dir,
            ply_name="m",
            demo_output=tmp_path / "demo",
            checkpoint=ckpt,
        ))


def test_ray_thickness_single_and_two_layer_synthetic():
    # Single layer at z=1.5 with alpha=0.5 (background=0): thickness must be 0.0
    alpha = torch.full((1, 4, 4), 0.5)
    z0 = 1.5
    z1_single = alpha * z0
    z2_single = alpha * (z0 ** 2)
    sigma_single = ray_thickness_from_moments(z1_single, z2_single, alpha, scene_scale=0.3)
    assert torch.allclose(sigma_single, torch.zeros_like(sigma_single), atol=1e-6)

    # Two layers at z0 - d/2 and z0 + d/2 with equal weights w1 = w2 = 0.25 (alpha=0.5):
    # physical thickness must be (d / 2) / scene_scale
    d = 0.12  # normalized units -> 0.40m in physical units (since scene_scale=0.3), so half-spacing is 0.20m
    w = 0.25
    z_front, z_back = z0 - 0.5 * d, z0 + 0.5 * d
    z1_two = torch.full((1, 4, 4), w * z_front + w * z_back)
    z2_two = torch.full((1, 4, 4), w * (z_front ** 2) + w * (z_back ** 2))
    sigma_two = ray_thickness_from_moments(z1_two, z2_two, alpha, scene_scale=0.3)
    expected_m = (0.5 * d) / 0.3
    assert torch.allclose(sigma_two, torch.full_like(sigma_two, expected_m), atol=1e-5)


def test_depth_normals_and_surface_fscore_synthetic():
    # Front-parallel plane at both near (z=0.05, z=0.1) and mid (z=0.6) range -> normal must be [0, 0, -1] (0 deg error)
    intrinsics = torch.tensor([[200.0, 200.0, 8.0, 8.0]])
    for z_val in (0.05, 0.1, 0.6):
        depth = torch.full((1, 16, 16), z_val)
        normals, smooth = depth_map_to_camera_normals(depth, intrinsics)
        assert smooth[:, 1:-1, 1:-1].all()
        assert torch.allclose(normals[:, 1:-1, 1:-1, 2], torch.full((1, 14, 14), -1.0), atol=1e-5)

    # Introduce a >5% depth jump at column 8 -> adjacent columns masked out
    depth_jump = torch.full((1, 16, 16), 0.6)
    depth_jump[:, :, 8:] = 1.2
    _, smooth_jump = depth_map_to_camera_normals(depth_jump, intrinsics)
    assert not smooth_jump[0, 5, 7].item()
    assert not smooth_jump[0, 5, 8].item()

    # Identical point clouds -> Accuracy=0, Completeness=0, F-score@3cm=1, F-score@5cm=1
    pts = torch.rand(100, 3) * 0.5
    surf = compute_surface_fscore(pts, pts, scene_scale=0.3)
    assert surf["surf_acc_cm"] == pytest.approx(0.0, abs=1e-5)
    assert surf["surf_comp_cm"] == pytest.approx(0.0, abs=1e-5)
    assert surf["surf_fscore_3cm"] == pytest.approx(1.0)
    assert surf["surf_fscore_5cm"] == pytest.approx(1.0)

    # Empty valid mask (alpha=0 everywhere) must not crash compute_scene_geometry_metrics
    d_pred = torch.full((2, 8, 8), 0.5)
    a_pred = torch.zeros((2, 8, 8))
    sig_m = torch.zeros((2, 8, 8))
    d_gt = torch.full((2, 8, 8), 0.5)
    n_gt = torch.zeros((2, 8, 8, 3))
    n_gt[..., 2] = -1.0
    gt_ok = torch.ones((2, 8, 8), dtype=torch.bool)
    cam_view = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    intr_2 = torch.tensor([[50.0, 50.0, 4.0, 4.0], [50.0, 50.0, 4.0, 4.0]])
    f_geo, s_geo = compute_scene_geometry_metrics(d_pred, a_pred, sig_m, d_gt, n_gt, gt_ok, cam_view, intr_2)
    assert len(f_geo) == 2 and s_geo["depth_cov"] == 0.0


def test_write_csv_atomic_and_provider_raycast_and_s1_loss(tmp_path):
    import types
    from tokengs.data.provider import Provider
    from tokengs.models.input_types import ModelInputDecoder
    from tokengs.models.tokengs import TokenGS
    from tokengs.options import config_defaults

    csv_file = tmp_path / "metrics.csv"
    write_csv_atomic(csv_file, [{"scene_id": "s1", "psnr": 30.0}])
    assert csv_file.is_file() and not csv_file.with_name("metrics.csv.tmp").exists()

    # Synthetic triangle mesh in Provider._raycast_supervision_maps (tests uint32 indices & reflected=True/False)
    fake_prov = types.SimpleNamespace(
        opt=types.SimpleNamespace(img_size=(8, 8)),
        dataset=types.SimpleNamespace(root_path=tmp_path),
        _raw_mesh_cache={
            "sc0": (
                np.array([[-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [0.0, 2.0, 2.0]], dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.uint32),
            )
        },
    )
    w2n = torch.eye(4, dtype=torch.float32)
    cam_view = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    intr = torch.tensor([[8.0, 8.0, 4.0, 4.0]], dtype=torch.float32)
    for refl in (False, True):
        sup = Provider._raycast_supervision_maps(fake_prov, "sc0", w2n, refl, 0.3, cam_view, intr)
        assert sup["valid_gt_sup"].any()
        valid_d = sup["depths_gt_sup"][sup["valid_gt_sup"]]
        assert torch.allclose(valid_d, torch.full_like(valid_d, 0.6), atol=1e-5)
        valid_n = sup["normals_gt_sup"][sup["valid_gt_sup"]]
        assert torch.allclose(valid_n[:, 2], torch.full((valid_n.shape[0],), -1.0), atol=1e-5)

    # Test S1 compute_render_geo_supervision_loss gradient flow (depth + alpha + normal)
    opt_s1 = config_defaults["sft_scannetpp_render_geo"].evolve(lambda_render_thickness=0.0, img_size=(8, 8))
    dummy_self = types.SimpleNamespace(opt=opt_s1, img_size=(8, 8))
    d_pred = torch.full((1, 1, 1, 8, 8), 0.55, requires_grad=True)
    a_pred = torch.full((1, 1, 1, 8, 8), 0.8, requires_grad=True)
    n_gt = torch.zeros((1, 1, 8, 8, 3))
    n_gt[..., 2] = -1.0
    batch = {
        "depths_gt_sup": torch.full((1, 1, 8, 8), 0.6),
        "normals_gt_sup": n_gt,
        "valid_gt_sup": torch.ones((1, 1, 8, 8), dtype=torch.bool),
    }
    dec_in = ModelInputDecoder(cam_view=cam_view.unsqueeze(0), intrinsics=intr.unsqueeze(0))
    res = {"gaussians": torch.zeros((1, 16, 14)), "depths_pred": d_pred, "alphas_pred": a_pred}
    s1_losses = TokenGS.compute_render_geo_supervision_loss(dummy_self, batch, dec_in, res)
    s1_losses["loss_render_geo_total"].backward()
    assert d_pred.grad is not None and torch.isfinite(d_pred.grad).all() and d_pred.grad.abs().sum() > 0
    assert a_pred.grad is not None and torch.isfinite(a_pred.grad).all() and a_pred.grad.abs().sum() > 0

    # Interrupted backfill test: CSVs have geometry columns, but summary.json is still old RGB-only -> has_geometry_metrics must be False
    interrupted_dir = tmp_path / "interrupted_exp"
    interrupted_dir.mkdir()
    geo_defaults = {k: 1.0 for k in GEOMETRY_SCENE_METRICS}
    rows_50 = [
        {"scene_id": f"s{i:02d}", "psnr": 30.0, "ssim": 0.9, "lpips": 0.2, "forward_seconds": 0.7, **geo_defaults}
        for i in range(50)
    ]
    write_csv_atomic(interrupted_dir / "scene_metrics.csv", rows_50)
    write_json(interrupted_dir / "summary.json", {
        "num_scenes": 50,
        "geometry_metrics_status": "pending discussion; not computed",
    })
    assert is_experiment_complete(interrupted_dir)
    assert not has_geometry_metrics(interrupted_dir)

    # Once summary.json is atomically committed with GEOMETRY_COMPUTED_STATUS, alignment ratio, and geo means -> True
    write_json(interrupted_dir / "summary.json", {
        "num_scenes": 50,
        "geometry_metrics_status": GEOMETRY_COMPUTED_STATUS,
        "geometry_alignment_median_ratio": 1.0,
        **geo_defaults,
    })
    assert has_geometry_metrics(interrupted_dir)


def test_c0_and_s1_presets_and_parameter_freezing():
    from tokengs.options import config_defaults
    from tokengs.models.tokengs import TokenGS

    opt_c0 = config_defaults["sft_scannetpp_rgb_only"]
    opt_s1 = config_defaults["sft_scannetpp_render_geo"]
    assert opt_c0.sft_unfreeze_latent_tail and not opt_c0.use_render_geo_supervision
    assert opt_s1.sft_unfreeze_latent_tail and opt_s1.use_render_geo_supervision
    assert opt_s1.lambda_render_depth == 0.05
    assert opt_s1.lambda_render_alpha == 0.01
    assert opt_s1.lambda_render_thickness == 0.05
    assert opt_s1.lambda_render_normal == 0.02

    with torch.device("meta"):
        model = TokenGS(opt_c0)
    model.configure_sft_trainable_parameters()
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert "enc_dec_backbone.latents" in trainable_names
    assert "enc_dec_backbone.latent_kv_proj.weight" in trainable_names
    assert "enc_dec_backbone.k_proj_norm.weight" in trainable_names
    assert any(n.startswith("enc_dec_backbone.latent_blocks.8.") for n in trainable_names)
    assert not any(n.startswith("enc_dec_backbone.latent_blocks.7.") for n in trainable_names)
    assert not any(n.startswith("enc_dec_backbone.encoder.") for n in trainable_names)
    assert not any(n.startswith("activation_head.") for n in trainable_names)
    assert "gs_tokens" not in trainable_names


def test_ply_header_validation_rejects_unexpected_properties(tmp_path):
    from tokengs.data.static.scannetpp import fast_read_ply_vertices

    bad_ply = tmp_path / "bad.ply"
    bad_ply.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
        + b"\x00" * 24
    )
    with pytest.raises(ValueError, match="Unsupported PLY header"):
        fast_read_ply_vertices(bad_ply)
