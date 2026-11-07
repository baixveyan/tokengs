"""Geometry invariants and integration contracts for the scratch architecture."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from geometrygs import SpatialTokenGS
from geometrygs.data import SfmProvider, split_data_with_sfm
from geometrygs.options import config_defaults
from geometrygs.soft_chain import planar_map, surface_pair_loss
from geometrygs.spatial_model import options_from_architecture
from tokengs.models.tokengs import TokenGS
from test_model import fake_batch


def spatial_opt(**changes):
    base = dict(img_size=(32, 56), sfm_image_size=(64, 112), enc_embed_dim=32, token_dim=32,
                enc_num_heads=4, enc_depth=1, latent_cross_attn_depth=1, dec_depth=2,
                num_latents=8, num_gs_tokens=12, dec_patch_size=2, sfm_num_anchors=4,
                child_hidden_dim=16, spatial_query_chunk=5, child_token_chunk=5,
                sfm_dropout=0., sfm_point_dropout=0., num_input_views=3, num_views=4)
    base.update(changes)
    return config_defaults["spatial_small_sfm"].evolve(**base)


def records(n=3):
    return [{
        "points": torch.tensor([[0.0, 0., 1.], [0.1, 0., 1.], [-.1, 0., 1.]])[:n],
        "color": torch.ones(n, 3),
        "obs_mask": torch.ones(n, 3, dtype=torch.bool),
        "anchor_index": torch.arange(n),
        "track_len": torch.full((n,), 3), "angle_deg": torch.full((n,), 5.),
        "reproj_px": torch.full((n,), .2),
    }]


@pytest.mark.parametrize("n", [0, 3])
def test_bounds_fixed_centers_and_all_parameter_gradients(n):
    torch.manual_seed(7)
    opt = spatial_opt()
    model = SpatialTokenGS(opt).train()
    model.compute_anchors = lambda mi: records(n)
    mi, _ = split_data_with_sfm(fake_batch(), opt)
    g = model.forward_reconstruction(mi).gaussians
    info = model.last_spatial
    children = g.view(1, opt.num_gs_tokens, 4, 14)
    assert info["anchored"].sum() == n
    assert torch.equal(info["centers"][0, :n], records(n)[0]["points"])
    offsets = (children[..., :3] - info["centers"][:, :, None]).norm(dim=-1)
    assert torch.all(offsets <= info["radii"] + 1e-6)
    assert torch.all(children[..., 4:7] <= info["radii"][:, :, None] * opt.child_scale_radius_ratio + 1e-6)
    assert torch.allclose(children[..., 7:11].norm(dim=-1), torch.ones_like(offsets), atol=1e-6)
    g.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if n:
        for name in ("sfm_position.weight", "sfm_feature.weight", "scene_latents", "gs_tokens",
                     "initial_center.3.weight", "decoder_blocks.0.spatial.bias.0.weight"):
            assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name


def test_geometry_memory_influences_free_tokens():
    torch.manual_seed(4)
    model = SpatialTokenGS(spatial_opt()).eval()
    mi, _ = split_data_with_sfm(fake_batch(), model.opt)
    rec = records()
    model.compute_anchors = lambda mi: rec
    with torch.no_grad():
        a = model.forward_reconstruction(mi).gaussians.clone()
        rec[0]["points"] = rec[0]["points"] + torch.tensor([.05, 0., 0.])
        b = model.forward_reconstruction(mi).gaussians
    assert not torch.allclose(a[:, 12:, :3], b[:, 12:, :3])  # free output depends on anchors


def test_no_sfm_bypasses_detector_and_ignores_labels():
    model = SpatialTokenGS(spatial_opt(sfm_enabled=False)).eval()
    model.sfm.reconstruct = lambda *a: pytest.fail("disabled SfM must not run detector")
    batch = fake_batch()
    mi, _ = split_data_with_sfm(batch, model.opt)
    del mi.sfm
    with torch.no_grad():
        before = model.forward_reconstruction(mi).gaussians
        batch["depths_gt_sup"] = torch.randn(1, 4, 32, 56)
        batch["normals_gt_sup"] = torch.randn(1, 4, 32, 56, 3)
        batch["valid_gt_sup"] = torch.ones(1, 4, 32, 56, dtype=torch.bool)
        altered, _ = split_data_with_sfm(batch, model.opt)
        after = model.forward_reconstruction(altered).gaussians
    assert torch.equal(before, after)
    assert not model.last_spatial["anchored"].any()


def test_checkpoint_self_describes_architecture(tmp_path):
    opt = spatial_opt(sfm_enabled=False)
    model = SpatialTokenGS(opt).eval()
    path = tmp_path / "model.safetensors"
    save_file(model.state_dict(), str(path))
    state = load_file(str(path))
    restored_opt = options_from_architecture(state["_spatial_architecture"])
    restored = SpatialTokenGS(restored_opt).eval()
    restored.load_state_dict(state, strict=True)
    mi, _ = split_data_with_sfm(fake_batch(), opt)
    with torch.no_grad():
        assert torch.equal(model.forward_reconstruction(mi).gaussians, restored.forward_reconstruction(mi).gaussians)
    assert not any("enc_dec_backbone" in k or "deconv" in k for k in state)


def test_soft_chain_removes_normal_disagreement_not_tangential_spacing():
    opt = spatial_opt()
    q = torch.tensor([[0., 0., 1.], [.02, 0., 1.]])
    n = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
    valid = torch.ones(2, dtype=torch.bool)
    p = (q + torch.tensor([[0., 0., .01], [0., 0., -.01]])).requires_grad_()
    loss, edges = surface_pair_loss(p, q, n, valid, opt)
    assert edges == 1 and loss > 0
    loss.backward()
    assert p.grad[0, 2] > 0 and p.grad[1, 2] < 0
    assert torch.equal(p.grad[:, :2], torch.zeros_like(p.grad[:, :2]))
    assert surface_pair_loss(q + torch.tensor([.04, .02, .01]), q, n, valid, opt)[0] == 0
    q2 = q.clone()
    q2[1, 2] += .02  # parallel different surface: reject link
    assert surface_pair_loss(p, q2, n, valid, opt)[1] == 0
    assert surface_pair_loss(p, q, n, torch.zeros_like(valid), opt)[1] == 0


def test_planar_mask_accepts_slant_and_rejects_depth_edge():
    h, w = 8, 10
    x = torch.arange(w).float() + .5
    depth = (1 / (1 - .2 * (x-5)/20)).repeat(h, 1)
    normal = torch.tensor([-.2, 0., 1.])
    normal = (normal / normal.norm()).expand(h, w, 3)
    valid = torch.ones(h, w, dtype=torch.bool)
    intr = torch.tensor([20., 20., 5., 4.])
    _, plane = planar_map(depth, normal, valid, intr, .003, .98)
    assert plane[1:-1, 1:-1].all()
    depth[:, 5:] += .2
    _, plane = planar_map(depth, normal, valid, intr, .003, .98)
    assert not plane[:, 4:6].any()


def test_disjoint_view_sampling_handles_small_windows():
    provider = object.__new__(SfmProvider)
    provider.opt = spatial_opt()
    provider.min_gap, provider.max_gap = 2, 3
    provider.get_rng = lambda i: np.random.default_rng(i)
    provider.dataset = SimpleNamespace(count_frames=lambda i: 20)
    for n in (4, 6, 8):
        provider.opt = spatial_opt(num_input_views=n, num_views=n+4)
        indices, _ = provider._get_indices_static(n)
        assert len(indices) == n+4 and len(set(indices)) == n+4
    provider.dataset = SimpleNamespace(count_frames=lambda i: 3)
    with pytest.raises(ValueError):
        provider._get_indices_static(0)


def test_training_presets_reject_legacy_flags():
    for name in ("spatial_sfm", "spatial_free", "spatial_sfm_chain"):
        opt = config_defaults[name]
        assert opt.resume.endswith("dl3dv_latent_6v_ssim.safetensors") and opt.lambda_surface3d == 0
        assert not opt.sft_unfreeze_readout and not opt.sft_unfreeze_latent_tail
        assert opt.enc_embed_dim == opt.token_dim == 1024
        assert opt.enc_depth == opt.latent_cross_attn_depth == 12
        assert opt.child_hidden_dim == 1024 and opt.use_multiscale_encoder
    assert config_defaults["spatial_small_sfm"].resume is None
    with pytest.raises(ValueError):
        spatial_opt(lambda_surface3d=.1)
    with pytest.raises(ValueError):
        spatial_opt(batch_size=2)


def test_pretrained_scene_transfer_preserves_features_and_latents(monkeypatch):
    torch.manual_seed(2)
    opt = spatial_opt(scene_backbone="tokengs", enc_depth=2, use_multiscale_encoder=True,
                      multiscale_encoder_layers=(0, 1))
    original = TokenGS(opt).eval()
    model = SpatialTokenGS(opt).eval()
    queries_before = model.gs_tokens.detach().clone()
    import geometrygs.train as training
    monkeypatch.setattr(training.tokengs_train, "load_checkpoint_model", lambda path: original.state_dict())
    training.load_model_checkpoint(opt.evolve(resume="fixture.safetensors"), model,
                                   SimpleNamespace(print=lambda *args: None), 0)
    assert torch.equal(queries_before, model.gs_tokens)  # do not transfer official GS queries
    mi, _ = split_data_with_sfm(fake_batch(), opt)
    with torch.no_grad():
        reference_f = original.enc_dec_backbone._encode_features(original._embed_encoder_input(mi.encoder))
        reference_z = original.enc_dec_backbone._features_to_scene_latents(reference_f)
        f, z = model.encode_scene_backbone(mi.encoder)
        torch.testing.assert_close(f, reference_f, atol=0, rtol=0)
        torch.testing.assert_close(z, reference_z, atol=0, rtol=0)
        for block in model.scene_adapters:
            z = block(z, torch.randn(1, 3, 32), torch.ones(1, 3, dtype=torch.bool))
        torch.testing.assert_close(z, reference_z, atol=0, rtol=0)
    broken = original.state_dict()
    broken.pop("enc_dec_backbone.latents")
    with pytest.raises(ValueError, match="missing"):
        model.initialize_scene_backbone(broken)


def test_full_backbone_gradients_lr_groups_and_checkpoint(tmp_path):
    opt = spatial_opt(scene_backbone="tokengs")
    model = SpatialTokenGS(opt).train()
    model.configure_sft_trainable_parameters()
    model.compute_anchors = lambda mi: records(3)
    mi, _ = split_data_with_sfm(fake_batch(), opt)
    model.forward_reconstruction(mi).gaussians.square().sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p._lr_scale == (opt.backbone_lr_scale if model._is_backbone_parameter(name) else 1.)
    path = tmp_path / "full_backbone.safetensors"
    save_file(model.state_dict(), str(path))
    state = load_file(str(path))
    restored = SpatialTokenGS(options_from_architecture(state["_spatial_architecture"]))
    restored.load_state_dict(state, strict=True)
    assert restored.opt.scene_backbone == "tokengs"
    assert not hasattr(model, "image_blocks") and not hasattr(model, "scene_norm")


def test_grid_free_space_does_not_label_occluded_space_empty():
    from geometrygs.grid_probe import make_grid, grid_labels
    centers, lower, step = make_grid(4, 2., scale=1.)
    depth = torch.ones(2, 8, 8)
    normal = torch.zeros(2, 8, 8, 3)
    normal[..., 2] = -1
    intr = torch.tensor([[2., 2., 4., 4.]]).repeat(2, 1)
    cameras = torch.eye(4)[None].repeat(2, 1, 1)
    labels, hits, _, _, _, _ = grid_labels(
        centers, lower, step, 4, depth, depth.bool(), normal, intr, cameras, (centers[:, 2] - 1).abs())
    assert (labels == 1).any() and (labels == 0).any()
    assert (labels[centers[:, 2] > 1.5] == -1).all()
    assert (labels[hits > 0] == 1).all()


def test_bbox_grid_labels_every_hit_and_fixed_cell():
    from geometrygs.grid_probe import bbox_grid, cell_counts, grid_index
    hits = torch.tensor([[0., 0., 1.], [.29, .11, 1.05], [-.2, .05, 1.3]])
    centers, lo, dims = bbox_grid(hits, .03)
    assert torch.allclose(centers[1] - centers[0], torch.tensor([0., 0., .03]))
    ids, inside = grid_index(hits, lo, .03, dims)
    assert inside.all()
    positive = cell_counts(hits, lo, .03, dims) > 0
    assert positive.sum() == 3 and positive[ids].all()
    assert (lo < hits.min(0).values).all() and (lo + dims * .03 > hits.max(0).values).all()


def test_ray_crossing_and_near_surface_band():
    from geometrygs.grid_probe import ray_crossed_cells, dilate
    lo, dims = torch.zeros(3), torch.tensor([4, 4, 4])
    crossed = ray_crossed_cells(torch.tensor([[.5, .5, -1.]]), torch.tensor([[0., 0., 2.]]), lo, 1., dims, .0)
    cube = crossed.reshape(4, 4, 4)
    assert cube[0, 0].all() and cube.sum() == 4
    band = dilate(crossed & torch.arange(64).eq(0), dims, 1).reshape(4, 4, 4)
    assert band[:2, :2, :2].all() and band.sum() == 8


def test_occupancy_features_ignore_cells_outside_views():
    from geometrygs.grid_probe import occupancy_features
    maps = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    c2w = torch.eye(4)[None].repeat(2, 1, 1)
    intr = torch.tensor([[4., 4., 2., 2.]]).repeat(2, 1)
    centers = torch.tensor([[0., 0., 1.], [0., 0., -1.]])
    x = occupancy_features(maps, centers, .1, c2w, intr, (4, 4), .01, 1.)
    geo = 3 + 4 * 2 + 1  # 28 with the six input views of the experiment
    assert x.shape == (2, geo + 6)
    assert x[0, geo - 1] == 1 and x[1, geo - 1] == 0 and x[1, geo:].eq(0).all()
    assert torch.allclose(x[0, geo + 3:], torch.full((3,), 24.))  # views differ by 48 -> cross-view std 24


def test_grid_calibration_and_unknown_exclusion():
    from geometrygs.grid_probe import recall_threshold, classification_metrics
    scores = [0.1, 0.4, 0.7, 0.9, 0.99]
    labels = [0, 1, 1, 1, -1]
    threshold = recall_threshold(scores, labels, 2 / 3)
    assert threshold == .7
    metrics = classification_metrics(scores, labels, threshold)
    assert metrics["surface_recall"] == 2 / 3
    assert metrics["free_closure"] == metrics["surface_precision"] == 1
