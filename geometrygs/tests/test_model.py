import torch

import geometrygs  # noqa: F401
from geometrygs.data import split_data_with_sfm
from geometrygs.model import SfmTokenGS
from geometrygs.options import config_defaults
from tokengs.models.tokengs import TokenGS


def tiny_opt(**kw):
    base = dict(
        img_size=(32, 56), patch_size=8, dec_patch_size=2, enc_depth=2, dec_depth=1, enc_embed_dim=64, enc_num_heads=4,
        use_multiscale_encoder=False, use_latent_bottleneck=True, num_latents=16, latent_cross_attn_depth=1,
        num_gs_tokens=32, token_dim=64, sfm_num_anchors=8, sfm_image_size=(64, 112), lambda_lpips=0.0,
        lambda_surface3d=0.0, use_render_geo_supervision=False, readout_pos_mlp=True, num_input_views=2, num_views=3,
    )
    base.update(kw)
    return config_defaults["sfm_anchor_k1024"].evolve(**base)


def fake_batch(v_in=3, v_tgt=1, H=32, W=56):
    V = v_in + v_tgt
    c2w = torch.eye(4).repeat(V, 1, 1)
    c2w[:, 0, 3] = torch.linspace(0, 0.2, V)
    intr = torch.tensor([[50.0, 50.0, W / 2, H / 2]]).repeat(V, 1)
    imgs = torch.rand(V, 3, H, W)
    b = {
        "input": torch.rand(1, V, 9, H, W), "rays_os": torch.zeros(1, V, 3, H, W), "rays_ds": torch.zeros(1, V, 3, H, W),
        "images_all": imgs[None], "images_input": imgs[None, :v_in], "images_output": imgs[None, v_in:],
        "intrinsics_all": intr[None], "intrinsics": intr[None, v_in:], "intrinsics_input": intr[None, :v_in],
        "cam_view_all": torch.linalg.inv(c2w).transpose(1, 2)[None], "cam_view": torch.linalg.inv(c2w[v_in:]).transpose(1, 2)[None],
        "masks_all": torch.ones(1, V, 1, H, W), "masks_output": torch.ones(1, v_tgt, 1, H, W), "has_mask": torch.zeros(1, dtype=torch.bool),
        "cam_to_world_input": c2w[None, :v_in], "cam_view_input": torch.linalg.inv(c2w[:v_in]).transpose(1, 2)[None],
        "images_sfm": torch.rand(1, v_in, 3, 2 * H, 2 * W), "intrinsics_sfm": 2 * intr[None, :v_in],
    }
    return b


def test_split_data_follows_batch_view_count_and_carries_sfm():
    opt = tiny_opt(num_input_views=6)
    mi, _ = split_data_with_sfm(fake_batch(v_in=3), opt)
    assert mi.encoder.images_rgb.shape[1] == 3 and mi.sfm["images"].shape[1] == 3
    assert opt.num_input_views == 6


def test_anchored_children_stay_bounded_and_free_slots_are_untouched():
    torch.manual_seed(0)
    opt = tiny_opt()
    model = SfmTokenGS(opt).eval()
    anchors = torch.tensor([[0.1, 0.0, 1.0], [0.0, 0.1, 1.2], [0.2, -0.1, 0.9]])

    def fake_anchors(model_input):
        rec = {"points": anchors, "obs_mask": torch.ones(3, 3, dtype=torch.bool), "anchor_index": torch.arange(3)}
        model.last_sfm = [rec]
        return [rec]

    model.compute_anchors = fake_anchors
    mi, _ = split_data_with_sfm(fake_batch(v_in=3), opt)
    with torch.no_grad():
        g = model.forward_reconstruction(mi).gaussians
        model.num_anchor_slots = 0  # no anchored slot -> must reproduce the original TokenGS decoding exactly
        plain = model.forward_reconstruction(mi).gaussians
        ref = TokenGS.forward_reconstruction(model, mi).gaussians
    P = opt.dec_patch_size ** 2
    pos = g[0, :, :3].view(opt.num_gs_tokens, P, 3)
    off = (pos[:3] - anchors[:, None]).abs()
    assert off.max() <= opt.anchor_offset_max + 1e-6 and off.max() > 0
    assert torch.isfinite(g).all() and torch.equal(plain, ref)
    # slots 3..7 had no SfM point: they keep their own learned token (only self-attention couples them to anchors)
    assert not torch.equal(g[0, : 3 * P], plain[0, : 3 * P])


def test_gradients_reach_anchor_modules():
    torch.manual_seed(0)
    opt = tiny_opt()
    model = SfmTokenGS(opt).train()
    model.compute_anchors = lambda mi: [{"points": torch.rand(5, 3) + torch.tensor([0, 0, 1.0]),
                                         "obs_mask": torch.ones(5, 3, dtype=torch.bool), "anchor_index": torch.arange(5)}]
    mi, _ = split_data_with_sfm(fake_batch(v_in=3), opt)
    g = model.forward_reconstruction(mi).gaussians
    (g[..., :3].square().sum() + g[..., 3:].sum()).backward()
    for m in model.anchor_modules():
        params = [m] if isinstance(m, torch.nn.Parameter) else list(m.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params), m


def test_state_dict_has_no_detector_and_loads_tokengs_weights():
    opt = tiny_opt()
    model = SfmTokenGS(opt)
    keys = model.state_dict().keys()
    assert not any("sift" in k.lower() for k in keys)
    assert {"gs_tokens", "anchor_token_base", "anchor_pos_head.1.weight"} <= set(keys)
