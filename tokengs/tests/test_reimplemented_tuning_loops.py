"""CPU equivalence of the reimplemented tuning loops with the official TokenGS inner loops.

A small mock model provides the methods the loops call (encoder, latent -> key/value projection, decoder,
loss); the official TokenGS._forward_ttt_* functions are run unbound on the same mock, so the test checks that
the reimplemented loops (geometry weights zero) perform exactly the official operations, keep the tuned tensor
and decode it again to the returned Gaussians.
The joint scene latent + token loop has no official counterpart; with one of its two learning rates at zero it
must reduce to the corresponding official loop.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reimplemented_scene_latent_and_token_tuning import (  # noqa: E402
    TTT_MODE as JOINT_MODE,
    install_reimplemented_scene_latent_and_token_tuning,
)
from reimplemented_scene_latent_tuning import install_reimplemented_scene_latent_tuning  # noqa: E402
from reimplemented_token_tuning import install_reimplemented_token_tuning  # noqa: E402
from tokengs.models.input_types import EncoderLatent, Reconstruction  # noqa: E402
from tokengs.models.tokengs import TokenGS  # noqa: E402

TOKENS, DIM, CHILDREN, VIEWS = 6, 8, 4, 2


class MockTokenGS(nn.Module):
    """Same call surface as TokenGS for the tuning loops; Gaussians [B, TOKENS*CHILDREN, 14]."""

    def __init__(self, ttt_mode: str):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.opt = types.SimpleNamespace(ttt_mode=ttt_mode)
        self.gs_tokens = nn.Parameter(torch.randn(TOKENS, DIM, generator=g))
        self.enc = nn.Parameter(torch.randn(VIEWS * 3, TOKENS * DIM, generator=g) * 0.1)
        self.kv = nn.Parameter(torch.randn(DIM, 2 * DIM, generator=g) * 0.3)
        self.head = nn.Parameter(torch.randn(DIM, CHILDREN * 14, generator=g) * 0.3)
        self.target = torch.rand(1, VIEWS, 3, generator=g)

    def forward_encoder_to_scene_latents(self, encoder_input):
        return (encoder_input.reshape(1, -1) @ self.enc).reshape(1, TOKENS, DIM)

    def encoder_latent_from_scene_latents(self, scene_latents):
        k, v = (scene_latents @ self.kv).chunk(2, dim=-1)
        return EncoderLatent(keys=torch.tanh(k), values=v)

    def forward_encoder(self, encoder_input):
        return self.encoder_latent_from_scene_latents(self.forward_encoder_to_scene_latents(encoder_input))

    def get_gs_tokens(self, batch_size):
        return self.gs_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def forward_decoder(self, encoder_latent, decoder_input, gs_tokens=None):
        if gs_tokens is None:
            gs_tokens = self.get_gs_tokens(1)
        attn = torch.softmax(gs_tokens @ encoder_latent.keys.transpose(-1, -2), dim=-1)
        x = (gs_tokens + attn @ encoder_latent.values) @ self.head
        x = x.reshape(1, TOKENS * CHILDREN, 14)
        quat = nn.functional.normalize(x[..., 7:11] + torch.tensor([1.0, 0, 0, 0]), dim=-1)
        return torch.cat([x[..., 0:3], torch.sigmoid(x[..., 3:4]), torch.exp(0.2 * x[..., 4:7]) * 0.1,
                          quat, torch.sigmoid(x[..., 11:14])], dim=-1)

    def _reconstruction_from_gaussians(self, gaussians):
        return Reconstruction(gaussians=gaussians, background_color=torch.zeros(3))

    def compute_loss(self, reconstruction, decoder_input, supervision):
        g = reconstruction.gaussians
        pred = (g[..., 11:14] * g[..., 3:4]).mean(dim=1, keepdim=True).expand(1, VIEWS, 3)
        loss = (pred - self.target).square().mean() + 0.01 * g[..., 0:3].square().mean()
        return {"loss": loss, "psnr": -10 * torch.log10(loss.detach())}


class MockInput:
    def __init__(self):
        self.encoder = torch.rand(VIEWS, 3, generator=torch.Generator().manual_seed(1))
        self.decoder = None
        self.batch_size = 1

    def to_ttt(self):
        return self, None


OFFICIAL = {"scene-latent-tuning": TokenGS._forward_ttt_scene_latents, "token-tuning": TokenGS._forward_ttt_tokens}
INSTALL = {"scene-latent-tuning": install_reimplemented_scene_latent_tuning,
           "token-tuning": install_reimplemented_token_tuning}


@pytest.mark.parametrize("mode", ["scene-latent-tuning", "token-tuning"])
def test_reimplemented_loop_matches_official_and_keeps_tuned_tensor(mode):
    official_model, model = MockTokenGS(mode), MockTokenGS(mode)
    with torch.no_grad():
        official = OFFICIAL[mode](official_model, MockInput(), None, n_steps=7, lr=1e-2)
    state = {"enabled": False}
    INSTALL[mode](model, {"lambda_sw": 0.0, "lambda_packing": 0.0, "lambda_inter": 0.0}, CHILDREN, state)
    with torch.no_grad():
        mine = model.forward_ttt(MockInput(), n_steps=7, lr=1e-2)
    assert torch.equal(official, mine)
    assert not torch.equal(mine, model.forward_decoder(model.forward_encoder(MockInput().encoder), None))

    # the kept tensor plus the frozen counterpart decode again to the returned Gaussians
    tuned = state["tuned_tensor"]
    assert not tuned.requires_grad
    latents = model.forward_encoder_to_scene_latents(MockInput().encoder)
    tokens = model.get_gs_tokens(1)
    if mode == "scene-latent-tuning":
        latents = tuned
    else:
        tokens = tuned
    with torch.no_grad():
        again = model.forward_decoder(model.encoder_latent_from_scene_latents(latents), None, gs_tokens=tokens)
    assert torch.equal(again, mine)
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("mode", ["scene-latent-tuning", "token-tuning"])
def test_curve_rows_with_geometry_loss(mode):
    model = MockTokenGS(mode)
    state = {"enabled": True, "target": lambda g: {"target_all_psnr": float(g[..., 11].mean())}}
    INSTALL[mode](model, {"lambda_sw": 0.1, "lambda_packing": 0.001, "lambda_inter": 0.01}, CHILDREN, state)
    with torch.no_grad():
        model.forward_ttt(MockInput(), n_steps=3, lr=1e-2)
    rows = state["rows"]
    assert [r["step"] for r in rows] == [0, 1, 2, 3]
    for r in rows:
        assert {"total_loss", "context_loss", "geo_sw_loss", "geo_weighted_total", "target_all_psnr"} <= set(r)
        assert r["total_loss"] > r["context_loss"]


def test_wrong_mode_is_rejected():
    model = MockTokenGS("scene-latent-tuning")
    install_reimplemented_token_tuning(model, {"lambda_sw": 0.0}, CHILDREN, {"enabled": False})
    with pytest.raises(ValueError, match="token tuning"):
        model.forward_ttt(MockInput(), n_steps=1, lr=1e-2)
    joint = MockTokenGS("token-tuning")
    install_reimplemented_scene_latent_and_token_tuning(joint, {"lambda_sw": 0.0}, CHILDREN, {"enabled": False},
                                                        token_lr=1e-2)
    with pytest.raises(ValueError, match="joint loop"):
        joint.forward_ttt(MockInput(), n_steps=1, lr=1e-2)


ZERO_GEO = {"lambda_sw": 0.0, "lambda_packing": 0.0, "lambda_inter": 0.0}


@pytest.mark.parametrize("official_mode, lr, token_lr", [("scene-latent-tuning", 1e-2, 0.0),
                                                       ("token-tuning", 0.0, 1e-2)])
def test_joint_loop_with_one_learning_rate_zero_is_the_single_loop(official_mode, lr, token_lr):
    """Token lr 0 -> the Gaussian tokens never move -> official scene latent tuning; scene lr 0 -> token tuning."""
    official_model, model = MockTokenGS(official_mode), MockTokenGS(JOINT_MODE)
    with torch.no_grad():
        official = OFFICIAL[official_mode](official_model, MockInput(), None, n_steps=7, lr=max(lr, token_lr))
    state = {"enabled": False}
    install_reimplemented_scene_latent_and_token_tuning(model, ZERO_GEO, CHILDREN, state, token_lr=token_lr)
    with torch.no_grad():
        mine = model.forward_ttt(MockInput(), n_steps=7, lr=lr)
    assert torch.equal(official, mine)
    frozen = {"scene_latents": model.forward_encoder_to_scene_latents(MockInput().encoder),
              "gs_tokens": model.get_gs_tokens(1)}
    unchanged = "gs_tokens" if token_lr == 0 else "scene_latents"
    assert torch.equal(state["tuned_tensor"][unchanged], frozen[unchanged])


def test_joint_loop_tunes_both_tensors_and_keeps_them():
    model = MockTokenGS(JOINT_MODE)
    state = {"enabled": False}
    install_reimplemented_scene_latent_and_token_tuning(model, ZERO_GEO, CHILDREN, state, token_lr=1e-2)
    with torch.no_grad():
        mine = model.forward_ttt(MockInput(), n_steps=7, lr=1e-2)
    tuned = state["tuned_tensor"]
    assert not any(t.requires_grad for t in tuned.values())
    assert not torch.equal(tuned["scene_latents"], model.forward_encoder_to_scene_latents(MockInput().encoder))
    assert not torch.equal(tuned["gs_tokens"], model.get_gs_tokens(1))
    for mode in ("scene-latent-tuning", "token-tuning"):
        with torch.no_grad():
            single = OFFICIAL[mode](MockTokenGS(mode), MockInput(), None, n_steps=7, lr=1e-2)
        assert not torch.allclose(single, mine)
    with torch.no_grad():
        again = model.forward_decoder(model.encoder_latent_from_scene_latents(tuned["scene_latents"]), None,
                                      gs_tokens=tuned["gs_tokens"])
    assert torch.equal(again, mine)
    assert all(p.requires_grad for p in model.parameters())


def test_joint_loop_curve_rows_with_geometry_loss():
    model = MockTokenGS(JOINT_MODE)
    state = {"enabled": True, "target": lambda g: {"target_all_psnr": float(g[..., 11].mean())}}
    install_reimplemented_scene_latent_and_token_tuning(
        model, {"lambda_sw": 0.1, "lambda_packing": 0.001, "lambda_inter": 0.01}, CHILDREN, state, token_lr=1e-4)
    with torch.no_grad():
        model.forward_ttt(MockInput(), n_steps=3, lr=1e-2)
    rows = state["rows"]
    assert [r["step"] for r in rows] == [0, 1, 2, 3]
    assert all(r["total_loss"] > r["context_loss"] and "geo_weighted_total" in r for r in rows)
