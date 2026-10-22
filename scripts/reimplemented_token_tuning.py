"""Reimplemented token tuning loop (paper: TT; official ttt_mode="token-tuning").

All network weights and the scene latents (the decoder keys / values computed once from the input
views) stay frozen; only the Gaussian tokens ([1, 4096, 1024]) are optimized with Adam for
ttt_n_steps on the input views.

install_reimplemented_token_tuning replaces model.forward_ttt with a line-by-line copy of
TokenGS._forward_ttt_tokens that can add a geometry loss to the inner loop and record a per-step
curve (same rows as the scene latent loop). With all geometry weights at zero it runs exactly the
official operations. The tuned Gaussian tokens of the last scene are kept in
curve_state["tuned_tensor"] (demo: saved to disk).
"""

from __future__ import annotations

import types

import torch

from tokengs.utils.training import freeze_model_parameters

from evaluation_statistics import geo_losses_for
from reimplemented_scene_latent_tuning import curve_row


def install_reimplemented_token_tuning(model, geo_weights: dict, children: int, curve_state: dict) -> None:
    """Replace model.forward_ttt (instance attribute) with the reimplemented loop below.

    Arguments as in reimplemented_scene_latent_tuning.install_reimplemented_scene_latent_tuning.
    """
    geo_on = any(float(v) > 0 for v in geo_weights.values())

    def forward_ttt(self, model_input, n_steps=10, lr=1e-3):
        if self.opt.ttt_mode not in ("token-tuning", "tokens"):
            raise ValueError(f"reimplemented loop only covers token tuning, got {self.opt.ttt_mode!r}")
        model_input_ttt, supervision_ttt = model_input.to_ttt()
        with freeze_model_parameters(self):
            return _tune_tokens(self, model_input_ttt, supervision_ttt, n_steps, lr)

    @torch.no_grad()
    def _tune_tokens(self, model_input, supervision, n_steps, lr):
        # --- verbatim TokenGS._forward_ttt_tokens, plus the optional geometry term / curve
        encoder_latent = self.forward_encoder(model_input.encoder)
        curve = curve_state.get("enabled", False)
        rows = curve_state.setdefault("rows", [])
        rows.clear()

        with torch.set_grad_enabled(True):
            gs_tokens = self.get_gs_tokens(batch_size=model_input.batch_size).detach().requires_grad_(True)
            optim = torch.optim.Adam([gs_tokens], lr=lr)
            for step in range(n_steps):
                optim.zero_grad()
                gaussians = self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens)
                reconstruction = self._reconstruction_from_gaussians(gaussians)
                loss_dict = self.compute_loss(reconstruction, model_input.decoder, supervision)
                loss = loss_dict["loss"]
                geo = None
                if geo_on:
                    geo = geo_losses_for(gaussians, children, **geo_weights)
                    loss = loss + geo["total_geo_loss"]
                if curve:
                    rows.append(curve_row(step, loss_dict, loss, geo, gaussians, children, curve_state))
                loss.backward()
                optim.step()

        curve_state["tuned_tensor"] = gs_tokens.detach()
        final = self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens.detach())
        if curve:
            loss_dict = self.compute_loss(self._reconstruction_from_gaussians(final), model_input.decoder, supervision)
            geo = geo_losses_for(final, children, **geo_weights) if geo_on else None
            total = loss_dict["loss"] + (geo["total_geo_loss"] if geo is not None else 0.0)
            rows.append(curve_row(n_steps, loss_dict, total, geo, final, children, curve_state))
        return final

    model.forward_ttt = types.MethodType(forward_ttt, model)
