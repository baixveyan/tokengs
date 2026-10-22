"""Reimplemented scene latent tuning loop (official name: TTT, ttt_mode="scene-latent-tuning").

All network weights and Gaussian tokens stay frozen; only the per-scene latent bottleneck output
(scene_latents, [1, 4096, 1024]) is optimized with Adam for ttt_n_steps on the input views.

install_reimplemented_scene_latent_tuning replaces model.forward_ttt with a line-by-line copy of
TokenGS._forward_ttt_scene_latents that can add a geometry loss to the inner loop and record a
per-step curve. With all geometry weights at zero it runs exactly the official operations.
The tuned scene latents of the last scene are kept in curve_state["tuned_tensor"] (demo: saved to disk).
"""

from __future__ import annotations

import time
import types

import torch

from tokengs.utils.training import freeze_model_parameters

from evaluation_statistics import geo_losses_for, geometry_stats


def install_reimplemented_scene_latent_tuning(model, geo_weights: dict, children: int, curve_state: dict) -> None:
    """Replace model.forward_ttt (instance attribute) with the reimplemented loop below.

    geo_weights: {"lambda_sw", "lambda_packing", "lambda_inter"}; all zero -> no geometry ops at all,
    i.e. exactly the official _forward_ttt_scene_latents operations.
    curve_state: mutable dict; when curve_state["enabled"] the per-step curve is written to
    curve_state["rows"] (needs curve_state["target"] = callable(gaussians) -> dict of target metrics);
    curve_state["tuned_tensor"] always receives the detached tuned scene latents.
    """
    geo_on = any(float(v) > 0 for v in geo_weights.values())

    def forward_ttt(self, model_input, n_steps=10, lr=1e-3):
        if self.opt.ttt_mode not in ("scene-latent-tuning", "latents"):
            raise ValueError(f"reimplemented loop only covers scene latent tuning, got {self.opt.ttt_mode!r}")
        model_input_ttt, supervision_ttt = model_input.to_ttt()
        with freeze_model_parameters(self):
            return _tune_scene_latents(self, model_input_ttt, supervision_ttt, n_steps, lr)

    @torch.no_grad()
    def _tune_scene_latents(self, model_input, supervision, n_steps, lr):
        # --- verbatim TokenGS._forward_ttt_scene_latents, plus the optional geometry term / curve
        scene_latents = self.forward_encoder_to_scene_latents(model_input.encoder)
        gs_tokens = self.get_gs_tokens(batch_size=model_input.batch_size).detach()
        curve = curve_state.get("enabled", False)
        rows = curve_state.setdefault("rows", [])
        rows.clear()

        with torch.set_grad_enabled(True):
            scene_latents = scene_latents.detach().requires_grad_(True)
            optim = torch.optim.Adam([scene_latents], lr=lr)
            for step in range(n_steps):
                optim.zero_grad()
                encoder_latent = self.encoder_latent_from_scene_latents(scene_latents)
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

        curve_state["tuned_tensor"] = scene_latents.detach()
        encoder_latent = self.encoder_latent_from_scene_latents(scene_latents.detach())
        final = self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens)
        if curve:
            loss_dict = self.compute_loss(self._reconstruction_from_gaussians(final), model_input.decoder, supervision)
            geo = geo_losses_for(final, children, **geo_weights) if geo_on else None
            total = loss_dict["loss"] + (geo["total_geo_loss"] if geo is not None else 0.0)
            rows.append(curve_row(n_steps, loss_dict, total, geo, final, children, curve_state))
        return final

    model.forward_ttt = types.MethodType(forward_ttt, model)


@torch.no_grad()
def curve_row(step, loss_dict, total, geo, gaussians, children, curve_state) -> dict:
    """One tuning-curve row: context losses, slot geometry and the target metrics of these Gaussians."""
    t0 = time.perf_counter()
    row = {"step": step, "total_loss": float(total)}
    for key in ("loss", "loss_rgb", "loss_ssim", "loss_visibility", "psnr"):
        if key in loss_dict:
            row[f"context_{key}"] = float(loss_dict[key])
    stats = geometry_stats(gaussians.detach(), children)
    row.update({f"geo_{k}": v for k, v in stats.items()})
    if geo is not None:
        row["geo_weighted_total"] = float(geo["total_geo_loss"])
    row.update(curve_state["target"](gaussians.detach()))
    row["diagnostic_seconds"] = time.perf_counter() - t0
    return row
