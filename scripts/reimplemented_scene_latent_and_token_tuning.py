"""Scene latent and token tuning together (no official counterpart; our ttt_mode="scene-latent-and-token-tuning").

All network weights stay frozen; the per-scene scene latents ([1, 4096, 1024]) and the Gaussian tokens
([1, 4096, 1024]) are both optimized on the input views, with one Adam that has one parameter group per
tensor and each tensor's own learning rate (scene latents: ttt_lr, preset 1e-2; Gaussian tokens: token_lr,
paper 1e-4), for ttt_n_steps. Adam is element-wise, so this is the same as two separate Adams stepping
together.

install_reimplemented_scene_latent_and_token_tuning replaces model.forward_ttt with a loop that merges the
official TokenGS._forward_ttt_scene_latents and TokenGS._forward_ttt_tokens: per step the scene latents are
projected to decoder keys / values (as in scene latent tuning) and decoded with the tunable Gaussian tokens
(as in token tuning). With token_lr = 0 it reduces to scene latent tuning, with lr = 0 to token tuning
(tests/test_reimplemented_tuning_loops.py). Geometry loss and per-step curve as in the other two loops.
The tuned tensors of the last scene are kept in curve_state["tuned_tensor"] =
{"scene_latents": ..., "gs_tokens": ...} (demo: saved to disk).
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from tokengs.utils.training import freeze_model_parameters

from evaluation_statistics import ALIVE_OPACITY, analytic_frustum_oob, apply_core_opacity_gate, geo_losses_for
from reimplemented_scene_latent_tuning import curve_row

TTT_MODE = "scene-latent-and-token-tuning"


def install_reimplemented_scene_latent_and_token_tuning(model, geo_weights: dict, children: int, curve_state: dict,
                                                        token_lr: float, vis_mode: str = "official",
                                                        core_gate_tau: float = 0.0,
                                                        depth_prior_weight: float = 0.0) -> None:
    """Replace model.forward_ttt (instance attribute) with the joint loop below.

    forward_ttt's lr is the scene latent learning rate (opt.ttt_lr); token_lr is the Gaussian token one.
    Other arguments as in reimplemented_scene_latent_tuning.install_reimplemented_scene_latent_tuning.
    """
    geo_on = any(float(v) > 0 for v in geo_weights.values())

    def forward_ttt(self, model_input, n_steps=10, lr=1e-3):
        if self.opt.ttt_mode != TTT_MODE:
            raise ValueError(f"joint loop only covers {TTT_MODE}, got {self.opt.ttt_mode!r}")
        model_input_ttt, supervision_ttt = model_input.to_ttt()
        with freeze_model_parameters(self):
            return _tune_scene_latents_and_tokens(self, model_input_ttt, supervision_ttt, n_steps, lr)

    @torch.no_grad()
    def _tune_scene_latents_and_tokens(self, model_input, supervision, n_steps, lr):
        scene_latents = self.forward_encoder_to_scene_latents(model_input.encoder)
        curve = curve_state.get("enabled", False)
        rows = curve_state.setdefault("rows", [])
        rows.clear()

        with torch.set_grad_enabled(True):
            scene_latents = scene_latents.detach().requires_grad_(True)
            gs_tokens = self.get_gs_tokens(batch_size=model_input.batch_size).detach().requires_grad_(True)
            optim = torch.optim.Adam([{"params": [scene_latents], "lr": lr},
                                      {"params": [gs_tokens], "lr": token_lr}])
            for step in range(n_steps):
                optim.zero_grad()
                encoder_latent = self.encoder_latent_from_scene_latents(scene_latents)
                gaussians = self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens)
                if core_gate_tau > 0:
                    gaussians = apply_core_opacity_gate(gaussians, children, core_gate_tau)
                reconstruction = self._reconstruction_from_gaussians(gaussians)
                loss_dict = self.compute_loss(reconstruction, model_input.decoder, supervision)
                loss = loss_dict["loss"]
                if vis_mode != "official":
                    loss = loss - float(self.opt.lambda_visibility) * loss_dict.get("loss_visibility", 0.0)
                    oob = analytic_frustum_oob(gaussians, model_input.decoder.cam_view, model_input.decoder.intrinsics,
                                               self.img_size, float(self.opt.znear))
                    alive_m = (gaussians[..., 3].detach() >= ALIVE_OPACITY).float()
                    if vis_mode == "vis_xyz":
                        vis_term = 1.0 * (oob * alive_m).mean()
                    elif vis_mode == "vis_opac":
                        vis_term = 2.0 * ((oob.detach() > 0).float() * alive_m * gaussians[..., 3]).mean()
                    else:
                        raise ValueError(f"unknown vis_mode: {vis_mode!r}")
                    loss_dict["loss_visibility"] = vis_term.detach()
                    loss = loss + vis_term
                if depth_prior_weight > 0 and curve_state.get("depth_prior") is not None:
                    d_tgt, d_msk = curve_state["depth_prior"]
                    m = d_msk > 0
                    if bool(m.any()):
                        d_pred = loss_dict["depths_pred"]
                        d_loss = F.smooth_l1_loss(d_pred[m], d_tgt[m], beta=0.05)
                        loss = loss + float(depth_prior_weight) * d_loss
                geo = None
                if geo_on:
                    geo = geo_losses_for(gaussians, children, **geo_weights)
                    loss = loss + geo["total_geo_loss"]
                if curve:
                    rows.append(curve_row(step, loss_dict, loss, geo, gaussians, children, curve_state))
                loss.backward()
                optim.step()

        curve_state["tuned_tensor"] = {"scene_latents": scene_latents.detach(), "gs_tokens": gs_tokens.detach()}
        encoder_latent = self.encoder_latent_from_scene_latents(scene_latents.detach())
        final = self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens.detach())
        if core_gate_tau > 0:
            final = apply_core_opacity_gate(final, children, core_gate_tau)
        if curve:
            loss_dict = self.compute_loss(self._reconstruction_from_gaussians(final), model_input.decoder, supervision)
            if vis_mode != "official":
                loss_dict["loss"] = loss_dict["loss"] - float(self.opt.lambda_visibility) * loss_dict.get("loss_visibility", 0.0)
                oob = analytic_frustum_oob(final, model_input.decoder.cam_view, model_input.decoder.intrinsics,
                                           self.img_size, float(self.opt.znear))
                alive_m = (final[..., 3] >= ALIVE_OPACITY).float()
                vis_term = (1.0 * (oob * alive_m).mean()) if vis_mode == "vis_xyz" else (
                    2.0 * ((oob > 0).float() * alive_m * final[..., 3]).mean()
                )
                loss_dict["loss_visibility"] = vis_term
                loss_dict["loss"] = loss_dict["loss"] + vis_term
            geo = geo_losses_for(final, children, **geo_weights) if geo_on else None
            total = loss_dict["loss"] + (geo["total_geo_loss"] if geo is not None else 0.0)
            rows.append(curve_row(n_steps, loss_dict, total, geo, final, children, curve_state))
        return final

    model.forward_ttt = types.MethodType(forward_ttt, model)
