# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TokenGS: Encoder-decoder model for 3D scene reconstruction from sparse views.
"""

from typing import Optional
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange

from lpips import LPIPS

from tokengs.options import Options
from tokengs.models.input_types import (
    EncoderLatent,
    ModelInput,
    ModelInputDecoder,
    ModelInputEncoder,
    ModelSupervision,
    Reconstruction,
    split_data,
)
from .attention import PatchEmbed
from tokengs.models.enc_dec import EncDecBackbone
from tokengs.rendering.gs import GaussianRenderer
from tokengs.models.activations import ClipActivationHead
from tokengs.models.losses import (
    compute_loss_from_renders,
    compute_tokengs_loss,
    compute_visibility_loss_from_means2d,
)
from tokengs.losses.geo_losses import compute_geo_losses
from tokengs.utils.training import freeze_model_parameters


class TokenGS(nn.Module):
    """
    TokenGS model with encoder-decoder architecture.
    
    Uses separate encoder and decoder with cross-attention for Gaussian token processing.
    """
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt
        
        self.img_size = self.opt.img_size if not isinstance(self.opt.img_size, int) else [self.opt.img_size, self.opt.img_size]

        norm_layer_factory = partial(nn.LayerNorm, bias=True)

        self.patch_embed = PatchEmbed(
            img_size=self.opt.img_size,
            patch_size=self.opt.patch_size,
            in_chans=3,
            embed_dim=self.opt.enc_embed_dim,
            norm_layer=norm_layer_factory,
        )

        self.patch_plucker_embed = PatchEmbed(
            img_size=self.opt.img_size,
            patch_size=self.opt.patch_size,
            in_chans=6,
            embed_dim=self.opt.enc_embed_dim,
            norm_layer=norm_layer_factory,
        )

        if self.opt.time_embedding:
            self.patch_time_embed = PatchEmbed(
                img_size=self.opt.img_size,
                patch_size=self.opt.patch_size,
                in_chans=self.opt.time_embedding_dim,
                embed_dim=self.opt.enc_embed_dim,
                # no normalization for time embedding
                # since it could be zero-initialized
                norm_layer=None,
            )
            self.patch_time_embed_tgt = PatchEmbed(
                img_size=self.opt.img_size,
                patch_size=self.opt.patch_size,
                in_chans=self.opt.time_embedding_dim,
                embed_dim=self.opt.enc_embed_dim,
                # no normalization for time embedding
                # since it could be zero-initialized
                norm_layer=None,
            )

            for m in (self.patch_time_embed, self.patch_time_embed_tgt):
                m.proj.weight.data.fill_(0.0)
                m.proj.bias.data.fill_(0.0)

        # Encoder-decoder architecture
        self.enc_dec_backbone = EncDecBackbone(opt)

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # Activation head (always clip)
        self.activation_head = ClipActivationHead(opt)

        # Stamped by the trainer each step when dynamic auxiliary loss uses a schedule.
        self.lambda_dyn_aux_eff = float(opt.lambda_dyn_aux)

        # LPIPS loss
        self.lpips_loss: Optional[LPIPS] = None
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
        else:
            self.lpips_loss = None

        # Learnable GS tokens
        self.gs_tokens = nn.Parameter(
            self.opt.gs_token_std * torch.randn(self.opt.num_gs_tokens, self.opt.token_dim)
        )

        if self.opt.num_dynamic_gs_tokens > 0:
            self.gs_tokens_dynamic = nn.Parameter(
                self.opt.gs_token_std * torch.randn(self.opt.num_dynamic_gs_tokens, self.opt.token_dim)
            )

    def state_dict(self, **kwargs):
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if "lpips_loss" in k:
                del state_dict[k]
        return state_dict

    def _background_color(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.opt.bg_color == "white":
            return torch.ones(3, dtype=dtype, device=device)
        if self.opt.bg_color == "black":
            return torch.zeros(3, dtype=dtype, device=device)
        if self.opt.bg_color == "grey":
            return torch.ones(3, dtype=dtype, device=device) * 0.5
        raise ValueError(f"Invalid background color: {self.opt.bg_color}")

    def _reconstruction_from_gaussians(self, gaussians: torch.Tensor) -> Reconstruction:
        return Reconstruction(
            gaussians=gaussians,
            background_color=self._background_color(gaussians.dtype, gaussians.device),
        )

    def _embed_encoder_input(self, encoder_input: ModelInputEncoder) -> torch.Tensor:
        B, V, _, H, W = encoder_input.images_rgb.shape
        height = int(H // self.opt.patch_size)
        width = int(W // self.opt.patch_size)

        assert height * self.opt.patch_size == H, f"H={H} must be divisible by patch_size={self.opt.patch_size}"
        assert width * self.opt.patch_size == W, f"W={W} must be divisible by patch_size={self.opt.patch_size}"

        # Reshape for tokenization
        images_rgb_reshaped = encoder_input.images_rgb.reshape(B * V, 3, H, W)
        plucker_reshaped = encoder_input.plucker.reshape(B * V, 6, H, W)
        
        # Embed RGB patches
        x = self.patch_embed(images_rgb_reshaped)
        # Add Plucker embeddings
        x_plucker_emb = self.patch_plucker_embed(plucker_reshaped)
        x = x + x_plucker_emb  # B*V, N, C

        # Reshape from (B*V, N, C) to (B, V*N, C) so the encoder sees one sequence per batch
        x = rearrange(x, "(b v) n c -> b (v n) c", b=B, v=V)

        # Add time embeddings if enabled (processed in 2D image space, then reshaped)
        if self.opt.time_embedding:
            time_emb_reshaped = encoder_input.time_embedding_input.reshape(B * V, self.opt.time_embedding_dim, H, W)
            x_time_emb = self.patch_time_embed(time_emb_reshaped)  # (B*V, N, C)
            x = x + x_time_emb.reshape(B, V * (height * width), -1)  # Reshape to match x: (B, V*N, C)

        return x

    def forward_encoder(self, encoder_input: ModelInputEncoder) -> EncoderLatent:
        """
        Encode input views into latent representation (keys and values for cross-attention).

        Args:
            encoder_input: ModelInputEncoder containing input view data (only input views)

        Returns:
            EncoderLatent containing keys and values for cross-attention
        """
        x = self._embed_encoder_input(encoder_input)

        # Run encoder on joint views and produce keys/values
        # x shape: (B, V*N, C) where all views are processed jointly
        image_feature_keys, image_feature_values = self.enc_dec_backbone._encode_to_kv(x)
        
        return EncoderLatent(
            keys=image_feature_keys,
            values=image_feature_values,
        )

    def forward_encoder_to_scene_latents(self, encoder_input: ModelInputEncoder) -> torch.Tensor:
        """Encode input views to the scene latent bottleneck output."""
        if not self.opt.use_latent_bottleneck:
            raise ValueError("scene-latent tuning requires use_latent_bottleneck=True")

        x = self._embed_encoder_input(encoder_input)
        image_features = self.enc_dec_backbone._encode_features(x)
        return self.enc_dec_backbone._features_to_scene_latents(image_features)

    def encoder_latent_from_scene_latents(self, scene_latents: torch.Tensor) -> EncoderLatent:
        """Convert scene latent bottleneck tokens to decoder attention keys and values."""
        image_feature_keys, image_feature_values = self.enc_dec_backbone._latents_to_kv(scene_latents)
        return EncoderLatent(
            keys=image_feature_keys,
            values=image_feature_values,
        )

    def get_gs_tokens(self, batch_size: int) -> torch.Tensor:
        """
        Get initial GS tokens (without time conditioning).
        Useful for test-time training where you want to optimize the GS tokens.
        Time conditioning is applied later in forward_decoder.
        
        Args:
            batch_size: Batch size
            
        Returns:
            GS tokens with shape [B, num_gs_tokens, C]
        """
        batch_gs_tokens = self.gs_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if self.opt.num_dynamic_gs_tokens > 0:
            dyn = self.gs_tokens_dynamic.unsqueeze(0).expand(batch_size, -1, -1).clone()
            batch_gs_tokens = torch.cat([batch_gs_tokens, dyn], dim=1)
        return batch_gs_tokens
    
    def _apply_time_embedding_to_gs_tokens(
        self, 
        gs_tokens: torch.Tensor,
        decoder_input: ModelInputDecoder,
    ) -> torch.Tensor:
        """
        Apply target time embedding to GS tokens.
        This is called by forward_decoder to condition GS tokens on target time.
        
        Args:
            gs_tokens: Base GS tokens [B, num_gs_tokens, C]
            decoder_input: ModelInputDecoder containing target time embedding
            
        Returns:
            Time-conditioned GS tokens [B, num_gs_tokens (+dynamic), C]
        """
        if not self.opt.time_embedding or decoder_input.time_embedding_target is None:
            # No time embedding or no target time provided
            return gs_tokens
        
        B = gs_tokens.shape[0]
        _, _, T_dim, H_tgt, W_tgt = decoder_input.time_embedding_target.shape
        
        # Process target time embedding through patch embedding
        time_emb_target = decoder_input.time_embedding_target.reshape(B, T_dim, H_tgt, W_tgt)
        x_time_emb_tgt = self.patch_time_embed_tgt(time_emb_target)
        
        if self.opt.num_dynamic_gs_tokens > 0:
            # For dynamic tokens: add time embedding only to dynamic tokens
            x_time_emb_tgt = x_time_emb_tgt.reshape(B, -1, self.opt.enc_embed_dim)[:, :self.opt.num_dynamic_gs_tokens, :]
            gs_tokens[:, -self.opt.num_dynamic_gs_tokens:] = gs_tokens[:, -self.opt.num_dynamic_gs_tokens:] + x_time_emb_tgt
        else:
            # For standard learnable tokens: add time embedding to all GS tokens
            x_time_emb_tgt = x_time_emb_tgt.reshape(B, -1, self.opt.enc_embed_dim)[:, :self.opt.num_gs_tokens, :]
            gs_tokens = gs_tokens + x_time_emb_tgt
        
        return gs_tokens

    @staticmethod
    def compute_lambda_dyn_aux_eff(step: int, opt) -> float:
        """Schedule the auxiliary dynamic-only loss weight."""
        base = float(opt.lambda_dyn_aux)
        if base <= 0:
            return 0.0
        if opt.lambda_dyn_aux_decay_steps <= 0:
            return base
        if step <= opt.lambda_dyn_aux_warmup_steps:
            return base
        frac = min(1.0, (step - opt.lambda_dyn_aux_warmup_steps) / max(1, opt.lambda_dyn_aux_decay_steps))
        return base + frac * (float(opt.lambda_dyn_aux_min) - base)

    def forward_decoder(
        self, 
        encoder_latent: EncoderLatent,
        decoder_input: ModelInputDecoder,
        gs_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process latent representation (keys/values) to Gaussians using decoder with cross-attention.
        
        Args:
            encoder_latent: EncoderLatent containing keys and values from the encoder
            decoder_input: ModelInputDecoder containing rendering parameters and target time
            gs_tokens: Optional precomputed GS tokens [B, num_gs_tokens, C]. If None, creates new ones.
            
        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians
        """
        B = encoder_latent.keys.shape[0]
        
        # Get or use provided GS tokens (without time conditioning)
        if gs_tokens is None:
            gs_tokens = self.get_gs_tokens(batch_size=B)

        # Apply time embedding to GS tokens
        gs_tokens = self._apply_time_embedding_to_gs_tokens(gs_tokens, decoder_input)

        # Run decoder with precomputed keys/values
        for layer in self.enc_dec_backbone.decoder_blocks:
            gs_tokens = layer(
                gs_tokens=gs_tokens,
                keys=encoder_latent.keys,
                values=encoder_latent.values,
            )

        # Convert to Gaussians
        gaussians = self.activation_head(gs_tokens)

        # give gaussians an offset to the z axis so it is visible when initialized
        gaussians[..., 2] = gaussians[..., 2] + self.opt.gaussian_z_offset

        return gaussians

    def forward_reconstruction(self, model_input: ModelInput) -> Reconstruction:
        """
        Generate a reconstruction from model input.
        This is a convenience method that combines encoding and decoding.
        For test-time training, use forward_encoder(), get_gs_tokens(), and forward_decoder() separately.
        
        Args:
            model_input: ModelInput containing all necessary input data
            
        Returns:
            Reconstruction containing Gaussians [B, N, 14] and background color [3]
        """
        # Encode input views
        encoder_latent = self.forward_encoder(model_input.encoder)
        
        # Decode to Gaussians (time conditioning is applied inside forward_decoder)
        gaussians = self.forward_decoder(encoder_latent, model_input.decoder)
        
        return self._reconstruction_from_gaussians(gaussians)

    def forward_gaussians(self, model_input: ModelInput) -> torch.Tensor:
        """
        Generate Gaussians from model input.
        This compatibility method returns only the raw Gaussian tensor.

        Args:
            model_input: ModelInput containing all necessary input data

        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians
        """
        return self.forward_reconstruction(model_input).gaussians

    def render_reconstruction(
        self,
        reconstruction: Reconstruction,
        decoder_input: ModelInputDecoder,
    ) -> dict:
        """
        Render a reconstruction to images.

        Args:
            reconstruction: Model prediction containing Gaussian parameters and background color
            decoder_input: Decoder input containing camera view matrices and intrinsics

        Returns:
            Dictionary with 'images_pred', 'alphas_pred', etc.
        """
        if decoder_input.cam_view is None:
            raise ValueError("decoder_input.cam_view is required for rendering")
        if decoder_input.intrinsics is None:
            raise ValueError("decoder_input.intrinsics is required for rendering")

        return self.gs.render(
            reconstruction.gaussians,
            decoder_input.cam_view,
            bg_color=reconstruction.background_color,
            intrinsics=decoder_input.intrinsics,
        )

    def render_gaussians(
        self,
        gaussians: torch.Tensor,
        decoder_input_or_cam_view: ModelInputDecoder | torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Render Gaussians to images.

        Args:
            gaussians: Gaussian parameters [B, N, 14]
            decoder_input_or_cam_view: Decoder input or camera view matrices [B, V, 4, 4]
            intrinsics: Intrinsics [B, V, 4], required when camera views are passed directly

        Returns:
            Dictionary with 'images_pred', 'alphas_pred', etc.
        """
        if isinstance(decoder_input_or_cam_view, ModelInputDecoder):
            decoder_input = decoder_input_or_cam_view
        else:
            decoder_input = ModelInputDecoder(
                cam_view=decoder_input_or_cam_view,
                intrinsics=intrinsics,
            )
        return self.render_reconstruction(
            self._reconstruction_from_gaussians(gaussians),
            decoder_input,
        )

    def compute_loss(
        self,
        reconstruction: Reconstruction,
        decoder_input: ModelInputDecoder,
        supervision: ModelSupervision,
    ) -> dict:
        render_results = self.render_reconstruction(reconstruction, decoder_input)
        results = compute_tokengs_loss(
            opt=self.opt,
            img_size=self.img_size,
            render_results=render_results,
            supervision=supervision,
            decoder_input=decoder_input,
            gaussians=reconstruction.gaussians,
            lpips_loss=self.lpips_loss,
        )

        lambda_dyn_aux = float(getattr(self, "lambda_dyn_aux_eff", self.opt.lambda_dyn_aux))
        if (
            lambda_dyn_aux > 0
            and self.opt.num_dynamic_gs_tokens > 0
            and self.opt.num_gs_tokens > 0
        ):
            P2 = self.opt.dec_patch_size * self.opt.dec_patch_size
            n_dyn_gs = self.opt.num_dynamic_gs_tokens * P2
            gaussians_dyn = reconstruction.gaussians[:, -n_dyn_gs:].contiguous()
            dyn_render = self.render_gaussians(gaussians_dyn, decoder_input)
            dyn_results = compute_tokengs_loss(
                opt=self.opt,
                img_size=self.img_size,
                render_results=dyn_render,
                supervision=supervision,
                decoder_input=decoder_input,
                gaussians=gaussians_dyn,
                lpips_loss=self.lpips_loss,
            )
            results["loss_dyn_aux"] = dyn_results["loss"].detach()
            results["psnr_dyn_aux"] = dyn_results["psnr"].detach()
            results["lambda_dyn_aux_eff"] = torch.tensor(lambda_dyn_aux, device=reconstruction.gaussians.device)
            results["loss"] = results["loss"] + lambda_dyn_aux * dyn_results["loss"]

        return results

    def _mean_of_grads_chunk_sizes(self, batch_size: int, num_views: int) -> tuple[int, int]:
        scene_chunk_size = min(self.opt.mean_of_grads_scene_chunk_size, batch_size)
        if self.opt.mean_of_grads_view_chunk_size is None:
            view_chunk_size = num_views if self.opt.mean_of_grads == "per-scene" else 1
        else:
            view_chunk_size = self.opt.mean_of_grads_view_chunk_size
        return scene_chunk_size, min(view_chunk_size, num_views)

    def compute_loss_mean_of_grads(
        self,
        reconstruction: Reconstruction,
        decoder_input: ModelInputDecoder,
        supervision: ModelSupervision,
    ) -> dict:
        if self.opt.mean_of_grads not in ("per-scene", "per-view"):
            raise ValueError(f"Invalid mean_of_grads mode: {self.opt.mean_of_grads}")
        if decoder_input.cam_view is None:
            raise ValueError("decoder_input.cam_view is required for rendering")
        if decoder_input.intrinsics is None:
            raise ValueError("decoder_input.intrinsics is required for rendering")

        gaussians = reconstruction.gaussians
        batch_size = gaussians.shape[0]
        num_views = decoder_input.cam_view.shape[1]
        scene_chunk_size, view_chunk_size = self._mean_of_grads_chunk_sizes(batch_size, num_views)
        gaussians_leaf = gaussians.detach().requires_grad_(True)
        grad_accum = torch.zeros_like(gaussians_leaf)

        per_scene_chunks: dict[str, list[torch.Tensor]] = {}
        render_chunks: dict[str, list[torch.Tensor]] = {}
        raw_loss_mask_chunks: list[torch.Tensor] = []
        render_keys = {"images_pred", "alphas_pred", "depths_pred", "images_output"}
        valid_mask_count = supervision.has_mask.float().sum()
        per_scene_loss = partial(compute_loss_from_renders, self.opt, self.img_size, lpips_loss=self.lpips_loss)

        for scene_start in range(0, batch_size, scene_chunk_size):
            scene_end = min(scene_start + scene_chunk_size, batch_size)
            gaussians_scene = gaussians_leaf[scene_start:scene_end]
            reconstruction_scene = Reconstruction(
                gaussians=gaussians_scene,
                background_color=reconstruction.background_color,
            )

            scene_results: dict[str, torch.Tensor] = {}
            scene_render_chunks: dict[str, list[torch.Tensor]] = {}

            for view_start in range(0, num_views, view_chunk_size):
                view_end = min(view_start + view_chunk_size, num_views)
                view_count = view_end - view_start
                view_weight = view_count / num_views

                scene_slice = slice(scene_start, scene_end)
                view_slice = slice(view_start, view_end)
                decoder_chunk = decoder_input.select_batch(scene_slice, view_slice)
                supervision_chunk = supervision.select_batch(scene_slice, view_slice)
                render_chunk = self.render_reconstruction(reconstruction_scene, decoder_chunk)

                out_chunk = torch.vmap(per_scene_loss, in_dims=(0,) * 5)(
                    render_chunk["images_pred"].unsqueeze(1),
                    render_chunk["alphas_pred"].unsqueeze(1),
                    supervision_chunk.images_output.unsqueeze(1),
                    supervision_chunk.masks_output.unsqueeze(1),
                    supervision_chunk.has_mask.unsqueeze(1),
                )
                if "loss_lpips" in out_chunk:
                    out_chunk["loss"] = out_chunk["loss"] + float(self.opt.lambda_lpips) * out_chunk["loss_lpips"]

                scaled_loss = out_chunk["loss"].sum() * view_weight / batch_size
                grad_chunk = torch.autograd.grad(
                    scaled_loss,
                    reconstruction_scene.gaussians,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                grad_accum[scene_start:scene_end] += grad_chunk

                for key, value in out_chunk.items():
                    if torch.is_tensor(value):
                        weighted = value.detach() * view_weight
                        scene_results[key] = scene_results.get(key, torch.zeros_like(weighted)) + weighted
                for key in render_keys - {"images_output"}:
                    scene_render_chunks.setdefault(key, []).append(render_chunk[key].detach())
                scene_render_chunks.setdefault("images_output", []).append(supervision_chunk.images_output.detach())

            if self.opt.lambda_visibility > 0:
                decoder_scene = decoder_input.select_batch(slice(scene_start, scene_end), slice(0, num_views))
                visibility_render = self.render_reconstruction(reconstruction_scene, decoder_scene)
                means2d_pred = visibility_render["means2d_pred"]
                loss_visibility = compute_visibility_loss_from_means2d(self.opt, self.img_size, means2d_pred)
                scaled_visibility = self.opt.lambda_visibility * loss_visibility.sum() / batch_size
                grad_visibility = torch.autograd.grad(
                    scaled_visibility,
                    reconstruction_scene.gaussians,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                grad_accum[scene_start:scene_end] += grad_visibility
                scene_results["loss_visibility"] = loss_visibility.detach()
                scene_results["loss"] = scene_results["loss"] + self.opt.lambda_visibility * loss_visibility.detach()

            for key, value in scene_results.items():
                if key == "loss_mask":
                    raw_loss_mask_chunks.append(value * (supervision.has_mask[scene_start:scene_end].float() + 1e-6))
                else:
                    per_scene_chunks.setdefault(f"{key}_per_scene", []).append(value)

            for key, chunks in scene_render_chunks.items():
                render_chunks.setdefault(key, []).append(torch.cat(chunks, dim=1))

        results = {key: torch.cat(chunks, dim=0) for key, chunks in per_scene_chunks.items()}
        for key, value in list(results.items()):
            if key.endswith("_per_scene") and key != "loss_mask_per_scene":
                results[key.removesuffix("_per_scene")] = value.mean()
        results.update({key: torch.cat(chunks, dim=0) for key, chunks in render_chunks.items()})

        if raw_loss_mask_chunks:
            loss_mask_per_scene = torch.cat(raw_loss_mask_chunks, dim=0) / (valid_mask_count + 1e-6)
            results["loss_mask_per_scene"] = loss_mask_per_scene
            results["loss_mask"] = loss_mask_per_scene.mean()

        if self.opt.lambda_opacity > 0:
            opacity = gaussians_leaf[..., 3:4]
            loss_opacity_per_scene = -torch.log(opacity + 1e-6).mean(dim=(1, 2))
            scaled_opacity = self.opt.lambda_opacity * loss_opacity_per_scene.sum() / batch_size
            grad_opacity = torch.autograd.grad(
                scaled_opacity,
                gaussians_leaf,
                retain_graph=False,
                create_graph=False,
            )[0]
            grad_accum += grad_opacity
            results["loss_opacity_per_scene"] = loss_opacity_per_scene.detach()
            results["loss_opacity"] = loss_opacity_per_scene.detach().mean()
            results["loss"] = results["loss"] + self.opt.lambda_opacity * results["loss_opacity"]

        results["backward_loss"] = (gaussians * grad_accum.detach()).sum()
        return results

    def configure_sft_trainable_parameters(self, accelerator=None) -> None:
        """Freeze backbone except Stage-4 latent tail (`latents`, `latent_blocks[-4:]`, `latent_kv_proj`, `k_proj_norm`) when `sft_unfreeze_latent_tail=True`."""
        if not getattr(self.opt, "sft_unfreeze_latent_tail", False):
            return
        for p in self.parameters():
            p.requires_grad_(False)
        if getattr(self.enc_dec_backbone, "latents", None) is not None:
            self.enc_dec_backbone.latents.requires_grad_(True)
        num_lat_blocks = len(self.enc_dec_backbone.latent_blocks)
        for blk in self.enc_dec_backbone.latent_blocks[max(0, num_lat_blocks - 4):]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for attr in ("latent_kv_proj", "k_proj_norm"):
            mod = getattr(self.enc_dec_backbone, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(True)
        if accelerator is None or accelerator.is_main_process:
            tr_m = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
            fr_m = sum(p.numel() for p in self.parameters() if not p.requires_grad) / 1e6
            print(f"[TokenGS-SFT] sft_unfreeze_latent_tail=True: trainable={tr_m:.2f}M params, frozen={fr_m:.2f}M params")

    def compute_render_geo_supervision_loss(
        self,
        data: dict,
        decoder_input: ModelInputDecoder,
        results: dict,
    ) -> dict[str, torch.Tensor]:
        """Compute S1 rendered geometry supervision losses (L_d, L_alpha, L_sigma, L_n) against GT mesh maps."""
        from gsplat.rendering import rasterization

        gaussians = results["gaussians"]
        depths_pred = results["depths_pred"].squeeze(2).float()  # [B, V, H, W]
        alphas_pred = results["alphas_pred"].squeeze(2).float()  # [B, V, H, W]
        zero = gaussians.new_tensor(0.0, dtype=torch.float32)
        if "depths_gt_sup" not in data or "normals_gt_sup" not in data or "valid_gt_sup" not in data:
            raise RuntimeError("use_render_geo_supervision=True requires depths_gt_sup, normals_gt_sup, valid_gt_sup in batch")

        depths_gt = data["depths_gt_sup"].to(depths_pred.device, dtype=torch.float32)
        normals_gt = data["normals_gt_sup"].to(depths_pred.device, dtype=torch.float32)
        gt_ok = data["valid_gt_sup"].to(depths_pred.device).bool() & (depths_gt > 1e-4)
        valid_m = gt_ok & (alphas_pred.detach() > 0.5) & (depths_pred.detach() > 1e-4)

        out: dict[str, torch.Tensor] = {}
        total = zero

        # 1. L_d: log-depth L1 on valid surface pixels
        if self.opt.lambda_render_depth > 0:
            d_mask = gt_ok & (depths_pred.detach() > 1e-4)
            if bool(d_mask.any()):
                log_diff = (depths_pred[d_mask].clamp_min(1e-4).log() - depths_gt[d_mask].clamp_min(1e-4).log()).abs()
                loss_d = log_diff.mean()
            else:
                loss_d = zero
            out["loss_render_depth"] = loss_d.detach()
            total = total + float(self.opt.lambda_render_depth) * loss_d

        # 2. L_alpha: encourage solid opacity on pixels with real mesh surface
        if self.opt.lambda_render_alpha > 0:
            if bool(gt_ok.any()):
                loss_alpha = (1.0 - alphas_pred[gt_ok].clamp(0.0, 1.0)).mean()
            else:
                loss_alpha = zero
            out["loss_render_alpha"] = loss_alpha.detach()
            total = total + float(self.opt.lambda_render_alpha) * loss_alpha

        # 3. L_sigma: ray depth variance sigma_z^2 via background-free [z_cam, z_cam^2] pass
        if self.opt.lambda_render_thickness > 0:
            cam_view = decoder_input.cam_view.float()
            intrinsics = decoder_input.intrinsics.float()
            B, V = cam_view.shape[:2]
            H, W = self.img_size
            means3d = gaussians[..., 0:3].contiguous().float()
            opacity = gaussians[..., 3].contiguous().float()
            scales = gaussians[..., 4:7].contiguous().float()
            rotations = gaussians[..., 7:11].contiguous().float()
            viewmat = cam_view.transpose(3, 2).contiguous()
            Ks = torch.zeros((B, V, 3, 3), dtype=torch.float32, device=means3d.device)
            Ks[:, :, 0, 0] = intrinsics[:, :, 0]
            Ks[:, :, 1, 1] = intrinsics[:, :, 1]
            Ks[:, :, 0, 2] = intrinsics[:, :, 2]
            Ks[:, :, 1, 2] = intrinsics[:, :, 3]
            Ks[:, :, 2, 2] = 1.0
            var_list = []
            for b in range(B):
                r_w2c = viewmat[b, :, :3, :3]
                t_w2c = viewmat[b, :, :3, 3]
                pts_cam = means3d[b].unsqueeze(0) @ r_w2c.transpose(-1, -2) + t_w2c.unsqueeze(1)
                z_cam = pts_cam[..., 2].clamp_min(0.0)
                z_colors = torch.stack([z_cam, z_cam.square()], dim=-1).contiguous()
                bg_zero = torch.zeros((V, 2), dtype=torch.float32, device=means3d.device)
                z12_r, a_r, _ = rasterization(
                    means=means3d[b],
                    quats=rotations[b],
                    scales=scales[b],
                    opacities=opacity[b],
                    colors=z_colors,
                    viewmats=viewmat[b],
                    Ks=Ks[b],
                    width=W,
                    height=H,
                    near_plane=float(self.opt.znear),
                    far_plane=float(self.opt.zfar),
                    packed=False,
                    backgrounds=bg_zero,
                    render_mode="RGB",
                )
                a_safe = a_r[..., 0].clamp_min(1e-6)
                ez = z12_r[..., 0] / a_safe
                ez2 = z12_r[..., 1] / a_safe
                var_z = (ez2 - ez.square()).clamp_min(0.0) / (0.3 ** 2)
                var_list.append(var_z)
            var_m2 = torch.stack(var_list, dim=0)  # [B, V, H, W]
            if bool(valid_m.any()):
                loss_sigma = var_m2[valid_m].mean()
            else:
                loss_sigma = zero
            out["loss_render_thickness"] = loss_sigma.detach()
            total = total + float(self.opt.lambda_render_thickness) * loss_sigma

        # 4. L_n: depth-gradient normal cosine error on smooth valid surface pixels
        if self.opt.lambda_render_normal > 0:
            intrinsics = decoder_input.intrinsics.float()
            B, V, H, W = depths_pred.shape
            u, v = torch.meshgrid(
                torch.arange(W, device=depths_pred.device, dtype=torch.float32) + 0.5,
                torch.arange(H, device=depths_pred.device, dtype=torch.float32) + 0.5,
                indexing="xy",
            )
            fx = intrinsics[..., 0].view(B, V, 1, 1)
            fy = intrinsics[..., 1].view(B, V, 1, 1)
            cx = intrinsics[..., 2].view(B, V, 1, 1)
            cy = intrinsics[..., 3].view(B, V, 1, 1)
            x = (u.view(1, 1, H, W) - cx) / fx * depths_pred
            y = (v.view(1, 1, H, W) - cy) / fy * depths_pred
            pts_cam = torch.stack([x, y, depths_pred], dim=-1)  # [B, V, H, W, 3]
            dx_du = 0.5 * (pts_cam[:, :, 1:-1, 2:, :] - pts_cam[:, :, 1:-1, :-2, :])
            dx_dv = 0.5 * (pts_cam[:, :, 2:, 1:-1, :] - pts_cam[:, :, :-2, 1:-1, :])
            tu = torch.nn.functional.normalize(dx_du, dim=-1, eps=1e-12)
            tv = torch.nn.functional.normalize(dx_dv, dim=-1, eps=1e-12)
            cross_uv = torch.linalg.cross(tu, tv, dim=-1)
            cross_norm = cross_uv.detach().norm(dim=-1)
            n_in = torch.nn.functional.normalize(cross_uv, dim=-1, eps=1e-12)
            n_in = torch.where(n_in[..., 2:3] > 0, -n_in, n_in)

            d_c = depths_pred.detach()[:, :, 1:-1, 1:-1]
            d_r = depths_pred.detach()[:, :, 1:-1, 2:]
            d_l = depths_pred.detach()[:, :, 1:-1, :-2]
            d_d = depths_pred.detach()[:, :, 2:, 1:-1]
            d_u = depths_pred.detach()[:, :, :-2, 1:-1]
            pos_ok = (d_c > 1e-4) & (d_r > 1e-4) & (d_l > 1e-4) & (d_d > 1e-4) & (d_u > 1e-4) & (cross_norm > 1e-6)
            denom = d_c.clamp_min(1e-6)
            jump_u = torch.maximum((d_r - d_c).abs(), (d_l - d_c).abs()) / denom
            jump_v = torch.maximum((d_d - d_c).abs(), (d_u - d_c).abs()) / denom
            smooth_in = pos_ok & (jump_u <= 0.05) & (jump_v <= 0.05)
            n_gt_in = normals_gt[:, :, 1:-1, 1:-1, :]
            n_mask = valid_m[:, :, 1:-1, 1:-1] & smooth_in & (n_gt_in.norm(dim=-1) > 0.5)
            if bool(n_mask.any()):
                cos_n = (n_in[n_mask] * n_gt_in[n_mask]).sum(dim=-1).abs().clamp(0.0, 1.0)
                loss_n = (1.0 - cos_n).mean()
            else:
                loss_n = zero
            out["loss_render_normal"] = loss_n.detach()
            total = total + float(self.opt.lambda_render_normal) * loss_n

        out["loss_render_geo_total"] = total
        return out

    def forward(self, data, skip_loss=False):
        """
        Forward pass of the TokenGS model.
        
        Args:
            data: Dictionary from dataloader
            skip_loss: If True, skip loss computation
            
        Returns:
            Dictionary containing results including 'loss', 'gaussians', 'images_pred', etc.
        """
        # Split data into structured input and supervision
        model_input, supervision = split_data(data, self.opt)

        # Generate reconstruction from input
        if self.opt.use_ttt_for_eval:
            gaussians = self.forward_ttt(model_input, n_steps=self.opt.ttt_n_steps, lr=self.opt.ttt_lr)  # [B, N, 14]
            reconstruction = self._reconstruction_from_gaussians(gaussians)
        else:
            reconstruction = self.forward_reconstruction(model_input)

        results = {"gaussians": reconstruction.gaussians}

        # Compute loss or just render
        if skip_loss:
            results.update(self.render_reconstruction(reconstruction, model_input.decoder))
            return results
        
        # Compute loss
        use_mean_of_grads = (
            self.opt.mean_of_grads != "none"
            and torch.is_grad_enabled()
            and not self.opt.use_ttt_for_eval
        )
        if use_mean_of_grads:
            results.update(self.compute_loss_mean_of_grads(reconstruction, model_input.decoder, supervision))
        else:
            results.update(self.compute_loss(reconstruction, model_input.decoder, supervision))

        if self.opt.use_geo_losses:
            num_children = int(self.opt.dec_patch_size) ** 2
            batch_size, num_gaussians, num_attributes = reconstruction.gaussians.shape
            if num_gaussians % num_children != 0:
                raise ValueError(
                    f"Gaussian count {num_gaussians} is not divisible by "
                    f"children per slot {num_children}"
                )
            slot_gaussians = reconstruction.gaussians.reshape(
                batch_size,
                num_gaussians // num_children,
                num_children,
                num_attributes,
            )
            geo_results = compute_geo_losses(
                gs_centers=slot_gaussians[..., 0:3],
                gs_scales=slot_gaussians[..., 4:7],
                gs_opacities=slot_gaussians[..., 3:4],
                gs_rotations=slot_gaussians[..., 7:11],
                lambda_sw=self.opt.lambda_geo_sw,
                lambda_packing=self.opt.lambda_geo_packing,
                lambda_inter=self.opt.lambda_geo_inter,
                k_neighbors=self.opt.geo_k_neighbors,
                target_distance=self.opt.geo_target_distance,
                num_sw_directions=self.opt.geo_num_sw_directions,
                mass_weighting=self.opt.geo_mass_weighting,
                inter_form=self.opt.geo_inter_form,
            )
            total_geo_loss = geo_results["total_geo_loss"]
            for key in ("sw_loss", "packing_loss", "inter_loss", "total_geo_loss"):
                results[key] = geo_results[key]
            for key, value in geo_results["stats"].items():
                results[f"geo_{key}"] = value

            if use_mean_of_grads:
                results["backward_loss"] = results["backward_loss"] + total_geo_loss
                results["loss"] = results["loss"] + total_geo_loss.detach()
            else:
                results["loss"] = results["loss"] + total_geo_loss

        if self.training and getattr(self.opt, "use_render_geo_supervision", False):
            s1_out = self.compute_render_geo_supervision_loss(data, model_input.decoder, results)
            l_s1 = s1_out.pop("loss_render_geo_total")
            results["loss"] = results["loss"] + l_s1
            if "backward_loss" in results:
                results["backward_loss"] = results["backward_loss"] + l_s1
            results.update(s1_out)

        results['images_output'] = supervision.images_output
        return results

    def forward_ttt(
        self,
        model_input: ModelInput,
        n_steps: int = 10,
        lr: float = 1e-3,
    ) -> torch.Tensor:
        """Test-time training against input views."""
        if self.opt.ttt_mode in ("token-tuning", "tokens"):
            return self.forward_ttt_tokens(model_input, n_steps, lr)
        if self.opt.ttt_mode in ("scene-latent-tuning", "latents"):
            return self.forward_ttt_scene_latents(model_input, n_steps, lr)
        raise ValueError(
            f"Unknown ttt_mode: {self.opt.ttt_mode!r}. "
            "Expected 'token-tuning' or 'scene-latent-tuning'."
        )

    def forward_ttt_tokens(self, model_input: ModelInput, n_steps: int = 10, lr: float = 1e-3) -> torch.Tensor:
        """
        Forward pass of the TokenGS model using test-time training of the Gaussian tokens for `n_steps`.
        
        This method optimizes the Gaussian tokens by rendering to input views and 
        comparing against the input view images as supervision.

        Args:
            model_input: ModelInput containing encoder and decoder inputs (same signature as forward_gaussians)
            n_steps: Number of test-time training steps
            lr: Learning rate for the optimizer

        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians
        """
        # Convert to TTT inputs (render to input views, supervise with input images)
        model_input_ttt, supervision_ttt = model_input.to_ttt()
        
        with freeze_model_parameters(self):
            return self._forward_ttt_tokens(model_input_ttt, supervision_ttt, n_steps, lr)

    @torch.no_grad()
    def _forward_ttt_tokens(
        self, model_input: ModelInput, supervision: ModelSupervision, n_steps: int = 10, lr: float = 1e-3,
    ) -> torch.Tensor:
        """Inner loop for forward_ttt_tokens; caller holds model frozen."""
        encoder_latent = self.forward_encoder(model_input.encoder)

        with torch.set_grad_enabled(True):
            gs_tokens = self.get_gs_tokens(batch_size=model_input.batch_size).detach().requires_grad_(True)
            optim = torch.optim.Adam([gs_tokens], lr=lr)
            for _ in range(n_steps):
                optim.zero_grad()
                gaussians = self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens)
                reconstruction = self._reconstruction_from_gaussians(gaussians)
                loss = self.compute_loss(reconstruction, model_input.decoder, supervision)["loss"]
                loss.backward()
                optim.step()
        
        # compute the final result
        return self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens.detach())

    def forward_ttt_scene_latents(
        self, model_input: ModelInput, n_steps: int = 10, lr: float = 1e-3
    ) -> torch.Tensor:
        """
        Forward pass using scene latent tuning for `n_steps`.

        The encoder and latent bottleneck run once. TTT then optimizes the
        per-scene latent bottleneck output, re-projects it to decoder keys and
        values each step, and keeps the model weights and Gaussian tokens fixed.
        """
        model_input_ttt, supervision_ttt = model_input.to_ttt()

        with freeze_model_parameters(self):
            return self._forward_ttt_scene_latents(model_input_ttt, supervision_ttt, n_steps, lr)

    @torch.no_grad()
    def _forward_ttt_scene_latents(
        self, model_input: ModelInput, supervision: ModelSupervision, n_steps: int = 10, lr: float = 1e-3,
    ) -> torch.Tensor:
        """Inner loop for forward_ttt_scene_latents; caller holds model frozen."""
        scene_latents = self.forward_encoder_to_scene_latents(model_input.encoder)
        gs_tokens = self.get_gs_tokens(batch_size=model_input.batch_size).detach()

        with torch.set_grad_enabled(True):
            scene_latents = scene_latents.detach().requires_grad_(True)
            optim = torch.optim.Adam([scene_latents], lr=lr)
            for _ in range(n_steps):
                optim.zero_grad()
                encoder_latent = self.encoder_latent_from_scene_latents(scene_latents)
                gaussians = self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens)
                reconstruction = self._reconstruction_from_gaussians(gaussians)
                loss = self.compute_loss(reconstruction, model_input.decoder, supervision)["loss"]
                loss.backward()
                optim.step()

        encoder_latent = self.encoder_latent_from_scene_latents(scene_latents.detach())
        return self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens)
