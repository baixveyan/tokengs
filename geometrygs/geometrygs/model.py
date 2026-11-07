"""TokenGS with a part of the Gaussian tokens fixed at SfM points of the input views.

Scene latents and the remaining (free) Gaussian tokens are unchanged. For each scene, known-pose SfM is run on the
input views, at most `sfm_num_anchors` points are chosen by farthest point sampling, and the first GS-token slots are
replaced by anchored tokens: a shared learnable base + an embedding of the anchor position (Fourier features) and of
the encoder patch features at the anchor's observations. Anchored tokens go through the same decoder (cross-attention
to scene latents, self-attention with all GS tokens) and readout head for colour, opacity, scale and rotation, but
their 64 children are placed at anchor + offset_max * tanh(offset). Slots without an anchor (fewer SfM points than
slots) keep their original learned token and readout.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.models.input_types import EncoderLatent, ModelInput
from tokengs.models.tokengs import TokenGS

from geometrygs.sfm import KnownPoseSfM, SfmConfig, select_anchors


def anchor_quality_kwargs(opt) -> dict:
    return {"strong_track": opt.anchor_strong_track, "strong_angle_deg": opt.anchor_strong_angle_deg,
            "knn": opt.anchor_knn, "max_knn_dist": opt.anchor_max_knn_dist}


def fourier_features(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * math.pi
    angles = x[..., None] * freqs
    return torch.cat([x, angles.sin().flatten(-2), angles.cos().flatten(-2)], -1)


class SfmTokenGS(TokenGS):
    def __init__(self, opt):
        super().__init__(opt)
        if not opt.use_latent_bottleneck:
            raise ValueError("SfmTokenGS requires the latent-bottleneck architecture")
        dim = opt.enc_embed_dim
        feat_dim = opt.enc_embed_dim * (len(opt.multiscale_encoder_layers) if opt.use_multiscale_encoder else 1)
        children = opt.dec_patch_size ** 2
        self.num_anchor_slots = int(opt.sfm_num_anchors)
        self.anchor_pe_mlp = nn.Sequential(nn.Linear(3 + 6 * opt.anchor_pe_freqs, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor_feat_proj = (
            nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim)) if opt.anchor_image_feature else None
        )
        self.anchor_embed_norm = nn.LayerNorm(dim)
        self.anchor_token_base = nn.Parameter(torch.zeros(dim))
        self.anchor_pos_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3 * children))
        nn.init.normal_(self.anchor_pos_head[1].weight, std=0.01)
        nn.init.zeros_(self.anchor_pos_head[1].bias)
        self.sfm = KnownPoseSfM(SfmConfig(
            num_features=opt.sfm_num_features, match_ratio=opt.sfm_match_ratio,
            max_epipolar_px=opt.sfm_max_epipolar_px, max_reproj_px=opt.sfm_max_reproj_px,
            min_tri_angle_deg=opt.sfm_min_tri_angle_deg, min_track=opt.sfm_min_track,
            z_near=opt.znear, z_far=opt.zfar,
        ))
        self.last_sfm: list[dict] = []

    def anchor_modules(self) -> list[nn.Module | nn.Parameter]:
        mods = [self.anchor_pe_mlp, self.anchor_embed_norm, self.anchor_pos_head, self.anchor_token_base]
        return mods + ([self.anchor_feat_proj] if self.anchor_feat_proj is not None else [])

    def configure_sft_trainable_parameters(self, accelerator=None) -> None:
        super().configure_sft_trainable_parameters(accelerator)
        scale = float(self.opt.anchor_lr_scale)
        for m in self.anchor_modules():
            for p in ([m] if isinstance(m, nn.Parameter) else m.parameters()):
                p.requires_grad_(True)
                p._lr_scale = scale
        if accelerator is None or accelerator.is_main_process:
            n = sum(p.numel() for m in self.anchor_modules() for p in ([m] if isinstance(m, nn.Parameter) else m.parameters()))
            print(f"[SfmTokenGS] anchor modules: {n / 1e6:.2f}M params, lr scale {scale}")

    # ------------------------------------------------------------------ encoder (keeps image features)
    def _encode(self, encoder_input) -> tuple[EncoderLatent, torch.Tensor]:
        bb = self.enc_dec_backbone
        image_features = bb._encode_features(self._embed_encoder_input(encoder_input))
        keys, values = bb._latents_to_kv(bb._features_to_scene_latents(image_features))
        return EncoderLatent(keys=keys, values=values), image_features

    # ------------------------------------------------------------------ SfM anchors
    @torch.no_grad()
    def compute_anchors(self, model_input: ModelInput) -> list[dict[str, torch.Tensor]]:
        sfm_in = getattr(model_input, "sfm", None)
        if sfm_in is None:
            raise RuntimeError("SfmTokenGS needs batch['images_sfm'] / ['intrinsics_sfm'] (use geometrygs.data.SfmProvider)")
        c2w = model_input.encoder.cam_to_world_input
        out = []
        for b in range(c2w.shape[0]):
            rec = self.sfm.reconstruct(sfm_in["images"][b], c2w[b], sfm_in["intrinsics"][b])
            sel = select_anchors(rec, self.num_anchor_slots, **anchor_quality_kwargs(self.opt))
            rec["anchor_index"] = sel
            out.append(rec)
        self.last_sfm = out
        return out

    def _sample_anchor_features(self, feats: torch.Tensor, xyz: torch.Tensor, obs_mask: torch.Tensor,
                                c2w: torch.Tensor, intr: torch.Tensor) -> torch.Tensor:
        """Average of encoder patch features at the anchor's projections in the views where SfM observed it."""
        V = c2w.shape[0]
        H, W = self.img_size
        p = self.opt.patch_size
        fmap = feats.view(V, H // p, W // p, -1).permute(0, 3, 1, 2)  # [V, C, h, w]
        w2c = torch.linalg.inv(c2w.float())
        pc = torch.einsum("vij,nj->vni", w2c[:, :3, :3], xyz) + w2c[:, None, :3, 3]
        z = pc[..., 2].clamp_min(1e-6)
        u = intr[:, None, 0] * pc[..., 0] / z + intr[:, None, 2]
        v = intr[:, None, 1] * pc[..., 1] / z + intr[:, None, 3]
        grid = torch.stack([u / W * 2 - 1, v / H * 2 - 1], -1)  # [V, n, 2]
        sampled = F.grid_sample(fmap.float(), grid[:, None].float(), align_corners=False)[:, :, 0]  # [V, C, n]
        inside = (grid.abs() <= 1).all(-1) & (pc[..., 2] > 0)
        w = (obs_mask.T & inside).float()  # [V, n]
        return (sampled * w[:, None]).sum(0).T / w.sum(0).clamp_min(1.0)[:, None]

    def _anchor_tokens(self, xyz: torch.Tensor, feat: torch.Tensor | None) -> torch.Tensor:
        emb = self.anchor_pe_mlp(fourier_features(xyz, self.opt.anchor_pe_freqs))
        if self.anchor_feat_proj is not None:
            emb = emb + self.anchor_feat_proj(feat)
        return self.anchor_token_base + float(self.opt.gs_token_std) * self.anchor_embed_norm(emb)

    # ------------------------------------------------------------------ decoding
    def forward_reconstruction(self, model_input: ModelInput):
        encoder_latent, feats = self._encode(model_input.encoder)
        anchors = self.compute_anchors(model_input)
        enc = model_input.encoder
        B, K = feats.shape[0], self.num_anchor_slots
        dev = feats.device
        xyz = torch.zeros(B, K, 3, device=dev)
        mask = torch.zeros(B, K, dtype=torch.bool, device=dev)
        anchor_feat = torch.zeros(B, K, feats.shape[-1], device=dev) if self.anchor_feat_proj is not None else None
        for b, rec in enumerate(anchors):
            sel = rec["anchor_index"][:K]
            n = int(sel.numel())
            if n == 0:
                continue
            xyz[b, :n] = rec["points"][sel]
            mask[b, :n] = True
            if anchor_feat is not None:
                anchor_feat[b, :n] = self._sample_anchor_features(
                    feats[b], rec["points"][sel], rec["obs_mask"][sel], enc.cam_to_world_input[b], enc.intrinsics_input[b])

        tokens = self._apply_time_embedding_to_gs_tokens(self.get_gs_tokens(B), model_input.decoder)
        anchor_tok = self._anchor_tokens(xyz, anchor_feat)
        tokens = torch.cat([torch.where(mask[..., None], anchor_tok, tokens[:, :K]), tokens[:, K:]], 1)
        for layer in self.enc_dec_backbone.decoder_blocks:
            tokens = layer(gs_tokens=tokens, keys=encoder_latent.keys, values=encoder_latent.values)

        gaussians = self.activation_head(tokens)
        gaussians[..., 2] = gaussians[..., 2] + self.opt.gaussian_z_offset
        P = self.opt.dec_patch_size ** 2
        N = tokens.shape[1]
        offsets = float(self.opt.anchor_offset_max) * torch.tanh(self.anchor_pos_head(tokens[:, :K]).view(B, K, P, 3))
        pos = gaussians[..., :3].reshape(B, N, P, 3)
        anchored = torch.where(mask[..., None, None], xyz[:, :, None, :] + offsets, pos[:, :K])
        pos = torch.cat([anchored, pos[:, K:]], 1).reshape(B, N * P, 3)
        gaussians = torch.cat([pos, gaussians[..., 3:]], -1)
        return self._reconstruction_from_gaussians(gaussians)

    def forward(self, data, skip_loss=False):
        """TokenGS.forward with the per-batch input-view count (training samples 4-8 views)."""
        base = self.opt
        v_in = int(data["images_input"].shape[1])
        if v_in != base.num_input_views:
            self.opt = dataclasses.replace(base, num_input_views=v_in, num_views=int(data["images_all"].shape[1]))
        try:
            return super().forward(data, skip_loss=skip_loss)
        finally:
            self.opt = base
