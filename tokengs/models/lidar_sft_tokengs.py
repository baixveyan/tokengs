import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengs.losses.geo_losses import _quaternion_wxyz_to_rotation_matrix
from tokengs.models.input_types import EncoderLatent, ModelInputDecoder, ModelInputEncoder
from tokengs.models.tokengs import TokenGS
from tokengs.options import Options


def fourier_encode_3d(xyz: torch.Tensor, num_bands: int = 6) -> torch.Tensor:
    """3D Fourier positional encoding mapping [..., 3] -> [..., 3 + 3 * 2 * num_bands = 39]."""
    freqs = (2.0 ** torch.arange(num_bands, dtype=xyz.dtype, device=xyz.device)) * math.pi
    ang = xyz.unsqueeze(-1) * freqs
    sin_cos = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).flatten(-2)
    return torch.cat([xyz, sin_cos], dim=-1)


class LiDARSceneLatentProbeAdapter(nn.Module):
    """3D LiDAR Surface Introspection Probe & Zero-Init Residual Grounding Adapter at `scene_latents` (Stage 3 exit).

    1) Probes each of the 4096 `scene_latents` [B, 4096, 1024] to predict its 3D surface anchor point
       `pred_pts` [B, 4096, 3] and surface normal `pred_normals` [B, 4096, 3] (supervised by ScanNet++ LiDAR!).
    2) Encodes `(pred_pts, pred_normals)` via 3D Fourier PE + MLP + Zero-Initialized Linear projection and
       adds it back as a residual to `scene_latents`:
         `Z_final = Z_0 + Linear_zero(MLP(gamma_3D(pred_pts), pred_normals))`
       Because `Linear_zero` is initialized to exact zeros, at Step 0 `Z_final == Z_0` bit-for-bit!
    """

    def __init__(self, dim: int = 1024, num_bands: int = 6):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        pe_dim = 3 + 3 * 2 * num_bands + 3  # 39 + 3 = 42

        self.probe_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 6),  # [x, y, z, nx, ny, nz]
        )
        nn.init.trunc_normal_(self.probe_head[-1].weight, std=0.002)
        nn.init.zeros_(self.probe_head[-1].bias)
        with torch.no_grad():
            self.probe_head[-1].bias[2] = 0.25

        self.geo_encoder = nn.Sequential(
            nn.Linear(pe_dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim),
            nn.SiLU(),
        )
        self.zero_inject_proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.zero_inject_proj.weight)
        nn.init.zeros_(self.zero_inject_proj.bias)

    def forward(self, raw_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Fix 3: detach `raw_latents` entering `probe_head` so early random probe errors never shock pretrained latent_blocks!
        raw_6d = self.probe_head(raw_latents.detach())
        u_xyz = raw_6d[..., 0:3]
        pred_pts = torch.sign(u_xyz) * torch.expm1(u_xyz.abs().clamp(max=3.0))
        pred_normals = F.normalize(
            raw_6d[..., 3:6] + raw_latents.new_tensor([0.0, 0.0, -1.0]),
            dim=-1, eps=1e-6,
        )
        pe_3d = torch.cat([fourier_encode_3d(pred_pts, self.num_bands), pred_normals], dim=-1)
        geo_feat = self.zero_inject_proj(self.geo_encoder(pe_3d))
        enriched_latents = raw_latents + geo_feat
        return enriched_latents, pred_pts, pred_normals


class LiDARSFTTokenGS(TokenGS):
    """Post-Training SFT TokenGS with ScanNet++ LiDAR 3D Surface Grounding (`model_type = 'tokengs_lidar_sft'`).

    Supports three SFT modes (`opt.lidar_sft_mode`):
    1) `"scene_latent_probe"` (Experiment 1 / 方式一):
       Finetunes ONLY `[部位 2 后4层 + 部位 3]` + `LiDARSceneLatentProbeAdapter`; freezes `ViT`, `gs_tokens`, `decoder_blocks`, `deconv`.
    2) `"surface_self_attn"` (Experiment 2 / 方式二):
       In addition to Mode 1, unfreezes `[部位 4]` (`gs_tokens` + `decoder_blocks[0]`); freezes `ViT` and `deconv`.
    3) `"fused_full_decoder"` (Experiment 3 / 融合方案 + 解冻反卷积层 `deconv`, 100 Epochs):
       - Freezes: `[部位 1] 12-layer ViT` + `[部位 2 前8层] latent_blocks.0..7` + `decoder_blocks[0].gs_cross_attn` (protects high-freq 2D texture readout!).
       - Unfreezes with gentle `0.25x` LR (`1e-5`): `[部位 2 后4层 + 部位 3]` + `[部位 4] gs_tokens & decoder_blocks[0].(gs_self_attn + mlp)` + **`[部位 5] activation_head.deconv`**!
       - Unfreezes with full `1.0x` LR (`4e-5`): New SFT modules (`lidar_adapter` + `token_3d_grounding`).
       - Supervises both `scene_latents` AND individual decoded child Gaussians with:
         (a) 3D LiDAR surface distance (individual active children, not just token average!),
         (b) Soft needle elimination (`ReLU(s1/s2 - 2.0)`),
         (c) Soft blade-to-surface rotation alignment (penalizing long axes pointing along the camera ray).
    """

    def __init__(self, opt: Options):
        super().__init__(opt)
        self.lidar_adapter = LiDARSceneLatentProbeAdapter(dim=opt.enc_embed_dim)
        mode = getattr(opt, "lidar_sft_mode", "scene_latent_probe")
        if mode in ("surface_self_attn", "fused_full_decoder"):
            self.token_3d_grounding = nn.Linear(opt.enc_embed_dim, opt.enc_embed_dim, bias=True)
            nn.init.zeros_(self.token_3d_grounding.weight)
            nn.init.zeros_(self.token_3d_grounding.bias)
        else:
            self.token_3d_grounding = None

        self._latest_pred_pts: Optional[torch.Tensor] = None
        self._latest_pred_normals: Optional[torch.Tensor] = None
        self._latest_geo_delta: Optional[torch.Tensor] = None

    def configure_sft_trainable_parameters(self, accelerator=None) -> None:
        """Apply exact anatomical freezing and differential LR tagging."""
        mode = getattr(self.opt, "lidar_sft_mode", "scene_latent_probe")
        # 1. Freeze all parameters first
        for p in self.parameters():
            p.requires_grad_(False)

        def _unfreeze(params_iter, lr_scale: float = 1.0):
            for p in params_iter:
                p.requires_grad_(True)
                p._lr_scale = lr_scale

        # 2. Unfreeze [部位 2 后 4 层 + 部位 3] (`latents`, `latent_blocks.8..11`, `k_proj/v_proj`)
        pre_lr_scale = 0.25 if mode == "fused_full_decoder" else 1.0
        if getattr(self.enc_dec_backbone, "latents", None) is not None:
            _unfreeze([self.enc_dec_backbone.latents], lr_scale=pre_lr_scale)
        num_lat_blocks = len(self.enc_dec_backbone.latent_blocks)
        for blk in self.enc_dec_backbone.latent_blocks[max(0, num_lat_blocks - 4):]:
            _unfreeze(blk.parameters(), lr_scale=pre_lr_scale)
        for attr in ("latent_proj_k", "latent_proj_v", "latent_norm_k", "latent_norm_v", "k_proj_norm", "v_proj_norm"):
            mod = getattr(self.enc_dec_backbone, attr, None)
            if mod is not None:
                _unfreeze(mod.parameters(), lr_scale=pre_lr_scale)

        # 3. Unfreeze new SFT `lidar_adapter` at full 1.0x LR
        _unfreeze(self.lidar_adapter.parameters(), lr_scale=1.0)

        # 4. Mode-specific downstream decoder & deconv unfreezing
        if mode == "surface_self_attn":
            if self.gs_tokens is not None:
                _unfreeze([self.gs_tokens], lr_scale=1.0)
            for blk in self.enc_dec_backbone.decoder_blocks:
                _unfreeze(blk.parameters(), lr_scale=1.0)
            if self.token_3d_grounding is not None:
                _unfreeze(self.token_3d_grounding.parameters(), lr_scale=1.0)

        elif mode == "fused_full_decoder":
            # (a) Unfreeze gs_tokens & decoder_blocks[0].(gs_self_attn + mlp) while keeping gs_cross_attn FROZEN to protect texture readout
            if self.gs_tokens is not None:
                _unfreeze([self.gs_tokens], lr_scale=pre_lr_scale)
            for blk in self.enc_dec_backbone.decoder_blocks:
                for sub_name in ("gs_self_attn", "gs_self_attn_scale", "mlp", "mlp_scale"):
                    sub_mod = getattr(blk, sub_name, None)
                    if sub_mod is not None:
                        _unfreeze(sub_mod.parameters(), lr_scale=pre_lr_scale)
            # (b) Unfreeze [部位 5] `activation_head.deconv` at gentle 0.25x LR so it reshapes needles/blades into surface disks!
            if hasattr(self, "activation_head"):
                _unfreeze(self.activation_head.parameters(), lr_scale=pre_lr_scale)
            # (c) Unfreeze new `token_3d_grounding` at full 1.0x LR
            if self.token_3d_grounding is not None:
                _unfreeze(self.token_3d_grounding.parameters(), lr_scale=1.0)

        if accelerator is None or accelerator.is_main_process:
            tr_m = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
            fr_m = sum(p.numel() for p in self.parameters() if not p.requires_grad) / 1e6
            deconv_grad = hasattr(self, "activation_head") and any(p.requires_grad for p in self.activation_head.parameters())
            print(f"[LiDAR-SFT] mode={mode}: trainable={tr_m:.2f}M params, frozen={fr_m:.2f}M params (deconv_trainable={deconv_grad})")

    def forward_encoder_to_scene_latents(self, encoder_input: ModelInputEncoder) -> torch.Tensor:
        raw_latents = super().forward_encoder_to_scene_latents(encoder_input)
        enriched_latents, pred_pts, pred_normals = self.lidar_adapter(raw_latents)
        self._latest_pred_pts = pred_pts
        self._latest_pred_normals = pred_normals
        self._latest_geo_delta = enriched_latents - raw_latents
        return enriched_latents

    def forward_encoder(self, encoder_input: ModelInputEncoder) -> EncoderLatent:
        scene_latents = self.forward_encoder_to_scene_latents(encoder_input)
        return self.encoder_latent_from_scene_latents(scene_latents)

    def forward_decoder(
        self,
        encoder_latent: EncoderLatent,
        decoder_input: ModelInputDecoder,
        gs_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = encoder_latent.keys.shape[0]
        if gs_tokens is None:
            gs_tokens = self.get_gs_tokens(batch_size=B)
        if self.token_3d_grounding is not None and self._latest_geo_delta is not None:
            gs_tokens = gs_tokens + self.token_3d_grounding(self._latest_geo_delta)
        return super().forward_decoder(encoder_latent, decoder_input, gs_tokens=gs_tokens)

    def compute_lidar_grounding_loss(self, data: dict, gaussians: torch.Tensor) -> dict[str, torch.Tensor]:
        """Supervise `scene_latents` AND (in `fused_full_decoder` mode) individual active child Gaussians'
        3D surface adherence, aspect ratio (`s1/s2 <= 2.0`), and surface-tangent orientation!
        """
        if "lidar_points" not in data or self._latest_pred_pts is None or self._latest_pred_normals is None:
            zero = gaussians.new_tensor(0.0)
            return {"loss_lidar_total": zero}

        lidar_pts = data["lidar_points"].float()       # [B, M=8192, 3]
        pred_pts = self._latest_pred_pts.float()       # [B, N=4096, 3]
        pred_norms = self._latest_pred_normals.float() # [B, N=4096, 3]

        # 1. Stage-3 `scene_latents` Bidirectional 3D Chamfer Distance to LiDAR surface
        dists = torch.cdist(pred_pts, lidar_pts)  # [B, 4096, 8192]
        min_p2l, nn_idx = dists.min(dim=2)
        min_l2p = dists.min(dim=1).values
        loss_chamfer = F.smooth_l1_loss(min_p2l, torch.zeros_like(min_p2l), beta=0.02) + \
                       F.smooth_l1_loss(min_l2p, torch.zeros_like(min_l2p), beta=0.02)

        # 2. Stage-3 `scene_latents` Local LiDAR Tangent-Plane Normal Orthogonality
        B, N, _ = pred_pts.shape
        M = lidar_pts.shape[1]
        b_idx = torch.arange(B, device=lidar_pts.device).unsqueeze(1).expand(B, N)
        q0 = lidar_pts[b_idx, nn_idx]
        norm_err = pred_pts.new_tensor(0.0)
        for offset in (1, 2, 4):
            nb_idx = (nn_idx + offset).clamp(max=M - 1)
            q_nb = lidar_pts[b_idx, nb_idx]
            diff = q_nb - q0
            dist_nb = diff.norm(dim=-1, keepdim=True)
            valid_nb = (dist_nb.squeeze(-1) > 1e-4) & (dist_nb.squeeze(-1) < 0.15)
            tangent = diff / dist_nb.clamp_min(1e-5)
            cos_tn = (pred_norms * tangent).sum(dim=-1).abs()
            norm_err = norm_err + (cos_tn * valid_nb.float()).mean()
        loss_normal = norm_err / 3.0

        total_lidar_loss = 0.20 * loss_chamfer + 0.05 * loss_normal
        mode = getattr(self.opt, "lidar_sft_mode", "scene_latent_probe")

        if mode == "surface_self_attn":
            gs_slots = gaussians[:, :, :4].reshape(B, N, -1, 4)
            op_w = gs_slots[..., 3:4].detach().clamp_min(1e-4)
            tok_core_xyz = (gs_slots[..., 0:3] * op_w).sum(dim=2) / op_w.sum(dim=2).clamp_min(1e-4)
            d_core = torch.cdist(tok_core_xyz.float(), lidar_pts).min(dim=2).values
            loss_core_surf = F.smooth_l1_loss(d_core, torch.zeros_like(d_core), beta=0.02)
            total_lidar_loss = total_lidar_loss + 0.10 * loss_core_surf

        elif mode == "fused_full_decoder":
            # Unfrozen `deconv` supervision on active child Gaussians:
            # (a) Individual active child Gaussians (subsampled 16384 per scene for fast cdist) pulled onto LiDAR surface
            #     + Intra-token depth variance penalty so 64 children cannot split front/back!
            gs_f = gaussians.float()  # [B, 262144, 14]
            xyz_all = gs_f[:, :, 0:3]
            op_all = gs_f[:, :, 3]
            sc_all = gs_f[:, :, 4:7]
            rot_all = gs_f[:, :, 7:11]

            # Subsample 8192 high-opacity Gaussians per scene to compute individual child-to-LiDAR distance
            top_k_idx = op_all.detach().topk(8192, dim=1).indices  # [B, 8192]
            xyz_top = xyz_all.gather(1, top_k_idx.unsqueeze(-1).expand(-1, -1, 3))  # [B, 8192, 3]
            op_top = op_all.gather(1, top_k_idx).detach().clamp(0.01, 1.0)          # [B, 8192]
            d_child_to_lidar = torch.cdist(xyz_top, lidar_pts).min(dim=2).values    # [B, 8192]
            loss_child_surf = (F.smooth_l1_loss(d_child_to_lidar, torch.zeros_like(d_child_to_lidar), beta=0.02, reduction="none") * op_top).mean()

            # Intra-token child spread along depth (prevents 32 children in front of wall and 32 behind wall!)
            gs_slots = gs_f[:, :, :4].reshape(B, N, -1, 4)
            op_s = (gs_slots[..., 3].detach() >= 0.01).float()
            op_sum = op_s.sum(dim=2, keepdim=True).clamp_min(1.0)
            mean_xyz = (gs_slots[..., 0:3] * op_s.unsqueeze(-1)).sum(dim=2, keepdim=True) / op_sum.unsqueeze(-1)
            # Project child offset onto camera ray direction to penalize radial depth thickness per token
            c2w_in = data["cam_to_world_input"].float()  # [B, V_in, 4, 4]
            cam_center = c2w_in[:, :, :3, 3].mean(dim=1, keepdim=True).unsqueeze(2)  # [B, 1, 1, 3]
            ray_dir_tok = F.normalize(mean_xyz.detach() - cam_center, dim=-1, eps=1e-6)  # [B, N, 1, 3]
            radial_offset = ((gs_slots[..., 0:3] - mean_xyz.detach()) * ray_dir_tok).sum(dim=-1).abs()  # [B, N, 64]
            loss_radial_thick = (radial_offset * op_s).sum() / op_s.sum().clamp_min(1.0)

            # (b) Fix 2: Tangent-Width-Preserving Anisotropy Loss on unfrozen `deconv`!
            # Since `deconv` is shared across all 4096 tokens, pushing only `s1` down would create a net negative
            # bias gradient on all scale channels. We therefore symmetrically push `l1` down and `l2` up on needles
            # (`s1/s2 > 2.0`), AND anchor the tangent disk width `s2` above `0.0075m` (official median `0.00789m`)!
            alive_mask = op_all.detach() >= 0.01
            sc_sorted, sc_idx = sc_all.sort(dim=-1, descending=True)
            s1, s2 = sc_sorted[..., 0].clamp_min(1e-6), sc_sorted[..., 1].clamp_min(1e-6)
            l1, l2 = s1.log(), s2.log()
            half_log_cap = 0.5 * math.log(2.0)
            needle_mask = alive_mask & ((l1.detach() - l2.detach()) > (2.0 * half_log_cap))
            alive_w = alive_mask.float()
            alive_denom = alive_w.sum().clamp_min(1.0)
            if bool(needle_mask.any()):
                l_mid = 0.5 * (l1.detach() + l2.detach())
                l1_tgt = l_mid + half_log_cap
                l2_tgt = l_mid - half_log_cap
                n_denom = needle_mask.float().sum().clamp_min(1.0)
                loss_needle_sym = (
                    (0.4 * (l1 - l1_tgt).pow(2) + 0.8 * (l2 - l2_tgt).pow(2)) * needle_mask.float()
                ).sum() / n_denom
            else:
                loss_needle_sym = gs_f.new_tensor(0.0)
            # Prevent tangent radius `s2` from shrinking below the official pretrained disk radius (0.0078m)
            loss_s2_floor = (F.relu(math.log(0.0078) - l2).pow(2) * alive_w).sum() / alive_denom
            loss_aniso = loss_needle_sym + 0.5 * loss_s2_floor

            # (c) Fix 1: Pure-Rotation Blade Orientation Loss (zero gradient on `sc_all`, soft hinge at cos > 0.35 for side walls!)
            rot_mats = _quaternion_wxyz_to_rotation_matrix(rot_all.reshape(-1, 4)).reshape(B, -1, 3, 3)
            # Extract the unit eigenvector `e_long` corresponding to the largest scale `s1` (`sc_idx[..., 0]`)
            long_axis_idx = sc_idx[..., 0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 1)  # [B, 262144, 3, 1]
            e_long = torch.gather(rot_mats, dim=3, index=long_axis_idx).squeeze(-1)          # [B, 262144, 3]
            cam_c_flat = c2w_in[:, :, :3, 3].mean(dim=1, keepdim=True)
            unit_ray = F.normalize(xyz_all.detach() - cam_c_flat, dim=-1, eps=1e-6)
            cos_long_ray = (e_long * unit_ray).sum(dim=-1).abs()  # [B, 262144]
            blade_excess = F.relu(cos_long_ray - 0.35).pow(2)
            alive_w = alive_mask.float()
            loss_blade_orient = (blade_excess * alive_w).sum() / alive_w.sum().clamp_min(1.0)

            total_lidar_loss = (
                total_lidar_loss
                + 0.10 * loss_child_surf
                + 0.03 * loss_radial_thick
                + 0.02 * loss_aniso
                + 0.02 * loss_blade_orient
            )
            return {
                "loss_lidar_total": total_lidar_loss,
                "loss_lidar_chamfer": loss_chamfer.detach(),
                "loss_lidar_normal": loss_normal.detach(),
                "loss_child_surf": loss_child_surf.detach(),
                "loss_aniso": loss_aniso.detach(),
                "loss_blade_orient": loss_blade_orient.detach(),
            }

        return {
            "loss_lidar_total": total_lidar_loss,
            "loss_lidar_chamfer": loss_chamfer.detach(),
            "loss_lidar_normal": loss_normal.detach(),
        }

    def forward(self, data: dict, skip_loss: bool = False) -> dict:
        results = super().forward(data, skip_loss=skip_loss)
        if not skip_loss and "loss" in results and self.training:
            lidar_losses = self.compute_lidar_grounding_loss(data, results["gaussians"])
            l_lidar = lidar_losses["loss_lidar_total"]
            results["loss"] = results["loss"] + l_lidar
            if "backward_loss" in results:
                results["backward_loss"] = results["backward_loss"] + l_lidar
            results.update({k: v for k, v in lidar_losses.items() if k != "loss_lidar_total"})
        return results
