"""GeometryGS: pretrained scene memory and a new full-width spatial Gaussian decoder."""

import dataclasses
import json

import torch
from torch import nn

from geometrygs.model import SfmTokenGS, anchor_quality_kwargs, fourier_features
from geometrygs.sfm import KnownPoseSfM, SfmConfig, select_anchors
from geometrygs.soft_chain import soft_chain_loss
from geometrygs.spatial import Attention, ImageBlock, LocalGaussianHead, SceneBlock, SpatialDecoderBlock, run_block
from geometrygs.scene_backbone import TokenGSSceneBackbone, SfmSceneAdapter
from tokengs.models.attention import PatchEmbed
from tokengs.models.tokengs import TokenGS
from tokengs.rendering.gs import GaussianRenderer


# Embedded in every safetensors checkpoint: evaluation must not guess architecture
# from the experiment name or silently load only matching tensors.
BASE_ARCH_FIELDS = (
    "img_size", "patch_size", "dec_patch_size", "enc_embed_dim", "token_dim", "enc_num_heads",
    "enc_depth", "dec_depth", "latent_cross_attn_depth", "num_latents", "num_gs_tokens",
    "mlp_ratio", "gaussian_scale_cap", "opacity_bias", "bg_color", "znear", "zfar",
    "camera_normalization_method",
    "use_multiscale_encoder", "multiscale_encoder_layers",
)


def architecture_config(opt):
    from tokengs.options import Options
    base = {f.name for f in dataclasses.fields(Options)}
    extra = {f.name for f in dataclasses.fields(opt)} - base
    return {"version": 2, "options": {k: getattr(opt, k) for k in sorted(extra | set(BASE_ARCH_FIELDS))}}


def options_from_architecture(tensor):
    from geometrygs.options import config_defaults
    data = json.loads(bytes(tensor.cpu().tolist()))
    if data["version"] not in (1, 2):
        raise ValueError(f"unsupported GeometryGS architecture version {data['version']}")
    values = data["options"]
    if data["version"] == 1:
        values.update(scene_backbone="native", use_multiscale_encoder=False)
    for k in ("img_size", "sfm_image_size", "multiscale_encoder_layers"):
        if k in values:
            values[k] = tuple(values[k])
    return config_defaults["spatial_small_sfm"].evolve(**values, evaluating=True, use_input_supervision=False,
                                                use_render_geo_supervision=False, val_depth_metric=False,
                                                num_input_views=6, num_views=14)


class SpatialTokenGS(TokenGS):
    """Keep the official image/scene path; replace Gaussian decoding and readout."""

    def __init__(self, opt):
        nn.Module.__init__(self)
        opt.validate()
        self.opt, self.img_size = opt, opt.img_size
        d, heads = opt.enc_embed_dim, opt.enc_num_heads
        self.patch_embed = PatchEmbed(img_size=opt.img_size, patch_size=opt.patch_size, in_chans=3,
                                     embed_dim=d, norm_layer=nn.LayerNorm)
        self.patch_plucker_embed = PatchEmbed(img_size=opt.img_size, patch_size=opt.patch_size, in_chans=6,
                                             embed_dim=d, norm_layer=nn.LayerNorm)
        if opt.scene_backbone == "tokengs":
            self.scene_encoder = TokenGSSceneBackbone(opt)
            feature_dim = self.scene_encoder.feature_dim
            self.scene_adapters = nn.ModuleList([SfmSceneAdapter(opt) for _ in range(opt.sfm_fusion_depth)])
        else:
            feature_dim = d
            self.image_blocks = nn.ModuleList([ImageBlock(d, heads, opt.mlp_ratio) for _ in range(opt.enc_depth)])
            self.image_norm = nn.LayerNorm(d)
            self.scene_latents = nn.Parameter(torch.randn(opt.num_latents, d) * 0.02)
            self.scene_blocks = nn.ModuleList([SceneBlock(d, heads, opt.mlp_ratio)
                                              for _ in range(opt.latent_cross_attn_depth)])
            self.scene_norm = nn.LayerNorm(d)
        self.sfm_position = nn.Linear(3 + 6 * opt.spatial_pe_freqs, d)
        self.sfm_feature = nn.Linear(feature_dim, d)
        self.sfm_quality = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.sfm_norm = nn.LayerNorm(d)
        self.gs_tokens = nn.Parameter(torch.randn(opt.num_gs_tokens, d) * 0.02)
        self.token_types = nn.Embedding(2, d)
        self.initial_query = Attention(d, heads)
        self.initial_center = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        self.decoder_blocks = nn.ModuleList([SpatialDecoderBlock(opt, i) for i in range(opt.dec_depth)])
        self.activation_head = LocalGaussianHead(opt)
        self.gs = GaussianRenderer(opt)
        self.lpips_loss = None
        if opt.lambda_lpips > 0:
            from lpips import LPIPS
            self.lpips_loss = LPIPS(net="vgg").requires_grad_(False)
        self.lambda_dyn_aux_eff = 0.0
        self.sfm = KnownPoseSfM(SfmConfig(
            num_features=opt.sfm_num_features, match_ratio=opt.sfm_match_ratio,
            max_epipolar_px=opt.sfm_max_epipolar_px, max_reproj_px=opt.sfm_max_reproj_px,
            min_tri_angle_deg=opt.sfm_min_tri_angle_deg, min_track=opt.sfm_min_track,
            z_near=opt.znear, z_far=opt.zfar,
        ))
        encoded = json.dumps(architecture_config(opt), sort_keys=True).encode()
        self.register_buffer("_spatial_architecture", torch.tensor(list(encoded), dtype=torch.uint8))
        self.last_sfm, self.last_spatial = [], {}

    @staticmethod
    def _is_backbone_parameter(name):
        return name.startswith(("scene_encoder.", "patch_embed.", "patch_plucker_embed."))

    def initialize_scene_backbone(self, source):
        """Strict, complete image/scene transfer. Never load official GS queries/head."""
        if self.opt.scene_backbone != "tokengs":
            raise ValueError("pretrained scene transfer requires scene_backbone='tokengs'")
        current = self.state_dict()
        mapping = {
            k: ("enc_dec_backbone." + k[len("scene_encoder."):] if k.startswith("scene_encoder.") else k)
            for k in current if self._is_backbone_parameter(k)
        }
        missing = [src for src in mapping.values() if src not in source]
        bad_shapes = [src for dst, src in mapping.items()
                      if src in source and current[dst].shape != source[src].shape]
        source_prefixes = ("patch_embed.", "patch_plucker_embed.", "enc_dec_backbone.encoder.",
                           "enc_dec_backbone.multiscale_norms.", "enc_dec_backbone.encoder_norm.",
                           "enc_dec_backbone.latent_feature_kv_proj.", "enc_dec_backbone.latent_feature_k_norm.",
                           "enc_dec_backbone.latent_blocks.")
        unexpected = [k for k in source if (k.startswith(source_prefixes) or k == "enc_dec_backbone.latents")
                      and k not in mapping.values()]
        if missing or bad_shapes or unexpected:
            raise ValueError(f"incompatible scene checkpoint: missing={missing[:4]}, "
                             f"shape={bad_shapes[:4]}, unexpected={unexpected[:4]}")
        if any(not torch.isfinite(source[src]).all() for src in mapping.values()):
            raise ValueError("non-finite pretrained scene weights")
        with torch.no_grad():
            for dst, src in mapping.items():
                current[dst].copy_(source[src])
        return sum(current[k].numel() for k in mapping)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "_spatial_architecture"
        if key in state_dict:
            # Schema-1 small-model checkpoints remain loadable after adding schema-2 fields.
            self._spatial_architecture = self._spatial_architecture.new_empty(state_dict[key].shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def configure_sft_trainable_parameters(self, accelerator=None):
        for name, p in self.named_parameters():
            upstream = self.opt.scene_backbone == "tokengs" and self._is_backbone_parameter(name)
            p.requires_grad_(not name.startswith("lpips_loss.") and not (upstream and self.opt.freeze_backbone))
            p._lr_scale = self.opt.backbone_lr_scale if upstream else 1.0
        if accelerator is None or accelerator.is_main_process:
            n = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"[GeometryGS] {self.opt.scene_backbone} scene path: "
                  f"{total / 1e6:.2f}M total, {n / 1e6:.2f}M trainable parameters")

    _sample_anchor_features = SfmTokenGS._sample_anchor_features

    @torch.no_grad()
    def compute_anchors(self, model_input):
        images = getattr(model_input, "sfm", None)
        cameras = model_input.encoder.cam_to_world_input
        records = []
        for b in range(cameras.shape[0]):
            dropped = self.training and torch.rand((), device=cameras.device) < self.opt.sfm_dropout
            if not self.opt.sfm_enabled or dropped or self.opt.sfm_num_anchors == 0:
                rec = {"points": cameras.new_empty(0, 3), "color": cameras.new_empty(0, 3),
                       "obs_mask": torch.empty(0, cameras.shape[1], dtype=torch.bool, device=cameras.device),
                       **{k: cameras.new_empty(0) for k in ("track_len", "angle_deg", "reproj_px")}}
                selected = torch.empty(0, dtype=torch.long, device=cameras.device)
            else:
                if images is None:
                    raise ValueError("SfM-enabled reconstruction requires input-view SfM images/intrinsics")
                rec = self.sfm.reconstruct(images["images"][b], cameras[b], images["intrinsics"][b])
                selected = select_anchors(rec, self.opt.sfm_num_anchors, **anchor_quality_kwargs(self.opt))
                if self.training:
                    selected = selected[torch.rand(len(selected), device=selected.device) >= self.opt.sfm_point_dropout]
            rec["anchor_index"] = selected
            records.append(rec)
        self.last_sfm = records
        return records

    def _memory(self, features, encoder, records):
        b = len(records)
        k = max(1, max(len(r["anchor_index"]) for r in records))
        xyz = features.new_zeros(b, k, 3)
        feat = features.new_zeros(b, k, features.shape[-1])
        quality = features.new_zeros(b, k, 3)
        valid = torch.zeros(b, k, dtype=torch.bool, device=features.device)
        for ib, rec in enumerate(records):
            sel = rec["anchor_index"]
            n = len(sel)
            if not n:
                continue
            xyz[ib, :n] = rec["points"][sel]
            valid[ib, :n] = True
            feat[ib, :n] = self._sample_anchor_features(
                features[ib], xyz[ib, :n], rec["obs_mask"][sel],
                encoder.cam_to_world_input[ib], encoder.intrinsics_input[ib])
            quality[ib, :n] = torch.stack([
                rec["track_len"][sel].float() / encoder.cam_to_world_input.shape[1],
                rec["angle_deg"][sel] / 90.0,
                rec["reproj_px"][sel] / self.opt.sfm_max_reproj_px,
            ], -1)
        memory = self.sfm_norm(self.sfm_position(fourier_features(xyz, self.opt.spatial_pe_freqs))
                               + self.sfm_feature(feat) + self.sfm_quality(quality))
        return memory, xyz, valid

    def encode_scene_backbone(self, encoder_input):
        """Return the original multiscale features and scene latents before SfM adapters."""
        if self.opt.scene_backbone != "tokengs":
            raise ValueError("this method requires the TokenGS scene backbone")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.opt.freeze_backbone):
            features = self._embed_encoder_input(encoder_input)
            return self.scene_encoder(features, checkpointed=self.training and self.opt.spatial_checkpoint)

    def forward_reconstruction(self, model_input):
        cp = self.training and self.opt.spatial_checkpoint
        if self.opt.scene_backbone == "tokengs":
            features, z = self.encode_scene_backbone(model_input.encoder)
        else:
            features = self._embed_encoder_input(model_input.encoder)
            for block in self.image_blocks:
                features = run_block(block, features, checkpointed=cp)
            features = self.image_norm(features)
        records = self.compute_anchors(model_input)
        memory, anchor_xyz, valid = self._memory(features, model_input.encoder, records)
        b = features.shape[0]
        if self.opt.scene_backbone == "tokengs":
            for block in self.scene_adapters:
                z = run_block(block, z, memory, valid, checkpointed=cp)
        else:
            z = self.scene_latents[None].expand(b, -1, -1)
            for block in self.scene_blocks:
                z = run_block(block, z, features, memory, valid, checkpointed=cp)
            z = self.scene_norm(z)
        queries = self.gs_tokens[None].expand(b, -1, -1)
        h = queries + self.initial_query(queries, z)
        centers = self.opt.free_center_init_extent * self.initial_center(h)
        k = memory.shape[1]
        anchored = torch.zeros(b, self.opt.num_gs_tokens, dtype=torch.bool, device=features.device)
        anchored[:, :k] = valid
        h = torch.cat([torch.where(valid[..., None], memory, h[:, :k]), h[:, k:]], 1)
        centers = torch.cat([torch.where(valid[..., None], anchor_xyz, centers[:, :k]), centers[:, k:]], 1)
        h = h + self.token_types(anchored.long())
        for block in self.decoder_blocks:
            h, centers, radii = run_block(block, h, centers, z, anchored, checkpointed=cp)
        gaussians = self.activation_head(h, centers, radii)
        self.last_spatial = {"centers": centers.detach(), "radii": radii.detach(), "anchored": anchored.detach()}
        return self._reconstruction_from_gaussians(gaussians)

    def forward(self, data, skip_loss=False):
        # Same per-batch view handling as the legacy SfM model, without legacy modules.
        base = self.opt
        v = int(data["images_input"].shape[1])
        self.opt = dataclasses.replace(base, num_input_views=v, num_views=int(data["images_all"].shape[1]))
        try:
            results = super().forward(data, skip_loss=skip_loss)
            if self.training and not skip_loss and self.opt.lambda_soft_chain:
                chain, count = soft_chain_loss(results["gaussians"], data, self.opt)
                results["loss"] = results["loss"] + self.opt.lambda_soft_chain * chain
                results["loss_soft_chain"] = chain.detach()
                results["soft_chain_edges"] = chain.new_tensor(count)
            return results
        finally:
            self.opt = base
