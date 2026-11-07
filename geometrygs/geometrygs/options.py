"""Options for the SfM-anchored model and its training presets."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any
from typing import Literal

import tyro

from tokengs.options import Options, config_defaults as tokengs_defaults


@dataclass
class SfmOptions(Options):
    # --- SfM on the input views (computed on the GPU inside the forward pass)
    sfm_image_size: tuple[int, int] = (1024, 1792)
    sfm_num_features: int = 8192
    sfm_match_ratio: float = 0.9
    sfm_max_epipolar_px: float = 2.0
    sfm_max_reproj_px: float = 2.0
    sfm_min_tri_angle_deg: float = 1.5
    sfm_min_track: int = 2
    # anchor reliability filter (see geometrygs.sfm.anchor_quality_mask); 0.03 normalized = 10 cm
    anchor_strong_track: int = 3
    anchor_strong_angle_deg: float = 3.0
    anchor_knn: int = 5
    anchor_max_knn_dist: float = 0.03
    # --- anchored tokens: the first `sfm_num_anchors` GS-token slots are replaced by tokens fixed at selected SfM points
    sfm_num_anchors: int = 1024
    anchor_offset_max: float = 0.045  # bound of child offsets around the anchor in normalized units (0.3 / m -> 15 cm)
    anchor_pe_freqs: int = 8
    anchor_image_feature: bool = True  # add encoder patch features sampled at the anchor's observations
    anchor_lr_scale: float = 5.0  # LR multiplier of the newly initialised anchor modules
    # --- number of input views sampled per training scene (evaluation keeps num_input_views)
    min_input_views: int = 4
    max_input_views: int = 8

    def validate(self) -> None:
        super().validate()
        if not 2 <= self.min_input_views <= self.max_input_views:
            raise ValueError("need 2 <= min_input_views <= max_input_views")
        if self.sfm_num_anchors < 0 or self.sfm_num_anchors > self.num_gs_tokens:
            raise ValueError("sfm_num_anchors must be within [0, num_gs_tokens]")
        if self.anchor_offset_max <= 0:
            raise ValueError("anchor_offset_max must be positive")
        for big, small in zip(self.sfm_image_size, self.img_size):
            if big % small:
                raise ValueError("sfm_image_size must be an integer multiple of img_size")

    def evolve(self, **changes: Any) -> "SfmOptions":
        values = {f.name: copy.deepcopy(getattr(self, f.name)) for f in dataclasses.fields(self)}
        for key, value in changes.items():
            if key not in values:
                raise AttributeError(f"SfmOptions has no attribute '{key}'")
            values[key] = value
        return type(self)(**values)


def from_tokengs(base: Options, **changes: Any) -> SfmOptions:
    values = {f.name: copy.deepcopy(getattr(base, f.name)) for f in dataclasses.fields(base)}
    values.update(changes)
    return SfmOptions(**values)


config_defaults: dict[str, SfmOptions] = {}
config_doc: dict[str, str] = {}

config_doc["sfm_anchor_k1024"] = (
    "SfM-anchored TokenGS: official 6-view latent checkpoint post-trained on ScanNet++ with the strongest SFT recipe "
    "(scene-latent tail + readout chain, RGB + rendered GT depth + token 3D surface loss, position MLP, 10 epochs), "
    "4-8 random input views per training scene, 1024 of the 4096 GS tokens fixed at FPS-selected SfM points."
)
config_defaults["sfm_anchor_k1024"] = from_tokengs(
    tokengs_defaults["sft_scannetpp_latenttail_readout_rgb_depth_surface3d_posmlp"],
    model_type="tokengs_sfm",
    experiment_name="sfm_anchor_k1024",
)

config_doc["eval_sfm_anchor"] = "Evaluation options for SfM-anchored checkpoints (6 inputs, 50 ScanNet++ val scenes)."
config_defaults["eval_sfm_anchor"] = from_tokengs(
    tokengs_defaults["eval_scannetpp_latent_6view_ssim"], model_type="tokengs_sfm"
)

@dataclass
class SpatialOptions(SfmOptions):
    """Spatial architecture. Lengths below use normalized coordinates (metres * 0.3)."""

    scene_backbone: Literal["native", "tokengs"] = "native"
    sfm_fusion_depth: int = 2
    backbone_lr_scale: float = 0.1
    freeze_backbone: bool = False
    sfm_enabled: bool = True
    sfm_dropout: float = 0.1
    sfm_point_dropout: float = 0.2
    spatial_pe_freqs: int = 6
    spatial_query_chunk: int = 128
    spatial_checkpoint: bool = True
    child_hidden_dim: int = 128
    child_token_chunk: int = 128
    anchor_radius_min: float = 0.003  # 1 cm
    anchor_radius_max: float = 0.024  # 8 cm
    free_radius_min: float = 0.009    # 3 cm
    free_radius_max: float = 0.15     # 50 cm
    child_scale_min: float = 0.0001
    child_scale_radius_ratio: float = 0.3333333333
    free_center_init_extent: float = 0.6
    center_update_max: float = 0.15
    scene_unit_scale: float = 0.3
    disjoint_training_views: bool = True
    lambda_soft_chain: float = 0.0
    chain_neighbors: int = 8
    chain_radius_m: float = 0.2
    chain_match_m: float = 0.15
    chain_plane_tolerance_m: float = 0.01
    chain_normal_angle_deg: float = 10.0
    chain_huber_m: float = 0.02

    def validate(self):
        super().validate()
        if self.enc_embed_dim != self.token_dim or self.enc_embed_dim % self.enc_num_heads:
            raise ValueError("spatial model needs token_dim=enc_embed_dim divisible by enc_num_heads")
        if min(self.enc_depth, self.latent_cross_attn_depth, self.dec_depth, self.num_latents, self.num_gs_tokens) <= 0:
            raise ValueError("spatial model needs positive depths and token counts")
        if not self.use_latent_bottleneck or self.time_embedding or self.num_dynamic_gs_tokens:
            raise ValueError("spatial model supports static scenes and a scene bottleneck")
        if self.scene_backbone not in ("native", "tokengs"):
            raise ValueError("unknown scene backbone")
        if self.scene_backbone == "native" and self.use_multiscale_encoder:
            raise ValueError("native spatial backbone is single-scale")
        if self.scene_backbone == "native" and self.freeze_backbone:
            raise ValueError("freeze_backbone is for the separate TokenGS image/scene path")
        if self.use_multiscale_encoder:
            layers = self.multiscale_encoder_layers
            if not layers or tuple(sorted(set(layers))) != tuple(layers) or min(layers) < 0 or max(layers) >= self.enc_depth:
                raise ValueError("multiscale layers must be sorted unique encoder layer indices")
        if self.sfm_fusion_depth < 1 or self.backbone_lr_scale <= 0:
            raise ValueError("fusion depth and backbone LR scale must be positive")
        if self.sft_unfreeze_latent_tail or self.sft_unfreeze_readout:
            raise ValueError("legacy SFT freeze flags are invalid; use freeze_backbone explicitly")
        if self.lambda_surface3d or self.use_geo_losses or self.readout_pos_mlp or self.readout_attr_mlp or self.readout_surfel or self.readout_sh_degree:
            raise ValueError("legacy heads/surface/compactness losses are not supported by the spatial architecture")
        if self.load_lidar_pc or self.use_ttt_for_eval or self.camera_scale_method != "constant":
            raise ValueError("spatial model uses input-only SfM, no lidar input or scene optimization, fixed scene scale")
        if self.mean_of_grads != "none" or self.deferred_bp:
            raise ValueError("spatial training requires ordinary backpropagation")
        if not (0 <= self.sfm_dropout <= 1 and 0 <= self.sfm_point_dropout <= 1):
            raise ValueError("SfM dropout probabilities must be in [0, 1]")
        if min(self.spatial_query_chunk, self.child_token_chunk, self.child_hidden_dim, self.chain_neighbors) < 1:
            raise ValueError("chunk sizes, hidden width and neighbor count must be positive")
        if self.spatial_pe_freqs < 0:
            raise ValueError("spatial_pe_freqs must be nonnegative")
        if not (0 < self.anchor_radius_min <= self.anchor_radius_max and 0 < self.free_radius_min <= self.free_radius_max):
            raise ValueError("invalid local radius bounds")
        if not 0 < self.child_scale_min < min(self.anchor_radius_min, self.free_radius_min) * self.child_scale_radius_ratio:
            raise ValueError("scale floor must be below the smallest radius-dependent scale cap")
        if min(self.scene_unit_scale, self.center_update_max, self.free_center_init_extent,
               self.chain_radius_m, self.chain_match_m, self.chain_plane_tolerance_m, self.chain_huber_m) <= 0:
            raise ValueError("spatial distances must be positive")
        if self.lambda_soft_chain < 0 or not 0 < self.chain_normal_angle_deg < 90:
            raise ValueError("invalid soft-chain weight or normal threshold")
        if self.lambda_soft_chain and not self.evaluating and not self.use_render_geo_supervision:
            raise ValueError("soft chain needs training-only laser depth/normal maps")
        if self.batch_size != 1 and self.min_input_views != self.max_input_views:
            raise ValueError("variable view counts require batch_size=1 (no ragged-view collation)")


_scratch = dataclasses.asdict(config_defaults["sfm_anchor_k1024"])
_scratch.update(
    model_type="geometrygs_spatial", experiment_name="spatial_sfm", project_name="GeometryGS",
    resume=None, enc_embed_dim=512, token_dim=512, enc_num_heads=8, enc_depth=6,
    latent_cross_attn_depth=6, dec_depth=4, num_latents=4096, num_gs_tokens=4096,
    dec_patch_size=8, use_multiscale_encoder=False, use_latent_bottleneck=True,
    sft_unfreeze_latent_tail=False, sft_unfreeze_readout=False, readout_pos_mlp=False,
    lambda_surface3d=0.0, use_geo_losses=False, lambda_render_thickness=0.0,
    lambda_render_normal=0.0, lambda_render_alpha=0.0, use_input_supervision=True,
    num_epochs=100, lr=2e-4, pct_start_steps=1000, mixed_precision="no",
    render_geo_warmup_epochs=1.0,
)
config_defaults["spatial_small_sfm"] = SpatialOptions(**_scratch).evolve(experiment_name="spatial_small_sfm")
from geometrygs import TOKENGS_ROOT

config_defaults["spatial_sfm"] = config_defaults["spatial_small_sfm"].evolve(
    experiment_name="spatial_sfm", scene_backbone="tokengs",
    enc_embed_dim=1024, token_dim=1024, enc_num_heads=16, enc_depth=12,
    latent_cross_attn_depth=12, use_multiscale_encoder=True,
    multiscale_encoder_layers=(5, 7, 9, 11), child_hidden_dim=1024,
    resume=str(TOKENGS_ROOT / "checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors"),
)
config_defaults["spatial_free"] = config_defaults["spatial_sfm"].evolve(
    experiment_name="spatial_free", sfm_enabled=False,
)
config_defaults["spatial_sfm_chain"] = config_defaults["spatial_sfm"].evolve(
    experiment_name="spatial_sfm_chain", lambda_soft_chain=0.05,
)
for _name in ("spatial_sfm", "spatial_free", "spatial_sfm_chain"):
    config_doc[_name] = "Full-width spatial model with strict pretrained TokenGS image/scene initialization."
config_doc["spatial_small_sfm"] = "Optional 75M random-initialized capacity ablation, not the main model."

AllSfmConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
