"""TokenGS release image/scene path, without its Gaussian decoder or readout.

Uses the original blocks and normalization order. Module names below intentionally
match enc_dec_backbone.* in the official checkpoint for strict upstream-only loading.
"""

import torch
from torch import nn

from geometrygs.spatial import run_block
from tokengs.models.attention import Block
from tokengs.models.enc_dec import DecoderBlock


class TokenGSSceneBackbone(nn.Module):
    def __init__(self, opt):
        super().__init__()
        d = opt.enc_embed_dim
        self.heads = opt.enc_num_heads
        self.multiscale_layers = opt.multiscale_encoder_layers if opt.use_multiscale_encoder else ()
        self.encoder = nn.Sequential(*[
            Block(d, self.heads, opt.mlp_ratio, qkv_bias=True, proj_bias=True, ffn_bias=True,
                  init_values=0.01, qk_norm=True, rope=None, flex_attn_block_mask=None)
            for _ in range(opt.enc_depth)
        ])
        if self.multiscale_layers:
            self.multiscale_norms = nn.ModuleList([nn.LayerNorm(d) for _ in self.multiscale_layers])
            self.feature_dim = d * len(self.multiscale_layers)
        else:
            self.encoder_norm = nn.LayerNorm(d)
            self.feature_dim = d
        self.latents = nn.Parameter(torch.randn(opt.num_latents, d) * 0.02)
        self.latents._no_weight_decay = True
        self.latent_feature_kv_proj = nn.Linear(self.feature_dim, 2 * d)
        self.latent_feature_k_norm = nn.LayerNorm(d // self.heads)
        self.latent_blocks = nn.ModuleList([
            DecoderBlock(d, self.heads, opt.mlp_ratio, qkv_bias=True, ffn_bias=True,
                         qk_norm=True, init_values=0.01)
            for _ in range(opt.latent_cross_attn_depth)
        ])

    def forward(self, x, checkpointed=False):
        features = []
        for i, block in enumerate(self.encoder):
            x = run_block(block, x, checkpointed=checkpointed)
            if i in self.multiscale_layers:
                features.append(self.multiscale_norms[len(features)](x))
        features = torch.cat(features, -1) if self.multiscale_layers else self.encoder_norm(x)
        b, n, _ = features.shape
        kv = self.latent_feature_kv_proj(features).view(b, n, 2, self.heads, -1)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        k = self.latent_feature_k_norm(k)
        z = self.latents[None].expand(b, -1, -1)
        for block in self.latent_blocks:
            z = run_block(block, z, k, v, checkpointed=checkpointed)
        return features, z


class SfmSceneAdapter(nn.Module):
    """Identity-initialized residual, preserving the pretrained scene representation."""

    def __init__(self, opt):
        super().__init__()
        from geometrygs.spatial import Attention, ffn
        d = opt.enc_embed_dim
        self.geometry = Attention(d, opt.enc_num_heads)
        self.ffn = ffn(d, opt.mlp_ratio)
        self.geometry_scale = nn.Parameter(torch.zeros(d))
        self.ffn_scale = nn.Parameter(torch.zeros(d))

    def forward(self, z, memory, valid):
        z = z + self.geometry(z, memory, valid) * self.geometry_scale
        return z + self.ffn(z) * self.ffn_scale * valid.any(1)[:, None, None]
