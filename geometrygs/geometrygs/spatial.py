"""Spatial building blocks; no renderer, SfM, or ground-truth dependencies."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from geometrygs.model import fourier_features


def run_block(module, *args, checkpointed=False):
    if checkpointed and torch.is_grad_enabled():
        return checkpoint(module, *args, use_reentrant=False)
    return module(*args)


def ball(x):
    """Smooth radial map into the open unit ball, with finite derivative at zero."""
    return x / torch.sqrt(1 + x.square().sum(-1, keepdim=True))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q_norm, self.kv_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, memory, valid=None):
        b, n, d = x.shape
        q = self.q(self.q_norm(x)).view(b, n, self.heads, -1).transpose(1, 2)
        k, v = self.kv(self.kv_norm(memory)).view(b, -1, 2, self.heads, d // self.heads).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        mask = None
        if valid is not None:
            # All-invalid scenes use a dummy key and then zero the update. All parameters
            # remain in the graph, including on DDP ranks whose scene has no SfM points.
            safe = valid.clone()
            safe[:, 0] |= ~valid.any(1)
            mask = safe[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = self.out(y.transpose(1, 2).reshape(b, n, d))
        return y if valid is None else y * valid.any(1)[:, None, None]


def ffn(dim, ratio):
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, int(dim * ratio)), nn.GELU(),
                         nn.Linear(int(dim * ratio), dim))


class ImageBlock(nn.Module):
    def __init__(self, dim, heads, ratio):
        super().__init__()
        self.attn, self.ffn = Attention(dim, heads), ffn(dim, ratio)

    def forward(self, x):
        x = x + self.attn(x, x)
        return x + self.ffn(x)


class SceneBlock(nn.Module):
    def __init__(self, dim, heads, ratio):
        super().__init__()
        self.image = Attention(dim, heads)
        self.geometry = Attention(dim, heads)
        self.self_attn = Attention(dim, heads)
        self.ffn = ffn(dim, ratio)

    def forward(self, z, images, sfm, valid):
        z = z + self.image(z, images)
        z = z + self.geometry(z, sfm, valid)
        z = z + self.self_attn(z, z)
        return z + self.ffn(z)


class SpatialAttention(nn.Module):
    """Full GS-token attention with a learned relative-3D/radius/type bias.

    Query chunks + checkpointing avoid retaining an N x N x hidden pair tensor.
    No nearest-neighbor restriction: initially misplaced free tokens can read any anchor.
    """

    def __init__(self, opt):
        super().__init__()
        d = opt.enc_embed_dim
        self.heads, self.chunk = opt.enc_num_heads, opt.spatial_query_chunk
        self.checkpointed = opt.spatial_checkpoint
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        # Relative displacement (3), log radius ratio (1), query/key anchor flags (2).
        self.bias = nn.Sequential(nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, self.heads))

    def _attend(self, q, k, v, cq, centers, rq, radii, tq, types):
        delta = (centers[:, None] - cq[:, :, None]) / (rq[:, :, None] + radii[:, None]).clamp_min(1e-4)
        delta = torch.sign(delta) * torch.log1p(delta.abs())
        ratio = torch.log(radii[:, None] / rq[:, :, None])
        shape = ratio.shape
        pair = torch.cat([delta, ratio, tq[:, :, None, None].expand(shape),
                          types[:, None, :, None].expand(shape)], -1)
        bias = self.bias(pair).permute(0, 3, 1, 2)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    def forward(self, h, centers, radii, anchored):
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm(h)).view(b, n, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        types = anchored.to(h.dtype)
        pieces = []
        for start in range(0, n, self.chunk):
            end = start + self.chunk
            args = (q[:, :, start:end], k, v, centers[:, start:end], centers,
                    radii[:, start:end], radii, types[:, start:end], types)
            pieces.append(run_block(self._attend, *args, checkpointed=self.checkpointed and self.training))
        return self.out(torch.cat(pieces, dim=2).transpose(1, 2).reshape(b, n, d))


class SpatialDecoderBlock(nn.Module):
    def __init__(self, opt, layer):
        super().__init__()
        d = opt.enc_embed_dim
        self.freqs = opt.spatial_pe_freqs
        self.position = nn.Sequential(nn.Linear(3 + 6 * self.freqs + 2, d), nn.GELU(), nn.Linear(d, d))
        self.scene = Attention(d, opt.enc_num_heads)
        self.spatial = SpatialAttention(opt)
        self.ffn = ffn(d, opt.mlp_ratio)
        self.update = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))
        self.step = opt.center_update_max / (1 + layer)
        self.radius = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.bounds = (opt.anchor_radius_min, opt.anchor_radius_max, opt.free_radius_min, opt.free_radius_max)

    def radii(self, h, anchored):
        amin, amax, fmin, fmax = self.bounds
        lo = torch.where(anchored[..., None], amin, fmin)
        hi = torch.where(anchored[..., None], amax, fmax)
        return lo + (hi - lo) * torch.sigmoid(self.radius(h))

    def forward(self, h, centers, z, anchored):
        radii = self.radii(h, anchored)
        pos = self.position(torch.cat([fourier_features(centers, self.freqs),
                                       radii.log(), anchored[..., None].to(h.dtype)], -1))
        h = h + self.scene(h + pos, z)
        h = h + self.spatial(h + pos, centers, radii, anchored)
        h = h + self.ffn(h)
        delta = self.step * ball(self.update(h))
        centers = centers + torch.where(anchored[..., None], torch.zeros_like(delta), delta)
        return h, centers, self.radii(h, anchored)


class LocalGaussianHead(nn.Module):
    """Shared child MLP with explicit local positions and radius-dependent scale caps."""

    def __init__(self, opt):
        super().__init__()
        d, hidden = opt.enc_embed_dim, opt.child_hidden_dim
        self.num_children = opt.dec_patch_size ** 2
        self.chunk, self.checkpointed = opt.child_token_chunk, opt.spatial_checkpoint
        self.child = nn.Parameter(torch.randn(self.num_children, hidden) * 0.02)
        # Linear([h,e]) = W_h h + W_e e; project h before broadcasting to save memory.
        self.token = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden))
        self.shared = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.position = nn.Linear(hidden, 3)
        self.scale = nn.Linear(hidden, 3)
        self.rotation = nn.Linear(hidden, 4)
        self.opacity = nn.Linear(hidden, 1)
        self.color = nn.Linear(hidden, 3)
        self.scale_floor = opt.child_scale_min
        self.scale_ratio = opt.child_scale_radius_ratio
        self.scale_cap = opt.gaussian_scale_cap
        self.opacity_bias = opt.opacity_bias
        nn.init.normal_(self.rotation.weight, std=0.01)
        nn.init.zeros_(self.rotation.bias)
        with torch.no_grad():
            self.rotation.bias[0] = 1

    def _decode(self, h, centers, radii):
        x = self.shared(self.token(h)[..., None, :] + self.child)
        positions = centers[..., None, :] + radii[..., None, :] * ball(self.position(x))
        cap = (radii[..., None, :] * self.scale_ratio).clamp_max(self.scale_cap)
        scales = self.scale_floor + (cap - self.scale_floor) * torch.sigmoid(self.scale(x))
        rotations = F.normalize(self.rotation(x), dim=-1, eps=1e-8)
        opacity = torch.sigmoid(self.opacity(x) - self.opacity_bias)
        colors = torch.sigmoid(self.color(x))
        return torch.cat([positions, opacity, scales, rotations, colors], -1)

    def forward(self, h, centers, radii):
        chunks = []
        for i in range(0, h.shape[1], self.chunk):
            chunks.append(run_block(self._decode, h[:, i:i+self.chunk], centers[:, i:i+self.chunk],
                                    radii[:, i:i+self.chunk], checkpointed=self.checkpointed and self.training))
        return torch.cat(chunks, 1).flatten(1, 2)
