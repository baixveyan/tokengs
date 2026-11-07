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

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_

from tokengs.options import Options


class BaseActivationHead(nn.Module, ABC):
    """
    Base class for converting tokens to Gaussians.

    This is a minimal interface - subclasses can have completely different
    internal structures and initialization logic.
    """

    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt

        self.num_gaussians_per_token = self.opt.dec_patch_size**2

        scale_cap = self.opt.gaussian_scale_cap
        self.scale_shift = 1 - math.log(scale_cap)
        self.scale_cap = scale_cap

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        rays_os: Optional[torch.Tensor] = None,
        rays_ds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert tokens to Gaussians.

        Args:
            x: Input tokens [B, N, C] where N is number of tokens
            rays_os: Optional ray origins [B, V, 3, H, W] for depth prediction
            rays_ds: Optional ray directions [B, V, 3, H, W] for depth prediction

        Returns:
            Gaussians [B, N * num_gaussians_per_token, 14]
        """
        pass


class ClipActivationHead(BaseActivationHead):
    """
    Clip-based activation head that uses exponential activations with hard clipping.
    Uses a single deconv (linear) layer for all Gaussian components.
    Supports both xyz and depth prediction modes.
    """

    def __init__(self, opt: Options):
        super().__init__(opt)

        # Set output dimensions based on prediction mode
        self.output_dims = (
            3 + 1 + 3 + 4 + 3
        )  # x, y, z, opacity, scale, rotation, rgb

        # Single linear layer (deconv) for all components
        self.deconv = nn.Linear(
            self.opt.enc_embed_dim,
            self.output_dims * self.opt.dec_patch_size * self.opt.dec_patch_size,
            bias=True,
        )

        self._init_weights(self.deconv)
        # Optional nonlinear residual on the raw position channels (step 2a). Final layer is zero-initialised, so at
        # step 0 the head is bit-identical to the plain linear deconv.
        self.pos_mlp = None
        if getattr(self.opt, "readout_pos_mlp", False):
            P = self.opt.dec_patch_size * self.opt.dec_patch_size
            hidden = int(getattr(self.opt, "readout_pos_mlp_hidden", 1024))
            self.pos_mlp = nn.Sequential(
                nn.LayerNorm(self.opt.enc_embed_dim),
                nn.Linear(self.opt.enc_embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3 * P),
            )
            nn.init.zeros_(self.pos_mlp[-1].weight)
            nn.init.zeros_(self.pos_mlp[-1].bias)
        # Optional nonlinear residual on the 11 non-position raw channels (rgb, scale, rotation, opacity); zero-init final
        # layer, so at step 0 the head is unchanged.
        self.attr_mlp = None
        if getattr(self.opt, "readout_attr_mlp", False):
            P = self.opt.dec_patch_size * self.opt.dec_patch_size
            hidden = int(getattr(self.opt, "readout_pos_mlp_hidden", 1024))
            self.attr_mlp = nn.Sequential(
                nn.LayerNorm(self.opt.enc_embed_dim),
                nn.Linear(self.opt.enc_embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, self._attr_mlp_dims() * P),
            )
            nn.init.zeros_(self.attr_mlp[-1].weight)
            nn.init.zeros_(self.attr_mlp[-1].bias)
        # View-dependent colour: higher-order SH coefficients (degree >= 1) predicted from the token feature, appended
        # after the 14 standard channels. Zero-initialised, so at step 0 the rendered colour equals the RGB.
        self.sh_head = None
        self.sh_rest_dim = 0
        sh_degree = int(getattr(self.opt, "readout_sh_degree", 0))
        if sh_degree > 0:
            P = self.opt.dec_patch_size * self.opt.dec_patch_size
            self.sh_rest_dim = 3 * ((sh_degree + 1) ** 2 - 1)
            self.sh_head = nn.Linear(self.opt.enc_embed_dim, P * self.sh_rest_dim)
            nn.init.zeros_(self.sh_head.weight)
            nn.init.zeros_(self.sh_head.bias)
        # Step 2b: surfel constraint — children's offset along the token's local normal is soft-clamped to r_n
        # (annealed during training from surfel_rn_start to surfel_rn_end, stored as a buffer for evaluation).
        if getattr(self.opt, "readout_surfel", False):
            self.register_buffer("surfel_rn", torch.tensor(float(self.opt.surfel_rn_start)))
        if self.opt.clip_head_z_init is not None:
            bias_val = math.log(1.0 + self.opt.clip_head_z_init)
            with torch.no_grad():
                self.deconv.bias.view(
                    self.opt.dec_patch_size * self.opt.dec_patch_size,
                    self.output_dims,
                )[:, 2].fill_(bias_val)

        # Mark parameters to exclude from weight decay
        for param in self.parameters():
            param._no_weight_decay = True

    def _attr_mlp_dims(self) -> int:
        mode = getattr(self.opt, "readout_attr_mlp_channels", "all")
        if mode not in ("all", "opacity"):
            raise ValueError(f"readout_attr_mlp_channels must be 'all' or 'opacity', got {mode!r}")
        return self.output_dims - 3 if mode == "all" else 1

    def _init_weights(self, m):
        """Initialize deconv weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.opt.clip_head_readout_std)
            if m.bias is not None:
                m.bias.data.zero_()

    def pos_act(self, x: torch.Tensor) -> torch.Tensor:
        """Position activation without scaling factor (for xyz prediction)"""
        pos = torch.sign(x) * (torch.expm1(torch.abs(x)))  # inverse log transform
        return pos

    def scale_act(self, x: torch.Tensor) -> torch.Tensor:
        """Scale activation with exponential and hard clipping"""
        x = x - self.scale_shift
        return torch.minimum(
            torch.exp(x),
            torch.tensor(self.scale_cap, device=x.device, dtype=x.dtype),
        )

    def opacity_act(self, x: torch.Tensor) -> torch.Tensor:
        """Opacity activation with hard clipping"""
        return torch.sigmoid(x - self.opt.opacity_bias)

    def rot_act(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation normalization"""
        return F.normalize(x, dim=-1)

    def rgb_act(self, x: torch.Tensor) -> torch.Tensor:
        """RGB activation (always tanh for clip-based head)"""
        return 0.5 * torch.tanh(x) + 0.5

    def forward(
        self,
        x: torch.Tensor,
        rays_os: Optional[torch.Tensor] = None,
        rays_ds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert tokens to Gaussians using a single deconv layer.
        Supports both xyz and depth prediction modes.

        Args:
            x: Input tokens [B, N, C]
            rays_os: Ray origins [B, V, 3, H, W] (required for depth prediction)
            rays_ds: Ray directions [B, V, 3, H, W] (required for depth prediction)

        Returns:
            Gaussians [B, N * num_gaussians_per_token, 14]
        """
        B = x.shape[0]

        # Process through deconv
        x = x.permute(0, 2, 1).unsqueeze(-1)  # B, C, N, 1

        x = x.squeeze(-1).permute(0, 2, 1)  # B, N, C
        h = x
        x = self.deconv(x)  # [B, N, output_dims * P * P]
        if self.pos_mlp is not None or self.attr_mlp is not None:
            P = self.opt.dec_patch_size ** 2
            raw = x.view(B, x.shape[1], P, self.output_dims)
            pos_raw, attr_raw = raw[..., :3], raw[..., 3:]
            if self.pos_mlp is not None:
                pos_raw = pos_raw + self.pos_mlp(h).view(B, x.shape[1], P, 3)
            if self.attr_mlp is not None:
                d = self.attr_mlp(h).view(B, x.shape[1], P, self._attr_mlp_dims())
                if d.shape[-1] == 1:  # opacity only: raw channel order is rgb 3, scale 3, rotation 4, opacity 1
                    attr_raw = torch.cat([attr_raw[..., :-1], attr_raw[..., -1:] + d], dim=-1)
                else:
                    attr_raw = attr_raw + d
            x = torch.cat([pos_raw, attr_raw], dim=-1).reshape(B, x.shape[1], -1)
        x = rearrange(x, "b n (p c) -> b c (n p)", p=self.opt.dec_patch_size**2)

        x = x.reshape(B, self.output_dims, -1)  # B, output_dims, N * P * P
        x = x.permute(0, 2, 1).contiguous()  # B, N * P * P, output_dims

        # Apply activations based on prediction mode
        pos, rgb, scaling, rotation, opacity = x.split([3, 3, 3, 4, 1], dim=-1)
        pos = self.pos_act(pos)
        if hasattr(self, "surfel_rn"):
            P = self.opt.dec_patch_size ** 2
            pt = pos.view(B, -1, P, 3)
            anchor = pt.mean(2, keepdim=True)
            rel = pt - anchor
            cov = rel.detach().transpose(-1, -2) @ rel.detach() / P
            normal = torch.linalg.eigh(cov + 1e-9 * torch.eye(3, device=cov.device))[1][..., 0].unsqueeze(2)  # [B,N,1,3]
            d = (rel * normal).sum(-1, keepdim=True)
            rn = self.surfel_rn.to(pos.dtype)
            pos = (anchor + rel - d * normal + rn * torch.tanh(d / rn) * normal).reshape(B, -1, 3)
        

        opacity = self.opacity_act(opacity)
        scale = self.scale_act(scaling)
        rotation = self.rot_act(rotation)
        rgbs = self.rgb_act(rgb)

        gaussians = torch.cat(
            [pos, opacity, scale, rotation, rgbs], dim=-1
        )  # [B, N, 14]
        if self.sh_head is not None:
            # token order of the Gaussians is (n p), matching the deconv rearrange above
            sh_rest = rearrange(self.sh_head(h), "b n (p k) -> b (n p) k", p=self.opt.dec_patch_size ** 2)
            gaussians = torch.cat([gaussians, sh_rest], dim=-1)  # [B, N, 14 + 3*((d+1)^2-1)]

        return gaussians
