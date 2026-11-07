"""Data plumbing: variable input-view count, higher-resolution SfM images, and split_data that carries them."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

import tokengs.data as tokengs_data
import tokengs.data.provider as provider_module
import tokengs.models.tokengs as tokengs_model_module
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data as _split_data
from tokengs.utils.data import ImageTransform

_original_random_reflect = provider_module.random_reflect
_last_reflect = {"reflected": False}


def _recording_random_reflect(rgbs, c2ws, generator=None):
    out_rgbs, out_c2ws = _original_random_reflect(rgbs, c2ws, generator=generator)
    _last_reflect["reflected"] = out_rgbs is not rgbs
    return out_rgbs, out_c2ws


def split_data_with_sfm(batch: dict, opt):
    """split_data with the input-view count read from the batch, and SfM images attached to the model input."""
    num_input = int(batch["images_input"].shape[1])
    if opt.num_input_views != num_input:
        opt = dataclasses.replace(opt, num_input_views=num_input, num_views=int(batch["images_all"].shape[1]))
    model_input, supervision = _split_data(batch, opt)
    if "images_sfm" in batch:
        model_input.sfm = {"images": batch["images_sfm"], "intrinsics": batch["intrinsics_sfm"]}
    return model_input, supervision


class SfmProvider(Provider):
    """Training: samples min..max input views per scene from a random window. Every item additionally carries the
    input views at `sfm_image_size` (same centre crop, same reflection) with matching intrinsics."""

    def __init__(self, dataset_name, opt, training: bool = True, num_repeat: int = 1):
        super().__init__(dataset_name, opt, training=training, num_repeat=num_repeat)
        self._base_views = (int(opt.num_input_views), int(opt.num_views))
        size = tuple(getattr(opt, "sfm_image_size", (0, 0)))
        self.sfm_transform = ImageTransform(crop_size=size, sample_size=size, max_crop=True) if (
            all(size) and getattr(opt, "sfm_enabled", True)
        ) else None

    def _get_indices_static(self, idx):
        if not getattr(self.opt, "disjoint_training_views", False):
            return super()._get_indices_static(idx)
        rng = self.get_rng(idx)
        total = self.dataset.count_frames(idx)
        nv, ni = self.opt.num_views, self.opt.num_input_views
        if total < nv:
            raise ValueError(f"scene has {total} frames; need {nv} distinct input/target frames")
        gap = max(nv - 1, min(total - 1, int(rng.integers(self.min_gap, self.max_gap + 1))))
        start = int(rng.integers(0, total-gap))
        context = np.sort(np.concatenate(([start, start+gap],
                          rng.permutation(np.arange(start+1, start+gap))[:ni-2])))
        candidates = np.setdiff1d(np.arange(start, start+gap+1), context)
        target = rng.permutation(candidates)[:nv-ni]
        return np.concatenate([context, target]), []

    def get_item(self, idx, **kwargs):
        if not (self.training and hasattr(self.opt, "min_input_views")):
            return super().get_item(idx, **kwargs)
        base_in, base_all = self._base_views
        v = int(self.rng.integers(self.opt.min_input_views, self.opt.max_input_views + 1))
        self.opt.num_input_views, self.opt.num_views = v, v + (base_all - base_in)
        try:
            return super().get_item(idx, **kwargs)
        finally:
            self.opt.num_input_views, self.opt.num_views = base_in, base_all

    def _preprocess(self, rgbs, masks, depths, c2ws, intrinsics, timesteps, has_mask, raw_lidar_points=None, scene_id=None):
        v = self.opt.num_input_views
        sfm_images = shift = scale = None
        if self.sfm_transform is not None:
            sfm_images, shift, scale, _ = self.sfm_transform.preprocess_images(rgbs[:v].clone())
            raw_intr = intrinsics[:v]
        _last_reflect["reflected"] = False
        out = super()._preprocess(rgbs, masks, depths, c2ws, intrinsics, timesteps, has_mask,
                                  raw_lidar_points=raw_lidar_points, scene_id=scene_id)
        if sfm_images is not None:
            if _last_reflect["reflected"]:
                sfm_images = sfm_images.flip(-1)
            out["images_sfm"] = sfm_images.float().contiguous()
            out["intrinsics_sfm"] = torch.stack([
                raw_intr[..., 0] * scale[0], raw_intr[..., 1] * scale[1],
                (raw_intr[..., 2] + shift[0]) * scale[0], (raw_intr[..., 3] + shift[1]) * scale[1],
            ], dim=-1).float()
        return out


provider_module.random_reflect = _recording_random_reflect
tokengs_data.Provider = SfmProvider
tokengs_model_module.split_data = split_data_with_sfm
