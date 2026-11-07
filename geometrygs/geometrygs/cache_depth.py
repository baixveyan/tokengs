"""Build validation maps in a separate CPU process using the actual saved training options."""

import argparse
import dataclasses
import os
from pathlib import Path

import torch
import tyro

import geometrygs  # noqa: F401
from geometrygs.options import SfmOptions, SpatialOptions
from tokengs.data.provider import Provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = args.config.read_text()
    cls = SpatialOptions if "!dataclass:SpatialOptions" in text else SfmOptions
    base = tyro.extras.from_yaml(cls, text)
    opt = dataclasses.replace(base, val_depth_metric=True, val_depth_cache_build=True,
                              use_input_supervision=False, num_workers=0, evaluating=False,
                              dataset_kwargs=base.validation_dataset_kwargs)
    provider = Provider(opt.validation_data_mode, opt, training=False)
    cache = {}
    for i in range(len(provider)):
        sample = provider[i]
        cache[sample["val_depth_cache_key"]] = {
            "depth": sample["depths_gt_sup"].half().contiguous(), "valid": sample["valid_gt_sup"].bool(),
        }
        if (i + 1) % 10 == 0:
            print(f"[depth-cache] {i+1}/{len(provider)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    torch.save(cache, tmp)
    os.replace(tmp, args.out)


if __name__ == "__main__":
    main()
