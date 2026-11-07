"""Build the validation GT-depth cache used for per-epoch depth AbsRel during training.

Runs Open3D raycasting in THIS process only (never inside the training process, whose forked DataLoader
workers would deadlock afterwards). One entry per validation window: key "<scene_id>:<target frame ids>",
value {"depth": float16 [V_target, H, W] (normalized units), "valid": bool [V_target, H, W]}.

    python scripts/cache_scannetpp_val_depth.py --preset sft_scannetpp_readout_rgb_depth --out <path>.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
from pathlib import Path

import torch

from tokengs.data.provider import Provider
from tokengs.options import config_defaults


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    base = config_defaults[args.preset]
    opt = dataclasses.replace(
        base, val_depth_metric=True, val_depth_cache_build=True, use_input_supervision=False,
        num_workers=0, evaluating=False,
    )
    provider = Provider(opt.validation_data_mode, opt, training=False)
    cache: dict[str, dict[str, torch.Tensor]] = {}
    t0 = time.time()
    for idx in range(len(provider)):
        sample = provider[idx]
        key = sample["val_depth_cache_key"]
        cache[key] = {
            "depth": sample["depths_gt_sup"].to(torch.float16).contiguous(),
            "valid": sample["valid_gt_sup"].bool().contiguous(),
        }
        if (idx + 1) % 10 == 0 or idx + 1 == len(provider):
            print(f"[val-depth-cache] {idx + 1}/{len(provider)} windows, {time.time() - t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    torch.save(cache, tmp)
    os.replace(tmp, args.out)
    print(f"wrote {args.out} ({len(cache)} entries)")


if __name__ == "__main__":
    main()
