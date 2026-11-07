"""Decode a saved decoder_inputs.pt (written by evaluate_with_scene_latent_tuning.py --demo) back to Gaussians.

decoder_inputs.pt holds the two decoder inputs of one scene -- scene latents and Gaussian tokens -- with the
tuned one replaced by its tuned value, so the frozen decoder turns them into exactly the evaluated Gaussians
without re-running any tuning. This script rebuilds the model from the recorded preset + checkpoint
(sha256 checked), decodes, and writes <output-dir>/gaussians.pt, <name>.ply and <name>_slotcolor.ply
(same layout as the demo PLY files). If gaussians.pt sits next to decoder_inputs.pt, the max |diff| against it
is reported (FP32, same GPU type: expected 0 or float rounding).

    CUDA_VISIBLE_DEVICES=1 python scripts/decode_saved_decoder_inputs.py \
        --decoder-inputs <demo dir>/decoder_inputs.pt --output-dir /tmp/redecoded --name token_tuning
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from tokengs.evaluate import _copy_checkpoint_strict, _load_checkpoint_state_dict
from tokengs.models import model_registry
from tokengs.models.input_types import ModelInputDecoder
from tokengs.options import config_defaults

import demo_ply_export
import evaluation_statistics as es


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decoder-inputs", required=True, help="decoder_inputs.pt from a --demo run")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", default="gaussians", help="PLY file name stem")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None, help="override the recorded checkpoint path (sha256 still checked)")
    args = p.parse_args()

    src = Path(args.decoder_inputs).absolute()
    saved = torch.load(src, map_location="cpu", weights_only=False)
    out_dir = Path(args.output_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / "gaussians.pt", out_dir / f"{args.name}.ply", out_dir / f"{args.name}_slotcolor.ply"]
    if any(t.exists() for t in targets):
        print(f"[FATAL] refusing to overwrite {[str(t) for t in targets if t.exists()]}", file=sys.stderr)
        return 2

    resume = Path(args.resume or saved["checkpoint"]).absolute()
    sha = sha256_file(resume)
    if sha != saved["checkpoint_sha256"]:
        print(f"[FATAL] checkpoint sha256 {sha} != recorded {saved['checkpoint_sha256']}", file=sys.stderr)
        return 2
    opt = config_defaults[saved["preset"]].evolve(resume=str(resume))
    model = model_registry[opt.model_type](opt)
    _copy_checkpoint_strict(model, _load_checkpoint_state_dict(str(resume)))
    model = model.to(args.device).eval()

    with torch.no_grad():
        scene_latents = saved["scene_latents"].to(args.device)
        gs_tokens = saved["gs_tokens"].to(args.device)
        # the decoder only reads the decoder input for the target time embedding (off for static DL3DV models)
        if opt.time_embedding:
            raise RuntimeError("time-embedding models need the target time input, which is not saved")
        decoder_input = ModelInputDecoder(time_embedding_target=None, cam_view=None, intrinsics=None)
        gaussians = model.forward_decoder(model.encoder_latent_from_scene_latents(scene_latents), decoder_input,
                                          gs_tokens=gs_tokens).float()
        children = saved["children_per_slot"]
        _, per = es.needle_stats(gaussians[0], saved["cam_to_world_input"].to(args.device),
                                 saved["intrinsics_input"].to(args.device), tuple(saved["img_size"]),
                                 saved["znear"], detailed=False)

    torch.save(gaussians.cpu(), targets[0])
    demo_ply_export.export_gaussians_ply(targets[1], gaussians, children, per)
    demo_ply_export.export_gaussians_ply(targets[2], gaussians, children, per, color_by_slot=True)
    report = {"decoder_inputs": str(src), "scene_id": saved["scene_id"], "tuned_tensor": saved["tuned_tensor"],
              "checkpoint_sha256": sha, "device": args.device, "outputs": [str(t) for t in targets]}
    reference = src.with_name("gaussians.pt")
    if reference.exists():
        ref = torch.load(reference, map_location="cpu")
        report["max_abs_diff_vs_demo_gaussians"] = float((gaussians.cpu() - ref.float()).abs().max())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
