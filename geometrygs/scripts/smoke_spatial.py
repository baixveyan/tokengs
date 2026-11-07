"""One real-scene forward/backward, checkpoint reload and RGB-only inference smoke.

Use an explicitly idle CUDA_VISIBLE_DEVICES GPU. This is not a training experiment
or an evaluation of reconstruction quality. The selected val scene is a debug input.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from geometrygs import SpatialTokenGS
from geometrygs.data import SfmProvider, split_data_with_sfm
from geometrygs.options import config_defaults
from geometrygs.plyio import write_points_ply
from geometrygs.spatial_model import options_from_architecture
from tokengs.train import verify_idle_gpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--views", type=int, default=6)
    args = parser.parse_args()
    verify_idle_gpus()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite smoke artifacts: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    opt = config_defaults["spatial_sfm_chain" if args.full else "spatial_small_sfm"].evolve(
        num_workers=0, val_depth_metric=False, min_input_views=args.views, max_input_views=args.views,
        sfm_dropout=0., sfm_point_dropout=0., random_reflect=False, lambda_soft_chain=0.05,
    )
    if not args.full:
        opt = opt.evolve(enc_embed_dim=64, token_dim=64, enc_num_heads=4, enc_depth=1,
                         latent_cross_attn_depth=1, dec_depth=2, num_latents=32, num_gs_tokens=64,
                         sfm_num_anchors=16, dec_patch_size=2, child_hidden_dim=32,
                         sfm_image_size=(256, 448), sfm_num_features=512)
    provider = SfmProvider("scannetpp_val_scaled_0.3", opt, training=True)
    idx = next(i for i, p in enumerate(provider.dataset.sample_list) if Path(p).name == "7b6477cb95")
    sample = provider[idx]
    batch = torch.utils.data.default_collate([sample])
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    model = SpatialTokenGS(opt).cuda().train()
    loaded = 0
    if opt.resume:
        loaded = model.initialize_scene_backbone(load_file(opt.resume))
    model.configure_sft_trainable_parameters()
    from tokengs.train import setup_optimizer
    # The integration smoke uses the same per-module LR groups as the trainer.
    from accelerate import Accelerator
    optimizer = setup_optimizer(opt, model, Accelerator(), 0)
    start = time.monotonic()
    output = model(batch)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    report = {"full": args.full, "views": args.views, "gaussians": list(output["gaussians"].shape),
              "parameters": sum(p.numel() for p in model.parameters()),
              "pretrained_parameters": loaded,
              "loss": float(output["loss"]), "gradient_norm": float(grad_norm),
              "soft_chain_edges": float(output.get("soft_chain_edges", 0)),
              "anchors": int(model.last_spatial["anchored"].sum()),
              "step_seconds": time.monotonic()-start,
              "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
    del output, optimizer
    model.zero_grad(set_to_none=True)
    save_file(model.state_dict(), str(args.output / "model.safetensors"))
    model.eval()
    # Reconstruct without ANY supervision tensor; full camera/RGB/SfM-input path only.
    mi, _ = split_data_with_sfm(batch, opt)
    with torch.no_grad():
        expected = model.forward_reconstruction(mi).gaussians.cpu()
    rec = model.last_sfm[0]
    write_points_ply(args.output / "sfm_anchors.ply", rec["points"][rec["anchor_index"]].cpu().numpy(),
                     rec["color"][rec["anchor_index"]].cpu().numpy())
    del model
    state = load_file(str(args.output / "model.safetensors"))
    restored = SpatialTokenGS(options_from_architecture(state["_spatial_architecture"])).cuda().eval()
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        actual = restored.forward_reconstruction(mi).gaussians
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    torch.save({"gaussians": actual.cpu(), **{k: v.cpu() for k, v in restored.last_spatial.items()}},
               args.output / "gaussians.pt")
    report["checkpoint_reload_exact"] = True
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
