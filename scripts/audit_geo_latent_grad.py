"""Audit whether TokenGS geometry losses reach the TTT scene latents.

Read-only with respect to core code and checkpoints. For each requested scene it
checks, under FP32 and with all model weights frozen:

1. Existing outer path: `TokenGS.forward` with `use_ttt_for_eval=True` and
   `use_geo_losses=True` -- does the returned geometry loss carry any graph?
2. Inner path: scene_latents -> latent_kv_proj -> decoder -> activation head ->
   slot reshape -> `compute_geo_losses`; per-term gradient norm / finiteness on the
   latents, attribute-level gradient split, and cosine with the render-loss gradient.
3. Central finite differences along a fixed random direction vs autograd.
4. Short Adam descent (TTT optimizer, lr 1e-2) on each geometry term alone.

Example:
    CUDA_VISIBLE_DEVICES=7 python scripts/audit_geo_latent_grad.py \
        --resume checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors \
        --output /tmp/tokengs_geo_grad_audit --scene-ids <id> [<id> ...]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from accelerate import Accelerator

from tokengs.evaluate import _copy_checkpoint_strict, _load_checkpoint_state_dict
from tokengs.data import get_multi_dataloader
from tokengs.losses import geo_losses as geo_lib
from tokengs.losses.geo_losses import compute_geo_losses
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults
from tokengs.utils.training import freeze_model_parameters

TERMS = ("sw_loss", "packing_loss", "inter_loss")
ATTR_SLICES = {"center": (0, 3), "opacity": (3, 4), "scale": (4, 7), "rotation": (7, 11), "rgb": (11, 14)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slot_view(gaussians: torch.Tensor, num_children: int) -> torch.Tensor:
    b, n, a = gaussians.shape
    return gaussians.reshape(b, n // num_children, num_children, a)


def geo_terms(model, gaussians: torch.Tensor, lambdas: tuple[float, float, float]) -> dict:
    opt = model.opt
    slots = slot_view(gaussians, int(opt.dec_patch_size) ** 2)
    return compute_geo_losses(
        gs_centers=slots[..., 0:3],
        gs_scales=slots[..., 4:7],
        gs_opacities=slots[..., 3:4],
        gs_rotations=slots[..., 7:11],
        lambda_sw=lambdas[0],
        lambda_packing=lambdas[1],
        lambda_inter=lambdas[2],
        k_neighbors=opt.geo_k_neighbors,
        target_distance=opt.geo_target_distance,
        num_sw_directions=opt.geo_num_sw_directions,
    )


def decode(model, latents, decoder_input):
    encoder_latent = model.encoder_latent_from_scene_latents(latents)
    gs_tokens = model.get_gs_tokens(batch_size=latents.shape[0]).detach()
    return model.forward_decoder(encoder_latent, decoder_input, gs_tokens=gs_tokens)


def grad_stats(g: torch.Tensor | None) -> dict:
    if g is None:
        return {"is_none": True}
    return {
        "is_none": False,
        "norm": float(g.norm()),
        "finite": bool(torch.isfinite(g).all()),
        "nonzero_fraction": float((g != 0).float().mean()),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


# Inputs that geo_losses.py deliberately stop-gradients (opacity mass weights for
# every term; per-child scales inside the packing denominator). Freezing them at
# z0 makes the finite-difference target the same function autograd differentiates.
STOP_GRAD_COLUMNS = {"sw_loss": ("opacity",), "packing_loss": ("opacity", "scale"), "inter_loss": ("opacity",)}


def freeze_columns(gaussians: torch.Tensor, reference: torch.Tensor, attrs: tuple[str, ...]) -> torch.Tensor:
    out = gaussians.clone()
    for a in attrs:
        s, e = ATTR_SLICES[a]
        out[..., s:e] = reference[..., s:e]
    return out


def _inter_parents(model, gaussians: torch.Tensor):
    slots = slot_view(gaussians, int(model.opt.dec_patch_size) ** 2).float()
    weights, active = geo_lib._opacity_mass_weights(slots[..., 3:4])
    means, _, center_cov = geo_lib._weighted_slot_moments(slots[..., 0:3], weights)
    cov = geo_lib._aggregate_parent_covariance(center_cov, slots[..., 4:7], slots[..., 7:11], weights)
    return means, cov, active


def inter_neighbors(model, gaussians: torch.Tensor):
    """kNN graph exactly as `_inter_slot_loss` builds it (no_grad, discrete)."""
    means, _, active = _inter_parents(model, gaussians)
    n = means.shape[1]
    with torch.no_grad():
        dist = torch.cdist(means, means)
        valid = active.unsqueeze(-1) & active.unsqueeze(-2) & ~torch.eye(n, dtype=torch.bool, device=means.device)
        dist.masked_fill_(~valid, float("inf"))
        idx = dist.topk(min(model.opt.geo_k_neighbors, n - 1), dim=-1, largest=False).indices
        ok = torch.gather(torch.isfinite(dist), 2, idx)
    return idx, ok


def inter_fixed_graph(model, gaussians: torch.Tensor, idx: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """`_inter_slot_loss` with the neighbour graph held fixed."""
    means, cov, _ = _inter_parents(model, gaussians)
    b = torch.arange(means.shape[0], device=means.device).view(-1, 1, 1)
    delta = means.unsqueeze(2) - means[b, idx]
    pair_cov = 0.5 * (cov.unsqueeze(2) + cov[b, idx])
    solved = torch.cholesky_solve(delta.unsqueeze(-1), geo_lib._stable_cholesky(pair_cov)).squeeze(-1)
    dist = torch.sqrt((delta * solved).sum(dim=-1).clamp_min(geo_lib._EPS))
    return geo_lib._masked_mean((dist - model.opt.geo_target_distance).square(), ok)


def audit_outer_path(model, data) -> dict:
    """Current code: geometry loss computed on TTT output in TokenGS.forward."""
    with torch.enable_grad():
        results = model(data)
    geo = results.get("total_geo_loss")
    return {
        "gaussians_requires_grad": bool(results["gaussians"].requires_grad),
        "gaussians_grad_fn": None if results["gaussians"].grad_fn is None else type(results["gaussians"].grad_fn).__name__,
        "total_geo_loss_present": geo is not None,
        "total_geo_loss_requires_grad": None if geo is None else bool(geo.requires_grad),
        "total_geo_loss_value": None if geo is None else float(geo),
    }


def audit_inner_path(model, model_input_ttt, supervision_ttt, args) -> dict:
    out: dict = {}
    with torch.no_grad():
        z0 = model.forward_encoder_to_scene_latents(model_input_ttt.encoder)
    out["scene_latents_shape"] = list(z0.shape)
    out["scene_latents_dtype"] = str(z0.dtype)
    out["scene_latents_rms"] = float(z0.pow(2).mean().sqrt())

    with torch.enable_grad():
        z = z0.detach().clone().requires_grad_(True)
        gaussians = decode(model, z, model_input_ttt.decoder)
        out["gaussians_dtype"] = str(gaussians.dtype)
        out["num_gaussians"] = int(gaussians.shape[1])
        render = model.compute_loss(model._reconstruction_from_gaussians(gaussians), model_input_ttt.decoder, supervision_ttt)["loss"]
        unit = geo_terms(model, gaussians, (1.0, 1.0, 1.0))
        default = geo_terms(model, gaussians, (model.opt.lambda_geo_sw, model.opt.lambda_geo_packing, model.opt.lambda_geo_inter))

        losses = {"render_loss": render, **{t: unit[t] for t in TERMS}, "geo_total_default_lambda": default["total_geo_loss"]}
        out["values"] = {k: float(v) for k, v in losses.items()}
        out["geo_stats"] = {k: float(v) for k, v in unit["stats"].items()}
        out["loss_requires_grad"] = {k: bool(v.requires_grad) for k, v in losses.items()}

        grads = {}
        for name, loss in losses.items():
            grads[name] = torch.autograd.grad(loss, z, retain_graph=True, allow_unused=True)[0]
        out["latent_grad"] = {k: grad_stats(g) for k, g in grads.items()}
        g_render = grads["render_loss"]
        out["cosine_with_render_grad"] = {k: cosine(g, g_render) for k, g in grads.items() if k != "render_loss" and g is not None}
        out["grad_norm_ratio_vs_render"] = {
            k: float(g.norm() / g_render.norm()) for k, g in grads.items() if k != "render_loss" and g is not None
        }

        # Attribute-level split: which Gaussian parameters each term actually moves.
        attr = {}
        for name in ("render_loss", *TERMS):
            g = torch.autograd.grad(losses[name], gaussians, retain_graph=True, allow_unused=True)[0]
            attr[name] = {a: float(g[..., s:e].norm()) for a, (s, e) in ATTR_SLICES.items()}
        out["gaussian_attribute_grad_norm"] = attr

        scales = gaussians[..., 4:7].detach()
        out["scale_cap_saturated_fraction"] = float((scales >= model.opt.gaussian_scale_cap * (1 - 1e-6)).float().mean())
        opac = gaussians[..., 3].detach()
        out["opacity_quantiles"] = [float(q) for q in torch.quantile(opac.flatten()[::16], torch.tensor([0.01, 0.5, 0.99], device=opac.device))]

        params_with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
        params_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        out["model_params_with_grad"] = params_with_grad
        out["model_params_requires_grad"] = params_trainable
        del gaussians, render, unit, default, losses

    # Central finite differences. Latents have ~4M entries, so a unit direction
    # needs h large enough that h * |g.d| clears the FP32 resolution of the loss.
    gen = torch.Generator(device=z0.device).manual_seed(0)
    d_random = torch.randn(z0.shape, generator=gen, device=z0.device, dtype=z0.dtype)
    d_random = d_random / d_random.norm()
    fd = {}
    with torch.no_grad():
        g0 = decode(model, z0, model_input_ttt.decoder)
        for name in TERMS:
            fd[name] = {}
            directions = {
                "gradient": (grads[name] / grads[name].norm(), args.fd_steps_grad),
                "random": (d_random, args.fd_steps_random),
            }
            for dname, (d, steps) in directions.items():
                analytic = float((grads[name] * d).sum())
                for variant in ("raw", "stop_grad_inputs_frozen"):
                    rows = []
                    for h in steps:
                        vals = []
                        for sign in (1.0, -1.0):
                            g = decode(model, z0 + sign * h * d, model_input_ttt.decoder)
                            if variant != "raw":
                                g = freeze_columns(g, g0, STOP_GRAD_COLUMNS[name])
                            vals.append(float(geo_terms(model, g, (1.0, 1.0, 1.0))[name]))
                        numeric = (vals[0] - vals[1]) / (2 * h)
                        rows.append({
                            "h": h,
                            "per_element_rms_step": h / math.sqrt(z0.numel()),
                            "loss_delta": vals[0] - vals[1],
                            "numeric": numeric,
                            "rel_err": abs(numeric - analytic) / max(abs(analytic), 1e-12),
                        })
                    fd[name][f"{dname}/{variant}"] = {"analytic_directional": analytic, "central_differences": rows}

        # inter_loss with the kNN graph frozen at z0 (plus frozen opacity mass).
        idx0, ok0 = inter_neighbors(model, g0)
        fixed_at_z0 = float(inter_fixed_graph(model, g0, idx0, ok0))
        fd["inter_loss"]["fixed_graph_matches_module_at_z0"] = {
            "module": out["values"]["inter_loss"], "fixed_graph": fixed_at_z0,
        }
        for dname, (d, steps) in {
            "gradient": (grads["inter_loss"] / grads["inter_loss"].norm(), args.fd_steps_grad),
            "random": (d_random, args.fd_steps_random),
        }.items():
            analytic = float((grads["inter_loss"] * d).sum())
            rows = []
            for h in steps:
                vals, switched = [], []
                for sign in (1.0, -1.0):
                    g = freeze_columns(decode(model, z0 + sign * h * d, model_input_ttt.decoder), g0, ("opacity",))
                    idx, _ = inter_neighbors(model, g)
                    switched.append(float((idx != idx0).any(dim=-1).float().mean()))
                    vals.append(float(inter_fixed_graph(model, g, idx0, ok0)))
                numeric = (vals[0] - vals[1]) / (2 * h)
                rows.append({
                    "h": h,
                    "numeric": numeric,
                    "rel_err": abs(numeric - analytic) / max(abs(analytic), 1e-12),
                    "slots_with_neighbor_switch_fraction": switched,
                })
            fd["inter_loss"][f"{dname}/stop_grad_inputs_and_knn_frozen"] = {
                "analytic_directional": analytic, "central_differences": rows,
            }
        del g0
    out["finite_difference"] = fd
    out["stop_grad_columns"] = {k: list(v) for k, v in STOP_GRAD_COLUMNS.items()}
    del grads

    # Adam descent on each geometry term alone, TTT optimizer settings.
    descent = {}
    for name in TERMS:
        with torch.enable_grad():
            z = z0.detach().clone().requires_grad_(True)
            optim = torch.optim.Adam([z], lr=args.descent_lr)
            traj = []
            for _ in range(args.descent_steps):
                optim.zero_grad()
                loss = geo_terms(model, decode(model, z, model_input_ttt.decoder), (1.0, 1.0, 1.0))[name]
                traj.append(float(loss))
                loss.backward()
                optim.step()
            with torch.no_grad():
                g_final = decode(model, z, model_input_ttt.decoder)
                traj.append(float(geo_terms(model, g_final, (1.0, 1.0, 1.0))[name]))
                render_after = float(model.compute_loss(model._reconstruction_from_gaussians(g_final), model_input_ttt.decoder, supervision_ttt)["loss"])
        descent[name] = {
            "start": traj[0],
            "end": traj[-1],
            "relative_change": (traj[-1] - traj[0]) / max(abs(traj[0]), 1e-12),
            "monotone_fraction": sum(b <= a for a, b in zip(traj, traj[1:])) / (len(traj) - 1),
            "trajectory": traj,
            "render_loss_after": render_after,
            "latent_shift_rms": float((z.detach() - z0).pow(2).mean().sqrt()),
        }
    out["render_loss_before"] = out["values"]["render_loss"]
    out["geo_only_adam_descent"] = descent
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="eval_dl3dv_latent_6view_ssim")
    ap.add_argument("--resume", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scene-ids", nargs="+", required=True)
    ap.add_argument("--fd-steps-grad", type=float, nargs="+", default=[1.0, 0.3, 0.1])
    ap.add_argument("--fd-steps-random", type=float, nargs="+", default=[30.0, 10.0, 3.0])
    ap.add_argument("--descent-steps", type=int, default=20)
    ap.add_argument("--descent-lr", type=float, default=1e-2)
    args = ap.parse_args()

    out_dir = Path(args.output).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = config_defaults[args.preset]
    eval_json = REPO_ROOT / preset.dataset_kwargs["evaluation_json"]
    indices = json.load(open(eval_json, encoding="utf-8"))
    subset = {s: indices[s] for s in args.scene_ids}
    subset_json = out_dir / "evaluation_subset.json"
    subset_json.write_text(json.dumps(subset, indent=2))

    opt = preset.evolve(
        resume=str(Path(args.resume).absolute()),
        workspace=str(out_dir),
        use_ttt_for_eval=True,
        ttt_n_steps=1,  # outer-path check only needs a graph-free TTT output
        use_geo_losses=True,
        num_workers=0,
        eval_n_media_dumps=0,
        mixed_precision="no",
        use_input_supervision=False,
        dataset_kwargs={**preset.dataset_kwargs, "evaluation_json": str(subset_json)},
    )
    accelerator = Accelerator(mixed_precision="no")
    model = model_registry[opt.model_type](opt)
    _copy_checkpoint_strict(model, _load_checkpoint_state_dict(opt.resume))
    model = model.to(accelerator.device).eval()
    _, test_dl, _, _ = get_multi_dataloader(opt, accelerator)
    test_dl = accelerator.prepare(test_dl)

    report = {
        "preset": args.preset,
        "checkpoint": opt.resume,
        "mixed_precision": "no",
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "geo_defaults": {
            "lambda_sw": opt.lambda_geo_sw, "lambda_packing": opt.lambda_geo_packing, "lambda_inter": opt.lambda_geo_inter,
            "k_neighbors": opt.geo_k_neighbors, "target_distance": opt.geo_target_distance,
            "num_sw_directions": opt.geo_num_sw_directions, "children_per_slot": int(opt.dec_patch_size) ** 2,
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scenes": [],
    }
    for data in test_dl:
        scene_id = data["scene_id"][0]
        log(f"scene {scene_id[:12]}")
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        outer = audit_outer_path(model, data)
        model_input, _ = split_data(data, opt)
        model_input_ttt, supervision_ttt = model_input.to_ttt()
        with freeze_model_parameters(model):
            inner = audit_inner_path(model, model_input_ttt, supervision_ttt, args)
        row = {
            "scene_id": scene_id,
            "outer_forward_path": outer,
            "inner_ttt_path": inner,
            "seconds": time.perf_counter() - t0,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        }
        report["scenes"].append(row)
        lg = inner["latent_grad"]
        log("  outer geo requires_grad=%s | inner grad norms: %s" % (
            outer["total_geo_loss_requires_grad"],
            ", ".join(f"{k}={v.get('norm', float('nan')):.3e}" for k, v in lg.items())))
        for name, d in inner["geo_only_adam_descent"].items():
            log(f"  descent {name}: {d['start']:.4g} -> {d['end']:.4g} ({d['relative_change']:+.1%}), render {inner['render_loss_before']:.4f} -> {d['render_loss_after']:.4f}")
        (out_dir / "geo_grad_audit.json").write_text(json.dumps(report, indent=2))
    log(f"wrote {out_dir / 'geo_grad_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
