"""Per-frame DL3DV benchmark evaluation for TokenGS checkpoints, with or without scene latent tuning.

Scene latent tuning = what the official code calls TTT with ttt_mode="scene-latent-tuning":
all network weights and Gaussian tokens frozen, only the per-scene latent bottleneck output is
optimized with Adam on the input views (50 steps, lr 1e-2 in the latent preset).

Token tuning = the paper's TT (official ttt_mode="token-tuning", model.forward_ttt_tokens): all
network weights and the scene latents frozen, only the Gaussian tokens are optimized with Adam on the
input views (paper: 50 steps, lr 1e-4). `--token-tuning` needs an explicit `--tuning-lr` because the
latent preset's ttt_lr is the scene latent value.

Scene latent and token tuning (`--scene-latent-and-token-tuning`, no official counterpart): both tensors
optimized together, one Adam with each tensor's own learning rate (scene latents: preset ttt_lr or
`--tuning-lr`; Gaussian tokens: the explicit `--token-lr`); reimplemented loop only
(scripts/reimplemented_scene_latent_and_token_tuning.py).

Reuses the official `tokengs.evaluate` checkpoint loading and the official
dataloader, but records per-frame PSNR/SSIM/LPIPS so that both the official
`all_frames` metric (mean over all target views) and `novel_only` (target views
that are not context views) can be reported. Core code is not modified.

Also records per frame the FFRGS sharpness metrics (edge L1, HF ratio) and the
frame distance to the nearest context view; per scene the slot geometry
(FFRGS formulation) and needle scalars of the final Gaussians.

`--tuning-loop reimplemented` runs a line-by-line copy of the official tuning loop
(scripts/reimplemented_scene_latent_tuning.py, scripts/reimplemented_token_tuning.py) so that
`--geo-sw/--geo-packing/--geo-inter` can add the geometry term inside the 50 steps and `--demo` can
record the per-step curve and save the tuned tensor.
`--demo` additionally writes detailed needle tables, unpruned PLY files (slot_id field) and
decoder_inputs.pt (scene latents + Gaussian tokens that fully determine the final Gaussians; no tuning or
reimplemented loop only; scripts/decode_saved_decoder_inputs.py decodes it again); with
`--ply-name NAME` (one scene) every demo file goes straight into --output, the PLY files as
NAME{,_slotcolor}.ply plus cameras.ply and a NAME.json sidecar, so that several methods can be compared
side by side (EXP_TokenGS/demo/scene_<id[:12]>/<method>/, scripts/compare_same_scene_ply.py).

Example (no tuning, FP32, single GPU):
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_with_scene_latent_tuning.py \
        --preset eval_dl3dv_latent_6view_ssim \
        --resume checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors \
        --output EXP_TokenGS/no_tuning --no-tuning
"""

import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch
from accelerate import Accelerator
from scipy import stats

# Importing tokengs.evaluate applies the official torch.manual_seed(0) and warning filters.
from tokengs.evaluate import _copy_checkpoint_strict, _copy_matching_checkpoint, _load_checkpoint_state_dict
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults
from tokengs.utils import MetricsCalculator

import demo_ply_export
import evaluation_statistics as es
import reimplemented_scene_latent_and_token_tuning
import reimplemented_scene_latent_tuning
import reimplemented_token_tuning

METRICS = ("psnr", "ssim", "lpips")
SHARPNESS = ("edge_l1", "hf_ratio")
FRAME_METRICS = METRICS + SHARPNESS
# hf_ratio: 1 is ideal (pred gradient energy == GT); "better" means closer to 1.
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "edge_l1": False, "hf_ratio": None}
GEO_KEYS = ("sw_loss", "packing_loss", "inter_loss", "packing_ratio", "packing_target", "effective_children",
            "inter_distance", "inter_distance_median", "inter_too_close", "inter_too_far", "active_slot_ratio")
NEEDLE_KEYS = ("alive_ratio", "opacity_mean", "aniso_p50_weighted", "aniso_p50_unweighted",
               "cos_long_axis_ray_p50_weighted", "long_axis_along_ray_gt0.9_weighted",
               "outside_all_input_frustums_weighted")
SURFACE_KEYS = ("feat_pixel_ratio", "d4+_feat_psnr", "d4+_smooth_psnr",
                "warp_psnr_dt5", "mv_cycle_err_dt5", "ray_thickness_med")
SECTIONS = ("all_frames", "novel_only")
SECTION_PREFIX = {"all_frames": "all", "novel_only": "novel"}
JOINT = "scene_latents_and_gs_tokens"
TUNES_SCENE_LATENTS = ("scene_latents", JOINT)
TUNES_GS_TOKENS = ("gs_tokens", JOINT)

# Official release_2026.6.md table (DL3DV-10K benchmark, 140 scenes, all target frames);
# the table labels these rows "w/o TTT" and "w/ TTT".
OFFICIAL_RELEASE = {
    "eval_dl3dv_latent_6view_ssim": {
        "no_tuning": {"psnr": 25.120, "ssim": 0.806, "lpips": 0.257},
        "scene_latent_tuning": {"psnr": 27.304, "ssim": 0.866, "lpips": 0.185},
    },
}

REQUIRED_OUTPUTS = (
    "summary.json",
    "scene_metrics.jsonl",
    "frame_metrics.jsonl",
    "scene_metrics.csv",
    "launch_contract.json",
    "status.json",
    "resource_evidence.json",
)


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


class RunLog:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:  # noqa: BLE001 - evidence collection must not abort the run
        return f"<error: {exc!r}>"


def query_gpu(index: int) -> dict:
    fields = "index,uuid,memory.used,memory.free,utilization.gpu,power.draw"
    out = run_cmd(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", str(index)]).strip()
    idx, uuid, used, free, util, power = [x.strip() for x in out.split(",")]
    apps_out = run_cmd([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory,process_name", "--format=csv,noheader,nounits",
    ])
    apps = []
    for line in apps_out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 4 and parts[1] == uuid:
            apps.append({"pid": int(parts[0]), "gpu_uuid": parts[1], "used_memory_mib": float(parts[2]), "name": parts[3]})
    return {
        "at": now_iso(),
        "index": int(idx),
        "uuid": uuid,
        "memory_used_mib": float(used),
        "memory_free_mib": float(free),
        "utilization_percent": float(util),
        "power_draw_w": float(power),
        "compute_apps": apps,
    }


class GpuMonitor(threading.Thread):
    def __init__(self, index: int, interval_ms: int = 200):
        super().__init__(daemon=True)
        self.index = index
        self.interval_ms = interval_ms
        self.samples: list[dict] = []
        self.error: str | None = None
        self.proc: subprocess.Popen | None = None

    def run(self) -> None:
        try:
            self.proc = subprocess.Popen(
                [
                    "nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits", "-i", str(self.index), "-lms", str(self.interval_ms),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for line in self.proc.stdout:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 4:
                    continue
                try:
                    self.samples.append({
                        "timestamp": parts[0],
                        "memory_used_mib": float(parts[1]),
                        "utilization_percent": float(parts[2]),
                        "power_draw_w": float(parts[3]),
                    })
                except ValueError:
                    continue
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.join(timeout=5)


def metric_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else float("nan")
    t_crit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    half = t_crit * std / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n_scenes": n,
        "mean": mean,
        "sample_std": std,
        "confidence_level": 0.95,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
        "t_critical": t_crit,
    }


def _better(d: np.ndarray, a: np.ndarray, b: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (this_run_better, reference_better) masks."""
    if HIGHER_IS_BETTER[metric] is None:
        da, db = np.abs(a - 1.0), np.abs(b - 1.0)
        return da < db, da > db
    if HIGHER_IS_BETTER[metric]:
        return d > 0, d < 0
    return d < 0, d > 0


def rankings(rows: list[dict], key: str, metric: str) -> dict:
    ordered = sorted(rows, key=lambda r: r[key], reverse=HIGHER_IS_BETTER[metric])
    pick = lambda rs: [{"scene_id": r["scene_id"], "value": r[key]} for r in rs]  # noqa: E731
    return {"top3": pick(ordered[:3]), "bottom3": pick(ordered[::-1][:3])}


def paired_comparison(rows: list[dict], ref_rows: list[dict], label: str, ref_dir: str) -> dict:
    ref = {r["scene_id"]: r for r in ref_rows}
    ours = {r["scene_id"]: r for r in rows}
    if not set(ours) <= set(ref):
        return {
            "label": label,
            "reference_dir": ref_dir,
            "error": "this run has scenes missing from the reference",
            "only_in_this_run": sorted(set(ours) - set(ref)),
        }
    out = {
        "label": label,
        "reference_dir": ref_dir,
        "delta_definition": "this_run - reference",
        "n_paired_scenes": len(ours),
        "reference_restricted_to_this_run_scenes": set(ours) != set(ref),
    }
    ids = sorted(ours)

    def compare(key: str, metric: str | None) -> dict | None:
        if not all(key in ours[s] and key in ref[s] for s in ids):
            return None
        a = np.array([ours[s][key] for s in ids], dtype=np.float64)
        b = np.array([ref[s][key] for s in ids], dtype=np.float64)
        d = a - b
        d_stats = metric_stats(d.tolist())
        res = {
            "this_run_mean": float(a.mean()),
            "reference_mean": float(b.mean()),
            "delta_mean": float(d.mean()),
            "delta_median": float(np.median(d)),
            "paired_delta_std": d_stats["sample_std"],
            "paired_delta_ci95": [d_stats["ci95_lower"], d_stats["ci95_upper"]],
        }
        if metric is not None:
            better, worse = _better(d, a, b, metric)
            res.update({
                "scenes_this_run_better": int(better.sum()),
                "scenes_reference_better": int(worse.sum()),
                "scenes_tied": int(len(ids) - better.sum() - worse.sum()),
            })
        return res

    for section in SECTIONS:
        p = SECTION_PREFIX[section]
        out[section] = {}
        for m in FRAME_METRICS:
            res = compare(f"{p}_{m}", m)
            if res is not None:
                out[section][m] = res
    out["geometry"] = {k: r for k in GEO_KEYS if (r := compare(f"geo_{k}", None)) is not None}
    out["needle"] = {k: r for k in NEEDLE_KEYS if (r := compare(f"needle_{k}", None)) is not None}
    out["paired_psnr_deltas_by_scene"] = (
        {s: ours[s]["all_psnr"] - ref[s]["all_psnr"] for s in ids}
    )
    return out


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="eval_dl3dv_latent_6view_ssim", choices=sorted(config_defaults))
    p.add_argument("--resume", required=True, help="checkpoint path (.safetensors)")
    p.add_argument("--output", required=True, help="output directory (must not already contain results)")
    tuning = p.add_mutually_exclusive_group(required=True)
    tuning.add_argument("--scene-latent-tuning", dest="tuning_target", action="store_const", const="scene_latents",
                        help="tune scene latents per scene before rendering (official ttt_mode='scene-latent-tuning')")
    tuning.add_argument("--token-tuning", dest="tuning_target", action="store_const", const="gs_tokens",
                        help="tune the Gaussian tokens per scene before rendering (paper TT, official "
                             "ttt_mode='token-tuning'); requires --tuning-lr")
    tuning.add_argument("--scene-latent-and-token-tuning", dest="tuning_target", action="store_const", const=JOINT,
                        help="tune scene latents and Gaussian tokens together (no official counterpart); requires "
                             "--tuning-loop reimplemented and --token-lr")
    tuning.add_argument("--no-tuning", dest="tuning_target", action="store_const", const=None,
                        help="render the feed-forward prediction as is (official use_ttt_for_eval=False)")
    p.add_argument("--tuning-steps", type=int, default=None, help="override preset ttt_n_steps")
    p.add_argument("--tuning-lr", type=float, default=None, help="override preset ttt_lr")
    p.add_argument("--token-lr", type=float, default=None,
                   help="Gaussian token learning rate of --scene-latent-and-token-tuning (the scene latents use "
                        "--tuning-lr / preset ttt_lr)")
    p.add_argument("--mixed-precision", default="no", choices=["no", "bf16", "fp16"])
    p.add_argument("--evaluation-json", default=None, help="override preset evaluation_json")
    p.add_argument("--scene-ids", nargs="*", default=None, help="restrict to these scene ids (smoke tests)")
    p.add_argument("--expected-scenes", type=int, default=None, help="fail unless this many scenes are evaluated")
    p.add_argument("--compare", action="append", default=[], metavar="LABEL=DIR",
                   help="paired per-scene comparison against another run directory")
    p.add_argument("--max-idle-memory-mib", type=float, default=100.0,
                   help="refuse to start if the GPU already uses more memory than this")
    p.add_argument("--tuning-loop", default="official", choices=["official", "reimplemented"],
                   help="official: model.forward_ttt; reimplemented: line-by-line copy that accepts a geometry loss, "
                        "records the demo curve and keeps the tuned tensor")
    p.add_argument("--geo-sw", type=float, default=0.0, help="inner-loop SW weight (reimplemented loop only)")
    p.add_argument("--geo-packing", type=float, default=0.0, help="inner-loop packing weight (reimplemented loop only)")
    p.add_argument("--geo-inter", type=float, default=0.0, help="inner-loop inter weight (reimplemented loop only)")
    p.add_argument("--vis-mode", default="official", choices=["official", "vis_xyz", "vis_opac"],
                   help="inner-loop frustum visibility mode (reimplemented joint loop only)")
    p.add_argument("--core-gate-tau", type=float, default=0.0,
                   help="if > 0, attenuate opacity of intra-token children beyond tau * r_core (reimplemented joint loop)")
    p.add_argument("--depth-prior", default="none", choices=["none", "colmap", "colmap_raw", "colmap_clean", "vggt_omega"],
                   help="inner-loop rendered depth prior source (colmap defaults to denoised colmap_clean)")
    p.add_argument("--depth-prior-weight", type=float, default=0.0,
                   help="Smooth-L1 weight for --depth-prior rendered depth loss (reimplemented joint loop only)")
    p.add_argument("--demo", action="store_true",
                   help="write tuning curve, tuned tensor, needle tables, unpruned PLY and render strips per scene "
                        "(use with --scene-ids)")
    p.add_argument("--ply-name", default=None,
                   help="demo of one scene only: write every demo file straight into --output, the PLY files as "
                        "<ply-name>{,_slotcolor}.ply + cameras.ply + <ply-name>.json (instead of demo/<scene id>/)")
    args = p.parse_args()
    args.tuning = args.tuning_target is not None
    geo = (args.geo_sw, args.geo_packing, args.geo_inter)
    if min(geo) < 0:
        p.error("geometry weights must be non-negative")
    if args.tuning_loop == "reimplemented" and not args.tuning:
        p.error("--tuning-loop reimplemented requires --scene-latent-tuning, --token-tuning or "
                "--scene-latent-and-token-tuning")
    if args.tuning_target == JOINT and (args.tuning_loop != "reimplemented" or args.token_lr is None):
        p.error("--scene-latent-and-token-tuning requires --tuning-loop reimplemented and an explicit --token-lr")
    if args.tuning_target != JOINT and args.token_lr is not None:
        p.error("--token-lr only applies to --scene-latent-and-token-tuning (token tuning alone uses --tuning-lr)")
    if args.tuning_target == "gs_tokens" and args.tuning_lr is None:
        p.error("--token-tuning requires an explicit --tuning-lr (the preset ttt_lr is the scene latent value)")
    if max(geo) > 0 and args.tuning_loop != "reimplemented":
        p.error("--geo-* weights require --tuning-loop reimplemented")
    if args.core_gate_tau < 0 or args.depth_prior_weight < 0:
        p.error("--core-gate-tau and --depth-prior-weight must be non-negative")
    if (args.depth_prior != "none") != (args.depth_prior_weight > 0):
        p.error("--depth-prior != none and --depth-prior-weight > 0 must be specified together")
    if (args.vis_mode != "official" or args.core_gate_tau > 0 or args.depth_prior != "none") and (
        args.tuning_target != JOINT or args.tuning_loop != "reimplemented"
    ):
        p.error("--vis-mode != official, --core-gate-tau > 0 and --depth-prior != none require "
                "--scene-latent-and-token-tuning and --tuning-loop reimplemented")
    if args.ply_name is not None:
        if not args.demo or not args.scene_ids or len(args.scene_ids) != 1:
            p.error("--ply-name requires --demo and exactly one --scene-ids")
        if not args.ply_name.replace("_", "").replace("-", "").replace(".", "").isalnum():
            p.error(f"--ply-name must only use letters, digits, '_', '-', '.': {args.ply_name!r}")
    return args


def ply_paths(args: argparse.Namespace, demo_dir: Path) -> dict[str, Path]:
    """Where the demo PLY files of one scene go (named after --ply-name if given)."""
    stem = args.ply_name or "gaussians"
    return {"ply": demo_dir / f"{stem}.ply", "ply_slotcolor": demo_dir / f"{stem}_slotcolor.ply",
            "cameras_ply": demo_dir / "cameras.ply"}


def main() -> int:
    script_t0 = time.perf_counter()
    args = parse_args()
    out_dir = Path(args.output).absolute()
    # console.log: the launcher's complete stdout/stderr log of this run (scripts/one_click_launch_common.sh)
    if out_dir.exists() and any(p.name != "console.log" for p in out_dir.iterdir()):
        print(f"[FATAL] output directory {out_dir} is not empty; refusing to overwrite", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir / "run.log")
    started_at = now_iso()
    status_path = out_dir / "status.json"

    def set_status(status: str, **extra) -> dict:
        obj = {"status": status, "updated_at": now_iso(), **extra}
        write_json(status_path, obj)
        return obj

    set_status("starting", started_at=started_at)

    # ---- GPU binding & idle gate
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or not cvd.strip().isdigit():
        set_status("failed", error=f"CUDA_VISIBLE_DEVICES must be a single physical index, got {cvd!r}")
        log(f"[FATAL] CUDA_VISIBLE_DEVICES must be a single physical index, got {cvd!r}")
        return 2
    gpu_index = int(cvd)
    gpu_before = query_gpu(gpu_index)
    if gpu_before["compute_apps"] or gpu_before["memory_used_mib"] > args.max_idle_memory_mib:
        set_status("failed", error="GPU not idle", gpu_before=gpu_before)
        log(f"[FATAL] GPU {gpu_index} is not idle: {gpu_before}")
        return 3
    log(f"GPU {gpu_index} idle check passed: {gpu_before['memory_used_mib']} MiB used, no compute apps")

    monitor = GpuMonitor(gpu_index)
    monitor.start()

    try:
        return _run(args, out_dir, log, set_status, started_at, gpu_index, gpu_before, monitor, script_t0)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"[FATAL] {exc!r}\n{tb}")
        set_status("failed", started_at=started_at, error=repr(exc), traceback=tb)
        monitor.stop()
        _write_resource_evidence(out_dir, gpu_before, None, monitor, gpu_index, status="failed")
        return 1


def _write_resource_evidence(out_dir, gpu_before, pid_binding, monitor, gpu_index, status, torch_peaks=None):
    samples = monitor.samples
    evidence = {
        "gpu_before": gpu_before,
        "pid_binding_after_launch": pid_binding,
        "gpu_monitor_interval_seconds": monitor.interval_ms / 1000.0,
        "gpu_monitor_error": monitor.error,
        "gpu_sample_count": len(samples),
        "gpu_peak_memory_used_mib": max((s["memory_used_mib"] for s in samples), default=None),
        "gpu_peak_utilization_percent": max((s["utilization_percent"] for s in samples), default=None),
        "gpu_after": query_gpu(gpu_index),
        "torch_peak_allocated_mib": (torch_peaks or {}).get("allocated"),
        "torch_peak_reserved_mib": (torch_peaks or {}).get("reserved"),
        "status": status,
        "gpu_samples": samples,
    }
    write_json(out_dir / "resource_evidence.json", evidence)


def _decoder_inputs(model, data, opt, args, curve_state, gaussians) -> dict | None:
    """The two decoder inputs (scene latents, Gaussian tokens) that fully determine this scene's final Gaussians.

    Untuned side recomputed with the official encoder / token getters, tuned side taken from the reimplemented loop;
    None for the official tuning loop (it does not expose the tuned tensor). Decoding them again must reproduce the
    evaluated Gaussians; the max |diff| is returned as "redecode_max_abs_diff".
    """
    tuned = curve_state.get("tuned_tensor")
    if args.tuning and tuned is None:
        return None
    model_input, _ = split_data(data, opt)
    scene_latents = model.forward_encoder_to_scene_latents(model_input.encoder)
    gs_tokens = model.get_gs_tokens(batch_size=model_input.batch_size)
    if args.tuning_target == "scene_latents":
        scene_latents = tuned
    elif args.tuning_target == "gs_tokens":
        gs_tokens = tuned
    elif args.tuning_target == JOINT:
        scene_latents, gs_tokens = tuned["scene_latents"], tuned["gs_tokens"]
    redecoded = model.forward_decoder(model.encoder_latent_from_scene_latents(scene_latents), model_input.decoder,
                                      gs_tokens=gs_tokens)
    if args.core_gate_tau > 0:
        redecoded = es.apply_core_opacity_gate(redecoded, int(opt.dec_patch_size) ** 2, args.core_gate_tau)
    return {"scene_latents": scene_latents.detach(), "gs_tokens": gs_tokens.detach(),
            "redecode_max_abs_diff": float((redecoded.float() - gaussians).abs().max())}


def _write_demo(out_dir, scene_id, args, gaussians, needle, needle_per, children, data, pred, gt,
                target_ids, is_ctx, frame_distance, curve_state, geo, per, row, decoder_inputs, opt, ckpt_sha) -> dict:
    """Demo artefacts for one scene: unpruned PLY (true colour / slot colour), cameras, needle table,
    tuning curve (reimplemented loop only), decoder inputs (decoder_inputs.pt) and a GT-vs-render strip of
    the four farthest novel frames. With --ply-name everything goes straight into out_dir, plus a
    <ply-name>.json sidecar next to the PLY files."""
    from PIL import Image

    demo_dir = out_dir if args.ply_name is not None else out_dir / "demo" / scene_id
    demo_dir.mkdir(parents=True, exist_ok=True)
    media = {}
    torch.save(gaussians.cpu(), demo_dir / "gaussians.pt")
    media["gaussians_pt"] = "gaussians.pt"
    redecode = None
    if decoder_inputs is not None:
        redecode = {"max_abs_diff_vs_evaluated_gaussians": decoder_inputs["redecode_max_abs_diff"]}
        torch.save({
            "scene_latents": decoder_inputs["scene_latents"].cpu(),
            "gs_tokens": decoder_inputs["gs_tokens"].cpu(),
            "tuned_tensor": args.tuning_target,
            "scene_id": scene_id,
            "preset": args.preset,
            "checkpoint": str(Path(args.resume).absolute()),
            "checkpoint_sha256": ckpt_sha,
            "tuning_loop": args.tuning_loop if args.tuning else None,
            "tuning_steps": opt.ttt_n_steps if args.tuning else None,
            "tuning_lr": opt.ttt_lr if args.tuning else None,
            "token_lr": args.token_lr,
            "tuning_loop_geo_weights": {"lambda_sw": args.geo_sw, "lambda_packing": args.geo_packing,
                                        "lambda_inter": args.geo_inter},
            "vis_mode": args.vis_mode,
            "core_gate_tau": args.core_gate_tau,
            "depth_prior": args.depth_prior,
            "depth_prior_weight": args.depth_prior_weight,
            "children_per_slot": children,
            "img_size": [int(x) for x in opt.img_size],
            "znear": opt.znear,
            "cam_to_world_input": data["cam_to_world_input"][0].float().cpu(),
            "intrinsics_input": data["intrinsics_input"][0].float().cpu(),
            "redecode_check": redecode,
        }, demo_dir / "decoder_inputs.pt")
        media["decoder_inputs"] = "decoder_inputs.pt"
    paths = ply_paths(args, demo_dir)
    demo_ply_export.export_gaussians_ply(paths["ply"], gaussians, children, needle_per)
    demo_ply_export.export_gaussians_ply(paths["ply_slotcolor"], gaussians, children, needle_per, color_by_slot=True)
    target_c2w = torch.linalg.inv(data["cam_view"][0].float().transpose(-1, -2))
    demo_ply_export.export_camera_ply(paths["cameras_ply"], data["cam_to_world_input"][0].float(), target_c2w)
    zip_candidate = REPO_ROOT / "data" / "dl3dv_eval" / f"{scene_id}.zip"
    if zip_candidate.is_file():
        v_in = int(opt.num_input_views)
        ctx_ids = [int(x) for x in data["frame_ids"][0][:v_in].tolist()]
        colmap_files = demo_ply_export.export_colmap_points_ply(
            demo_dir, zip_candidate, scene_id, ctx_ids, data["intrinsics_input"], opt.img_size, 0.15, float(opt.znear)
        )
        media.update(colmap_files)
    write_json(demo_dir / "needle_stats.json", {
        "scene_id": scene_id,
        "definitions": {
            "alive": "opacity > 1/255; all percentiles/fractions opacity-weighted over alive children unless noted",
            "aniso": "max scale / min scale",
            "cos_long_axis_ray": "|cos| between the longest axis and the ray from the nearest input camera centre",
            "parallax_deg": "largest angle between the rays from the 6 input cameras to the child centre",
            "outside_all_input_frustums": "centre not inside any input view (z > znear and inside the image)",
        },
        "needle": needle,
        "geometry": geo,
    })
    media["needle_stats"] = "needle_stats.json"

    rows = curve_state.get("rows") if curve_state.get("enabled") else None
    if rows:
        with open(demo_dir / "tuning_curve.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        media["tuning_curve"] = "tuning_curve.jsonl"
        media["tuning_curve_png"] = _plot_curve(rows, demo_dir / "tuning_curve.png", scene_id,
                                                args.tuning_target.replace("_", " ").replace("gs ", "Gaussian "))

    novel_pos = [p for p in range(len(target_ids)) if not is_ctx[p]]
    pick = sorted(sorted(novel_pos, key=lambda p: -frame_distance[p])[:4])
    strip_rows = [torch.cat([x[0, p].clamp(0, 1).float().cpu() for p in pick], dim=-1) for x in (gt, pred)]
    img = (torch.cat(strip_rows, dim=-2).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(img).save(demo_dir / "render_strip.png")
    media["render_strip"] = "render_strip.png"
    write_json(demo_dir / "render_strip.json", {
        "layout": "top row GT, bottom row render",
        "frames": [{"target_position": p, "frame_id": target_ids[p], "nearest_context_distance": frame_distance[p],
                    "psnr": float(per["psnr"][p]), "hf_ratio": float(per["hf_ratio"][p])} for p in pick],
    })
    rel = demo_dir.relative_to(out_dir)
    media = {k: str(rel / v) for k, v in media.items()}
    media.update({k: str(rel / v.name) for k, v in paths.items()})
    if args.ply_name is not None:
        # file names only: the sidecar lives in the same folder as everything it refers to
        sidecar = paths["ply"].with_suffix(".json")
        write_json(sidecar, {
            "method": args.ply_name,
            "scene_id": scene_id,
            "files": {"ply": paths["ply"].name, "ply_slotcolor": paths["ply_slotcolor"].name,
                      "cameras_ply": paths["cameras_ply"].name, "gaussians_pt": "gaussians.pt"},
            "decoder_inputs": "decoder_inputs.pt" if decoder_inputs is not None else None,
            "redecode_check": redecode,
            "tuning_target": args.tuning_target,
            "tuning_loop": args.tuning_loop if args.tuning else None,
            "tuning_steps": opt.ttt_n_steps if args.tuning else None,
            "tuning_lr": opt.ttt_lr if args.tuning else None,
            "token_lr": args.token_lr,
            "tuning_loop_geo_weights": {"lambda_sw": args.geo_sw, "lambda_packing": args.geo_packing,
                                        "lambda_inter": args.geo_inter},
            "checkpoint_sha256": ckpt_sha,
            "children_per_slot": children,
            "metrics": {k: v for k, v in row.items() if k.startswith(("all_", "novel_", "bucket_", "geo_", "needle_"))},
        })
        media["ply_sidecar"] = sidecar.name
    return media


def _plot_curve(rows: list[dict], path: Path, scene_id: str, tuned: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    panels = [
        ("context loss (render)", [("context_loss", "render loss"), ("total_loss", "total (render+geo)")]),
        ("target PSNR", [("target_all_psnr", "all 50"), ("target_novel_psnr", "novel"), ("target_d4plus_psnr", "d>=4")]),
        ("target HF ratio (1 = GT)", [("target_all_hf_ratio", "all"), ("target_d4plus_hf_ratio", "d>=4")]),
        ("inter Mahalanobis distance (target 2)", [("geo_inter_distance", "mean"), ("geo_inter_distance_median", "median")]),
        ("inter too close / too far", [("geo_inter_too_close", "too close <1.5"), ("geo_inter_too_far", "too far >2.5")]),
        ("packing ratio vs target / SW", [("geo_packing_ratio", "packing ratio"), ("geo_packing_target", "target"),
                                         ("geo_sw_loss", "SW loss")]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, series) in zip(axes.flat, panels):
        for key, label in series:
            if all(key in r for r in rows):
                ax.plot(steps, [r[key] for r in rows], label=label)
        ax.set_title(title)
        ax.set_xlabel(f"{tuned} tuning step")
        if title.startswith("packing"):
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{tuned} tuning curve {scene_id[:16]}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def _run(args, out_dir, log, set_status, started_at, gpu_index, gpu_before, monitor, script_t0) -> int:
    resume = Path(args.resume).absolute()
    preset = config_defaults[args.preset]
    eval_json = Path(args.evaluation_json or preset.dataset_kwargs["evaluation_json"])
    if not eval_json.is_absolute():
        eval_json = (REPO_ROOT / eval_json).absolute()
    eval_indices = json.load(open(eval_json, encoding="utf-8"))
    if args.scene_ids:
        missing = [s for s in args.scene_ids if s not in eval_indices]
        if missing:
            raise ValueError(f"scene ids not in {eval_json}: {missing}")
        eval_indices = {s: eval_indices[s] for s in args.scene_ids}
        subset_json = out_dir / "evaluation_subset.json"
        write_json(subset_json, eval_indices)
        eval_json_used = subset_json
    else:
        eval_json_used = eval_json
    expected_scenes = args.expected_scenes if args.expected_scenes is not None else len(eval_indices)

    changes = dict(
        resume=str(resume),
        workspace=str(out_dir),
        use_ttt_for_eval=bool(args.tuning),
        num_workers=0,
        eval_n_media_dumps=0,
        mixed_precision=args.mixed_precision,
        use_input_supervision=False,
        dataset_kwargs={**preset.dataset_kwargs, "evaluation_json": str(eval_json_used)},
    )
    if args.tuning_steps is not None:
        changes["ttt_n_steps"] = args.tuning_steps
    if args.tuning_lr is not None:
        changes["ttt_lr"] = args.tuning_lr
    if args.tuning_target is not None:
        # the joint mode is ours: the official forward_ttt rejects it, so only the reimplemented loop can run it
        changes["ttt_mode"] = {"scene_latents": "scene-latent-tuning", "gs_tokens": "token-tuning",
                               JOINT: reimplemented_scene_latent_and_token_tuning.TTT_MODE}[args.tuning_target]
    # Inner-loop geometry weights are recorded in the options for the contract; the outer
    # use_geo_losses path stays off (it has no effect on scene latent tuning, see EXP_TokenGS.md).
    changes.update(lambda_geo_sw=args.geo_sw, lambda_geo_packing=args.geo_packing, lambda_geo_inter=args.geo_inter,
                   use_geo_losses=False)
    opt = preset.evolve(**changes)
    geo_weights = {"lambda_sw": opt.lambda_geo_sw, "lambda_packing": opt.lambda_geo_packing,
                   "lambda_inter": opt.lambda_geo_inter}
    if args.tuning_loop == "reimplemented" and (opt.geo_mass_weighting, opt.geo_inter_form, opt.geo_k_neighbors,
                                                opt.geo_target_distance, opt.geo_num_sw_directions) != (
            es.GEO_STAT_CONFIG["mass_weighting"], es.GEO_STAT_CONFIG["inter_form"], es.GEO_STAT_CONFIG["k_neighbors"],
            es.GEO_STAT_CONFIG["target_distance"], es.GEO_STAT_CONFIG["num_sw_directions"]):
        raise RuntimeError("preset geometry options differ from the pre-registered FFRGS configuration")
    children = int(opt.dec_patch_size) ** 2

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if accelerator.mixed_precision != args.mixed_precision:
        raise RuntimeError(f"accelerator precision {accelerator.mixed_precision} != requested {args.mixed_precision}")
    if accelerator.num_processes != 1:
        raise RuntimeError("this script is single-process only")

    log(f"hashing checkpoint {resume}")
    ckpt_sha = sha256_file(resume)
    git_head = run_cmd(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).strip()
    git_diff = run_cmd(["git", "diff", "HEAD", "--", "tokengs"], cwd=REPO_ROOT)
    contract = {
        "started_at": started_at,
        "cwd": os.getcwd(),
        "argv": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index": gpu_index,
        "physical_gpu_uuid": gpu_before["uuid"],
        "preset": args.preset,
        "entry": "scripts/evaluate_with_scene_latent_tuning.py "
                 "(official tokengs.evaluate checkpoint loading + get_multi_dataloader)",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "helper_sha256": {m.__name__: sha256_file(Path(m.__file__).resolve())
                          for m in (es, reimplemented_scene_latent_tuning, reimplemented_token_tuning,
                                    reimplemented_scene_latent_and_token_tuning, demo_ply_export)},
        # tokengs/losses/ is untracked, so the git diff hash below does not cover it
        "geo_losses_sha256": sha256_file(REPO_ROOT / "tokengs" / "losses" / "geo_losses.py"),
        "evaluation_json": str(eval_json),
        "evaluation_json_sha256": sha256_file(eval_json),
        "evaluation_json_used": str(eval_json_used),
        "scene_ids_restriction": args.scene_ids,
        "expected_scene_count": expected_scenes,
        "dataset_root_resolved": str((REPO_ROOT / "data" / "dl3dv_eval").resolve()),
        "checkpoint": str(resume),
        "checkpoint_sha256": ckpt_sha,
        "tuning_target": args.tuning_target,
        "scene_latent_tuning_enabled": args.tuning_target in TUNES_SCENE_LATENTS,
        "token_tuning_enabled": args.tuning_target in TUNES_GS_TOKENS,
        "tuning_loop": args.tuning_loop if opt.use_ttt_for_eval else None,
        "tuning_loop_geo_weights": geo_weights if args.tuning_loop == "reimplemented" else None,
        "geometry_stat_config": es.GEO_STAT_CONFIG,
        "demo": bool(args.demo),
        "ply_name": args.ply_name,
        "official_option_ttt_mode": opt.ttt_mode,
        "tuning_steps": opt.ttt_n_steps,
        "tuning_lr": opt.ttt_lr,
        "token_lr": args.token_lr,
        "num_input_views": opt.num_input_views,
        "batch_size": opt.batch_size,
        "num_workers": opt.num_workers,
        "mixed_precision": accelerator.mixed_precision,
        "tf32": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "media_enabled": False,
        "strict_checkpoint_loading": opt.strict_checkpoint_loading,
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gsplat": __import__("gsplat").__version__,
            "accelerate": __import__("accelerate").__version__,
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "PATH_head": os.environ.get("PATH", "").split(":")[:3],
            "ninja": run_cmd(["which", "ninja"]).strip(),
            "conda_prefix_of_python": str(Path(sys.executable).resolve().parents[1]),
        },
        "code_state": {
            "git_head": git_head,
            "tokengs_diff_vs_head_sha256": hashlib.sha256(git_diff.encode()).hexdigest(),
            "tokengs_diff_stat": run_cmd(["git", "diff", "--stat", "HEAD", "--", "tokengs"], cwd=REPO_ROOT),
        },
        "options": dataclasses.asdict(opt),
        "gpu_before": gpu_before,
    }
    write_json(out_dir / "launch_contract.json", contract)
    log(f"contract written: preset={args.preset} tuning_target={args.tuning_target} ttt_mode={opt.ttt_mode} "
        f"tuning_loop={args.tuning_loop if opt.use_ttt_for_eval else None} steps={opt.ttt_n_steps} lr={opt.ttt_lr} "
        f"token_lr={args.token_lr} "
        f"geo={geo_weights} demo={args.demo} precision={accelerator.mixed_precision} scenes={expected_scenes} "
        f"gpu={gpu_index} output={out_dir}")

    model = model_registry[opt.model_type](opt)
    ckpt = _load_checkpoint_state_dict(str(resume))
    if opt.strict_checkpoint_loading:
        _copy_checkpoint_strict(model, ckpt)
    else:
        _copy_matching_checkpoint(model, ckpt, log)
    del ckpt
    log("model loaded (strict)")

    _train_dl, test_dataloader, _, test_dataset = get_multi_dataloader(opt, accelerator)
    model, _train_dl, test_dataloader = accelerator.prepare(model, _train_dl, test_dataloader)
    model.eval()
    if len(test_dataset) != expected_scenes:
        raise RuntimeError(f"test dataset has {len(test_dataset)} scenes, expected {expected_scenes}")
    unwrapped = accelerator.unwrap_model(model)
    curve_state: dict = {"enabled": False}
    if args.tuning_loop == "reimplemented":
        if args.tuning_target == JOINT:
            reimplemented_scene_latent_and_token_tuning.install_reimplemented_scene_latent_and_token_tuning(
                unwrapped, geo_weights, children, curve_state, token_lr=args.token_lr,
                vis_mode=args.vis_mode, core_gate_tau=args.core_gate_tau,
                depth_prior_weight=args.depth_prior_weight)
        else:
            install = {"scene_latents": reimplemented_scene_latent_tuning.install_reimplemented_scene_latent_tuning,
                       "gs_tokens": reimplemented_token_tuning.install_reimplemented_token_tuning}[args.tuning_target]
            install(unwrapped, geo_weights, children, curve_state)
        log(f"reimplemented {args.tuning_target} tuning loop installed (geo weights {geo_weights}, "
            f"children/slot {children})")
    omega_model = es.load_vggt_omega_model(accelerator.device) if args.depth_prior == "vggt_omega" else None
    img_size = tuple(int(x) for x in opt.img_size)

    pid_binding = query_gpu(gpu_index)["compute_apps"]
    metrics_calc = MetricsCalculator(device=accelerator.device)
    set_status("running", started_at=started_at, completed_scenes=0, expected_scene_count=expected_scenes)

    scene_rows: list[dict] = []
    f_scene = open(out_dir / "scene_metrics.jsonl", "w", encoding="utf-8", buffering=1)
    f_frame = open(out_dir / "frame_metrics.jsonl", "w", encoding="utf-8", buffering=1)
    v_in = opt.num_input_views
    loop_t0 = time.perf_counter()
    t_prev = time.perf_counter()
    peak_alloc = peak_reserved = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            data_load = time.perf_counter() - t_prev
            torch.cuda.reset_peak_memory_stats()
            scene_id = data["scene_id"][0]
            frame_ids = [int(x) for x in data["frame_ids"][0].tolist()]
            context_ids, target_ids = frame_ids[:v_in], frame_ids[v_in:]
            spec = eval_indices[scene_id]
            if context_ids != list(spec["context"]) or target_ids != list(spec["target"]):
                raise RuntimeError(f"{scene_id}: loaded frames do not match evaluation json")
            frame_distance = [min(abs(fid - c) for c in context_ids) for fid in target_ids]
            far_mask = torch.tensor([d >= 4 for d in frame_distance], device=accelerator.device)
            novel_mask = torch.tensor([fid not in set(context_ids) for fid in target_ids], device=accelerator.device)

            if args.demo and opt.use_ttt_for_eval and args.tuning_loop == "reimplemented":
                gt_t, cam_t, intr_t = data["images_output"], data["cam_view"], data["intrinsics"]

                def target_metrics(gaussians, gt_t=gt_t, cam_t=cam_t, intr_t=intr_t):
                    pred_t = unwrapped.render_gaussians(gaussians, cam_t, intr_t)["images_pred"]
                    psnr_t = metrics_calc.calculate_psnr(pred_t, gt_t, reduction="none")[0]
                    edge_t, hf_t = es.edge_hf_per_frame(pred_t[0], gt_t[0])
                    out = {}
                    for name, m in (("all", torch.ones_like(novel_mask)), ("novel", novel_mask), ("d4plus", far_mask)):
                        if bool(m.any()):
                            out[f"target_{name}_psnr"] = float(psnr_t[m].mean())
                            out[f"target_{name}_edge_l1"] = float(edge_t[m].mean())
                            out[f"target_{name}_hf_ratio"] = float(hf_t[m].mean())
                    return out

                curve_state.update(enabled=True, target=target_metrics)
            else:
                curve_state.update(enabled=False, target=None)

            torch.cuda.synchronize()
            curve_state["tuned_tensor"] = None
            if args.depth_prior in ("colmap", "colmap_clean", "colmap_raw"):
                prov_eval = test_dataset.datasets[0] if hasattr(test_dataset, "datasets") else test_dataset
                curve_state["depth_prior"] = es.build_colmap_sparse_depth(
                    prov_eval.dataset, scene_id, context_ids, data["intrinsics_input"], img_size,
                    float(prov_eval.scene_scale), float(opt.znear), accelerator.device,
                    clean=(args.depth_prior != "colmap_raw"),
                )
            elif args.depth_prior == "vggt_omega":
                curve_state["depth_prior"] = es.build_vggt_omega_sim3_depth(
                    omega_model, data["images_input"], data["cam_to_world_input"],
                )
            else:
                curve_state["depth_prior"] = None
            t0 = time.perf_counter()
            results = model(data)
            torch.cuda.synchronize()
            inference = time.perf_counter() - t0

            t1 = time.perf_counter()
            pred, gt, masks = results["images_pred"], data["images_output"], data["masks_output"]
            if pred.shape[:2] != (1, len(target_ids)):
                raise RuntimeError(f"{scene_id}: pred shape {tuple(pred.shape)} vs {len(target_ids)} targets")
            mask_kw = {"mask": masks} if data["has_mask"].any() else {}
            per = metrics_calc.calculate_all_metrics(pred, gt, reduction="none", **mask_kw)
            per = {m: per[m][0].detach().double().cpu().numpy() for m in METRICS}
            edge, hf = es.edge_hf_per_frame(pred[0], gt[0])
            per["edge_l1"], per["hf_ratio"] = edge.double().cpu().numpy(), hf.double().cpu().numpy()
            metrics_time = time.perf_counter() - t1
            for m in FRAME_METRICS:
                if not np.all(np.isfinite(per[m])):
                    raise RuntimeError(f"{scene_id}: non-finite {m} values {per[m].tolist()}")

            t2 = time.perf_counter()
            gaussians = results["gaussians"].detach().float()
            geo = es.geometry_stats(gaussians, children)
            needle, needle_per = es.needle_stats(gaussians[0], data["cam_to_world_input"][0], data["intrinsics_input"][0],
                                                 img_size, opt.znear, detailed=args.demo)
            prov_eval = test_dataset.datasets[0] if hasattr(test_dataset, "datasets") else test_dataset
            feat_mask = es.build_target_colmap_feature_mask(
                prov_eval.dataset, scene_id, target_ids, data["intrinsics"], img_size,
                float(prov_eval.scene_scale), float(opt.znear), accelerator.device,
            )
            surf = es.surface_and_region_stats(
                unwrapped, gaussians, pred, gt, results["depths_pred"], data["cam_view"], data["intrinsics"],
                data["cam_view_input"], data["intrinsics_input"], far_mask, feat_mask, img_size,
            )
            geometry_seconds = time.perf_counter() - t2

            ctx_set = set(context_ids)
            is_ctx = np.array([fid in ctx_set for fid in target_ids])
            for pos, fid in enumerate(target_ids):
                f_frame.write(json.dumps({
                    "scene_index": i, "scene_id": scene_id, "target_position": pos, "frame_id": fid,
                    "is_context_view": bool(is_ctx[pos]),
                    "nearest_context_distance": frame_distance[pos],
                    "distance_bucket": es.frame_distance_bucket(frame_distance[pos]),
                    **{m: float(per[m][pos]) for m in FRAME_METRICS},
                }) + "\n")

            alloc = torch.cuda.max_memory_allocated() / 2**20
            reserved = torch.cuda.max_memory_reserved() / 2**20
            peak_alloc, peak_reserved = max(peak_alloc, alloc), max(peak_reserved, reserved)
            row = {
                "scene_index": i,
                "scene_id": scene_id,
                "all_frame_count": int(len(target_ids)),
                "novel_frame_count": int((~is_ctx).sum()),
                "context_frame_count": int(is_ctx.sum()),
                "data_load_seconds": data_load,
                "inference_seconds": inference,
                "metrics_seconds": metrics_time,
                "geometry_seconds": geometry_seconds,
            }
            for m in FRAME_METRICS:
                row[f"all_{m}"] = float(per[m].mean())
                row[f"novel_{m}"] = float(per[m][~is_ctx].mean())
            buckets = np.array([es.frame_distance_bucket(d) for d in frame_distance])
            for b in es.DISTANCE_BUCKETS:
                sel = buckets == b
                row[f"bucket_{b}_frame_count"] = int(sel.sum())
                for m in FRAME_METRICS:
                    row[f"bucket_{b}_{m}"] = float(per[m][sel].mean()) if sel.any() else None
            row.update({f"geo_{k}": v for k, v in geo.items()})
            row.update({f"needle_{k}": needle[k] for k in NEEDLE_KEYS})
            row.update({f"surf_{k}": surf[k] for k in SURFACE_KEYS})
            media_t0 = time.perf_counter()
            if args.demo:
                decoder_inputs = _decoder_inputs(unwrapped, data, opt, args, curve_state, gaussians)
                media = _write_demo(out_dir, scene_id, args, gaussians, needle, needle_per, children, data, pred, gt,
                                    target_ids, is_ctx, frame_distance, curve_state, geo, per, row, decoder_inputs,
                                    opt, ckpt_sha)
                if decoder_inputs is None:
                    log(f"{scene_id[:12]}: official tuning loop does not expose the tuned tensor; "
                        f"decoder_inputs.pt not written (use --tuning-loop reimplemented)")
                else:
                    log(f"{scene_id[:12]}: decoder_inputs.pt written; re-decoding it reproduces the evaluated "
                        f"Gaussians up to max |diff| {decoder_inputs['redecode_max_abs_diff']:.3e}")
                    del decoder_inputs
            else:
                media = {}
            row.update({
                "media": media,
                "media_seconds": time.perf_counter() - media_t0,
                "scene_total_seconds": time.perf_counter() - t_prev,
                "torch_peak_allocated_mib": alloc,
                "torch_peak_reserved_mib": reserved,
            })
            scene_rows.append(row)
            f_scene.write(json.dumps(row) + "\n")
            log(f"[{i + 1}/{expected_scenes}] {scene_id[:12]} all PSNR {row['all_psnr']:.3f} SSIM {row['all_ssim']:.4f} "
                f"LPIPS {row['all_lpips']:.4f} | novel PSNR {row['novel_psnr']:.3f} ({row['novel_frame_count']}f) "
                f"| edge {row['all_edge_l1']:.4f} HF {row['all_hf_ratio']:.3f} | inter d {geo['inter_distance']:.3f} "
                f"aniso {needle['aniso_p50_weighted']:.1f} | infer {inference:.2f}s")
            set_status("running", started_at=started_at, completed_scenes=i + 1, expected_scene_count=expected_scenes,
                       last_scene_id=scene_id)
            del results, pred, gt, masks, gaussians, needle_per
            torch.cuda.empty_cache()
            t_prev = time.perf_counter()
    f_scene.close()
    f_frame.close()
    loop_seconds = time.perf_counter() - loop_t0

    if len(scene_rows) != expected_scenes or len({r["scene_id"] for r in scene_rows}) != expected_scenes:
        raise RuntimeError(f"evaluated {len(scene_rows)} scenes, expected {expected_scenes} unique")

    with open(out_dir / "scene_metrics.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["scene_index", "scene_id", "all_frame_count", "novel_frame_count",
                *[f"all_{m}" for m in FRAME_METRICS], *[f"novel_{m}" for m in FRAME_METRICS],
                *[f"bucket_{b}_psnr" for b in es.DISTANCE_BUCKETS],
                *[f"geo_{k}" for k in GEO_KEYS], *[f"needle_{k}" for k in NEEDLE_KEYS],
                "data_load_seconds", "inference_seconds", "metrics_seconds", "geometry_seconds", "media_seconds",
                "scene_total_seconds", "torch_peak_allocated_mib", "torch_peak_reserved_mib"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(scene_rows)

    completed_at = now_iso()
    summary = {
        "status": "complete",
        "started_at": started_at,
        "completed_at": completed_at,
        "scene_count": len(scene_rows),
        "expected_scene_count": expected_scenes,
        "formal_complete": args.scene_ids is None and len(scene_rows) == expected_scenes,
        "preset": args.preset,
        "tuning_target": args.tuning_target,
        "scene_latent_tuning_enabled": args.tuning_target in TUNES_SCENE_LATENTS,
        "token_tuning_enabled": args.tuning_target in TUNES_GS_TOKENS,
        "tuning_steps": opt.ttt_n_steps if opt.use_ttt_for_eval else None,
        "tuning_lr": opt.ttt_lr if opt.use_ttt_for_eval else None,
        "token_lr": args.token_lr,
        "tuning_loop": args.tuning_loop if opt.use_ttt_for_eval else None,
        "tuning_loop_geo_weights": geo_weights if args.tuning_loop == "reimplemented" else None,
        "mixed_precision": accelerator.mixed_precision,
        "frame_counts": {
            "all_frames_per_scene": sorted({r["all_frame_count"] for r in scene_rows}),
            "novel_only_per_scene": sorted({r["novel_frame_count"] for r in scene_rows}),
        },
    }
    for section in SECTIONS:
        p = SECTION_PREFIX[section]
        summary[section] = {
            "metrics": {m: metric_stats([r[f"{p}_{m}"] for r in scene_rows]) for m in FRAME_METRICS},
            "rankings": {m: rankings(scene_rows, f"{p}_{m}", m) for m in METRICS},
        }
    summary["distance_buckets"] = {
        "definition": "frame-index distance to the nearest context view; per-scene bucket mean, then mean over "
                      "scenes that have the bucket",
    }
    for b in es.DISTANCE_BUCKETS:
        have = [r for r in scene_rows if r[f"bucket_{b}_frame_count"] > 0]
        summary["distance_buckets"][b] = {
            "n_scenes": len(have),
            "n_frames": int(sum(r[f"bucket_{b}_frame_count"] for r in have)),
            **({m: metric_stats([r[f"bucket_{b}_{m}"] for r in have]) for m in FRAME_METRICS} if have else {}),
        }

    def describe(key: str) -> dict:
        vals = np.array([r[key] for r in scene_rows], dtype=np.float64)
        return {"mean": float(vals.mean()), "median": float(np.median(vals)),
                "p10": float(np.percentile(vals, 10)), "p90": float(np.percentile(vals, 90))}

    summary["geometry"] = {"config": es.GEO_STAT_CONFIG, **{k: describe(f"geo_{k}") for k in GEO_KEYS}}
    summary["needle"] = {k: describe(f"needle_{k}") for k in NEEDLE_KEYS}
    # The release table has no token tuning row (alone or together with the scene latents).
    official = OFFICIAL_RELEASE.get(args.preset) if args.tuning_target in (None, "scene_latents") else None
    if official is not None:
        key = "scene_latent_tuning" if opt.use_ttt_for_eval else "no_tuning"
        summary["official_release_comparison"] = {
            "source": "release_2026.6.md (all target frames, 140 scenes; rows labelled w/o TTT and w/ TTT)",
            "reference_setting": key,
            "reference": official[key],
            "delta_all_frames": {m: summary["all_frames"]["metrics"][m]["mean"] - official[key][m] for m in METRICS},
            "delta_definition": "local all_frames mean - official",
            "official_table": official,
        }
    comparisons = []
    for spec in args.compare:
        label, ref_dir = spec.split("=", 1)
        ref_rows = load_jsonl(Path(ref_dir) / "scene_metrics.jsonl")
        comparisons.append(paired_comparison(scene_rows, ref_rows, label, str(Path(ref_dir).absolute())))
    if comparisons:
        summary["paired_comparisons"] = comparisons
    summary["timing"] = {
        "evaluation_loop_seconds": loop_seconds,
        "scene_total_seconds_sum": float(sum(r["scene_total_seconds"] for r in scene_rows)),
        "mean_scene_total_seconds": float(np.mean([r["scene_total_seconds"] for r in scene_rows])),
        "total_script_seconds": time.perf_counter() - script_t0,
    }
    write_json(out_dir / "summary.json", summary)

    monitor.stop()
    _write_resource_evidence(out_dir, gpu_before, pid_binding, monitor, gpu_index, status="complete",
                             torch_peaks={"allocated": peak_alloc, "reserved": peak_reserved})
    set_status("complete", started_at=started_at, completed_at=completed_at, scene_count=len(scene_rows),
               expected_scene_count=expected_scenes, formal_complete=summary["formal_complete"])

    missing = [name for name in REQUIRED_OUTPUTS if not (out_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing outputs: {missing}")
    for section in SECTIONS:
        ms = summary[section]["metrics"]
        log(f"{section}: PSNR {ms['psnr']['mean']:.6f} SSIM {ms['ssim']['mean']:.6f} LPIPS {ms['lpips']['mean']:.6f} "
            f"edge {ms['edge_l1']['mean']:.6f} HF {ms['hf_ratio']['mean']:.6f}")
    for b in es.DISTANCE_BUCKETS:
        bs = summary["distance_buckets"][b]
        if bs["n_scenes"]:
            log(f"bucket {b}: {bs['n_frames']} frames PSNR {bs['psnr']['mean']:.4f} HF {bs['hf_ratio']['mean']:.4f}")
    g, nd = summary["geometry"], summary["needle"]
    log(f"geometry (mean over scenes): SW {g['sw_loss']['mean']:.4f} packing {g['packing_ratio']['mean']:.2f}/"
        f"{g['packing_target']['mean']:.2f} inter d {g['inter_distance']['mean']:.3f} close "
        f"{g['inter_too_close']['mean']:.3f} far {g['inter_too_far']['mean']:.3f} | aniso p50 "
        f"{nd['aniso_p50_weighted']['mean']:.2f} needle>0.9 {nd['long_axis_along_ray_gt0.9_weighted']['mean']:.3f}")
    if official is not None:
        log(f"delta vs official release {summary['official_release_comparison']['reference_setting']}: "
            f"{summary['official_release_comparison']['delta_all_frames']}")
    for c in comparisons:
        if "error" in c:
            log(f"compare {c['label']}: {c['error']}")
            continue
        for section in SECTIONS:
            log(f"compare {c['label']} {section}: " + ", ".join(
                f"{m} {c[section][m]['delta_mean']:+.6f}" for m in FRAME_METRICS if m in c[section]))
    log(f"torch peak memory: allocated {peak_alloc:.0f} MiB, reserved {peak_reserved:.0f} MiB")
    log(f"outputs in {out_dir}")
    log(f"done in {time.perf_counter() - script_t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
