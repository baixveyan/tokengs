"""ScanNet++ 50-scene RGB-only feed-forward evaluation and same-scene Demo bundle exporter."""

import argparse
import csv
import dataclasses
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fused_ssim import fused_ssim
from lpips import LPIPS
from PIL import Image
from safetensors.torch import load_file

import demo_ply_export
import evaluation_statistics as es
from scannetpp_evaluation_protocol import (
    DEFAULT_DEMO_SCENE, GEOMETRY_COMPUTED_STATUS, GEOMETRY_FRAME_METRICS,
    GEOMETRY_SCENE_METRICS, METRICS, build_raycasting_scene,
    compute_scene_geometry_metrics, mean_metrics, raycast_gt_depth_and_normals,
    render_ray_thickness_map, sha256_file, validate_frame_split,
    write_csv_atomic, write_json,
)
from tokengs.data.provider import Provider
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults


ROOT = Path(__file__).resolve().parents[1]
NAME = "baseline_pretrained_6view"
CHECKPOINT = ROOT / "checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors"
CHECKPOINT_SHA = "5913da347475049cb58e552634ee1c527293c2571576806db0d4da702ce978c3"
PRUNE_THRESHOLDS = (0.01, 0.05)


def idle_gpu():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Select one idle GPU using EXP_ScanNetPP/run_50_scenes.sh or scripts/run_scannetpp_baseline.sh")
    query = ["nvidia-smi", "-i", visible, "--query-gpu=index,uuid,memory.used",
             "--format=csv,noheader,nounits"]
    fields = subprocess.check_output(query, text=True).strip().split(",")
    gpu = dict(zip(("index", "uuid", "memory_used_mib"), map(str.strip, fields)))
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True).strip()
    busy = any(line.split(",")[0].strip() == gpu["uuid"]
               and int(line.split(",")[1]) != os.getpid() for line in apps.splitlines())
    if float(gpu["memory_used_mib"]) > 100 or busy:
        raise RuntimeError("Selected GPU is no longer idle")
    return gpu


def frame_splits(provider):
    dataset = provider.dataset
    expected = (dataset.root_path / "splits" / dataset.split).read_text().split()
    actual = [Path(path).name for path in dataset.sample_list]
    if len(actual) != 50 or len(set(actual)) != 50 or set(actual) != set(expected):
        raise RuntimeError("Dataset must contain every official validation scene exactly once")
    result = []
    for index, scene_id in enumerate(actual):
        context, target = dataset.get_context_target_frames(index)
        validate_frame_split(context, target)
        result.append({"scene_id": scene_id, "context": context, "target": target})
    return result


def save_render_strip(demo_dir: Path, truth: torch.Tensor, prediction: torch.Tensor,
                      context_ids: list[int], target_ids: list[int], frames: list[dict]) -> None:
    strip = torch.cat((truth.detach().cpu(), prediction.detach().cpu()), dim=2).permute(2, 0, 3, 1)
    strip = strip.reshape(strip.shape[0], -1, 3).clamp(0, 1)
    Image.fromarray((strip.numpy() * 255).round().astype(np.uint8)).save(demo_dir / "render_strip.png")
    write_json(demo_dir / "render_strip.json", {
        "layout": "top row GT, bottom row render (all 8 novel target views)",
        "frames": [
            {
                "target_position": f["target_position"],
                "frame_id": f["frame_id"],
                "nearest_context_distance": min(abs(int(f["frame_id"]) - int(c)) for c in context_ids),
                "psnr": f["psnr"],
                "ssim": f["ssim"],
                "lpips": f["lpips"],
            }
            for f in frames
        ],
    })


def export_demo_bundle(
    demo_dir: Path,
    method_name: str,
    scene_id: str,
    gaussians: torch.Tensor,
    cam_to_world_input: torch.Tensor,
    cam_view: torch.Tensor,
    cam_view_input: torch.Tensor,
    intrinsics_input: torch.Tensor,
    raw_context_c2w: torch.Tensor,
    img_size: tuple[int, int] | list[int],
    znear: float,
    scene_scale: float,
    ckpt_sha: str,
    row: dict,
    frames: list[dict],
    context_ids: list[int],
    target_ids: list[int],
    dataset_root: Path,
    truth: torch.Tensor | None = None,
    prediction: torch.Tensor | None = None,
    existing_render_strip: Path | None = None,
) -> None:
    """Write the EXP_TokenGS-aligned demo folder EXP_ScanNetPP/demo/scene_<id>/<method_name>/."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = demo_dir.parent
    demo_root = scene_dir.parent
    demo_scene_json = demo_root / "demo_scene.json"
    if not demo_scene_json.is_file():
        write_json(demo_scene_json, {
            "rule": "first scene of the official ScanNet++ v2 validation split (nvs_sem_val.txt)",
            "scene_id": scene_id,
            "n_scenes": 50,
        })

    if gaussians.ndim == 2:
        gaussians = gaussians.unsqueeze(0)
    gaussians_cpu = gaussians.detach().float().cpu()
    children = 64
    H, W = int(img_size[0]), int(img_size[1])

    needle, needle_per = es.needle_stats(
        gaussians_cpu[0],
        cam_to_world_input.detach().float().cpu(),
        intrinsics_input.detach().float().cpu(),
        (H, W),
        float(znear),
        detailed=True,
    )
    geo = es.geometry_stats(gaussians_cpu, children)

    torch.save(gaussians_cpu, demo_dir / "gaussians.pt")
    ply_path = demo_dir / f"{method_name}.ply"
    slot_path = demo_dir / f"{method_name}_slotcolor.ply"
    cam_path = demo_dir / "cameras.ply"
    demo_ply_export.export_gaussians_ply(ply_path, gaussians_cpu, children, needle_per, color_by_slot=False)
    demo_ply_export.export_gaussians_ply(slot_path, gaussians_cpu, children, needle_per, color_by_slot=True)
    for thr in PRUNE_THRESHOLDS:
        demo_ply_export.export_pruned_ply(ply_path, demo_dir / f"{method_name}_prune{thr:g}.ply", thr)
        demo_ply_export.export_pruned_ply(slot_path, demo_dir / f"{method_name}_prune{thr:g}_slotcolor.ply", thr)

    target_c2w = torch.linalg.inv(cam_view.detach().float().cpu().transpose(-1, -2))
    demo_ply_export.export_camera_ply(cam_path, cam_to_world_input.detach().float().cpu(), target_c2w)

    gt_lidar_ply = scene_dir / "gt_lidar_mesh_frustum.ply"
    mesh_ply = dataset_root / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    if not gt_lidar_ply.is_file() and mesh_ply.is_file():
        demo_ply_export.export_scannetpp_lidar_ply(
            gt_lidar_ply, mesh_ply, raw_context_c2w, cam_view_input, intrinsics_input, (H, W), float(scene_scale)
        )

    if truth is not None and prediction is not None:
        save_render_strip(demo_dir, truth, prediction, context_ids, target_ids, frames)
    elif existing_render_strip is not None and existing_render_strip.is_file():
        shutil.copyfile(existing_render_strip, demo_dir / "render_strip.png")
        write_json(demo_dir / "render_strip.json", {
            "layout": "top row GT, bottom row render (all 8 novel target views)",
            "frames": [
                {
                    "target_position": f["target_position"],
                    "frame_id": f["frame_id"],
                    "nearest_context_distance": min(abs(int(f["frame_id"]) - int(c)) for c in context_ids),
                    "psnr": f["psnr"],
                    "ssim": f["ssim"],
                    "lpips": f["lpips"],
                }
                for f in frames
            ],
        })
    else:
        raise ValueError("Demo export requires either (truth, prediction) tensors or an existing render_strip.png")

    write_json(demo_dir / "needle_stats.json", {
        "scene_id": scene_id,
        "needle": needle,
        "geometry": geo,
    })

    with (demo_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frames[0]))
        writer.writeheader()
        writer.writerows(frames)
    with (demo_dir / "scene_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    completed_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "name": method_name,
        "scene_id": scene_id,
        "psnr": row["psnr"],
        "ssim": row["ssim"],
        "lpips": row["lpips"],
        "num_input_views": len(context_ids),
        "num_target_views": len(target_ids),
        "image_size": [H, W],
        "precision": "float32",
        "tf32": False,
        "point_cloud_input": False,
        "optimization_steps": 0,
        "checkpoint_sha256": ckpt_sha,
        "completed_at": completed_at,
    }
    write_json(demo_dir / "summary.json", summary)

    metrics = {
        "all_psnr": row["psnr"],
        "novel_psnr": row["psnr"],
        "all_ssim": row["ssim"],
        "novel_ssim": row["ssim"],
        "all_lpips": row["lpips"],
        "novel_lpips": row["lpips"],
        **{f"geo_{k}": v for k, v in geo.items()},
        **{f"needle_{k}": v for k, v in needle.items() if not isinstance(v, (dict, list))},
    }
    sidecar_path = demo_dir / f"{method_name}.json"
    write_json(sidecar_path, {
        "method": method_name,
        "scene_id": scene_id,
        "files": {
            "ply": ply_path.name,
            "ply_slotcolor": slot_path.name,
            "cameras_ply": cam_path.name,
            "gaussians_pt": "gaussians.pt",
        },
        "decoder_inputs": None,
        "redecode_check": None,
        "tuning_target": None,
        "tuning_loop": None,
        "tuning_steps": None,
        "tuning_lr": None,
        "token_lr": None,
        "tuning_loop_geo_weights": {"lambda_sw": 0.0, "lambda_packing": 0.0, "lambda_inter": 0.0},
        "checkpoint_sha256": ckpt_sha,
        "children_per_slot": children,
        "metrics": metrics,
    })
    for required in (ply_path, slot_path, cam_path, demo_dir / "gaussians.pt",
                     demo_dir / "render_strip.png", demo_dir / "needle_stats.json", sidecar_path):
        if not required.is_file():
            raise RuntimeError(f"Demo bundle missing required file: {required}")
    write_json(demo_dir / "status.json", {
        "status": "complete",
        "scene_id": scene_id,
        "completed_at": completed_at,
    })


def export_demo_from_artifact(args):
    """Build the full demo folder on CPU from an existing saved .pt artifact and experiment CSVs."""
    if args.render_strip_png is None or not args.render_strip_png.is_file():
        raise ValueError("--from-artifact requires a valid --render-strip-png file")
    payload = torch.load(args.from_artifact, map_location="cpu", weights_only=True)
    scene_id = payload["scene_id"]
    exp_dir = args.output
    method_name = args.ply_name or exp_dir.name
    demo_dir = args.demo_output or (ROOT / "EXP_ScanNetPP/demo" / f"scene_{scene_id[:12]}" / method_name)
    if demo_dir.exists() and any(p.name != "run.log" for p in demo_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing Demo directory: {demo_dir}")

    exp_summary_path = exp_dir / "summary.json"
    if not exp_summary_path.is_file():
        raise RuntimeError(f"Missing experiment summary.json in {exp_dir}")
    exp_summary = json.loads(exp_summary_path.read_text(encoding="utf-8"))
    ckpt_sha = sha256_file(args.checkpoint)
    if exp_summary.get("checkpoint_sha256") != ckpt_sha:
        raise ValueError("Experiment summary.json checkpoint_sha256 does not match --checkpoint")
    art_sha = payload.get("checkpoint_sha256")
    if art_sha is None:
        for cand in (args.from_artifact.parent / "summary.json", args.from_artifact.parent.parent / "summary.json"):
            if cand.is_file():
                art_sha = json.loads(cand.read_text(encoding="utf-8")).get("checkpoint_sha256")
                break
    if not art_sha:
        raise ValueError("Artifact is missing verifiable checkpoint_sha256 provenance")
    if art_sha != ckpt_sha:
        raise ValueError("Artifact checkpoint_sha256 provenance does not match --checkpoint")

    with (exp_dir / "scene_metrics.csv").open("r", encoding="utf-8", newline="") as stream:
        scene_rows = [r for r in csv.DictReader(stream) if r["scene_id"] == scene_id]
    with (exp_dir / "frame_metrics.csv").open("r", encoding="utf-8", newline="") as stream:
        frame_rows = [r for r in csv.DictReader(stream) if r["scene_id"] == scene_id]
    if len(scene_rows) != 1 or len(frame_rows) != 8:
        raise RuntimeError(f"Missing scene/frame metrics for {scene_id} in {exp_dir}")

    s_row = scene_rows[0]
    expected_ctx = [int(x) for x in s_row["context_frames"].split()]
    expected_tgt = [int(x) for x in s_row["target_frames"].split()]
    payload_ctx = [int(x) for x in payload["context_ids"]]
    payload_tgt = [int(x) for x in payload["target_ids"]]
    csv_tgt = [int(f["frame_id"]) for f in frame_rows]
    if payload_ctx != expected_ctx or payload_tgt != expected_tgt or csv_tgt != expected_tgt:
        raise ValueError("Artifact context/target frame IDs do not match scene_metrics.csv / frame_metrics.csv")
    row = {
        "scene_id": scene_id,
        "context_frames": s_row["context_frames"],
        "target_frames": s_row["target_frames"],
        "num_target_frames": int(s_row["num_target_frames"]),
        "psnr": float(s_row["psnr"]),
        "ssim": float(s_row["ssim"]),
        "lpips": float(s_row["lpips"]),
        "forward_seconds": float(s_row["forward_seconds"]),
    }
    frames = [
        {
            "scene_id": f["scene_id"],
            "frame_id": int(f["frame_id"]),
            "target_position": int(f["target_position"]),
            "is_context_view": f["is_context_view"] == "True",
            "psnr": float(f["psnr"]),
            "ssim": float(f["ssim"]),
            "lpips": float(f["lpips"]),
        }
        for f in frame_rows
    ]
    export_demo_bundle(
        demo_dir=demo_dir,
        method_name=method_name,
        scene_id=scene_id,
        gaussians=payload["gaussians"],
        cam_to_world_input=payload["cam_to_world_input"],
        cam_view=payload["cam_view"],
        cam_view_input=payload["cam_view_input"],
        intrinsics_input=payload["intrinsics_input"],
        raw_context_c2w=payload["raw_context_c2w"],
        img_size=payload["image_size"],
        znear=float(config_defaults["eval_scannetpp_latent_6view_ssim"].znear),
        scene_scale=float(payload["scene_scale"]),
        ckpt_sha=ckpt_sha,
        row=row,
        frames=frames,
        context_ids=[int(x) for x in payload["context_ids"]],
        target_ids=[int(x) for x in payload["target_ids"]],
        dataset_root=args.dataset_root,
        existing_render_strip=args.render_strip_png,
    )
    print(f"Exported demo bundle from artifact to {demo_dir}")


def backfill_geometry_from_reconstructions(args):
    """Backfill G1-G4 geometry metrics into an existing 50-scene evaluation directory from saved .pt reconstructions."""
    output = args.output
    recons_dir = args.backfill_geometry_from_reconstructions
    summary_path = output / "summary.json"
    scene_csv_path = output / "scene_metrics.csv"
    frame_csv_path = output / "frame_metrics.csv"
    if not (summary_path.is_file() and scene_csv_path.is_file() and frame_csv_path.is_file()):
        raise RuntimeError(f"Cannot backfill geometry: missing summary/CSV files in {output}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ckpt_sha = sha256_file(args.checkpoint)
    if summary.get("checkpoint_sha256") != ckpt_sha:
        raise ValueError("Target summary.json checkpoint_sha256 does not match --checkpoint")
    art_summary_path = recons_dir.parent / "summary.json"
    if not art_summary_path.is_file():
        raise ValueError(f"Missing provenance summary.json at {art_summary_path}")
    if json.loads(art_summary_path.read_text(encoding="utf-8")).get("checkpoint_sha256") != ckpt_sha:
        raise ValueError("Reconstructions provenance checkpoint_sha256 does not match --checkpoint")

    gpu = idle_gpu()
    device = torch.device("cuda:0")
    opt = config_defaults["eval_scannetpp_latent_6view_ssim"]

    with scene_csv_path.open("r", encoding="utf-8", newline="") as stream:
        scene_rows = list(csv.DictReader(stream))
    with frame_csv_path.open("r", encoding="utf-8", newline="") as stream:
        frame_rows = list(csv.DictReader(stream))
    if len(scene_rows) != 50 or len(frame_rows) != 400:
        raise RuntimeError("Expected 50 scenes and 400 target frames in existing CSVs")

    frames_by_scene: dict[str, list[dict]] = {}
    for f in frame_rows:
        frames_by_scene.setdefault(f["scene_id"], []).append(f)

    new_scene_rows = []
    new_frame_rows = []
    for idx, s_row in enumerate(scene_rows):
        scene_id = s_row["scene_id"]
        pt_path = recons_dir / f"{scene_id}.pt"
        if not pt_path.is_file():
            raise FileNotFoundError(f"Missing reconstruction file: {pt_path}")
        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        expected_ctx = [int(x) for x in s_row["context_frames"].split()]
        expected_tgt = [int(x) for x in s_row["target_frames"].split()]
        if [int(x) for x in payload["context_ids"]] != expected_ctx or [int(x) for x in payload["target_ids"]] != expected_tgt:
            raise ValueError(f"Frame split mismatch in {pt_path}")

        gaussians = payload["gaussians"].to(device=device, dtype=torch.float32).unsqueeze(0)
        cam_view = payload["cam_view"].to(device=device, dtype=torch.float32).unsqueeze(0)
        intrinsics = payload["intrinsics"].to(device=device, dtype=torch.float32).unsqueeze(0)
        scene_scale = float(payload["scene_scale"])
        img_size = tuple(int(x) for x in payload["image_size"])
        with torch.inference_mode():
            sigma_m, _ = render_ray_thickness_map(
                gaussians, cam_view, intrinsics, img_size, float(opt.znear), float(opt.zfar), scene_scale
            )
        mesh_ply = args.dataset_root / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
        rc_scene = build_raycasting_scene(mesh_ply, payload["raw_context_c2w"], scene_scale)
        d_gt, n_gt, gt_ok = raycast_gt_depth_and_normals(rc_scene, payload["cam_view"], payload["intrinsics"], img_size)
        f_geo_list, s_geo = compute_scene_geometry_metrics(
            payload["depths_pred"], payload["alphas_pred"], sigma_m[0],
            d_gt, n_gt, gt_ok, payload["cam_view"], payload["intrinsics"],
            scene_scale=scene_scale, nn_device=device,
        )
        s_frames = frames_by_scene[scene_id]
        for pos, f_dict in enumerate(s_frames):
            merged_f = {
                "scene_id": f_dict["scene_id"],
                "frame_id": int(f_dict["frame_id"]),
                "target_position": int(f_dict["target_position"]),
                "is_context_view": f_dict["is_context_view"] == "True",
                "psnr": float(f_dict["psnr"]),
                "ssim": float(f_dict["ssim"]),
                "lpips": float(f_dict["lpips"]),
                **f_geo_list[pos],
            }
            new_frame_rows.append(merged_f)

        merged_s = {
            "scene_id": scene_id,
            "context_frames": s_row["context_frames"],
            "target_frames": s_row["target_frames"],
            "num_target_frames": int(s_row["num_target_frames"]),
            "psnr": float(s_row["psnr"]),
            "ssim": float(s_row["ssim"]),
            "lpips": float(s_row["lpips"]),
            **s_geo,
            "forward_seconds": float(s_row["forward_seconds"]),
        }
        new_scene_rows.append(merged_s)
        print(
            f"[Geo {idx + 1:02d}/50] {scene_id}: AbsRel={s_geo['depth_absrel']:.4f} "
            f"RMSE={s_geo['depth_rmse_m']:.3f}m RayThick={s_geo['ray_thick_med_m']:.3f}m "
            f"F@3cm={s_geo['surf_fscore_3cm']:.4f} F@5cm={s_geo['surf_fscore_5cm']:.4f} "
            f"NormMed={s_geo['normal_med_deg']:.2f}deg",
            flush=True,
        )

    for r in new_scene_rows:
        for k in GEOMETRY_SCENE_METRICS:
            if not math.isfinite(float(r[k])):
                raise RuntimeError(f"Non-finite geometry metric {k} in scene {r['scene_id']}")
    scale_ratios = [r["depth_scale_ratio_med"] for r in new_scene_rows]
    median_scale_ratio = statistics.median(scale_ratios)
    if not (0.90 <= median_scale_ratio <= 1.10):
        raise RuntimeError(f"Scale alignment gate failed: median D_pred/D_gt = {median_scale_ratio:.4f} outside [0.90, 1.10]")

    for k in GEOMETRY_SCENE_METRICS:
        summary[k] = statistics.mean(r[k] for r in new_scene_rows)
    summary["geometry_metrics_status"] = GEOMETRY_COMPUTED_STATUS
    summary["geometry_alignment_median_ratio"] = median_scale_ratio

    write_csv_atomic(frame_csv_path, new_frame_rows)
    write_csv_atomic(scene_csv_path, new_scene_rows)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


def evaluate(args):
    if args.from_artifact is not None:
        export_demo_from_artifact(args)
        return
    if args.backfill_geometry_from_reconstructions is not None:
        backfill_geometry_from_reconstructions(args)
        return

    output = args.output
    method_name = args.ply_name or output.name
    if not args.demo_only and output.exists() and any(
        path.name not in ("run.log", "model.safetensors") and not path.name.startswith(".")
        for path in output.iterdir()
    ):
        raise RuntimeError("Refusing to overwrite existing results")
    if args.window_size < 14:
        raise ValueError("Window is too small for six inputs and eight distinct targets")
    gpu = idle_gpu()
    ckpt_sha = sha256_file(args.checkpoint)
    if args.checkpoint.resolve() == CHECKPOINT.resolve() and ckpt_sha != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected official checkpoint SHA256")
    if args.demo_only:
        exp_summary_path = ROOT / "EXP_ScanNetPP" / method_name / "summary.json"
        if exp_summary_path.is_file():
            exp_summary = json.loads(exp_summary_path.read_text(encoding="utf-8"))
            if exp_summary.get("checkpoint_sha256") != ckpt_sha:
                raise ValueError(
                    f"--demo-only checkpoint SHA256 does not match {exp_summary_path}: "
                    f"{ckpt_sha} != {exp_summary.get('checkpoint_sha256')}"
                )
    if not (Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth").is_file():
        raise RuntimeError("Local LPIPS backbone missing; no automatic download is allowed")
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    opt = dataclasses.replace(
        config_defaults["eval_scannetpp_latent_6view_ssim"],
        model_type=args.model_type,
        evaluating=True, use_ttt_for_eval=False, ttt_n_steps=0,
        use_geo_losses=False, load_lidar_pc=False, use_input_supervision=False,
        mixed_precision="no", num_workers=0,
        dataset_kwargs={"window_size": args.window_size, "root_path": str(args.dataset_root)},
    )
    provider = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    splits = frame_splits(provider)
    demo_scene = args.demo_scene or splits[0]["scene_id"]
    if demo_scene not in {scene["scene_id"] for scene in splits}:
        raise ValueError("Demo scene is not in the validation split")
    demo_output = args.demo_output or (ROOT / "EXP_ScanNetPP/demo" / f"scene_{demo_scene[:12]}" / method_name)
    if demo_output.exists() and any(p.name not in ("run.log", "console.log") for p in demo_output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite an existing Demo: {demo_output}")
    if not args.demo_only:
        output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    model = model_registry[opt.model_type](opt).eval().requires_grad_(False)
    model.load_state_dict(load_file(str(args.checkpoint), device="cpu"), strict=True)
    model = model.to(device=device, dtype=torch.float32)
    lpips_metric = LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)

    indices = [i for i, s in enumerate(splits) if not args.demo_only or s["scene_id"] == demo_scene]
    subset = torch.utils.data.Subset(provider, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    frame_fields = ["scene_id", "frame_id", "target_position", "is_context_view", *METRICS, *GEOMETRY_FRAME_METRICS]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU {gpu['index']} ({gpu['uuid']}): {len(indices)} scene(s), FP32, no TF32, no point cloud")
    frame_csv_context = (
        (output / "frame_metrics.csv").open("w", newline="", encoding="utf-8")
        if not args.demo_only else open(os.devnull, "w", encoding="utf-8")
    )
    with frame_csv_context as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=frame_fields)
        if not args.demo_only:
            writer.writeheader()
        for step_idx, raw in enumerate(loader):
            spec = splits[indices[step_idx]]
            scene_id = raw["scene_id"][0]
            if scene_id != spec["scene_id"] or (
                raw["frame_ids"][0].tolist() != spec["context"] + spec["target"]
            ):
                raise RuntimeError("Loaded scene/frame identity differs from the expected split")
            if "lidar_points" in raw:
                raise RuntimeError("Point cloud unexpectedly loaded during RGB evaluation")
            data = {key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in raw.items()}
            model_input, _ = split_data(data, opt)
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
            reconstruction = model.forward_reconstruction(model_input)
            rendered = model.render_reconstruction(reconstruction, model_input.decoder)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_start
            gaussians = reconstruction.gaussians
            if gaussians.shape != (1, 262144, 14) or not torch.isfinite(gaussians).all():
                raise RuntimeError("Unexpected or non-finite Gaussian output")
            prediction = rendered["images_pred"][0].float().clamp(0, 1)
            truth = data["images_output"][0].float()
            mse = (prediction - truth).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
            psnr = -10 * mse.log10()
            lpips = lpips_metric(truth, prediction, normalize=True).flatten()

            raw_ctx_c2w = torch.from_numpy(
                provider.dataset._scene_meta_cache[scene_id]["c2ws"][spec["context"]].copy()
            )
            sigma_m, _ = render_ray_thickness_map(
                gaussians, data["cam_view"], data["intrinsics"],
                opt.img_size, float(opt.znear), float(opt.zfar), float(provider.scene_scale),
            )
            mesh_ply = provider.dataset.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
            rc_scene = build_raycasting_scene(mesh_ply, raw_ctx_c2w, float(provider.scene_scale))
            d_gt, n_gt, gt_ok = raycast_gt_depth_and_normals(
                rc_scene, data["cam_view"][0], data["intrinsics"][0], opt.img_size
            )
            frame_geo_list, scene_geo = compute_scene_geometry_metrics(
                rendered["depths_pred"][0], rendered["alphas_pred"][0], sigma_m[0],
                d_gt, n_gt, gt_ok, data["cam_view"][0], data["intrinsics"][0],
                scene_scale=float(provider.scene_scale), nn_device=device,
            )

            frames = []
            for position, frame_id in enumerate(spec["target"]):
                frames.append({
                    "scene_id": scene_id, "frame_id": frame_id, "target_position": position,
                    "is_context_view": False, "psnr": float(psnr[position]),
                    "ssim": float(fused_ssim(
                        prediction[position:position + 1], truth[position:position + 1],
                        train=False,
                    )),
                    "lpips": float(lpips[position]),
                    **frame_geo_list[position],
                })
            row = {
                "scene_id": scene_id,
                "context_frames": " ".join(map(str, spec["context"])),
                "target_frames": " ".join(map(str, spec["target"])),
                "num_target_frames": len(frames), **mean_metrics(frames),
                **scene_geo,
                "forward_seconds": forward_seconds,
            }
            if not args.demo_only:
                writer.writerows(frames)
                stream.flush()
            rows.append(row)
            if scene_id == demo_scene:
                export_demo_bundle(
                    demo_dir=demo_output,
                    method_name=method_name,
                    scene_id=scene_id,
                    gaussians=gaussians,
                    cam_to_world_input=data["cam_to_world_input"][0],
                    cam_view=data["cam_view"][0],
                    cam_view_input=data["cam_view_input"][0],
                    intrinsics_input=data["intrinsics_input"][0],
                    raw_context_c2w=raw_ctx_c2w,
                    img_size=opt.img_size,
                    znear=float(opt.znear),
                    scene_scale=float(provider.scene_scale),
                    ckpt_sha=ckpt_sha,
                    row=row,
                    frames=frames,
                    context_ids=spec["context"],
                    target_ids=spec["target"],
                    dataset_root=provider.dataset.root_path,
                    truth=truth,
                    prediction=prediction,
                )
            print(f"[{step_idx + 1:02d}/{len(indices)}] {scene_id}: "
                  f"PSNR={row['psnr']:.4f} SSIM={row['ssim']:.5f} "
                  f"LPIPS={row['lpips']:.5f} AbsRel={scene_geo['depth_absrel']:.4f} "
                  f"F@3cm={scene_geo['surf_fscore_3cm']:.4f}", flush=True)
    if args.demo_only:
        return
    if len(rows) != 50:
        raise RuntimeError("Incomplete evaluation")
    for r in rows:
        for k in GEOMETRY_SCENE_METRICS:
            if not math.isfinite(float(r[k])):
                raise RuntimeError(f"Non-finite geometry metric {k} in scene {r['scene_id']}")
    scale_ratios = [r["depth_scale_ratio_med"] for r in rows]
    median_scale_ratio = statistics.median(scale_ratios)
    if not (0.90 <= median_scale_ratio <= 1.10):
        raise RuntimeError(f"Scale alignment gate failed: median D_pred/D_gt = {median_scale_ratio:.4f} outside [0.90, 1.10]")
    geo_means = {
        k: statistics.mean(r[k] for r in rows)
        for k in GEOMETRY_SCENE_METRICS
    }
    summary = {
        "name": output.name, **mean_metrics(rows), **geo_means, "num_scenes": 50,
        "num_target_frames": 400, "num_input_views": 6, "num_target_views": 8,
        "window_size": args.window_size, "image_size": list(opt.img_size),
        "precision": "float32", "tf32": False, "point_cloud_input": False,
        "optimization_steps": 0,
        "geometry_metrics_status": GEOMETRY_COMPUTED_STATUS,
        "geometry_alignment_median_ratio": median_scale_ratio,
        "forward_seconds_mean": sum(row["forward_seconds"] for row in rows) / len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib_including_lpips": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint_sha256": ckpt_sha, "gpu": gpu,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_csv_atomic(output / "scene_metrics.csv", rows)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "EXP_ScanNetPP" / NAME)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--model-type", default="tokengs")
    parser.add_argument("--demo-output", type=Path)
    parser.add_argument("--demo-scene", default=DEFAULT_DEMO_SCENE)
    parser.add_argument("--demo-only", action="store_true")
    parser.add_argument("--ply-name")
    parser.add_argument("--from-artifact", type=Path)
    parser.add_argument("--backfill-geometry-from-reconstructions", type=Path)
    parser.add_argument("--render-strip-png", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/jms/xveyan/data/4dgs/scannetpp"))
    parser.add_argument("--window-size", type=int, default=18)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
