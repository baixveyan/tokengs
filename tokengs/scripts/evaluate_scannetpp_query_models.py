"""ScanNet++ 50-scene RGB-only zero-shot evaluation for the paper-architecture query models loaded with the LocusGS code.

Same protocol as scripts/evaluate_scannetpp.py (official 50 validation scenes, 6 inputs, 8 interpolation targets in the
18-frame window, 256x448, FP32, TF32 off, laser mesh G1-G4, scale gate) except that these DL3DV models were trained with
first_cam camera normalization, so the cameras and the laser mesh are normalized by the first context camera
(scene scale 0.3 as for the release model).  No Demo bundle.

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/evaluate_scannetpp_query_models.py --model locusgs_finetune_4v
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from fused_ssim import fused_ssim
from lpips import LPIPS

from diag_dl3dv import MODELS, load_any
from evaluate_scannetpp import ROOT, frame_splits, idle_gpu
from scannetpp_evaluation_protocol import (
    GEOMETRY_COMPUTED_STATUS, GEOMETRY_FRAME_METRICS, GEOMETRY_SCENE_METRICS, METRICS,
    compute_scene_geometry_metrics, mean_metrics, raycast_gt_depth_and_normals, render_ray_thickness_map,
    write_csv_atomic, write_json,
)
from tokengs.data.provider import Provider
from tokengs.options import config_defaults

NAMES = {"locusgs_finetune_4v": "baseline_locusgs_dl3dv_4v", "tokengs_paper_4v": "baseline_tokengs_paper_dl3dv_4v"}


def build_raycasting_scene_first_cam(mesh_ply: Path, raw_context_c2w: torch.Tensor, scene_scale: float):
    import open3d as o3d

    mesh = o3d.t.io.read_triangle_mesh(str(mesh_ply))
    mat_inv = torch.linalg.inv(raw_context_c2w[0].double()).float()
    v_w = torch.from_numpy(mesh.vertex.positions.numpy()).float()
    v_n = (v_w @ mat_inv[:3, :3].T + mat_inv[:3, 3].unsqueeze(0)) * float(scene_scale)
    mesh.vertex.positions = o3d.core.Tensor.from_numpy(v_n.numpy())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(NAMES), required=True)
    ap.add_argument("--dataset-root", type=Path, default=Path("/home/jms/xveyan/data/4dgs/scannetpp"))
    args = ap.parse_args()
    output = ROOT / "EXP_ScanNetPP" / NAMES[args.model]
    if output.exists() and any(p.name != "run.log" and not p.name.startswith(".") for p in output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing results in {output}")
    if not (Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth").is_file():
        raise RuntimeError("Local LPIPS backbone missing; no automatic download is allowed")
    gpu = idle_gpu()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    model, lopt, _dopt, _kind = load_any(args.model, device)
    from locusgs.models.input_types import split_data as locus_split

    opt = dataclasses.replace(
        config_defaults["eval_scannetpp_latent_6view_ssim"], camera_normalization_method="first_cam",
        evaluating=True, use_ttt_for_eval=False, ttt_n_steps=0, use_geo_losses=False, load_lidar_pc=False,
        use_input_supervision=False, mixed_precision="no", num_workers=0,
        dataset_kwargs={"window_size": 18, "root_path": str(args.dataset_root)})
    if tuple(opt.img_size) != tuple(lopt.img_size) or opt.num_input_views != lopt.num_input_views:
        raise RuntimeError("data / model options disagree")
    provider = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    splits = frame_splits(provider)
    output.mkdir(parents=True, exist_ok=True)
    lpips_metric = LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    rows, started = [], time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    frame_fields = ["scene_id", "frame_id", "target_position", "is_context_view", *METRICS, *GEOMETRY_FRAME_METRICS]
    print(f"GPU {gpu['index']}: {args.model} -> {output}, 50 scenes, FP32, no TF32, first_cam", flush=True)
    with (output / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=frame_fields)
        writer.writeheader()
        for idx, spec in enumerate(splits):
            raw = provider[idx]
            scene_id = raw["scene_id"]
            if scene_id != spec["scene_id"] or raw["frame_ids"].tolist() != spec["context"] + spec["target"]:
                raise RuntimeError("Loaded scene/frame identity differs from the expected split")
            data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
            mi, _ = locus_split(data, lopt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gaussians = model.forward_decoder(model.forward_encoder(mi.encoder), mi.decoder)
            rendered = model.render_gaussians(gaussians, mi.decoder)
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - t0
            if gaussians.shape != (1, 262144, 14) or not torch.isfinite(gaussians).all():
                raise RuntimeError("Unexpected or non-finite Gaussian output")
            prediction = rendered["images_pred"][0].float().clamp(0, 1)
            truth = data["images_output"][0].float()
            psnr = -10 * (prediction - truth).square().mean(dim=(1, 2, 3)).clamp_min(1e-12).log10()
            lpips = lpips_metric(truth, prediction, normalize=True).flatten()
            raw_ctx_c2w = torch.from_numpy(provider.dataset._scene_meta_cache[scene_id]["c2ws"][spec["context"]].copy())
            sigma_m, _ = render_ray_thickness_map(gaussians, data["cam_view"], data["intrinsics"], opt.img_size,
                                                  float(opt.znear), float(opt.zfar), float(provider.scene_scale))
            rc_scene = build_raycasting_scene_first_cam(
                provider.dataset.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply", raw_ctx_c2w,
                float(provider.scene_scale))
            d_gt, n_gt, gt_ok = raycast_gt_depth_and_normals(rc_scene, data["cam_view"][0], data["intrinsics"][0], opt.img_size)
            frame_geo, scene_geo = compute_scene_geometry_metrics(
                rendered["depths_pred"][0], rendered["alphas_pred"][0], sigma_m[0], d_gt, n_gt, gt_ok,
                data["cam_view"][0], data["intrinsics"][0], scene_scale=float(provider.scene_scale), nn_device=device,
                gaussians=gaussians[0], rc_scene=rc_scene)
            frames = [{"scene_id": scene_id, "frame_id": fid, "target_position": pos, "is_context_view": False,
                       "psnr": float(psnr[pos]),
                       "ssim": float(fused_ssim(prediction[pos:pos + 1], truth[pos:pos + 1], train=False)),
                       "lpips": float(lpips[pos]), **frame_geo[pos]} for pos, fid in enumerate(spec["target"])]
            row = {"scene_id": scene_id, "context_frames": " ".join(map(str, spec["context"])),
                   "target_frames": " ".join(map(str, spec["target"])), "num_target_frames": len(frames),
                   **mean_metrics(frames), **scene_geo, "forward_seconds": forward_seconds}
            writer.writerows(frames)
            stream.flush()
            rows.append(row)
            print(f"[{idx + 1:02d}/50] {scene_id}: PSNR={row['psnr']:.4f} SSIM={row['ssim']:.5f} LPIPS={row['lpips']:.5f} "
                  f"AbsRel={scene_geo['depth_absrel']:.4f} F@3cm={scene_geo['surf_fscore_3cm']:.4f}", flush=True)
    if len(rows) != 50:
        raise RuntimeError("Incomplete evaluation")
    for r in rows:
        for k in GEOMETRY_SCENE_METRICS:
            if not math.isfinite(float(r[k])):
                raise RuntimeError(f"Non-finite geometry metric {k} in scene {r['scene_id']}")
    median_scale_ratio = statistics.median(r["depth_scale_ratio_med"] for r in rows)
    if not 0.90 <= median_scale_ratio <= 1.10:
        raise RuntimeError(f"Scale alignment gate failed: median D_pred/D_gt = {median_scale_ratio:.4f}")
    summary = {
        "name": output.name, **mean_metrics(rows), **{k: statistics.mean(r[k] for r in rows) for k in GEOMETRY_SCENE_METRICS},
        "num_scenes": 50, "target_mode": "interp", "num_target_frames": 400, "num_input_views": 6, "num_target_views": 8,
        "window_size": 18, "image_size": list(opt.img_size), "precision": "float32", "tf32": False,
        "point_cloud_input": False, "optimization_steps": 0, "camera_normalization": "first_cam",
        "model": args.model, "code": "LocusGS repository (4DGS/LocusGS) model classes",
        "geometry_metrics_status": GEOMETRY_COMPUTED_STATUS, "geometry_alignment_median_ratio": median_scale_ratio,
        "scale_gate_enforced": True, "forward_seconds_mean": statistics.mean(r["forward_seconds"] for r in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib_including_lpips": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint": str(MODELS[args.model]["ckpt"]), "checkpoint_sha256": MODELS[args.model]["sha"], "gpu": gpu,
        "completed_at": datetime.now(timezone.utc).isoformat()}
    write_csv_atomic(output / "scene_metrics.csv", rows)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
