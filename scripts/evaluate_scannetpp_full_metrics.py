import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from lpips import LPIPS
from plyfile import PlyData
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation_statistics as es
from tokengs.data.provider import Provider
from tokengs.models import model_registry, split_data
from tokengs.options import config_defaults
from tokengs.utils.training import freeze_model_parameters


def chunked_min_dist(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    mins = []
    for i in range(0, a.shape[0], chunk):
        mins.append(torch.cdist(a[i:i + chunk], b).min(dim=1).values)
    return torch.cat(mins, dim=0)


def fast_read_ply_vertices(ply_path: Path, stride: int = 4) -> np.ndarray:
    with open(ply_path, "rb") as f:
        n_verts = 0
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(b"element vertex"):
                n_verts = int(line.split()[-1])
            elif line.strip() == b"end_header":
                break
        v_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        verts = np.fromfile(f, dtype=v_dtype, count=n_verts)
    return np.stack([verts["x"][::stride], verts["y"][::stride], verts["z"][::stride]], axis=1)


def load_normalized_frustum_lidar_points(
    prov: Provider,
    sc_root: Path,
    scene_id: str,
    fids: list[int],
    raw_item: dict,
    img_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    H, W = img_size
    mesh_ply = sc_root / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    raw_c2ws = torch.from_numpy(prov.dataset._scene_meta_cache[scene_id]["c2ws"][fids]).float()
    in_c2ws = raw_c2ws[:6]
    pos_avg = in_c2ws[:, :3, 3].mean(0)
    fwd_avg = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=0)
    down_avg = in_c2ws[:, :3, 1].mean(0)
    down_avg = F.normalize(down_avg - down_avg.dot(fwd_avg) * fwd_avg, dim=0)
    right_avg = torch.linalg.cross(down_avg, fwd_avg)
    mat = torch.eye(4)
    mat[:3, 0], mat[:3, 1], mat[:3, 2], mat[:3, 3] = right_avg, down_avg, fwd_avg, pos_avg
    mat_inv = torch.linalg.inv(mat)

    pts_w = torch.from_numpy(fast_read_ply_vertices(mesh_ply, stride=4)).float()
    pts_n = (pts_w @ mat_inv[:3, :3].T + mat_inv[:3, 3].unsqueeze(0)) * float(prov.scene_scale)

    w2c_in = raw_item["cam_view_input"][0].float().transpose(-1, -2)
    intr_in = raw_item["intrinsics_input"][0].float()
    vis = torch.zeros(pts_n.shape[0], dtype=torch.bool)
    for v_i in range(6):
        xc = pts_n @ w2c_in[v_i, :3, :3].T + w2c_in[v_i, :3, 3]
        z = xc[:, 2]
        fx, fy, cx, cy = intr_in[v_i]
        u = fx * xc[:, 0] / z.clamp_min(1e-4) + cx
        v = fy * xc[:, 1] / z.clamp_min(1e-4) + cy
        vis |= (z > 0.05) & (z < 3.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pts_f = pts_n[vis]
    if pts_f.shape[0] > 50000:
        idx_s = torch.linspace(0, pts_f.shape[0] - 1, 50000).long()
        pts_f = pts_f[idx_s]
    return pts_f.to(device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TokenGS checkpoint on the 50 ScanNet++ val scenes with full photometric + 3D geometry metrics.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors")
    parser.add_argument("--model-type", type=str, default="tokengs")
    parser.add_argument("--lidar-sft-mode", type=str, default="scene_latent_probe", choices=["scene_latent_probe", "surface_self_attn", "fused_full_decoder"])
    parser.add_argument("--window-size", type=int, default=18)
    parser.add_argument("--ttt-mode", type=str, default="none", choices=["none", "scene", "scene_token"])
    parser.add_argument("--ttt-steps", type=int, default=50)
    parser.add_argument("--scene-lr", type=float, default=1e-2)
    parser.add_argument("--token-lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    opt = dataclasses.replace(
        config_defaults["eval_scannetpp_latent_6view_ssim"],
        model_type=args.model_type,
        lidar_sft_mode=args.lidar_sft_mode,
        evaluating=True,
        use_input_supervision=False,
        dataset_kwargs={"window_size": args.window_size},
    )
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    dl = torch.utils.data.DataLoader(prov, batch_size=1, shuffle=False)

    model = model_registry[opt.model_type](opt).to(device).eval()
    model.load_state_dict(load_file(args.checkpoint), strict=True)

    lpips_fn = LPIPS(net="vgg").to(device).eval()
    lpips_fn.requires_grad_(False)
    H, W = opt.img_size
    sc_root = Path("/home/jms/xveyan/data/4dgs/scannetpp")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for idx_s, raw in enumerate(dl):
        sid = raw["scene_id"][0]
        fids = raw["frame_ids"][0].tolist()
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}

        if args.ttt_mode == "none":
            with torch.no_grad():
                out = model(data)
                gs = out["gaussians"]
                preds = out["images_pred"][0].clamp(0, 1).float()
                gts = data["images_output"][0].float()
                depths_tgt = out["depths_pred"]
                images_pred_all = out["images_pred"]
        else:
            model_input, _ = split_data(data, opt)
            model_input_ttt, supervision_ttt = model_input.to_ttt()
            with torch.no_grad():
                sl0 = model.forward_encoder_to_scene_latents(model_input_ttt.encoder)
                gt0 = model.get_gs_tokens(batch_size=1)
            with freeze_model_parameters(model), torch.set_grad_enabled(True):
                sl = sl0.detach().clone().requires_grad_(True)
                if args.ttt_mode == "scene_token":
                    gt = gt0.detach().clone().requires_grad_(True)
                    optim = torch.optim.Adam([
                        {"params": [sl], "lr": args.scene_lr},
                        {"params": [gt], "lr": args.token_lr},
                    ])
                else:
                    gt = gt0.detach()
                    optim = torch.optim.Adam([{"params": [sl], "lr": args.scene_lr}])
                for _ in range(args.ttt_steps):
                    optim.zero_grad()
                    enc_lat = model.encoder_latent_from_scene_latents(sl)
                    gs_step = model.forward_decoder(enc_lat, model_input_ttt.decoder, gs_tokens=gt)
                    rec = model._reconstruction_from_gaussians(gs_step)
                    ld = model.compute_loss(rec, model_input_ttt.decoder, supervision_ttt)
                    ld["loss"].backward()
                    optim.step()
            with torch.no_grad():
                enc_lat = model.encoder_latent_from_scene_latents(sl.detach())
                gs = model.forward_decoder(enc_lat, model_input.decoder, gs_tokens=gt.detach())
                ren = model.render_reconstruction(model._reconstruction_from_gaussians(gs), model_input.decoder)
                images_pred_all = ren["images_pred"]
                preds = images_pred_all[0].clamp(0, 1).float()
                gts = data["images_output"][0].float()
                depths_tgt = ren["depths_pred"]

        with torch.no_grad():
            mse = (preds - gts).square().mean(dim=(-1, -2, -3)).clamp_min(1e-12)
            psnr_val = float((-10.0 * torch.log10(mse)).mean())
            ssim_val = float(fused_ssim(preds, gts, train=False).mean())
            lpips_val = float(lpips_fn(gts, preds, normalize=True).mean())

            far_mask = torch.ones(preds.shape[0], dtype=torch.bool, device=device)
            feat_mask = torch.ones((preds.shape[0], H, W), dtype=torch.bool, device=device)
            surf = es.surface_and_region_stats(
                model, gs, images_pred_all, data["images_output"], depths_tgt,
                data["cam_view"], data["intrinsics"], data["cam_view_input"], data["intrinsics_input"],
                far_mask, feat_mask, (H, W),
            )

            g0 = gs[0]
            vis05 = g0[:, 3] >= 0.05
            xyz05 = g0[vis05, 0:3]
            sc05 = g0[vis05, 4:7].sort(dim=-1, descending=True).values
            needles_gt3 = float(((sc05[:, 0] / sc05[:, 1].clamp_min(1e-6)) > 3.0).float().mean()) * 100
            needles_gt2 = float(((sc05[:, 0] / sc05[:, 1].clamp_min(1e-6)) > 2.0).float().mean()) * 100
            needle_sum, _ = es.needle_stats(g0.cpu(), raw["cam_to_world_input"][0], raw["intrinsics_input"][0], (H, W), opt.znear, detailed=False)

            pts_gt = load_normalized_frustum_lidar_points(prov, sc_root, sid, fids, raw, (H, W), device)
            d_to_gt = chunked_min_dist(xyz05, pts_gt)
            dist_p50 = float(d_to_gt.median())
            dist_p75 = float(d_to_gt.quantile(0.75))
            within_3cm = float((d_to_gt < 0.03).float().mean()) * 100
            within_5cm = float((d_to_gt < 0.05).float().mean()) * 100

            row = {
                "scene_id": sid,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
                "warp_psnr": float(surf["warp_psnr_dt5"]),
                "mv_cycle_err_pct": float(surf["mv_cycle_err_dt5"]) * 100,
                "ray_thick_m": float(surf["ray_thickness_med"]),
                "alive_op05_count": int(vis05.sum()),
                "needles_gt3_pct": needles_gt3,
                "needles_gt2_pct": needles_gt2,
                "blade_cos_ray_p50": float(needle_sum["cos_long_axis_ray_p50_weighted"]),
                "blade_along_ray_gt09_pct": float(needle_sum["long_axis_along_ray_gt0.9_weighted"]) * 100,
                "lidar_dist_p50_m": dist_p50,
                "lidar_dist_p75_m": dist_p75,
                "lidar_within_3cm_pct": within_3cm,
                "lidar_within_5cm_pct": within_5cm,
            }
            rows.append(row)
            if (idx_s + 1) % 10 == 0 or (idx_s + 1) == len(dl):
                print(
                    f"[{idx_s+1:2d}/50] {sid}: PSNR={psnr_val:.3f}, SSIM={ssim_val:.3f}, LPIPS={lpips_val:.3f}, "
                    f"RayThick={row['ray_thick_m']:.4f}m, Needles={needles_gt3:.1f}%, "
                    f"BladeCos={row['blade_cos_ray_p50']:.3f}, LiDAR_p50={dist_p50:.4f}m"
                )

    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "scene_id"}
    summary["num_scenes"] = len(rows)
    summary["window_size"] = args.window_size
    summary["checkpoint"] = args.checkpoint
    summary["elapsed_sec"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "scene_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== FULL METRICS SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:28s}: {v:.4f}")
        else:
            print(f"  {k:28s}: {v}")


if __name__ == "__main__":
    main()
