"""Training-free checks of whether input-view SfM anchors carry geometry the model does not already have.

On the 50 ScanNet++ validation windows (6 inputs, 8 novel targets, FP32, no TF32) and for each given checkpoint:

A. At every anchor observed in an input view: the anchor's depth error vs. the model's rendered-depth error at the
   same pixel (both against the laser mesh).
B. A non-learned propagation: rendered log-depth is corrected by a 3D-distance-weighted average of anchor residuals,
       log D'(p) = log D(p) + sum_a w_a(p) s_a / (sum_a w_a(p) + exp(-2)),  w_a(p) = exp(-|X(p) - a|^2 / 2 sigma^2),
   where X(p) is the model's back-projected point and s_a = log z_a - log D(u_a) is the anchor residual in that view
   (input views: anchors observed there; targets: anchors within 25 % of the rendered depth). Also a per-view
   global scale (median residual).

Depth errors are stratified by the laser surface point's distance to the nearest anchor, separately for input and
target views. Pixel-pooled (not the protocol's per-frame mean).
"""

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

import geometrygs  # noqa: F401
from geometrygs.data import SfmProvider, split_data_with_sfm
from geometrygs.model import anchor_quality_kwargs
from geometrygs.options import config_defaults
from geometrygs.sfm import KnownPoseSfM, SfmConfig, select_anchors
from scannetpp_evaluation_protocol import ALPHA_VALID_THRESHOLD, build_raycasting_scene, raycast_gt_depth_and_normals
from tokengs.models import model_registry

BINS_CM = [0, 5, 10, 20, 50, float("inf")]
BIN_NAMES = ["0-5", "5-10", "10-20", "20-50", ">50"]
SIGMAS_CM = [5, 10, 20, 40]
METHODS = ["model", "global_scale"] + [f"local_{s}cm" for s in SIGMAS_CM]


def load_model(path: Path, opt, device):
    state = load_file(str(path), device="cpu")
    if any(k.startswith("activation_head.pos_mlp.") for k in state):
        opt = dataclasses.replace(opt, readout_pos_mlp=True,
                                  readout_pos_mlp_hidden=int(state["activation_head.pos_mlp.1.weight"].shape[0]))
    if "activation_head.attr_mlp.5.weight" in state:
        n = state["activation_head.attr_mlp.5.weight"].shape[0] // (int(opt.dec_patch_size) ** 2)
        opt = dataclasses.replace(opt, readout_attr_mlp=True, readout_attr_mlp_channels="opacity" if n == 1 else "all")
    if "activation_head.surfel_rn" in state:
        opt = dataclasses.replace(opt, readout_surfel=True)
    if "activation_head.sh_head.weight" in state:
        k_rest = state["activation_head.sh_head.weight"].shape[0] // (opt.dec_patch_size ** 2 * 3)
        opt = dataclasses.replace(opt, readout_sh_degree=int(round((k_rest + 1) ** 0.5)) - 1)
    model = model_registry[opt.model_type](opt).eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.float32)


def pixel_rays(intr: torch.Tensor, H: int, W: int, device) -> torch.Tensor:
    u, v = torch.meshgrid(torch.arange(W, device=device) + 0.5, torch.arange(H, device=device) + 0.5, indexing="xy")
    fx, fy, cx, cy = intr.tolist()
    return torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], -1)  # [H, W, 3], z = 1


def project(points: torch.Tensor, w2c: torch.Tensor, intr: torch.Tensor):
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    fx, fy, cx, cy = intr.tolist()
    z = pc[:, 2]
    return torch.stack([fx * pc[:, 0] / z + cx, fy * pc[:, 1] / z + cy], -1), z


class Accumulator:
    def __init__(self):
        self.n = {}
        self.err = {}
        self.bad = {}

    def add(self, key, relerr: torch.Tensor):
        self.n[key] = self.n.get(key, 0) + int(relerr.numel())
        self.err[key] = self.err.get(key, 0.0) + float(relerr.sum())
        self.bad[key] = self.bad.get(key, 0) + int((relerr > 0.05).sum())


def main():
    tokengs_root = Path(__file__).resolve().parents[1].parent / "TokenGS"
    if not tokengs_root.exists() and (Path(__file__).resolve().parents[1].parent / "tokengs").exists():
        tokengs_root = Path(__file__).resolve().parents[1].parent / "tokengs"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", default=[
        f"official={tokengs_root}/checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors",
        f"sft_latenttail_rgb={tokengs_root}/EXP_ScanNetPP/sft_latenttail_rgb/model.safetensors",
        f"sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20={tokengs_root}/EXP_ScanNetPP/"
        "sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20/model.safetensors",
    ])
    ap.add_argument("--output", type=Path, default=Path("EXP/anchor_propagation_val6v"))
    ap.add_argument("--max-scenes", type=int, default=None)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    opt = config_defaults["eval_sfm_anchor"].evolve(
        model_type="tokengs", evaluating=True, use_ttt_for_eval=False, ttt_n_steps=0, use_geo_losses=False,
        load_lidar_pc=False, use_input_supervision=False, mixed_precision="no", num_workers=0,
        dataset_kwargs={"window_size": 18})
    provider = SfmProvider("scannetpp_val_scaled_0.3", opt, training=False)
    scale = float(provider.scene_scale)
    H, W = opt.img_size
    models = {}
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        models[name] = load_model(Path(path), opt, device)
    sfm = KnownPoseSfM(SfmConfig(num_features=opt.sfm_num_features, match_ratio=opt.sfm_match_ratio,
                                 max_epipolar_px=opt.sfm_max_epipolar_px, max_reproj_px=opt.sfm_max_reproj_px,
                                 min_tri_angle_deg=opt.sfm_min_tri_angle_deg, min_track=opt.sfm_min_track,
                                 z_near=opt.znear, z_far=opt.zfar))
    quality = anchor_quality_kwargs(opt)
    acc = {m: {vs: Accumulator() for vs in ("input", "target")} for m in models}
    anchor_rows = {m: [] for m in models}
    scene_rows = []
    n_scenes = len(provider) if args.max_scenes is None else min(args.max_scenes, len(provider))
    t0 = time.time()
    with torch.inference_mode():
        for idx in range(n_scenes):
            raw = provider[idx]
            scene_id = raw["scene_id"]
            batch = {k: v[None].to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
            mi, _ = split_data_with_sfm(batch, opt)
            V = opt.num_input_views
            rec = sfm.reconstruct(mi.sfm["images"][0], mi.encoder.cam_to_world_input[0], mi.sfm["intrinsics"][0])
            sel = select_anchors(rec, opt.sfm_num_anchors, **quality)
            anchors = rec["points"][sel].float()
            obs = rec["obs_mask"][sel]  # [A, V]
            ctx = raw["frame_ids"][:V].tolist()
            raw_c2w = torch.from_numpy(provider.dataset._scene_meta_cache[scene_id]["c2ws"][ctx].copy())
            mesh = provider.dataset.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
            rc = build_raycasting_scene(mesh, raw_c2w, scale)
            views = {
                "input": (batch["cam_view_input"][0], batch["intrinsics_input"][0]),
                "target": (batch["cam_view"][0], batch["intrinsics"][0]),
            }
            gt = {}
            for vs, (cv, intr) in views.items():
                d_gt, _, ok = raycast_gt_depth_and_normals(rc, cv, intr, opt.img_size)
                d_gt, ok = d_gt.to(device), ok.to(device)
                bins = torch.full(ok.shape, len(BIN_NAMES) - 1, dtype=torch.long, device=device)
                for i in range(cv.shape[0]):
                    c2w = torch.linalg.inv(cv[i].T)
                    x = (pixel_rays(intr[i], H, W, device) * d_gt[i][..., None]) @ c2w[:3, :3].T + c2w[:3, 3]
                    if len(anchors):
                        dist = torch.cdist(x[ok[i]], anchors).min(1).values / scale * 100
                        bins[i][ok[i]] = torch.bucketize(dist, torch.tensor(BINS_CM[1:-1], device=device), right=True)
                gt[vs] = (d_gt, ok, bins)
            scene_rows.append({"scene_id": scene_id, "num_points": int(rec["points"].shape[0]), "num_anchors": int(len(anchors)),
                               **{f"input_px_{b}": float(((gt["input"][2] == k) & gt["input"][1]).sum() / gt["input"][1].sum().clamp_min(1))
                                  for k, b in enumerate(BIN_NAMES)}})

            for mname, model in models.items():
                recon = model.forward_reconstruction(mi)
                for vs, (cv, intr) in views.items():
                    out = model.render_reconstruction(recon, SimpleNamespace(cam_view=cv[None], intrinsics=intr[None]))
                    D = out["depths_pred"][0].float().reshape(-1, H, W)
                    Al = out["alphas_pred"][0].float().reshape(-1, H, W)
                    d_gt, ok, bins = gt[vs]
                    for i in range(cv.shape[0]):
                        m = ok[i] & (Al[i] > ALPHA_VALID_THRESHOLD) & (D[i] > 0)
                        if not m.any():
                            continue
                        w2c = cv[i].T
                        c2w = torch.linalg.inv(w2c)
                        logD = D[i].clamp_min(1e-6).log()
                        preds = {"model": D[i]}
                        s = torch.zeros(0, device=device)
                        if len(anchors):
                            uv, z = project(anchors, w2c, intr[i])
                            px = uv.floor().long()
                            inside = (z > 0) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
                            pxc = px.clamp(min=torch.zeros(2, dtype=torch.long, device=device),
                                           max=torch.tensor([W - 1, H - 1], device=device))
                            d_at = D[i][pxc[:, 1], pxc[:, 0]]
                            a_at = Al[i][pxc[:, 1], pxc[:, 0]]
                            res = z.clamp_min(1e-6).log() - d_at.clamp_min(1e-6).log()
                            if vs == "input":
                                use = inside & obs[:, i] & (a_at > ALPHA_VALID_THRESHOLD)
                                g_at = d_gt[i][pxc[:, 1], pxc[:, 0]]
                                ok_at = ok[i][pxc[:, 1], pxc[:, 0]] & use
                                if ok_at.any():
                                    anchor_rows[mname].append(torch.stack([
                                        ((z - g_at).abs() / g_at)[ok_at], ((d_at - g_at).abs() / g_at)[ok_at]], 1).cpu())
                            else:
                                use = inside & (a_at > ALPHA_VALID_THRESHOLD) & (res.abs() < np.log(1.25))
                            s, a_use = res[use], anchors[use]
                        if s.numel():
                            preds["global_scale"] = D[i] * s.median().exp()
                            x = (pixel_rays(intr[i], H, W, device) * D[i][..., None]) @ c2w[:3, :3].T + c2w[:3, 3]
                            d2 = torch.cdist(x[m], a_use).square()
                            for sig in SIGMAS_CM:
                                w = torch.exp(-d2 / (2 * (sig / 100 * scale) ** 2))
                                corr = (w * s).sum(1) / (w.sum(1) + np.exp(-2.0))
                                Dc = D[i].clone()
                                Dc[m] = (logD[m] + corr).exp()
                                preds[f"local_{sig}cm"] = Dc
                        else:
                            preds.update({k: D[i] for k in METHODS if k != "model"})
                        g = d_gt[i][m]
                        b = bins[i][m]
                        for meth, P in preds.items():
                            rel = (P[m] - g).abs() / g
                            acc[mname][vs].add((meth, "all"), rel)
                            for k, bn in enumerate(BIN_NAMES):
                                acc[mname][vs].add((meth, bn), rel[b == k])
            print(f"[{idx + 1:02d}/{n_scenes}] {scene_id}: points={scene_rows[-1]['num_points']} "
                  f"anchors={scene_rows[-1]['num_anchors']} ({time.time() - t0:.0f}s)", flush=True)

    summary = {"bins_cm": BIN_NAMES, "sigmas_cm": SIGMAS_CM, "num_scenes": n_scenes, "models": {}}
    rows = []
    for mname in models:
        a = torch.cat(anchor_rows[mname]) if anchor_rows[mname] else torch.zeros(0, 2)
        ms = {"anchor_check": {
            "n_observations": int(a.shape[0]),
            "anchor_relerr_median": float(a[:, 0].median()) if len(a) else None,
            "model_relerr_median_at_anchor": float(a[:, 1].median()) if len(a) else None,
            "model_relerr_mean_at_anchor": float(a[:, 1].mean()) if len(a) else None,
            "anchor_better_fraction": float((a[:, 0] < a[:, 1]).float().mean()) if len(a) else None,
            "model_bad5_fraction_at_anchor": float((a[:, 1] > 0.05).float().mean()) if len(a) else None,
            "anchor_bad5_fraction": float((a[:, 0] > 0.05).float().mean()) if len(a) else None,
        }}
        for vs in ("input", "target"):
            A = acc[mname][vs]
            total = A.n.get(("model", "all"), 1)
            for bn in ["all", *BIN_NAMES]:
                for meth in METHODS:
                    n = A.n.get((meth, bn), 0)
                    row = {"model": mname, "views": vs, "bin_cm": bn, "method": meth, "pixels": n,
                           "pixel_fraction": n / total, "absrel": A.err.get((meth, bn), 0) / max(n, 1),
                           "bad5_fraction": A.bad.get((meth, bn), 0) / max(n, 1)}
                    rows.append(row)
                    ms.setdefault(vs, {}).setdefault(bn, {})[meth] = {k: row[k] for k in ("pixel_fraction", "absrel", "bad5_fraction")}
        summary["models"][mname] = ms
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(args.output / "bins.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(args.output / "scene_metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(scene_rows[0]))
        wr.writeheader()
        wr.writerows(scene_rows)
    for mname, ms in summary["models"].items():
        print(mname, json.dumps(ms["anchor_check"]))
        for vs in ("input", "target"):
            print(f"  {vs}: " + " | ".join(
                f"{bn}: " + " ".join(f"{meth}={ms[vs][bn][meth]['absrel']:.4f}" for meth in ("model", "global_scale", "local_10cm", "local_20cm"))
                for bn in ["all", *BIN_NAMES]))


if __name__ == "__main__":
    main()
