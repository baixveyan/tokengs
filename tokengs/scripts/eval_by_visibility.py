"""Evaluation split by input visibility for the ScanNet++ 50-scene evaluation (adds metrics; the official protocol is untouched).

dump   (1 idle GPU): per scene, store per-pixel squared RGB error, rendered alpha and depth of one model, plus (once per
       target mode) GT mesh depth, how many of the 6 input views see each GT surface point (mesh-occlusion tested by
       casting a ray from every input camera centre to the point) and its distance to the nearest seeing input camera.
       Cache: /mnt/disk/xveyan/workspace_scannetpp/eval_by_visibility/<mode>/{_gt,<experiment>}/<scene>.npz
report (CPU): pooled metrics per observability bucket, hole-inclusive depth accuracy, Completeness split into GT
       surface seen / unseen by the inputs, and a pixel-level decomposition of the PSNR gap between two models.
       Output: EXP_ScanNetPP/eval_by_visibility/eval_by_visibility.{json,md}

  CUDA_VISIBLE_DEVICES=7 python scripts/eval_by_visibility.py dump --experiment sft_readout_rgb_depth --mode ext_near
  python scripts/eval_by_visibility.py report --experiments baseline_pretrained_6view sft_readout_rgb_depth \
      --modes interp ext_near --pair sft_readout_rgb_depth_thick sft_readout_rgb_depth
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch

import evaluate_scannetpp as ev
from scannetpp_evaluation_protocol import (
    ALPHA_VALID_THRESHOLD, VOXEL_SIZE_METERS, _chunked_nn_dist, build_raycasting_scene,
    raycast_gt_depth_and_normals, voxel_downsample_points,
)

CACHE = Path("/mnt/disk/xveyan/workspace_scannetpp/eval_by_visibility")
OUT = ev.ROOT / "EXP_ScanNetPP" / "eval_by_visibility"
DATASET_ROOT = Path("/home/jms/xveyan/data/4dgs/scannetpp")
VIS_TOL_REL, VIS_TOL_M = 0.02, 0.02  # a point is seen if the first mesh hit is within 2 % or 2 cm of it
VIEW_BUCKETS = (("0 views", 0, 0), ("1 view", 1, 1), ("2 views", 2, 2), ("3 views", 3, 3), ("4+ views", 4, 6))
DIST_BUCKETS = (("<1.5 m", 0.0, 1.5), ("1.5-3.5 m", 1.5, 3.5), (">3.5 m", 3.5, 1e9))


def checkpoint_of(experiment: str) -> Path:
    return ev.CHECKPOINT if experiment == ev.NAME else ev.ROOT / "EXP_ScanNetPP" / experiment / "model.safetensors"


def build(experiment: str, mode: str):
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    opt = dataclasses.replace(
        ev.config_defaults["eval_scannetpp_latent_6view_ssim"], model_type="tokengs",
        evaluating=True, use_ttt_for_eval=False, ttt_n_steps=0, use_geo_losses=False, load_lidar_pc=False,
        use_input_supervision=False, mixed_precision="no", num_workers=0,
        dataset_kwargs={"window_size": 18, "root_path": str(DATASET_ROOT)},
    )
    state = ev.load_file(str(checkpoint_of(experiment)), device="cpu")
    if any(k.startswith("activation_head.pos_mlp.") for k in state):
        opt = dataclasses.replace(opt, readout_pos_mlp=True,
                                  readout_pos_mlp_hidden=int(state["activation_head.pos_mlp.1.weight"].shape[0]))
    if "activation_head.surfel_rn" in state:
        opt = dataclasses.replace(opt, readout_surfel=True)
    if "activation_head.sh_head.weight" in state:
        k_rest = state["activation_head.sh_head.weight"].shape[0] // (opt.dec_patch_size ** 2 * 3)
        opt = dataclasses.replace(opt, readout_sh_degree=int(round((k_rest + 1) ** 0.5)) - 1)
    provider = ev.Provider("scannetpp_val_scaled_0.3", opt, training=False)
    if mode != "interp":
        ev.set_extrapolation_targets(provider.dataset, mode)
    model = ev.model_registry[opt.model_type](opt).eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    return opt, provider, model.to(device="cuda", dtype=torch.float32)


def pixel_rays(cam_view_i: torch.Tensor, intr_i: torch.Tensor, H: int, W: int):
    u, v = torch.meshgrid(torch.arange(W, dtype=torch.float32) + 0.5, torch.arange(H, dtype=torch.float32) + 0.5,
                          indexing="xy")
    c2w = torch.linalg.inv(cam_view_i.float().T)
    fx, fy, cx, cy = intr_i.float().tolist()
    dirs = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1) @ c2w[:3, :3].T
    return c2w[:3, 3].expand_as(dirs), dirs


def input_visibility(rc_scene, pts: torch.Tensor, cam_view_in: torch.Tensor, intr_in: torch.Tensor,
                     H: int, W: int, znear: float, scale: float):
    """Number of input views that see each point (in frustum and not occluded by the mesh) and the distance (m)
    to the nearest seeing input camera (nearest input camera if none sees it)."""
    import open3d as o3d

    n_vis = torch.zeros(pts.shape[0], dtype=torch.uint8)
    d_seen = torch.full((pts.shape[0],), float("inf"))
    d_any = torch.full((pts.shape[0],), float("inf"))
    for j in range(cam_view_in.shape[0]):
        w2c = cam_view_in[j].float().T
        xc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        fx, fy, cx, cy = intr_in[j].float().tolist()
        z = xc[:, 2]
        u = fx * xc[:, 0] / z.clamp_min(1e-6) + cx
        v = fy * xc[:, 1] / z.clamp_min(1e-6) + cy
        in_fr = (z > znear) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        centre = torch.linalg.inv(w2c)[:3, 3]
        ray_d = pts - centre
        length = ray_d.norm(dim=-1)
        rays = torch.cat([centre.expand_as(ray_d), ray_d], dim=-1).numpy().astype(np.float32)
        t_hit = torch.from_numpy(rc_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()).float()
        gap = (1.0 - torch.nan_to_num(t_hit, posinf=1.0)) * length
        seen = in_fr & (gap <= torch.maximum(VIS_TOL_REL * length, torch.tensor(VIS_TOL_M * scale)))
        n_vis += seen.to(torch.uint8)
        d_m = length / scale
        d_seen = torch.where(seen, torch.minimum(d_seen, d_m), d_seen)
        d_any = torch.minimum(d_any, d_m)
    return n_vis, torch.where(torch.isfinite(d_seen), d_seen, d_any)


def dump(args):
    gpu = ev.idle_gpu()
    opt, provider, model = build(args.experiment, args.mode)
    splits = ev.frame_splits(provider)
    H, W = map(int, opt.img_size)
    scale = float(provider.scene_scale)
    gt_dir, out_dir = CACHE / args.mode / "_gt", CACHE / args.mode / args.experiment
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite {out_dir}")
    gt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = torch.utils.data.DataLoader(provider, batch_size=1, shuffle=False, num_workers=0)
    print(f"GPU {gpu['index']}: {args.experiment} / {args.mode}, FP32, no TF32", flush=True)
    with torch.inference_mode():
        for step, raw in enumerate(loader):
            spec = splits[step]
            scene_id = raw["scene_id"][0]
            if scene_id != spec["scene_id"] or raw["frame_ids"][0].tolist() != spec["context"] + spec["target"]:
                raise RuntimeError("Loaded scene/frame identity differs from the expected split")
            data = {k: v.cuda() if torch.is_tensor(v) else v for k, v in raw.items()}
            model_input, _ = ev.split_data(data, opt)
            rec = model.forward_reconstruction(model_input)
            rendered = model.render_reconstruction(rec, model_input.decoder)
            pred = rendered["images_pred"][0].float().clamp(0, 1)
            truth = data["images_output"][0].float()
            sq = (pred - truth).square().mean(dim=1).cpu()
            alpha = rendered["alphas_pred"][0].float().reshape(-1, H, W).cpu()
            depth = rendered["depths_pred"][0].float().reshape(-1, H, W).cpu()
            if not (torch.isfinite(sq).all() and torch.isfinite(depth).all()):
                raise RuntimeError(f"non-finite render in {scene_id}")
            np.savez_compressed(out_dir / f"{scene_id}.npz", sq=sq.numpy().astype(np.float16),
                                alpha=alpha.numpy().astype(np.float16), depth=depth.numpy().astype(np.float32))
            gt_file = gt_dir / f"{scene_id}.npz"
            if not gt_file.is_file():
                cam_view, intr = data["cam_view"][0].cpu(), data["intrinsics"][0].cpu()
                raw_ctx = torch.from_numpy(provider.dataset._scene_meta_cache[scene_id]["c2ws"][spec["context"]].copy())
                mesh_ply = provider.dataset.root_path / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
                rc_scene = build_raycasting_scene(mesh_ply, raw_ctx, scale)
                d_gt, _, gt_ok = raycast_gt_depth_and_normals(rc_scene, cam_view, intr, opt.img_size)
                n_vis = torch.zeros((cam_view.shape[0], H, W), dtype=torch.uint8)
                dist = torch.zeros((cam_view.shape[0], H, W))
                for i in range(cam_view.shape[0]):
                    orig, dirs = pixel_rays(cam_view[i], intr[i], H, W)
                    m = gt_ok[i]
                    pts = orig[m] + d_gt[i][m].unsqueeze(-1) * dirs[m]
                    nv, dm = input_visibility(rc_scene, pts, data["cam_view_input"][0].cpu(),
                                              data["intrinsics_input"][0].cpu(), H, W, float(opt.znear), scale)
                    n_vis[i][m], dist[i][m] = nv, dm
                np.savez_compressed(gt_file, depth=d_gt.numpy(), valid=gt_ok.numpy(), n_vis=n_vis.numpy(),
                                    dist_m=dist.numpy().astype(np.float16), cam_view=cam_view.numpy(),
                                    intrinsics=intr.numpy(), scene_scale=np.float32(scale))
            print(f"[{step + 1:02d}/50] {scene_id}", flush=True)
    (out_dir / "DONE").write_text(json.dumps({"experiment": args.experiment, "mode": args.mode,
                                              "checkpoint": str(checkpoint_of(args.experiment)), "gpu": gpu}))


class Acc:
    """Pooled pixel accumulators for one bucket."""

    def __init__(self):
        self.n = self.sse = self.cov = self.absrel = self.hit105 = self.hit110 = 0.0

    def add(self, sq, covered, ratio, absrel):
        self.n += sq.size
        self.sse += float(sq.sum(dtype=np.float64))
        self.cov += float(covered.sum())
        self.absrel += float(absrel[covered].sum(dtype=np.float64))
        self.hit105 += float((covered & (ratio < 1.05)).sum())
        self.hit110 += float((covered & (ratio < 1.10)).sum())

    def result(self, n_total):
        n = max(self.n, 1.0)
        return {
            "pixel_share": self.n / max(n_total, 1.0),
            "psnr_pooled": -10 * math.log10(max(self.sse / n, 1e-12)),
            "coverage": self.cov / n,
            "absrel_covered": self.absrel / max(self.cov, 1.0),
            "delta105_all": self.hit105 / n,
            "delta110_all": self.hit110 / n,
        }


def bucket_masks(n_vis, dist):
    out = {"all": np.ones_like(n_vis, dtype=bool), "seen (>=1 view)": n_vis >= 1}
    out.update({name: (n_vis >= lo) & (n_vis <= hi) for name, lo, hi in VIEW_BUCKETS})
    out.update({name: (dist >= lo) & (dist < hi) for name, lo, hi in DIST_BUCKETS})
    return out


def load_scene(mode, experiment, scene_id):
    g = np.load(CACHE / mode / "_gt" / f"{scene_id}.npz")
    p = np.load(CACHE / mode / experiment / f"{scene_id}.npz")
    valid = g["valid"]
    d_gt, d_pred = g["depth"], p["depth"]
    alpha = p["alpha"].astype(np.float32)
    covered = valid & (alpha > ALPHA_VALID_THRESHOLD) & (d_pred > 0)
    safe_pred = np.where(covered, d_pred, 1.0)
    safe_gt = np.where(valid, d_gt, 1.0)
    ratio = np.maximum(safe_pred / safe_gt, safe_gt / safe_pred)
    absrel = np.abs(safe_pred - safe_gt) / safe_gt
    return g, p["sq"].astype(np.float32), covered, ratio, absrel


def completeness_split(g, covered, d_pred):
    """Completeness (cm) and recall@5cm of target-view GT surface points, split by input visibility (official
    voxel size); predicted surface = backprojected covered pixels, as in the official G3."""
    scale = float(g["scene_scale"])
    H, W = covered.shape[1:]
    pred_pts, gt_pts, gt_seen = [], [], []
    for i in range(covered.shape[0]):
        orig, dirs = pixel_rays(torch.from_numpy(g["cam_view"][i]), torch.from_numpy(g["intrinsics"][i]), H, W)
        orig, dirs = orig.numpy(), dirs.numpy()
        c, v = covered[i], g["valid"][i]
        pred_pts.append(orig[c] + d_pred[i][c][:, None] * dirs[c])
        gt_pts.append(orig[v] + g["depth"][i][v][:, None] * dirs[v])
        gt_seen.append(g["n_vis"][i][v] >= 1)
    voxel = VOXEL_SIZE_METERS * scale
    p_ds = voxel_downsample_points(torch.from_numpy(np.concatenate(pred_pts)).float(), voxel)
    gt_all, seen = np.concatenate(gt_pts), np.concatenate(gt_seen)
    out = {}
    for name, m in (("seen", seen), ("unseen", ~seen)):
        g_ds = voxel_downsample_points(torch.from_numpy(gt_all[m]).float(), voxel)
        if g_ds.shape[0] == 0 or p_ds.shape[0] == 0:
            out[f"comp_{name}_cm"], out[f"rec5_{name}"] = float("nan"), float("nan")
            continue
        d = _chunked_nn_dist(g_ds, p_ds) / scale
        out[f"comp_{name}_cm"] = float(d.mean() * 100)
        out[f"rec5_{name}"] = float((d < 0.05).float().mean())
    return out


def scenes_of(mode, experiment):
    d = CACHE / mode / experiment
    if not (d / "DONE").is_file():
        raise RuntimeError(f"missing dump {d}")
    return sorted(p.stem for p in d.glob("*.npz"))


def report(args):
    result = {"definitions": {
        "visibility": f"GT surface point seen by an input view if inside its frustum and the first mesh hit from that "
                      f"camera centre is within {VIS_TOL_REL:.0%} or {VIS_TOL_M * 100:.0f} cm of the point",
        "distance": "to the nearest input camera that sees the point (nearest input camera if none)",
        "pixels": "target-view pixels with a GT mesh hit; metrics pooled over all pixels of all 50 scenes",
        "coverage": f"rendered alpha > {ALPHA_VALID_THRESHOLD}",
        "delta110_all": "share of ALL GT pixels that are covered AND within 10 % depth ratio (holes count as errors)",
        "comp_split": "official G3 Completeness restricted to GT points seen / unseen by the inputs (scene mean)",
    }, "modes": {}}
    md = ["# 可观测性拆分（ScanNet++ 50 验证场景）", "",
          "像素 = 目标视角中有激光网格交点的像素，全部 50 场景合并统计（pooled PSNR 与官方逐帧平均 PSNR 口径不同）。",
          "“看到” = 该表面点在某张输入图视锥内且未被网格遮挡；距离 = 到最近一台看到它的输入相机。", ""]
    for mode in args.modes:
        result["modes"][mode] = {}
        for exp in args.experiments:
            scenes = scenes_of(mode, exp)
            accs, comp = {}, []
            for sid in scenes:
                g, sq, covered, ratio, absrel = load_scene(mode, exp, sid)
                valid = g["valid"]
                for name, m in bucket_masks(g["n_vis"], g["dist_m"].astype(np.float32)).items():
                    sel = valid & m
                    accs.setdefault(name, Acc()).add(sq[sel], covered[sel], ratio[sel], absrel[sel])
                comp.append(completeness_split(g, covered, np.load(CACHE / mode / exp / f"{sid}.npz")["depth"]))
            n_total = accs["all"].n
            res = {name: a.result(n_total) for name, a in accs.items()}
            for k in comp[0]:
                vals = [c[k] for c in comp if math.isfinite(c[k])]
                res.setdefault("completeness", {})[k] = float(np.mean(vals)) if vals else float("nan")
            result["modes"][mode][exp] = res
        md += [f"## {mode}", ""]
        for name in ["all", "seen (>=1 view)"] + [b[0] for b in VIEW_BUCKETS] + [b[0] for b in DIST_BUCKETS]:
            share = result["modes"][mode][args.experiments[0]][name]["pixel_share"]
            md += [f"### {name}（像素占比 {share:.1%}）", "",
                   "| 实验 | PSNR(pooled) | 覆盖率 | AbsRel(覆盖像素) | δ<1.05(全部像素) | δ<1.10(全部像素) |",
                   "|---|---:|---:|---:|---:|---:|"]
            for exp in args.experiments:
                r = result["modes"][mode][exp][name]
                md.append(f"| {exp} | {r['psnr_pooled']:.2f} | {r['coverage']:.3f} | {r['absrel_covered']:.3f} | "
                          f"{r['delta105_all']:.3f} | {r['delta110_all']:.3f} |")
            md.append("")
        md += ["### Completeness 按输入可见性拆分（场景平均）", "",
               "| 实验 | Comp 看得到 (cm) | Comp 看不到 (cm) | Recall@5cm 看得到 | Recall@5cm 看不到 |", "|---|---:|---:|---:|---:|"]
        for exp in args.experiments:
            c = result["modes"][mode][exp]["completeness"]
            md.append(f"| {exp} | {c['comp_seen_cm']:.1f} | {c['comp_unseen_cm']:.1f} | {c['rec5_seen']:.3f} | "
                      f"{c['rec5_unseen']:.3f} |")
        md.append("")
        if args.pair:
            a_exp, b_exp = args.pair
            result["modes"][mode]["pair"] = pair = decompose(mode, a_exp, b_exp)
            md += [f"### 画质差拆解：{a_exp} 相对 {b_exp}", "",
                   f"合并 PSNR 差 {pair['delta_psnr_pooled']:+.3f} dB；下表为各格子对总误差增量（ΔSSE）的贡献。"
                   f"“A 未覆盖”= {a_exp} 渲染 α≤0.5。", "",
                   "| 输入可见视角数 | A 覆盖? | 像素占比 | ΔSSE 贡献占比 | 该格 ΔPSNR |", "|---|---|---:|---:|---:|"]
            for cell in pair["cells"]:
                md.append(f"| {cell['views']} | {'是' if cell['a_covered'] else '否'} | {cell['pixel_share']:.1%} | "
                          f"{cell['dsse_share']:+.1%} | {cell['delta_psnr']:+.2f} |")
            md.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_by_visibility.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "eval_by_visibility.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def decompose(mode, a_exp, b_exp):
    """Pixel-level split of the pooled SSE difference A - B by input visibility and by A's coverage."""
    scenes = scenes_of(mode, a_exp)
    if scenes != scenes_of(mode, b_exp):
        raise RuntimeError("pair dumps cover different scenes")
    cells, n_total, sse_a, sse_b = {}, 0.0, 0.0, 0.0
    for sid in scenes:
        g, sq_a, cov_a, _, _ = load_scene(mode, a_exp, sid)
        _, sq_b, _, _, _ = load_scene(mode, b_exp, sid)
        valid = g["valid"]
        n_total += float(valid.sum())
        sse_a += float(sq_a[valid].sum(dtype=np.float64))
        sse_b += float(sq_b[valid].sum(dtype=np.float64))
        for name, lo, hi in VIEW_BUCKETS:
            vm = valid & (g["n_vis"] >= lo) & (g["n_vis"] <= hi)
            for a_cov in (True, False):
                m = vm & (cov_a == a_cov)
                c = cells.setdefault((name, a_cov), [0.0, 0.0, 0.0])
                c[0] += float(m.sum())
                c[1] += float(sq_a[m].sum(dtype=np.float64))
                c[2] += float(sq_b[m].sum(dtype=np.float64))
    d_total = sse_a - sse_b
    out = []
    for (name, a_cov), (n, sa, sb) in cells.items():
        out.append({"views": name, "a_covered": a_cov, "pixel_share": n / n_total,
                    "dsse_share": (sa - sb) / d_total if d_total else float("nan"),
                    "delta_psnr": (-10 * math.log10(max(sa / max(n, 1), 1e-12))) - (-10 * math.log10(max(sb / max(n, 1), 1e-12)))})
    return {"a": a_exp, "b": b_exp,
            "delta_psnr_pooled": -10 * math.log10(sse_a / n_total) + 10 * math.log10(sse_b / n_total), "cells": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--experiment", required=True)
    d.add_argument("--mode", choices=["interp", "ext_near", "ext_far"], required=True)
    r = sub.add_parser("report")
    r.add_argument("--experiments", nargs="+", required=True)
    r.add_argument("--modes", nargs="+", required=True)
    r.add_argument("--pair", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    dump(args) if args.cmd == "dump" else report(args)


if __name__ == "__main__":
    main()
