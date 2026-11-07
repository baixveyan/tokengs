"""Which tokens render a pixel, and where are their Gaussians?  (official model, demo scene, no training)

10 pixels on laser surfaces in input view 0 of scene 7b6477cb95.  The exact per-token rendering weight at each pixel is
obtained by rasterising one-hot token channels.  Every token contributing >= 1 % of a pixel's alpha is exported with
all 64 of its Gaussians, one colour per token (3DGS PLY), plus the 10 pixel rays and their laser hit points (point PLY).

  ALLOW_SHARED_GPU=1 PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/viz_pixel_tokens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

import diag_thickness_split as TS
from diag_token_render_ownership import rasterize_channels
from evaluate_scannetpp import ROOT, idle_gpu
from probe_scene_latent_limit import S, make_opt
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data

SCENE, VIEW, MIN_SHARE = "7b6477cb95", 0, 0.01
OUT = ROOT / "EXP_ScanNetPP/demo/scene_7b6477cb95/baseline_pretrained_6view"


def write_gaussians(path, g, color, extra):
    f_dc = (color - 0.5) / 0.28209479177387814
    cols = {"x": g[:, 0], "y": g[:, 1], "z": g[:, 2], "f_dc_0": f_dc[:, 0], "f_dc_1": f_dc[:, 1], "f_dc_2": f_dc[:, 2],
            "opacity": torch.logit(g[:, 3], eps=1e-6), "scale_0": g[:, 4].clamp_min(1e-8).log(),
            "scale_1": g[:, 5].clamp_min(1e-8).log(), "scale_2": g[:, 6].clamp_min(1e-8).log(),
            "rot_0": g[:, 7], "rot_1": g[:, 8], "rot_2": g[:, 9], "rot_3": g[:, 10], "opacity_activated": g[:, 3], **extra}
    arr = np.empty(g.shape[0], dtype=[(k, "f4") for k in cols])
    for k, v in cols.items():
        arr[k] = v.cpu().numpy().astype(np.float32)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def write_points(path, xyz, rgb):
    arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


@torch.no_grad()
def main():
    idle_gpu()
    dev = torch.device("cuda:0")
    opt = make_opt("val")
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov.dataset._offset_frac = None
    i = next(k for k, p in enumerate(prov.dataset.sample_list) if Path(p).name == SCENE)
    data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in prov[i].items()}
    model = TS.load_model(TS.MODELS["official"], opt, dev)
    g = model.forward_reconstruction(split_data(data, opt)[0]).gaussians[0].float()
    H, W = map(int, opt.img_size)
    cv = data["cam_view_input"][0, VIEW].float()
    viewmat = cv.T.contiguous()
    fx, fy, cx, cy = data["intrinsics_input"][0, VIEW].float().tolist()
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=dev)
    ctx = prov.dataset.get_context_target_frames(i)[0]
    rc = build_raycasting_scene(prov.dataset.root_path / "data" / SCENE / "scans" / "mesh_aligned_0.05.ply",
                                torch.from_numpy(prov.dataset._scene_meta_cache[SCENE]["c2ws"][ctx].copy()), S)
    dgt, _n, ok = raycast_gt_depth_and_normals(rc, data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu(), opt.img_size)
    _img, alpha = rasterize_channels(g, torch.zeros((g.shape[0], 1), device=dev), viewmat, K, H, W, opt)
    pix = []
    for r in np.linspace(0.2, 0.8, 2):           # 2 x 5 grid, nudged to the nearest valid laser pixel
        for c in np.linspace(0.12, 0.88, 5):
            y0, x0 = int(r * H), int(c * W)
            m = ok[VIEW].numpy() & (alpha.cpu().numpy() > 0.5)
            ys, xs = np.nonzero(m)
            j = np.argmin((ys - y0) ** 2 + (xs - x0) ** 2)
            pix.append((int(ys[j]), int(xs[j])))
    ntok = g.shape[0] // 64
    tok = torch.arange(g.shape[0], device=dev) // 64
    wpix = torch.zeros((len(pix), ntok), device=dev)
    for c0 in range(0, ntok, 512):
        k = min(512, ntok - c0)
        sel = (tok >= c0) & (tok < c0 + k)
        oh = torch.zeros((g.shape[0], k), device=dev)
        oh[sel, tok[sel] - c0] = 1.0
        img, _ = rasterize_channels(g, oh, viewmat, K, H, W, opt)
        for p, (y, x) in enumerate(pix):
            wpix[p, c0:c0 + k] = img[y, x]
    c2w = torch.linalg.inv(cv.T)
    cam = c2w[:3, 3]
    rng = np.random.default_rng(0)
    keep, tok_color, tok_pixel, rows = [], {}, {}, []
    ray_xyz, ray_rgb = [], []
    for p, (y, x) in enumerate(pix):
        share = wpix[p] / wpix[p].sum().clamp_min(1e-12)
        ts = torch.nonzero(share >= MIN_SHARE).flatten().tolist()
        d = torch.tensor([(x + 0.5 - cx) / fx, (y + 0.5 - cy) / fy, 1.0], device=dev)
        dw = c2w[:3, :3] @ d
        zgt = float(dgt[VIEW, y, x])
        cen = [(g[t * 64:(t + 1) * 64, :3] * g[t * 64:(t + 1) * 64, 3:4]).sum(0) / g[t * 64:(t + 1) * 64, 3].sum().clamp_min(1e-8) for t in ts]
        zc = [float((cv.T[:3, :3] @ c + cv.T[:3, 3])[2]) if False else float((viewmat[:3, :3] @ c + viewmat[:3, 3])[2]) for c in cen]
        wsh = share[ts].tolist()
        rows.append({"pixel": [y, x], "laser_depth_m": round(zgt / S, 3), "tokens_ge_1pct": len(ts),
                     "tokens_for_90pct": int((share.sort(descending=True).values.cumsum(0) < 0.9).sum()) + 1,
                     "token_centre_depth_minus_laser_cm": sorted(round((z - zgt) / S * 100, 1) for z in zc),
                     "weighted_spread_of_token_depths_cm": round(float(np.sqrt(np.average((np.array(zc) - np.average(zc, weights=wsh)) ** 2, weights=wsh))) / S * 100, 1)})
        for t in ts:
            tok_color.setdefault(t, rng.random(3) * 0.8 + 0.1)
            tok_pixel.setdefault(t, p)
        tt = np.linspace(0, zgt * 1.4, 300)
        ray_xyz.append((cam[None] + torch.tensor(tt, device=dev, dtype=torch.float32)[:, None] * dw[None]).cpu().numpy())
        ray_rgb.append(np.tile([255, 255, 255], (300, 1)))
        hit = (cam + zgt * dw).cpu().numpy()
        ray_xyz.append(hit[None] + rng.normal(0, 0.002, (40, 3)))
        ray_rgb.append(np.tile([255, 0, 0], (40, 1)))
    ts = sorted(tok_color)
    idx = torch.cat([torch.arange(t * 64, (t + 1) * 64, device=dev) for t in ts])
    color = torch.tensor(np.repeat(np.stack([tok_color[t] for t in ts]), 64, 0), device=dev, dtype=torch.float32)
    extra = {"token_id": (idx // 64).float(), "pixel_id": torch.tensor(np.repeat([tok_pixel[t] for t in ts], 64), device=dev).float()}
    write_gaussians(OUT / "baseline_pretrained_6view_pixel10_tokens.ply", g[idx], color, extra)
    write_points(OUT / "baseline_pretrained_6view_pixel10_rays.ply", np.concatenate(ray_xyz), np.concatenate(ray_rgb).astype(np.uint8))
    (OUT / "pixel10_tokens.json").write_text(json.dumps({"view": VIEW, "min_share": MIN_SHARE, "pixels": rows}, indent=1), encoding="utf-8")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
