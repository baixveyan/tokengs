"""Demo-style visual diagnosis of frozen query-based feed-forward GS models on one scene (no training).

Every problem found by scripts/diag_dl3dv.py / EXP_DL3DV/diag_locusgs_report.md is exported as a viewable 3D demo
(PLY, kleinicke.ply-visualizer) plus an index image, for several models side by side.  All models are put into ONE
coordinate frame (the release model's mean_cam normalized frame; first_cam models are rigidly transformed, Gaussian
rotations included), so their PLYs overlay each other and the scene-level laser mesh / COLMAP points / cameras.
Pixels used to pick tokens and ray probes are model independent (fixed grids on the laser surface), except the
"worst thickness" probes, which are chosen per model.

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/demo_query_diagnosis.py --source scannetpp
  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/demo_query_diagnosis.py --source dl3dv
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement
from safetensors.torch import load_file
from scipy.spatial import cKDTree

import demo_ply_export as DPE
import diag_thickness_split as TS
from diag_dl3dv import FAR_NORM, MODELS as DMODELS, colmap_norm, kmat, load_any, make_opt, project, run_decoder, run_release
from diag_thinning_and_parallax import parallax_deg
from diag_token_render_ownership import rasterize_channels
from diagnose_latent_attention_geometry import ALIVE_OPACITY
from evaluate_scannetpp import ROOT, idle_gpu
from probe_gs_token_geometry import child_centre, load_model
from scannetpp_evaluation_protocol import build_raycasting_scene, normalize_world_transform, raycast_gt_depth_and_normals
from tokengs.data.provider import Provider
from tokengs.options import config_defaults

OUT = ROOT / "EXP_QueryGS_Diagnosis"
SCANNETPP_ROOT = "/home/jms/xveyan/data/4dgs/scannetpp"
SCENES = {"scannetpp": "7b6477cb95", "dl3dv": "2f3e1c0f688c84cec67f9a1ea219c54c14ffabf31a046e620dacc690cac2f1bd"}
SFT12 = "sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20"
MODEL_SETS = {"scannetpp": ["tokengs_release_latent_6v", "tokengs_paper_4v", "locusgs_finetune_4v", SFT12],
              "dl3dv": ["tokengs_release_latent_6v", "tokengs_paper_4v", "locusgs_finetune_4v"]}
CH, CHUNK, TOPK = 64, 512, 32
TURBO = matplotlib.colormaps["turbo"]


# ------------------------------------------------------------------------------------------------ small helpers
def rot_to_quat(R):
    """3x3 rotation -> unit quaternion (w, x, y, z)."""
    m = R.double()
    w = torch.sqrt(torch.clamp(1 + m[0, 0] + m[1, 1] + m[2, 2], min=1e-12)) / 2
    q = torch.stack([w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)])
    return F.normalize(q, dim=0).float()


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def rigid(g, R, t):
    """Apply x -> R x + t to Gaussians [N, 14] (centres and rotations)."""
    out = g.clone()
    out[:, :3] = g[:, :3] @ R.T + t
    out[:, 7:11] = quat_mul(rot_to_quat(R).to(g.device).expand(g.shape[0], 4), F.normalize(g[:, 7:11], dim=-1))
    return out


def palette(n, seed=0):
    return torch.rand(n, 3, generator=torch.Generator().manual_seed(seed)) * 0.75 + 0.2


def heat(x, lo, hi):
    return torch.from_numpy(TURBO(((x - lo) / (hi - lo)).clamp(0, 1).cpu().numpy())[:, :3]).float()


def write_gaussians(path, g, color, extra=None):
    g, color = g.detach().float().cpu(), color.detach().float().cpu()
    f_dc = (color - 0.5) / 0.28209479177387814
    cols = {"x": g[:, 0], "y": g[:, 1], "z": g[:, 2], "f_dc_0": f_dc[:, 0], "f_dc_1": f_dc[:, 1], "f_dc_2": f_dc[:, 2],
            "opacity": torch.logit(g[:, 3].clamp(1e-6, 1 - 1e-6)), "scale_0": g[:, 4].clamp_min(1e-8).log(),
            "scale_1": g[:, 5].clamp_min(1e-8).log(), "scale_2": g[:, 6].clamp_min(1e-8).log(),
            "rot_0": g[:, 7], "rot_1": g[:, 8], "rot_2": g[:, 9], "rot_3": g[:, 10], "opacity_activated": g[:, 3],
            **{k: v.detach().float().cpu() for k, v in (extra or {}).items()}}
    arr = np.empty(g.shape[0], dtype=[(k, "f4") for k in cols])
    for k, v in cols.items():
        arr[k] = v.numpy().astype(np.float32)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def write_points(path, xyz, rgb, extra=None):
    xyz = xyz.detach().float().cpu().numpy()
    rgb = (rgb.detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    extra = {k: v.detach().float().cpu().numpy() for k, v in (extra or {}).items()}
    arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                    *[(k, "f4") for k in extra]])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for k, v in extra.items():
        arr[k] = v
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def segment(a, b, n):
    s = torch.linspace(0, 1, n, device=a.device).unsqueeze(-1)
    return a.unsqueeze(0) * (1 - s) + b.unsqueeze(0) * s


def sphere(c, r, n=160):
    i = torch.arange(n, device=c.device).float() + 0.5
    phi, th = torch.acos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    d = torch.stack([torch.cos(th) * torch.sin(phi), torch.sin(th) * torch.sin(phi), torch.cos(phi)], -1)
    return c + r * d


def ball(c, r, n=60):
    return torch.cat([sphere(c, r * s, n // 3) for s in (0.35, 0.7, 1.0)])


def to_img(x):
    return Image.fromarray((x.detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8))


def rgb_img(t):  # [3, H, W]
    return to_img(t.permute(1, 2, 0))


def map_img(m, lo, hi, mask=None):
    c = torch.from_numpy(TURBO(((m - lo) / (hi - lo)).clamp(0, 1).cpu().numpy())[..., :3]).float()
    if mask is not None:
        c[~mask.cpu()] = 0.0
    return to_img(c)


def label(img, text, xy=(4, 2)):
    d = ImageDraw.Draw(img)
    for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        d.text((xy[0] + dx, xy[1] + dy), text, fill=(0, 0, 0))
    d.text(xy, text, fill=(255, 255, 255))
    return img


def grid_image(rows, pad=2):
    w, h = rows[0][0].size
    canvas = Image.new("RGB", (len(rows[0]) * (w + pad), len(rows) * (h + pad)), (40, 40, 40))
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            canvas.paste(im, (j * (w + pad), i * (h + pad)))
    return canvas


# ------------------------------------------------------------------------------------------------ rendering diagnostics
def token_topk(g, w2c, intr, H, W, opt):
    """Per pixel: alpha, top-32 token weights/ids (exact, one-hot token rasterisation), #tokens with >= 1 % of alpha."""
    N, dev = g.shape[0], g.device
    tid = torch.arange(N, device=dev) // CH
    K = kmat(intr, dev)
    _, alpha = rasterize_channels(g, torch.ones((N, 1), device=dev), w2c, K, H, W, opt)
    a = alpha.flatten()
    topw = torch.zeros((H * W, 0), device=dev)
    topi = torch.zeros((H * W, 0), dtype=torch.long, device=dev)
    cnt = torch.zeros(H * W, device=dev)
    for c0 in range(0, N // CH, CHUNK):
        k = min(CHUNK, N // CH - c0)
        col = torch.zeros((N, k), device=dev)
        s = (tid >= c0) & (tid < c0 + k)
        col[s, tid[s] - c0] = 1.0
        im, _ = rasterize_channels(g, col, w2c, K, H, W, opt)
        w = im.reshape(-1, k)
        cnt += (w >= 0.01 * a.unsqueeze(1)).sum(-1)
        allw = torch.cat([topw, w], 1)
        alli = torch.cat([topi, torch.arange(c0, c0 + k, device=dev).expand(H * W, k)], 1)
        topw, ix = allw.topk(min(TOPK, allw.shape[1]), 1)
        topi = alli.gather(1, ix)
        del im, w, col, allw, alli
    return alpha, topw, topi, cnt.reshape(H, W)


def grid_pixels(valid, nx, ny):
    """Model-independent grid pixels snapped to the nearest valid pixel."""
    H, W = valid.shape
    yy, xx = torch.nonzero(valid.cpu(), as_tuple=True)
    out = []
    for j in range(ny):
        for i in range(nx):
            py, px = H * (j + 0.5) / ny, W * (i + 0.5) / nx
            k = int(((yy - py) ** 2 + (xx - px) ** 2).argmin())
            out.append((int(yy[k]), int(xx[k])))
    return out


def worst_pixels(score, valid, n, radius=28):
    s = torch.where(valid, score, torch.full_like(score, -1)).clone()
    out = []
    H, W = s.shape
    for _ in range(n):
        k = int(s.argmax())
        y, x = divmod(k, W)
        if s[y, x] <= 0:
            break
        out.append((y, x))
        s[max(0, y - radius):y + radius, max(0, x - radius):x + radius] = -1
    return out


def mark(img, pts, colors, r=5, numbers=True):
    d = ImageDraw.Draw(img)
    for n, ((y, x), c) in enumerate(zip(pts, colors)):
        c = tuple(int(v * 255) for v in c)
        d.ellipse((x - r, y - r, x + r, y + r), outline=c, width=2)
        if numbers:
            d.text((x + r + 1, y - r - 1), str(n), fill=c)
    return img


# ------------------------------------------------------------------------------------------------ scene / model setup
def data_opts(src):
    if src == "dl3dv":
        return {"mean_cam": make_opt("release"), "first_cam": make_opt("tokengs_paper")}, "dl3dv_eval_scaled_0.15", 0.15
    base = dataclasses.replace(
        config_defaults["eval_scannetpp_latent_6view_ssim"], evaluating=True, use_ttt_for_eval=False, ttt_n_steps=0,
        use_geo_losses=False, load_lidar_pc=False, use_input_supervision=False, mixed_precision="no", num_workers=0,
        val_depth_metric=False, dataset_kwargs={"window_size": 18, "root_path": SCANNETPP_ROOT})
    if base.camera_normalization_method != "mean_cam":
        raise RuntimeError("release ScanNet++ preset is expected to use mean_cam")
    return {"mean_cam": base, "first_cam": dataclasses.replace(base, camera_normalization_method="first_cam")}, \
        "scannetpp_val_scaled_0.3", 0.3


def load_demo_model(name, dev):
    if name == SFT12:
        ck = ROOT / "EXP_ScanNetPP" / SFT12 / "model.safetensors"
        st = load_file(str(ck), device="cpu")
        opt = dataclasses.replace(make_opt("release"), readout_pos_mlp=True,
                                  readout_pos_mlp_hidden=int(st["activation_head.pos_mlp.1.weight"].shape[0]))
        return load_model(ck, opt, dev), opt, "release"
    model, mopt, _dopt, kind = load_any(name, dev)
    return model, mopt, kind


# ------------------------------------------------------------------------------------------------ main
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(SCENES), required=True)
    ap.add_argument("--scene")
    ap.add_argument("--models", nargs="*")
    a = ap.parse_args()
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")
    src, scene = a.source, a.scene or SCENES[a.source]
    models = a.models or MODEL_SETS[src]
    opts, dname, S = data_opts(src)
    provs = {k: Provider(dname, o, training=False) for k, o in opts.items()}
    ds = provs["mean_cam"].dataset
    names = [Path(p).name if src == "scannetpp" else Path(p).stem for p in ds.sample_list]
    idx = names.index(scene)
    data = {k: {kk: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for kk, v in p[idx].items()} for k, p in provs.items()}
    ctx, tgt = ds.get_context_target_frames(idx)
    if src == "scannetpp":
        raw_c2w = torch.from_numpy(ds._scene_meta_cache[scene]["c2ws"][ctx].copy()).float()
    else:
        _c, _n, _i, c2w_all, zh, _js = ds.load_video_reader(idx)
        zh.close()
        raw_c2w = torch.from_numpy(c2w_all[ctx]).float()
    A = normalize_world_transform(raw_c2w) @ raw_c2w[0]                   # first-cam frame -> mean-cam frame (world units)
    R_fm, t_fm = A[:3, :3].to(dev), (A[:3, 3] * S).to(dev)
    dm = data["mean_cam"]
    H, W = map(int, opts["mean_cam"].img_size)
    p = int(opts["mean_cam"].patch_size)
    hp, wp = H // p, W // p
    V = len(ctx)
    cam_in, intr_in, cam_t, intr_t = dm["cam_view_input"][0], dm["intrinsics_input"][0], dm["cam_view"][0], dm["intrinsics"][0]
    cams = torch.stack([torch.linalg.inv(c.float().T)[:3, 3] for c in cam_in])
    sdir = OUT / f"scene_{scene[:12]}"
    sdir.mkdir(parents=True, exist_ok=True)

    # ---- scene-level geometry reference (mean-cam frame)
    if src == "scannetpp":
        mesh = Path(SCANNETPP_ROOT) / "data" / scene / "scans" / "mesh_aligned_0.05.ply"
        rc = build_raycasting_scene(mesh, raw_c2w, S)
        DPE.export_scannetpp_lidar_ply(sdir / "gt_lidar_mesh_frustum.ply", mesh, raw_c2w, cam_in, intr_in, (H, W), S)
        views = [0, 3, 6]
        d_gt_t, _n, ok_t = raycast_gt_depth_and_normals(rc, cam_t.cpu(), intr_t.cpu(), (H, W))
        _d, _n, ok_in = raycast_gt_depth_and_normals(rc, cam_in.cpu(), intr_in.cpu(), (H, W))
        import open3d as o3d

        def surf_dist(x):  # metres
            return torch.from_numpy(rc.compute_distance(o3d.core.Tensor(x.cpu().numpy().astype(np.float32))).numpy()).to(dev) / S
        unit, far = "m", 0.30
    else:
        pn, vis = colmap_norm(ds, idx, ctx, list(ctx) + list(tgt), "mean_cam")
        tree = cKDTree(pn.numpy())
        nvis = torch.zeros(len(pn))
        for v_ in vis[:V]:
            nvis[v_] += 1
        write_points(sdir / "colmap_points.ply", pn[nvis > 0], heat(nvis[nvis > 0], 1, V), {"n_input_views": nvis[nvis > 0]})
        novel = [k for k, f in enumerate(tgt) if f not in set(ctx)]
        views = [novel[int(i)] for i in np.linspace(0, len(novel) - 1, 3).round()]
        d_gt_t = torch.zeros((len(tgt), H, W))
        for k in views:
            ids = vis[V + k]
            u, vv, z, okp = project(pn[ids], cam_t[k].T.cpu(), intr_t[k].cpu(), H, W)
            order = torch.argsort(z[okp], descending=True)
            d_gt_t[k, vv[okp].long()[order], u[okp].long()[order]] = z[okp][order]
        ok_t = d_gt_t > 0
        ok_in = torch.ones((V, H, W), dtype=torch.bool)

        def surf_dist(x):  # normalized units (nearest COLMAP point)
            return torch.from_numpy(tree.query(x.cpu().numpy(), k=1, workers=8)[0]).float().to(dev)
        unit, far = "norm", FAR_NORM
    DPE.export_camera_ply(sdir / "cameras.ply", dm["cam_to_world_input"][0], torch.linalg.inv(cam_t.float().transpose(1, 2)))
    # model-independent pixels
    tok_px = grid_pixels(ok_in[0], 5, 4)                                     # 20 token picks in input view 0
    pv = views[0]                                                            # ray-probe target view
    probe_fixed = grid_pixels(ok_t[pv] if src == "scannetpp" else torch.ones((H, W), dtype=torch.bool), 3, 2)
    inp0 = rgb_img(dm["images_input"][0, 0])
    tokpal = palette(len(tok_px), 7)
    mark(inp0.copy(), tok_px, tokpal.tolist()).save(sdir / "tokens20_pixels_input_view0.png")

    summary = {"scene": scene, "source": src, "gpu": gpu["index"], "frame": "release mean_cam normalized frame, scale "
               f"{S}; first_cam models rigidly transformed", "distance_unit": unit, "models": {}}
    for name in models:
        mdir = sdir / name
        mdir.mkdir(exist_ok=True)
        model, mopt, kind = load_demo_model(name, dev)
        first = kind != "release"
        d_in = data["first_cam" if first else "mean_cam"]
        mr = run_release(model, mopt, d_in) if kind == "release" else run_decoder(model, mopt, d_in, kind)
        g, T, render = mr["g"], mr["T"], mr["render"]
        st = {"checkpoint": str(DMODELS[name]["ckpt"]) if name in DMODELS else str(ROOT / "EXP_ScanNetPP" / SFT12 / "model.safetensors")}
        if first:   # frame change check: same image before/after the rigid transform
            r0 = render(g[None], d_in["cam_view"][:, pv:pv + 1], d_in["intrinsics"][:, pv:pv + 1])["images_pred"]
            g = rigid(g, R_fm, t_fm)
            r1 = render(g[None], cam_t[None, pv:pv + 1], intr_t[None, pv:pv + 1])["images_pred"]
            st["frame_transform_max_abs_image_diff"] = float((r0 - r1).abs().max())
        anchor = (mr["anchor"] @ R_fm.T + t_fm) if mr["anchor"] is not None and first else mr["anchor"]
        radius = mr["radius"]
        N = g.shape[0]
        ntok = N // CH
        g64 = g.reshape(ntok, CH, 14)
        op = g[:, 3]
        op64 = g64[..., 3]
        walive = op64 * (op64 >= ALIVE_OPACITY)
        alive_tok = walive.sum(-1) > 0.5
        Xp = child_centre(g64)
        vm = T.reshape(ntok, V, hp * wp).sum(-1)
        use = vm > 0.1
        dist = surf_dist(g[:, :3])
        junk = (op < 0.05) & (dist > far)
        tok_err = surf_dist(Xp)
        par = parallax_deg(Xp, cams, use)
        ccent = (g64[..., :3] - g64[..., :3].mean(1, keepdim=True)).norm(dim=-1).mean(1)
        tok_id = torch.arange(N, device=dev) // CH
        extra = {"token_id": tok_id.float(), "child_id": (torch.arange(N, device=dev) % CH).float(),
                 f"dist_to_surface_{unit}": dist, "is_junk": junk.float(), "token_parallax_deg": par[tok_id],
                 f"token_centre_dist_{unit}": tok_err[tok_id], "token_c_centroid": ccent[tok_id]}
        if anchor is not None:
            extra["anchor_radius"] = radius[tok_id]
            extra["offset_over_radius"] = (g[:, :3] - anchor[tok_id]).norm(dim=-1) / radius[tok_id]

        # 1. all Gaussians (true colours) + pruned + token colours
        write_gaussians(mdir / "gaussians.ply", g, g[:, 11:14], extra)
        DPE.export_pruned_ply(mdir / "gaussians.ply", mdir / "gaussians_prune0.05.ply", 0.05)
        al = op >= ALIVE_OPACITY
        tp = palette(ntok, 1).to(dev)
        write_gaussians(mdir / "tokencolor_alive.ply", g[al], tp[tok_id[al]], {k: v[al] for k, v in extra.items()})
        # 2. distance-to-surface heat (visible Gaussians) and junk slots
        vis_g = op >= 0.05
        write_points(mdir / "surface_distance_heat_points.ply", g[vis_g, :3], heat(dist[vis_g], 0, far),
                     {f"dist_{unit}": dist[vis_g], "opacity": op[vis_g]})
        ctx_pts = torch.nonzero(vis_g).squeeze(1)
        ctx_pts = ctx_pts[torch.randperm(ctx_pts.numel(), device=dev)[:60000]]
        write_points(mdir / "junk_slots_points.ply", torch.cat([g[junk, :3], g[ctx_pts, :3]]),
                     torch.cat([torch.tensor([[1.0, 0.1, 0.1]]).expand(int(junk.sum()), 3),
                                torch.tensor([[0.55, 0.55, 0.55]]).expand(ctx_pts.numel(), 3)]),
                     {"is_junk": torch.cat([torch.ones(int(junk.sum())), torch.zeros(ctx_pts.numel())]),
                      "opacity": torch.cat([op[junk], op[ctx_pts]]), f"dist_{unit}": torch.cat([dist[junk], dist[ctx_pts]])})
        # 3. token centres: parallax and error
        tc = alive_tok
        write_points(mdir / "token_centres_parallax.ply", Xp[tc], heat(par[tc], 0, 20), {"parallax_deg": par[tc], "n_views": use.sum(-1)[tc].float()})
        write_points(mdir / "token_centres_surface_distance.ply", Xp[tc], heat(tok_err[tc], 0, far), {f"dist_{unit}": tok_err[tc], "parallax_deg": par[tc]})

        # 4. per-pixel token maps: input view 0 (token picks) and the three target views
        a0, tw0, ti0, _c0 = token_topk(g, cam_in[0].float().T.contiguous(), intr_in[0], H, W, mopt)
        picks = []
        for (y, x) in tok_px:
            for t_ in ti0[y * W + x].tolist():
                if t_ not in picks:
                    picks.append(t_)
                    break
        sel = torch.tensor(picks, device=dev)
        mem = torch.isin(tok_id, sel)
        col20 = torch.zeros((N, 3), device=dev)
        for k, t_ in enumerate(picks):
            col20[tok_id == t_] = tokpal[k].to(dev)
        dead = op < ALIVE_OPACITY
        write_gaussians(mdir / "tokens20_gaussians.ply", g[mem], col20[mem], {k: v[mem] for k, v in extra.items()})
        write_points(mdir / "tokens20_points.ply", g[mem, :3], torch.where(dead[mem].unsqueeze(1), col20[mem] * 0.25, col20[mem]),
                     {"token_rank": torch.tensor([picks.index(int(t)) for t in tok_id[mem]], dtype=torch.float32),
                      "opacity": op[mem], "is_dead": dead[mem].float()})
        cen = torch.cat([ball(Xp[t_], 0.004) for t_ in picks])
        write_points(mdir / "tokens20_centres.ply", cen, tokpal.repeat_interleave(cen.shape[0] // len(picks), 0))
        if anchor is not None:
            sph = torch.cat([sphere(anchor[t_], radius[t_]) for t_ in picks])
            write_points(mdir / "tokens20_anchor_spheres.ply", sph, (tokpal * 0.6 + 0.4).repeat_interleave(160, 0),
                         {"token_rank": torch.arange(len(picks)).repeat_interleave(160).float()})
        # ownership image: attention (red) vs render weight (green) of the 20 tokens, on their dominant view
        oh = torch.zeros((N, len(picks)), device=dev)
        for k, t_ in enumerate(picks):
            oh[tok_id == t_, k] = 1.0
        Rimg = torch.stack([rasterize_channels(g, oh, cam_in[v].float().T.contiguous(), kmat(intr_in[v], dev), H, W, mopt)[0]
                            for v in range(V)])                              # [V, H, W, 20]
        tiles, own = [], []
        for k, t_ in enumerate(picks):
            att = T[t_].reshape(V, hp, wp)
            v = int(vm[t_].argmax())
            att_up = F.interpolate(att[v][None, None], size=(H, W), mode="nearest")[0, 0]
            rw = Rimg[v, ..., k]
            base = dm["images_input"][0, v].permute(1, 2, 0) * 0.45
            ia = base.clone()
            ia[..., 0] += 0.55 * (att_up / att_up.max().clamp_min(1e-12))
            ir = base.clone()
            ir[..., 1] += 0.55 * (rw / rw.max().clamp_min(1e-12))
            Rt = Rimg[..., k].reshape(V, hp, p, wp, p).sum((2, 4))
            a90 = att.flatten() >= att.flatten().sort(descending=True).values[(att.flatten().sort(descending=True).values.cumsum(0)
                                                                                < 0.9 * att.sum()).sum().clamp(max=att.numel() - 1)]
            inside = float((Rt.flatten() * a90).sum() / Rt.sum().clamp_min(1e-12))
            own.append(inside)
            tiles += [label(to_img(ia).resize((W // 2, H // 2)), f"#{k} attention v{v}"),
                      label(to_img(ir).resize((W // 2, H // 2)), f"#{k} render {inside:.0%} in att90")]
        grid_image([tiles[i:i + 8] for i in range(0, len(tiles), 8)]).save(mdir / "tokens20_attention_vs_render.png")

        # 5. render strip + thickness / token maps on three target views
        rr = render(g[None], cam_t[None], intr_t[None])
        img, dep, alp = rr["images_pred"][0].clamp(0, 1), rr["depths_pred"][0, :, 0], rr["alphas_pred"][0, :, 0]
        psnr = (-10 * (img - dm["images_output"][0]).square().mean((1, 2, 3)).log10())
        rows, thick_maps = [], {}
        for v in views:
            w2c = cam_t[v].float().T.contiguous()
            alpha, vt, _vw, vb, _ve = TS.split_view(g, w2c, kmat(intr_t[v], dev), H, W, mopt)
            _a, twv, tiv, cnt = token_topk(g, w2c, intr_t[v], H, W, mopt)
            thick = vt.clamp_min(0).sqrt() / (S if src == "scannetpp" else dep[v].clamp_min(1e-6))
            share = (vb / vt.clamp_min(1e-18)).clamp(0, 1)
            m = alpha > 0.5
            thick_maps[v] = (thick, m, twv, tiv)
            dg = d_gt_t[v].to(dev)
            okv = ok_t[v].to(dev)
            err = torch.where(okv, (dep[v] - dg).abs() / dg.clamp_min(1e-6), torch.zeros_like(dg))
            dmax = float(dg[okv].quantile(0.98)) if okv.any() else float(dep[v].quantile(0.98))
            dg_show = F.max_pool2d(dg[None, None], 3, 1, 1)[0, 0] if src == "dl3dv" else dg
            tlo, thi = (0, 0.3) if src == "scannetpp" else (0, 0.3)
            rows.append([label(rgb_img(dm["images_output"][0, v]), f"GT target {v}"),
                         label(rgb_img(img[v]), f"pred {psnr[v]:.2f} dB"),
                         label(map_img((img[v] - dm["images_output"][0, v]).abs().mean(0), 0, 0.25), "|RGB err|"),
                         label(map_img(dep[v], 0, dmax, alp[v] > 0.05), "depth pred"),
                         label(map_img(dg_show, 0, dmax, dg_show > 0), "depth GT (laser)" if src == "scannetpp" else "depth COLMAP"),
                         label(map_img(F.max_pool2d(err[None, None], 3, 1, 1)[0, 0] if src == "dl3dv" else err, 0, 0.3,
                                       (F.max_pool2d(okv.float()[None, None], 3, 1, 1)[0, 0] > 0) if src == "dl3dv" else okv), "AbsRel 0-0.3"),
                         label(map_img(thick, tlo, thi, m), "ray thickness 0-30cm" if src == "scannetpp" else "thickness/depth 0-0.3"),
                         label(map_img(share, 0, 1, m), "between-token share"),
                         label(map_img(cnt, 0, 20, m), "#tokens >=1% (0-20)")])
        grid_image(rows).save(mdir / "render_strip.png")

        # 6. ray probes: fixed grid pixels + worst-thickness pixels of this model, in target view pv
        thick, m, twv, tiv = thick_maps[pv]
        valid = m & (ok_t[pv].to(dev) if src == "scannetpp" else m)
        probes = probe_fixed + worst_pixels(thick, valid, 6)
        kinds = ["fixed"] * len(probe_fixed) + ["worst_thickness"] * (len(probes) - len(probe_fixed))
        c2w_t = torch.linalg.inv(cam_t[pv].float().T)
        o = c2w_t[:3, 3]
        fx, fy, cx, cy = intr_t[pv].tolist()
        ray_pts, ray_col, ptoks, pinfo = [], [], [], []
        for n_, (y, x) in enumerate(probes):
            dcam = torch.tensor([(x + 0.5 - cx) / fx, (y + 0.5 - cy) / fy, 1.0], device=dev)
            dirw = c2w_t[:3, :3] @ dcam
            zr, zg = float(dep[pv, y, x]), float(d_gt_t[pv, y, x])
            zmax = max(zr, zg) * 1.6
            ray_pts.append(segment(o, o + dirw * zmax, 300))
            ray_col.append(torch.tensor([[1.0, 1.0, 1.0]]).expand(300, 3))
            ray_pts.append(ball(o + dirw * zr, 0.006))
            ray_col.append(torch.tensor([[1.0, 0.9, 0.0]]).expand(60, 3))
            if zg > 0:
                ray_pts.append(ball(o + dirw * zg, 0.006))
                ray_col.append(torch.tensor([[0.0, 1.0, 0.2]]).expand(60, 3))
            w = twv[y * W + x]
            ids = tiv[y * W + x]
            keep = w >= 0.01 * float(alp[pv, y, x])
            toks = ids[keep].tolist()
            tok_depth = [float((cam_t[pv].float().T[:3, :3] @ Xp[t_] + cam_t[pv].float().T[:3, 3])[2]) for t_ in toks]
            pinfo.append({"probe": n_, "kind": kinds[n_], "pixel_yx": [y, x], "rendered_depth": zr, "gt_depth": zg,
                          "thickness": float(thick[y, x]), "n_tokens_ge_1pct": len(toks),
                          "tokens": [{"token": int(t_), "weight": round(float(wt), 4), "token_centre_depth": round(dz, 4)}
                                     for t_, wt, dz in zip(toks, w[keep].tolist(), tok_depth)]})
            ptoks += [(n_, t_) for t_ in toks]
        write_points(mdir / "ray_probes_rays.ply", torch.cat(ray_pts), torch.cat(ray_col))
        ut = sorted({t_ for _, t_ in ptoks})
        pmem = torch.isin(tok_id, torch.tensor(ut, device=dev))
        pcol = palette(ntok, 3).to(dev)[tok_id]
        pr_of = {t_: n_ for n_, t_ in ptoks}
        write_gaussians(mdir / "ray_probes_tokens.ply", g[pmem], pcol[pmem],
                        {"probe": torch.tensor([pr_of[int(t)] for t in tok_id[pmem]], dtype=torch.float32), **{k: v[pmem] for k, v in extra.items()}})
        write_points(mdir / "ray_probes_tokens_points.ply", g[pmem, :3], torch.where(dead[pmem].unsqueeze(1), pcol[pmem] * 0.25, pcol[pmem]),
                     {"probe": torch.tensor([pr_of[int(t)] for t in tok_id[pmem]], dtype=torch.float32), "opacity": op[pmem]})
        pim = rgb_img(dm["images_output"][0, pv])
        mark(pim, probes, [(0.0, 1.0, 1.0)] * len(probe_fixed) + [(1.0, 0.0, 1.0)] * (len(probes) - len(probe_fixed)))
        pim.save(mdir / "ray_probes_pixels.png")
        (mdir / "ray_probes.json").write_text(json.dumps(pinfo, indent=1), encoding="utf-8")

        # 7. LocusGS anchors
        if anchor is not None:
            dd = torch.cdist(anchor, anchor)
            dd.fill_diagonal_(float("inf"))
            nin = (dd < radius.unsqueeze(1)).sum(-1).float()
            depth_a = (anchor - cams[vm.argmax(-1)]).norm(dim=-1)
            write_points(mdir / "anchors_radius.ply", anchor, heat(radius / depth_a, 0.05, 0.25),
                         {"radius": radius, "radius_over_depth": radius / depth_a, "alive": alive_tok.float()})
            write_points(mdir / "anchors_crowding.ply", anchor, heat(torch.log1p(nin), 0, np.log1p(300)),
                         {"anchors_within_own_radius": nin, "nn_over_radius": dd.min(-1).values / radius})
            off = (Xp - anchor).norm(dim=-1) / depth_a
            far_t = torch.nonzero(alive_tok & (off > 0.1)).squeeze(1)
            if far_t.numel():
                lines = torch.cat([segment(anchor[t_], Xp[t_], 40) for t_ in far_t.tolist()])
                write_points(mdir / "anchor_to_gaussian_centre_offsets.ply", lines, torch.tensor([[1.0, 0.5, 0.0]]).expand(lines.shape[0], 3))
            st["anchor"] = {"radius_over_depth_median": float((radius / depth_a)[alive_tok].median()),
                            "anchors_within_own_radius_median": float(nin[alive_tok].median()),
                            "tokens_with_anchor_to_centre_gt_0.1_depth": int(far_t.numel()),
                            "alive_children_beyond_1r_frac": float(((extra["offset_over_radius"] > 1) & al).sum() / al.sum())}
        # scene numbers
        okd = ok_t[views].to(dev) & (alp[views] > 0.5)
        ab = ((dep[views] - d_gt_t[views].to(dev)).abs() / d_gt_t[views].to(dev).clamp_min(1e-6))[okd]
        tks = torch.cat([thick_maps[v][0][thick_maps[v][1]] for v in views])
        st.update({"psnr_all_targets": float(psnr.mean()), "psnr_shown_views": [round(float(psnr[v]), 3) for v in views],
                   "depth_absrel_shown_views": float(ab.mean()), "thickness_shown_views_p50_p90": [float(tks.median()), float(tks.quantile(0.9))],
                   "junk_share": float(junk.float().mean()), "visible_gaussians_dist_p50_p90": [float(dist[vis_g].median()), float(dist[vis_g].quantile(0.9))],
                   "c_centroid_mean": float(ccent.mean()), "token_parallax_lt2deg_alive": float((par[alive_tok] < 2).float().mean()),
                   "tokens20": picks, "tokens20_render_in_attention90": [round(x, 3) for x in own],
                   "ray_probe_tokens_per_pixel": [len(pi["tokens"]) for pi in pinfo]})
        (mdir / "summary.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
        summary["models"][name] = st
        print(f"[{scene[:12]}] {name}: done ({json.dumps({k: st[k] for k in ('psnr_all_targets', 'junk_share', 'c_centroid_mean')})})", flush=True)
        del model, mr, g, T, Rimg
        torch.cuda.empty_cache()
    (sdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
