"""Is there multi-view evidence the models fail to use for correcting geometry?  (pre-registered protocol:
EXP_GeometryCorrectability/README.md; ScanNet++ laser mesh as ground truth; FP32, TF32 off; frozen models)

  conflict  E1  laser-planar interior pixels of the 50 val target views: normal-direction error of Gaussian centre depths
                split exactly into common bias^2 + between-token spread^2 + within-token spread^2
  recover   E2  LocusGS: perturb the anchors of 15 % of alive tokens at the input of decoder layer 3 / 7 / 12
                (along the dominant-view ray +-5 % / +-10 % depth, or 10 % perpendicular) and measure recovery
                r = 1 - <final displacement, delta> / |delta|^2 of the final anchor and the final Gaussian centre
  fitpca / extract / probe   E3  frozen features of the 6 input views, 31 depth candidates z(1+o), o in [-0.3, 0.3]
                along each target pixel ray around the model depth; a small scorer sees only multi-view agreement
                (mean / variance / pairwise cosine / #valid views of the sampled features, no offset, no depth) and is
                trained on 160 ScanNet++ train scenes, evaluated on the 50 val scenes.  Controls: RGB 3x3 scorer,
                direct residual regression from the same features at the model point, photometric 3x3 NCC variance
                (no training), nearest-candidate oracle.
  summarize / demo

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/verify_geometry_correctability.py <cmd> ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import probe_scene_latent_limit as P
from demo_query_diagnosis import (SFT12, ball, grid_image, heat, label, load_demo_model, map_img, rgb_img, segment,
                                  write_points)
from diag_dl3dv import kmat, load_any
from diag_locusgs_deep import locus_forward
from diag_token_render_ownership import rasterize_channels
from diagnose_latent_attention_geometry import ALIVE_OPACITY
from evaluate_scannetpp import ROOT, idle_gpu
from evaluate_scannetpp_query_models import build_raycasting_scene_first_cam
from probe_gs_token_geometry import child_centre
from scannetpp_evaluation_protocol import build_raycasting_scene, normalize_world_transform, raycast_gt_depth_and_normals
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data

OUT = ROOT / "EXP_GeometryCorrectability"
CACHE = Path("/mnt/disk/xveyan/workspace_scannetpp/geometry_correctability")
S, CH, CHUNK = P.S, 64, 512
DEMO_SCENE = "7b6477cb95"
MODELS = {"release": ("tokengs_release_latent_6v", "mean_cam"), "sft12": (SFT12, "mean_cam"),
          "locus": ("locusgs_finetune_4v", "first_cam"), "paper": ("tokengs_paper_4v", "first_cam")}
FAMILIES = {"tokengs": ("release", ["release", "sft12"]), "locus": ("locus", ["locus"])}   # feature model, depth sources
OFFS = torch.linspace(-0.3, 0.3, 31)
N_TRAIN, N_PCA, PCA_DIM = 160, 12, 256
PLANAR_WIN, PLANAR_TOL_M, PLANAR_ANG = 9, 0.01, 10.0
PERTURB_LAYERS, PERTURB_FRAC = (3, 7, 12), 0.15
PERTURB_KINDS = (("depth", 0.05), ("depth", 0.10), ("lateral", 0.10))
VARIANTS = ("feat_cand", "rgb_cand", "feat_direct")


def setup_torch():
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return gpu, torch.device("cuda:0")


# ------------------------------------------------------------------------------------------------ models / scenes
class Model:
    def __init__(self, key, dev):
        name, self.frame = MODELS[key]
        self.key, self.dev = key, dev
        if self.frame == "mean_cam":
            self.net, self.opt, self.kind = load_demo_model(name, dev)
        else:
            self.net, self.opt, _d, self.kind = load_any(name, dev)

    def _locus_input(self, data):
        from locusgs.models.input_types import split_data as ls
        return ls(data, self.opt)[0]

    def gaussians(self, data):
        if self.kind == "release":
            rec = self.net.forward_reconstruction(split_data(data, self.opt)[0])
            return rec.gaussians.float(), rec.background_color
        mi = self._locus_input(data)
        G = self.net.forward_decoder(self.net.forward_encoder(mi.encoder), mi.decoder)
        return G.float(), self.net._background_color(torch.float32, self.dev)

    def features(self, data, hp, wp):
        if self.kind == "release":
            mi = split_data(data, self.opt)[0]
            f = self.net.enc_dec_backbone._encode_features(self.net._embed_encoder_input(mi.encoder))[0]
            return f.reshape(-1, hp, wp, f.shape[-1])
        return self.net.forward_encoder(self._locus_input(data).encoder).image_features[0]

    def render(self, G, bg, cam, intr):
        return self.net.gs.render(G, cam, bg_color=bg, intrinsics=intr)


def provider(split, frame):
    opt = P.make_opt(split)
    if frame == "first_cam":
        opt = dataclasses.replace(opt, camera_normalization_method="first_cam")
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov.dataset._offset_frac = None
    return prov, opt


def train_subset(n_total):
    return sorted(set(np.linspace(0, n_total - 1, N_TRAIN).round().astype(int).tolist()))


def load_scene(prov, idx, frame, dev):
    ds = prov.dataset
    sid = Path(ds.sample_list[idx]).name
    data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in prov[idx].items()}
    ctx, _t = ds.get_context_target_frames(idx)
    raw_c2w = torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][ctx].copy()).float()
    mesh = ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply"
    rc = build_raycasting_scene(mesh, raw_c2w, S) if frame == "mean_cam" else build_raycasting_scene_first_cam(mesh, raw_c2w, S)
    return sid, data, raw_c2w, rc


def shard(items, k, n):
    return items[k::n]


def pixel_dirs(intr, H, W, dev):
    fx, fy, cx, cy = intr.float().tolist()
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    return torch.stack([(xx + 0.5 - cx) / fx, (yy + 0.5 - cy) / fy, torch.ones_like(xx, dtype=torch.float32)], -1)


# ------------------------------------------------------------------------------------------------ E1 same-surface conflict
@torch.no_grad()
def depth_moments(g, viewmat, K, H, W, opt):
    """Per pixel alpha, weighted mean centre depth, within-token and between-token centre-depth variance (exact)."""
    N = g.shape[0]
    tok = torch.arange(N, device=g.device) // CH
    z = (g[:, :3] @ viewmat[:3, :3].T + viewmat[:3, 3])[:, 2].clamp_min(0)
    tot, alpha = rasterize_channels(g, torch.stack([z, z * z], -1).contiguous(), viewmat, K, H, W, opt)
    S1 = torch.zeros((H, W), device=g.device)
    for c0 in range(0, N // CH, CHUNK):
        k = min(CHUNK, N // CH - c0)
        sel = (tok >= c0) & (tok < c0 + k)
        oh = torch.zeros((N, k), device=g.device)
        oh[sel, tok[sel] - c0] = 1.0
        Wt, _ = rasterize_channels(g, oh, viewmat, K, H, W, opt)
        At, _ = rasterize_channels(g, oh * z.unsqueeze(-1), viewmat, K, H, W, opt)
        S1 += (At.square() / Wt.clamp_min(1e-12)).sum(-1)
    a = alpha.clamp_min(1e-6)
    A, B = tot[..., 0], tot[..., 1]
    return alpha, A / a, ((B - S1) / a).clamp_min(0), ((S1 - A * A / a) / a).clamp_min(0)


def planar_mask(d, n, ok, dcam):
    """Laser planar interior: whole 9x9 window valid, every point within 1 cm of the centre tangent plane, normals < 10 deg."""
    H, W = d.shape
    Pt = (dcam * d.unsqueeze(-1)).permute(2, 0, 1)[None]
    Nt = n.permute(2, 0, 1)[None]
    k, pad = PLANAR_WIN, PLANAR_WIN // 2
    Q = F.unfold(Pt, k, padding=pad).view(3, k * k, H, W)
    Nq = F.unfold(Nt, k, padding=pad).view(3, k * k, H, W)
    okq = F.unfold(ok.float()[None, None], k, padding=pad).view(k * k, H, W) > 0.5
    dist = ((Q - Pt[0].unsqueeze(1)) * Nt[0].unsqueeze(1)).sum(0).abs()
    cosang = (Nq * Nt[0].unsqueeze(1)).sum(0)
    return ok & okq.all(0) & (dist < PLANAR_TOL_M * S).all(0) & (cosang > math.cos(math.radians(PLANAR_ANG))).all(0)


@torch.no_grad()
def cmd_conflict(a):
    gpu, dev = setup_torch()
    rec = {"gpu": gpu["index"], "models": {}}
    for key in a.models:
        m = Model(key, dev)
        prov, opt = provider("val", m.frame)
        H, W = map(int, opt.img_size)
        acc = {g: {k: [] for k in ("bias", "between", "within")} for g in ("planar", "other")}
        sums = {g: torch.zeros(3, dtype=torch.float64) for g in acc}
        for n_, idx in enumerate(shard(list(range(len(prov))), a.shard, a.num_shards)):
            sid, data, _raw, rc = load_scene(prov, idx, m.frame, dev)
            G, _bg = m.gaussians(data)
            cam, intr = data["cam_view"][0], data["intrinsics"][0]
            d_gt, n_gt, ok = raycast_gt_depth_and_normals(rc, cam.cpu(), intr.cpu(), opt.img_size)
            for v in range(cam.shape[0]):
                viewmat = cam[v].float().T.contiguous()
                alpha, zbar, vw, vb = depth_moments(G[0], viewmat, kmat(intr[v], dev), H, W, m.opt)
                dcam = pixel_dirs(intr[v], H, W, dev)
                d, nn_, okv = d_gt[v].to(dev), n_gt[v].to(dev), ok[v].to(dev)
                s = ((dcam * nn_).sum(-1)).abs()
                pl = planar_mask(d, nn_, okv, dcam)
                valid = okv & (alpha > 0.5)
                vals = {"bias": s * (zbar - d) / S, "between": s * vb.sqrt() / S, "within": s * vw.sqrt() / S}
                for g, mk in (("planar", valid & pl), ("other", valid & ~pl)):
                    for k, x in vals.items():
                        acc[g][k].append(x[mk].half().cpu())
                    sums[g] += torch.tensor([float((vals["bias"][mk] ** 2).sum()), float((vals["between"][mk] ** 2).sum()),
                                             float((vals["within"][mk] ** 2).sum())], dtype=torch.float64)
            print(f"[{key} shard {a.shard}] {n_ + 1} {sid}", flush=True)
            del G
            torch.cuda.empty_cache()
        rec["models"][key] = {"values": {g: {k: torch.cat(x) for k, x in d.items()} for g, d in acc.items()}, "sums": sums}
        del m
        torch.cuda.empty_cache()
    (CACHE / "conflict").mkdir(parents=True, exist_ok=True)
    torch.save(rec, CACHE / "conflict" / f"shard_{a.shard}_{'_'.join(a.models)}.pt")


# ------------------------------------------------------------------------------------------------ E2 recovery (LocusGS)
@torch.no_grad()
def cmd_recover(a):
    gpu, dev = setup_torch()
    m = Model("locus", dev)
    from locusgs.models.input_types import split_data as ls
    prov, opt = provider("val", "first_cam")
    out = []
    for n_, idx in enumerate(shard(list(range(len(prov))), a.shard, a.num_shards)):
        sid, data, _raw, rc = load_scene(prov, idx, "first_cam", dev)
        import open3d as o3d

        def sd(x):
            return torch.from_numpy(rc.compute_distance(o3d.core.Tensor(x.cpu().numpy().astype(np.float32))).numpy()).to(dev) / S
        mi = ls(data, m.opt)[0]
        base = locus_forward(m.net, m.opt, mi, capture=(12,))
        An, G = base["anchors"], base["G"]
        ntok = G.shape[1] // CH
        V = data["cam_view_input"].shape[1]
        vm = base["caps"][12]["attn"].reshape(ntok, V, -1).sum(-1)
        dom, nviews = vm.argmax(-1), (vm > 0.1).sum(-1)
        del base
        g64 = G[0].reshape(ntok, CH, 14)
        alive = (g64[..., 3] * (g64[..., 3] >= ALIVE_OPACITY)).sum(-1) > 0.5
        Xb = child_centre(g64)
        cams = torch.stack([torch.linalg.inv(c.float().T)[:3, 3] for c in data["cam_view_input"][0]])
        gen = torch.Generator(device="cpu").manual_seed(1000 + idx)
        al = torch.nonzero(alive).squeeze(1)
        sel = al[torch.randperm(al.numel(), generator=gen)[: int(PERTURB_FRAC * al.numel())].to(dev)]
        others = al[~torch.isin(al, sel)]
        dist_b = sd(Xb[sel])
        rgb_t = data["images_output"][0]

        def psnr(Gx):
            r = m.render(Gx, m.net._background_color(torch.float32, dev), data["cam_view"], data["intrinsics"])["images_pred"][0].clamp(0, 1)
            return float((-10 * (r - rgb_t).square().mean((1, 2, 3)).log10()).mean())
        rec = {"scene_id": sid, "psnr_base": psnr(G), "nviews": nviews[sel].cpu().short(), "dist_base_m": dist_b.cpu().half(), "conds": {}}
        for layer in PERTURB_LAYERS:
            st = An[layer - 1][sel]
            ray = st - cams[dom[sel]]
            depth = ray.norm(dim=-1, keepdim=True)
            ray = ray / depth
            sign = (torch.randint(0, 2, (sel.numel(), 1), generator=gen).float() * 2 - 1).to(dev)
            rnd = torch.randn(sel.numel(), 3, generator=gen).to(dev)
            lat = F.normalize(rnd - (rnd * ray).sum(-1, keepdim=True) * ray, dim=-1)
            for kind, mag in PERTURB_KINDS:
                delta = (ray * sign if kind == "depth" else lat) * mag * depth
                run = locus_forward(m.net, m.opt, mi, perturb={layer: (sel, delta)})
                Gp = run["G"]
                Xp = child_centre(Gp[0].reshape(ntok, CH, 14))
                dA, dG = run["anchors"][12][sel] - An[12][sel], Xp[sel] - Xb[sel]
                d2 = (delta ** 2).sum(-1)
                rec["conds"][f"L{layer}_{kind}{int(mag * 100)}"] = {
                    "r_anchor": (1 - (dA * delta).sum(-1) / d2).cpu().half(), "r_gauss": (1 - (dG * delta).sum(-1) / d2).cpu().half(),
                    "delta_m": (d2.sqrt() / S).cpu().half(), "dist_pert_m": sd(Xp[sel]).cpu().half(),
                    "collateral_median_m": float((Xp[others] - Xb[others]).norm(dim=-1).median() / S), "psnr": psnr(Gp)}
                del run, Gp
        out.append(rec)
        print(f"[recover shard {a.shard}] {n_ + 1} {sid} L7 depth10 r_gauss median "
              f"{float(rec['conds']['L7_depth10']['r_gauss'].float().median()):.3f}", flush=True)
        torch.cuda.empty_cache()
    (CACHE / "recover").mkdir(parents=True, exist_ok=True)
    torch.save(out, CACHE / "recover" / f"shard_{a.shard}.pt")


# ------------------------------------------------------------------------------------------------ E3 extraction
@torch.no_grad()
def cmd_fitpca(a):
    gpu, dev = setup_torch()
    fkey, _src = FAMILIES[a.family]
    m = Model(fkey, dev)
    prov, opt = provider("train", m.frame)
    H, W = map(int, opt.img_size)
    xs = []
    gen = torch.Generator(device="cpu").manual_seed(0)
    for idx in train_subset(len(prov))[:N_PCA]:
        data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in prov[idx].items()}
        f = m.features(data, H // 8, W // 8)
        f = f.reshape(-1, f.shape[-1])
        xs.append(f[torch.randperm(f.shape[0], generator=gen)[:4096].to(dev)].double())
    X = torch.cat(xs)
    mu = X.mean(0)
    evals, evecs = torch.linalg.eigh(((X - mu).T @ (X - mu)) / (X.shape[0] - 1))
    comps = evecs[:, -PCA_DIM:].flip(-1)
    expl = float(evals[-PCA_DIM:].sum() / evals.sum())
    (CACHE / a.family).mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mu.float().cpu(), "comps": comps.float().cpu(), "explained": expl, "dim_in": X.shape[1]}, CACHE / a.family / "pca.pt")
    print(f"PCA {a.family}: {X.shape[1]} -> {PCA_DIM}, explained variance {expl:.3f}", flush=True)


@torch.no_grad()
def cmd_extract(a):
    gpu, dev = setup_torch()
    fkey, srcs = FAMILIES[a.family]
    fm = Model(fkey, dev)
    models = {s: (fm if s == fkey else Model(s, dev)) for s in srcs}
    prov, opt = provider(a.split, fm.frame)
    H, W = map(int, opt.img_size)
    pca = torch.load(CACHE / a.family / "pca.pt")
    mu, comps = pca["mean"].to(dev), pca["comps"].to(dev)
    items = train_subset(len(prov)) if a.split == "train" else list(range(len(prov)))
    odir = CACHE / a.family / a.split
    odir.mkdir(parents=True, exist_ok=True)
    for n_, idx in enumerate(shard(items, a.shard, a.num_shards)):
        sid, data, _raw, rc = load_scene(prov, idx, fm.frame, dev)
        if (odir / f"{sid}.pt").is_file():
            continue
        f = fm.features(data, H // 8, W // 8)
        feats = ((f - mu) @ comps).permute(0, 3, 1, 2).half().cpu()
        cam, intr = data["cam_view"][0], data["intrinsics"][0]
        cam_in, intr_in = data["cam_view_input"][0], data["intrinsics_input"][0]
        pred = {}
        for s, m in models.items():
            G, bg = m.gaussians(data)
            r = m.render(G, bg, data["cam_view"], data["intrinsics"])
            pred[s] = {"depth": r["depths_pred"][0, :, 0].half().cpu(), "alpha": r["alphas_pred"][0, :, 0].half().cpu()}
        d_gt, _n, ok = raycast_gt_depth_and_normals(rc, cam.cpu(), intr.cpu(), opt.img_size)
        d_in, _n2, ok_in = raycast_gt_depth_and_normals(rc, cam_in.cpu(), intr_in.cpu(), opt.img_size)
        vis = torch.zeros_like(d_gt, dtype=torch.uint8)
        for v in range(cam.shape[0]):
            dcam = pixel_dirs(intr[v], H, W, "cpu")
            c2w = torch.linalg.inv(cam[v].float().T.cpu())
            Pw = (dcam * d_gt[v].unsqueeze(-1)) @ c2w[:3, :3].T + c2w[:3, 3]
            for j in range(cam_in.shape[0]):
                w2c = cam_in[j].float().T.cpu()
                Xc = Pw @ w2c[:3, :3].T + w2c[:3, 3]
                fx, fy, cx, cy = intr_in[j].float().cpu().tolist()
                z = Xc[..., 2]
                u = (fx * Xc[..., 0] / z.clamp_min(1e-6) + cx).floor().long()
                vv = (fy * Xc[..., 1] / z.clamp_min(1e-6) + cy).floor().long()
                inb = (z > 1e-3) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
                uc, vc = u.clamp(0, W - 1), vv.clamp(0, H - 1)
                dz = d_in[j][vc, uc]
                vis[v] += (inb & ok_in[j][vc, uc] & ((z - dz).abs() < 0.02 * z) & ok[v]).to(torch.uint8)
        torch.save({"scene_id": sid, "feats": feats, "pred": pred, "d_gt": d_gt.half(), "ok": ok, "vis": vis,
                    "images_input": data["images_input"][0].half().cpu(), "cam_view_input": cam_in.float().cpu(),
                    "intrinsics_input": intr_in.float().cpu(), "cam_view": cam.float().cpu(), "intrinsics": intr.float().cpu()},
                   odir / f"{sid}.tmp")
        (odir / f"{sid}.tmp").rename(odir / f"{sid}.pt")
        print(f"[extract {a.family}/{a.split} shard {a.shard}] {n_ + 1} {sid}", flush=True)
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ E3 probe
def cand_grid(c, pv, py, px, z0, offs):
    """Projections of candidates z0 * (1 + offs) on target pixels into the input views -> grid [V, P, K, 2], valid [V, P, K]."""
    intr = c["intrinsics"][pv]
    d = torch.stack([(px + 0.5 - intr[:, 2]) / intr[:, 0], (py + 0.5 - intr[:, 3]) / intr[:, 1], torch.ones_like(z0)], -1)
    z = z0[:, None] * (1 + offs)[None]
    Xc = d[:, None, :] * z[..., None]
    c2w = torch.linalg.inv(c["cam_view"][pv].transpose(1, 2))
    Xw = torch.einsum("pij,pkj->pki", c2w[:, :3, :3], Xc) + c2w[:, None, :3, 3]
    w2c = c["cam_view_input"].transpose(1, 2)
    Xi = torch.einsum("vij,pkj->vpki", w2c[:, :3, :3], Xw) + w2c[:, None, None, :3, 3]
    ii = c["intrinsics_input"]
    zi = Xi[..., 2]
    u = ii[:, 0, None, None] * Xi[..., 0] / zi.clamp_min(1e-6) + ii[:, 2, None, None]
    v = ii[:, 1, None, None] * Xi[..., 1] / zi.clamp_min(1e-6) + ii[:, 3, None, None]
    H, W = c["d_gt"].shape[-2:]
    valid = (zi > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return torch.stack([2 * u / W - 1, 2 * v / H - 1], -1), valid


class Scorer(nn.Module):
    def __init__(self, cin, direct=False, d=96):
        super().__init__()
        self.proj = nn.Conv2d(cin, d, 1)
        self.direct = direct
        self.mlp = nn.Sequential(nn.Linear(2 * d + 2, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))

    def agg(self, maps, grid, valid):
        f = F.normalize(F.grid_sample(self.proj(maps), grid, mode="bilinear", align_corners=False), dim=1)  # [V, d, P, K]
        m = valid.float().unsqueeze(1)
        n = m.sum(0).clamp_min(1)
        mu = (f * m).sum(0) / n
        var = ((f - mu) ** 2 * m).sum(0) / n
        s = (f * m).sum(0)
        nv = valid.float().sum(0)
        cos = ((s * s).sum(0) - nv) / (nv * (nv - 1)).clamp_min(1)
        return torch.cat([mu, var, cos[None], nv[None] / valid.shape[0]], 0).permute(1, 2, 0)   # [P, K, 2d + 2]

    def forward(self, maps, grid, valid):
        out = self.mlp(self.agg(maps, grid, valid)).squeeze(-1)
        return 0.3 * torch.tanh(out[:, 0]) if self.direct else out


def rgb_maps(img):  # [V, 3, H, W] -> [V, 27, H, W]
    V, _c, H, W = img.shape
    return F.unfold(img, 3, padding=1).view(V, 27, H, W)


def photometric_select(rmaps, grid, valid):
    p = F.grid_sample(rmaps, grid, mode="bilinear", align_corners=False)   # [V, 27, P, K]
    p = p - p.mean(1, keepdim=True)
    p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-6)
    m = valid.float().unsqueeze(1)
    n = m.sum(0).clamp_min(1)
    mu = (p * m).sum(0) / n
    cost = (((p - mu) ** 2) * m).sum((0, 1)) / n[0]
    cost = torch.where(valid.sum(0) >= 2, cost, torch.full_like(cost, float("inf")))
    k = cost.argmin(-1)
    return torch.where(torch.isinf(cost).all(-1), torch.full_like(k, len(OFFS) // 2), k)


def to_dev(c, dev, src):
    out = {k: (v.to(dev).float() if torch.is_tensor(v) and v.is_floating_point() else (v.to(dev) if torch.is_tensor(v) else v)) for k, v in c.items() if k != "pred"}
    out["z0"], out["alpha"] = c["pred"][src]["depth"].to(dev).float(), c["pred"][src]["alpha"].to(dev).float()
    return out


def sample_pixels(c, n, gen, in_range_only):
    valid = c["ok"] & (c["alpha"] > 0.5) & (c["z0"] > 1e-3)
    if in_range_only:
        valid &= ((c["d_gt"] / c["z0"].clamp_min(1e-6) - 1).abs() <= 0.31)
    idx = torch.nonzero(valid)
    if n and idx.shape[0] > n:
        idx = idx[torch.randperm(idx.shape[0], generator=gen, device="cpu")[:n].to(idx.device)]
    return idx[:, 0], idx[:, 1], idx[:, 2]


def run_variant(model, variant, c, pv, py, px, offs):
    z0 = c["z0"][pv, py, px]
    if variant == "feat_direct":
        grid, valid = cand_grid(c, pv, py, px, z0, offs[len(offs) // 2:len(offs) // 2 + 1])
        return model(c["feats"], grid, valid)
    grid, valid = cand_grid(c, pv, py, px, z0, offs)
    return model(c["feats"] if variant == "feat_cand" else c["rgb"], grid, valid)


def cmd_probe(a):
    gpu, dev = setup_torch()
    fam_dir = CACHE / a.family
    train = sorted((fam_dir / "train").glob("*.pt"))
    val = sorted((fam_dir / "val").glob("*.pt"))
    if (len(train) != N_TRAIN or len(val) != 50) and not a.smoke:
        raise RuntimeError(f"incomplete caches: train {len(train)} val {len(val)}")
    load = lambda f: torch.load(f, map_location="cpu", weights_only=False)  # noqa: E731
    tr = [load(f) for f in train]
    offs = OFFS.to(dev)
    gen = torch.Generator(device="cpu").manual_seed(0)
    models = {}
    for variant in VARIANTS:
        torch.manual_seed(0)
        cin = 27 if variant == "rgb_cand" else tr[0]["feats"].shape[1]
        net = Scorer(cin, direct=variant == "feat_direct").to(dev)
        optim = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=1e-3, total_steps=a.steps, pct_start=0.05)
        t0, run_loss = time.time(), 0.0
        for step in range(a.steps):
            c = to_dev(tr[int(torch.randint(len(tr), (1,), generator=gen))], dev, a.source)
            if variant == "rgb_cand":
                c["rgb"] = rgb_maps(c["images_input"])
            pv, py, px = sample_pixels(c, a.pixels, gen, in_range_only=True)
            if pv.numel() < 64:
                continue
            z0, zg = c["z0"][pv, py, px], c["d_gt"][pv, py, px]
            out = run_variant(net, variant, c, pv, py, px, offs)
            if variant == "feat_direct":
                loss = (out - torch.log(zg / z0).clamp(-0.3, 0.3)).abs().mean()
            else:
                lab = (z0[:, None] * (1 + offs)[None] - zg[:, None]).abs().argmin(-1)
                loss = F.cross_entropy(out, lab)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            run_loss += float(loss)
            if (step + 1) % 500 == 0:
                print(f"[probe {a.family}/{a.source}] {variant} step {step + 1}/{a.steps} loss {run_loss / 500:.4f} {time.time() - t0:.0f}s", flush=True)
                run_loss = 0.0
        net.eval()
        models[variant] = net
        (CACHE / "probe").mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), CACHE / "probe" / f"{a.family}_{a.source}_{variant}.pt")
    del tr
    rows = []
    with torch.no_grad():
        for f in val:
            c = to_dev(load(f), dev, a.source)
            c["rgb"] = rgb_maps(c["images_input"])
            valid = c["ok"] & (c["alpha"] > 0.5) & (c["z0"] > 1e-3)
            valid[:, 1::2] = False
            valid[:, :, 1::2] = False
            pv, py, px = torch.nonzero(valid, as_tuple=True)
            for c0 in range(0, pv.numel(), 8192):
                sl = slice(c0, c0 + 8192)
                v_, y_, x_ = pv[sl], py[sl], px[sl]
                z0, zg = c["z0"][v_, y_, x_], c["d_gt"][v_, y_, x_]
                zk = z0[:, None] * (1 + offs)[None]
                rec = {"model": (z0 - zg).abs() / zg, "oracle": (zk - zg[:, None]).abs().min(-1).values / zg}
                for variant in ("feat_cand", "rgb_cand"):
                    k = run_variant(models[variant], variant, c, v_, y_, x_, offs).argmax(-1)
                    rec[variant] = (zk.gather(1, k[:, None])[:, 0] - zg).abs() / zg
                rec["feat_direct"] = (z0 * torch.exp(run_variant(models["feat_direct"], "feat_direct", c, v_, y_, x_, offs)) - zg).abs() / zg
                grid, vmask = cand_grid(c, v_, y_, x_, z0, offs)
                kp = photometric_select(c["rgb"], grid, vmask)
                rec["photometric"] = (zk.gather(1, kp[:, None])[:, 0] - zg).abs() / zg
                rec["in_range"] = ((zg / z0 - 1).abs() <= 0.31).float()
                rec["vis"] = c["vis"][v_, y_, x_].float()
                rec["scene"] = torch.full_like(z0, val.index(f))
                rows.append({k: v.half().cpu() if k not in ("scene",) else v.short().cpu() for k, v in rec.items()})
    cols = {k: torch.cat([r[k] for r in rows]).float() for k in rows[0]}
    torch.save(cols, CACHE / "probe" / f"{a.family}_{a.source}_val_pixels.pt")
    print(json.dumps(summarize_probe(cols), indent=1), flush=True)


def summarize_probe(c):
    meth = ("model", "feat_cand", "rgb_cand", "feat_direct", "photometric", "oracle")
    ir = c["in_range"] > 0.5
    A = ir & (c["model"] > 0.05) & (c["vis"] >= 2)
    B = c["model"] < 0.02
    res = {"pixels": int(c["model"].numel()), "in_range_frac": float(ir.float().mean()),
           "group_A_frac_of_all(model>5%,vis>=2,in_range)": float(A.float().mean()),
           "mean_absrel_all_pixels": {k: round(float(c[k].mean()), 4) for k in meth},
           "mean_absrel_in_range": {k: round(float(c[k][ir].mean()), 4) for k in meth},
           "group_A": {k: {"mean_absrel": round(float(c[k][A].mean()), 4), "reduction_vs_model": round(1 - float(c[k][A].mean() / c["model"][A].mean()), 4),
                           "corrected_lt_2pct": round(float((c[k][A] < 0.02).float().mean()), 4)} for k in meth},
           "harm_on_model_lt_2pct(err>5%)": {k: round(float((c[k][B] > 0.05).float().mean()), 4) for k in meth if k != "model"},
           "evidence_but_uncorrected_frac_of_all(A & feat_cand<2%)": float((A & (c["feat_cand"] < 0.02)).float().mean())}
    byvis = {}
    for lo, hi, nm in ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 99, "3+")):
        mk = ir & (c["vis"] >= lo) & (c["vis"] <= hi) & (c["model"] > 0.05)
        if mk.any():
            byvis[nm] = {"share": round(float(mk.float().mean()), 4), **{k: round(float(c[k][mk].mean()), 4) for k in meth}}
    res["model_err_gt_5pct_in_range_by_visible_views"] = byvis
    byerr = {}
    for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, 0.31)):
        mk = ir & (c["model"] >= lo) & (c["model"] < hi)
        if mk.any():
            byerr[f"{lo}-{hi}"] = {"share": round(float(mk.float().mean()), 4), **{k: round(float(c[k][mk].mean()), 4) for k in meth}}
    res["in_range_by_model_error"] = byerr
    res["H1_supported"] = bool(res["group_A"]["feat_cand"]["reduction_vs_model"] >= 0.30 and res["harm_on_model_lt_2pct(err>5%)"]["feat_cand"] <= 0.10)
    res["H1b_supported"] = bool(res["mean_absrel_in_range"]["feat_cand"] <= 0.9 * res["mean_absrel_in_range"]["feat_direct"])
    return res


@torch.no_grad()
def cmd_resolution(a):
    """POST-HOC (not pre-registered): how far does one 2 % depth step move the candidate in the input images?
    Same pixel order as the probe evaluation; joined with the saved per-pixel errors and binned."""
    _gpu, dev = setup_torch()
    offs = torch.tensor([-0.01, 0.01, -0.3, 0.3], device=dev)
    step_px, range_px, nvalid = [], [], []
    for f in sorted((CACHE / a.family / "val").glob("*.pt")):
        c = to_dev(torch.load(f, map_location="cpu", weights_only=False), dev, a.source)
        valid = c["ok"] & (c["alpha"] > 0.5) & (c["z0"] > 1e-3)
        valid[:, 1::2] = False
        valid[:, :, 1::2] = False
        pv, py, px = torch.nonzero(valid, as_tuple=True)
        H, W = c["d_gt"].shape[-2:]
        scale = torch.tensor([W / 2, H / 2], device=dev)
        for c0 in range(0, pv.numel(), 32768):
            sl = slice(c0, c0 + 32768)
            grid, vm = cand_grid(c, pv[sl], py[sl], px[sl], c["z0"][pv[sl], py[sl], px[sl]], offs)
            uv = grid * scale
            ok2 = vm[..., 0] & vm[..., 1]
            okr = vm[..., 2] & vm[..., 3]
            st = torch.where(ok2, (uv[..., 1, :] - uv[..., 0, :]).norm(dim=-1), torch.zeros_like(ok2, dtype=torch.float32))
            rg = torch.where(okr, (uv[..., 3, :] - uv[..., 2, :]).norm(dim=-1), torch.zeros_like(okr, dtype=torch.float32))
            step_px.append(st.max(0).values.cpu()); range_px.append(rg.max(0).values.cpu()); nvalid.append(ok2.sum(0).cpu())
    cols = torch.load(CACHE / "probe" / f"{a.family}_{a.source}_val_pixels.pt")
    st, rg, nv = torch.cat(step_px), torch.cat(range_px), torch.cat(nvalid)
    if st.numel() != cols["model"].numel():
        raise RuntimeError("pixel order mismatch with the probe evaluation")
    meth = ("model", "feat_cand", "rgb_cand", "feat_direct", "photometric", "oracle")
    ir = cols["in_range"] > 0.5
    res = {"note": "POST-HOC, not pre-registered", "step_px_(max over input views, per 2% depth step)": q(st, (0.1, 0.25, 0.5, 0.75, 0.9)),
           "range_px_(max over input views, whole +-30%)": q(rg, (0.1, 0.25, 0.5, 0.75, 0.9)), "bins": {}}
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 1e9)):
        mk = ir & (st >= lo) & (st < hi)
        if mk.any():
            A = mk & (cols["model"] > 0.05) & (cols["vis"] >= 2)
            res["bins"][f"{lo}-{hi if hi < 1e9 else 'inf'}px"] = {
                "share_in_range": round(float(mk.float().sum() / ir.float().sum()), 4),
                "mean_absrel": {k: round(float(cols[k][mk].mean()), 4) for k in meth},
                "group_A_mean_absrel": {k: round(float(cols[k][A].mean()), 4) for k in meth} if A.any() else {}}
    out = OUT / "results_posthoc_resolution.json"
    allr = json.loads(out.read_text()) if out.is_file() else {}
    allr[f"{a.family}_{a.source}"] = res
    out.write_text(json.dumps(allr, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------ summarize
def q(x, ps=(0.5, 0.9)):
    x = x[torch.isfinite(x)].double().numpy()   # numpy: torch.quantile refuses > 2^24 elements
    return {f"p{int(p * 100)}": round(float(np.quantile(x, p)), 4) for p in ps} if x.size else {}


def cmd_summarize(_a):
    res = {"protocol": "EXP_GeometryCorrectability/README.md (pre-registered)"}
    files = sorted((CACHE / "conflict").glob("shard_*.pt"))
    conf = {}
    for f in files:
        r = torch.load(f, map_location="cpu", weights_only=False)
        for key, d in r["models"].items():
            e = conf.setdefault(key, {"values": {g: {k: [] for k in ("bias", "between", "within")} for g in ("planar", "other")},
                                      "sums": {g: torch.zeros(3, dtype=torch.float64) for g in ("planar", "other")}})
            for g in ("planar", "other"):
                for k in ("bias", "between", "within"):
                    e["values"][g][k].append(d["values"][g][k])
                e["sums"][g] += d["sums"][g]
    res["E1_conflict"] = {}
    for key, e in conf.items():
        out = {}
        for g in ("planar", "other"):
            v = {k: torch.cat(x).float() for k, x in e["values"][g].items()}
            s = e["sums"][g]
            sh = (s / s.sum()).tolist()
            out[g] = {"pixels": int(v["bias"].numel()), "abs_bias_m": q(v["bias"].abs()), "between_m": q(v["between"]), "within_m": q(v["within"]),
                      "normal_mse_share": {"bias": round(sh[0], 4), "between": round(sh[1], 4), "within": round(sh[2], 4)},
                      "rms_normal_error_m": round(float((s.sum() / max(v["bias"].numel(), 1)) ** 0.5), 4),
                      "between_gt_1cm_frac": round(float((v["between"] > 0.01).float().mean()), 4),
                      "between_gt_2cm_frac": round(float((v["between"] > 0.02).float().mean()), 4),
                      "between_gt_5cm_frac": round(float((v["between"] > 0.05).float().mean()), 4),
                      "pixels_where_between_exceeds_abs_bias": round(float((v["between"] > v["bias"].abs()).float().mean()), 4)}
        pl = out["planar"]
        out["verdict"] = ("same-surface conflict real and substantial" if pl["normal_mse_share"]["between"] >= 0.30 and pl["between_m"]["p50"] >= 0.01
                          else "mainly common bias, not between-token conflict" if pl["normal_mse_share"]["between"] < 0.15 else "intermediate")
        res["E1_conflict"][key] = out
    rec = []
    for f in sorted((CACHE / "recover").glob("shard_*.pt")):
        rec.extend(torch.load(f, map_location="cpu", weights_only=False))
    if rec:
        e2 = {"scenes": len(rec), "psnr_base_mean": round(float(np.mean([r["psnr_base"] for r in rec])), 3)}
        nv = torch.cat([r["nviews"].float() for r in rec])
        for cond in rec[0]["conds"]:
            rg = torch.cat([r["conds"][cond]["r_gauss"].float() for r in rec])
            ra = torch.cat([r["conds"][cond]["r_anchor"].float() for r in rec])
            dp = torch.cat([r["conds"][cond]["dist_pert_m"].float() for r in rec])
            db = torch.cat([r["dist_base_m"].float() for r in rec])
            groups = {"all": torch.ones_like(nv, dtype=torch.bool), "1 view": nv == 1, ">=2 views": nv >= 2}
            e2[cond] = {"delta_m": q(torch.cat([r["conds"][cond]["delta_m"].float() for r in rec])),
                        "psnr_mean": round(float(np.mean([r["conds"][cond]["psnr"] for r in rec])), 3),
                        "collateral_median_m": round(float(np.median([r["conds"][cond]["collateral_median_m"] for r in rec])), 4),
                        **{g: {"tokens": int(mk.sum()), "r_gauss": q(rg[mk], (0.25, 0.5, 0.75)), "r_anchor": q(ra[mk], (0.25, 0.5, 0.75)),
                               "surface_dist_increase_m": q((dp - db)[mk])} for g, mk in groups.items()}}
        g7 = e2["L7_depth10"][">=2 views"]["r_gauss"]["p50"]
        e2["verdict_L7_depth10_>=2views"] = ("cannot correct from observations" if g7 < 0.5 else "recovers" if g7 >= 0.8 else "partial")
        res["E2_recover_locus"] = e2
    res["E3_probe"] = {}
    for f in sorted((CACHE / "probe").glob("*_val_pixels.pt")):
        res["E3_probe"][f.stem.replace("_val_pixels", "")] = summarize_probe(torch.load(f, map_location="cpu"))
    for fam in FAMILIES:
        p = CACHE / fam / "pca.pt"
        if p.is_file():
            res.setdefault("E3_pca_explained_variance", {})[fam] = round(torch.load(p)["explained"], 4)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------ demo
@torch.no_grad()
def cmd_demo(_a):
    gpu, dev = setup_torch()
    ddir = OUT / f"demo_scene_{DEMO_SCENE}"
    ddir.mkdir(parents=True, exist_ok=True)
    provm, optm = provider("val", "mean_cam")
    idx = [Path(p).name for p in provm.dataset.sample_list].index(DEMO_SCENE)
    H, W = map(int, optm.img_size)
    sid, dm, raw_c2w, rcm = load_scene(provm, idx, "mean_cam", dev)
    A = normalize_world_transform(raw_c2w) @ raw_c2w[0]
    Rm, tm = A[:3, :3].to(dev), (A[:3, 3] * S).to(dev)
    from demo_ply_export import export_camera_ply, export_scannetpp_lidar_ply
    mesh = provm.dataset.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply"
    export_scannetpp_lidar_ply(ddir / "gt_lidar_mesh_frustum.ply", mesh, raw_c2w, dm["cam_view_input"][0], dm["intrinsics_input"][0], optm.img_size, S)
    export_camera_ply(ddir / "cameras.ply", dm["cam_to_world_input"][0], torch.linalg.inv(dm["cam_view"][0].float().transpose(1, 2)))
    views = (0, 4)
    # E1: bias / between / within maps on laser-planar interiors + 3D planar points coloured by between-token spread
    for key in ("release", "sft12", "locus", "paper"):
        m = Model(key, dev)
        prov, opt = (provm, optm) if m.frame == "mean_cam" else provider("val", "first_cam")
        _s, data, _r, rc = (sid, dm, raw_c2w, rcm) if m.frame == "mean_cam" else load_scene(prov, idx, "first_cam", dev)
        G, _bg = m.gaussians(data)
        cam, intr = data["cam_view"][0], data["intrinsics"][0]
        d_gt, n_gt, ok = raycast_gt_depth_and_normals(rc, cam.cpu(), intr.cpu(), opt.img_size)
        rows, pts, cols = [], [], []
        for v in range(cam.shape[0]):
            viewmat = cam[v].float().T.contiguous()
            alpha, zbar, vw, vb = depth_moments(G[0], viewmat, kmat(intr[v], dev), H, W, m.opt)
            dcam = pixel_dirs(intr[v], H, W, dev)
            d, nn_, okv = d_gt[v].to(dev), n_gt[v].to(dev), ok[v].to(dev)
            s = (dcam * nn_).sum(-1).abs()
            pl = planar_mask(d, nn_, okv, dcam) & (alpha > 0.5)
            bias, betw, within = (s * (zbar - d) / S).abs(), s * vb.sqrt() / S, s * vw.sqrt() / S
            if v in views:
                rows.append([label(rgb_img(data["images_output"][0, v]), f"{key} target {v}"),
                             label(map_img(pl.float(), 0, 1, okv), "laser planar interior (red)"),
                             label(map_img(bias, 0, 0.05, pl), "|common bias| 0-5cm"),
                             label(map_img(betw, 0, 0.05, pl), "between-token spread 0-5cm"),
                             label(map_img(within, 0, 0.05, pl), "within-token spread 0-5cm")])
            c2w = torch.linalg.inv(viewmat)
            Pw = (dcam * d.unsqueeze(-1))[pl] @ c2w[:3, :3].T + c2w[:3, 3]
            if m.frame == "first_cam":
                Pw = Pw @ Rm.T + tm
            keep = torch.randperm(Pw.shape[0], device=dev)[:20000]
            pts.append(Pw[keep]); cols.append(betw[pl][keep])
        grid_image(rows).save(ddir / f"E1_conflict_maps_{key}.png")
        Pc, bc = torch.cat(pts), torch.cat(cols)
        write_points(ddir / f"E1_planar_points_between_spread_{key}.ply", Pc, heat(bc, 0, 0.05), {"between_m": bc})
        del m, G
        torch.cuda.empty_cache()
    # E2: LocusGS layer-7 depth perturbation: start -> final anchor, coloured by recovery of the Gaussian centre
    m = Model("locus", dev)
    from locusgs.models.input_types import split_data as ls
    prov1, _o1 = provider("val", "first_cam")
    _s, d1, _r, rc1 = load_scene(prov1, idx, "first_cam", dev)
    mi = ls(d1, m.opt)[0]
    base = locus_forward(m.net, m.opt, mi, capture=(12,))
    An, G = base["anchors"], base["G"]
    ntok = G.shape[1] // CH
    vm = base["caps"][12]["attn"].reshape(ntok, 6, -1).sum(-1)
    dom = vm.argmax(-1)
    del base
    g64 = G[0].reshape(ntok, CH, 14)
    alive = (g64[..., 3] * (g64[..., 3] >= ALIVE_OPACITY)).sum(-1) > 0.5
    Xb = child_centre(g64)
    cams = torch.stack([torch.linalg.inv(c.float().T)[:3, 3] for c in d1["cam_view_input"][0]])
    gen = torch.Generator(device="cpu").manual_seed(1000 + idx)
    al = torch.nonzero(alive).squeeze(1)
    sel = al[torch.randperm(al.numel(), generator=gen)[: int(PERTURB_FRAC * al.numel())].to(dev)]
    st = An[6][sel]
    ray = st - cams[dom[sel]]
    depth = ray.norm(dim=-1, keepdim=True)
    sign = (torch.randint(0, 2, (sel.numel(), 1), generator=gen).float() * 2 - 1).to(dev)
    delta = ray / depth * sign * 0.10 * depth
    run = locus_forward(m.net, m.opt, mi, perturb={7: (sel, delta)})
    Xp = child_centre(run["G"][0].reshape(ntok, CH, 14))
    rG = 1 - ((Xp[sel] - Xb[sel]) * delta).sum(-1) / (delta ** 2).sum(-1)
    to_m = lambda x: x @ Rm.T + tm  # noqa: E731
    pick = torch.randperm(sel.numel(), generator=torch.Generator().manual_seed(0))[:120].tolist()
    lines, lcol, mk, mcol = [], [], [], []
    for i in pick:
        col = heat(rG[i:i + 1].clamp(0, 1), 0, 1)
        lines.append(segment(to_m(Xb[sel[i]]), to_m(Xb[sel[i]] + delta[i]), 20)); lcol.append(torch.tensor([[1.0, 1.0, 1.0]]).expand(20, 3))
        lines.append(segment(to_m(Xb[sel[i]] + delta[i]), to_m(Xp[sel[i]]), 20)); lcol.append(col.expand(20, 3))
        mk.append(ball(to_m(Xb[sel[i]]), 0.004)); mcol.append(torch.tensor([[0.0, 1.0, 0.3]]).expand(60, 3))
        mk.append(ball(to_m(Xp[sel[i]]), 0.004)); mcol.append(col.expand(60, 3))
    write_points(ddir / "E2_locus_L7_depth10_perturbation.ply", torch.cat(lines + mk), torch.cat(lcol + mcol))
    info = {"E2_demo_r_gauss_median": float(rG.median()), "E2_demo_tokens": int(sel.numel())}
    del m, run
    torch.cuda.empty_cache()
    # E3: per-pixel error maps and candidate scores for each depth source
    for fam, (_fk, srcs) in FAMILIES.items():
        c_cpu = torch.load(CACHE / fam / "val" / f"{DEMO_SCENE}.pt", map_location="cpu", weights_only=False)
        for src in srcs:
            nets = {}
            for variant in VARIANTS:
                cin = 27 if variant == "rgb_cand" else c_cpu["feats"].shape[1]
                n = Scorer(cin, direct=variant == "feat_direct").to(dev)
                n.load_state_dict(torch.load(CACHE / "probe" / f"{fam}_{src}_{variant}.pt"))
                nets[variant] = n.eval()
            c = to_dev(c_cpu, dev, src)
            c["rgb"] = rgb_maps(c["images_input"])
            offs = OFFS.to(dev)
            rows = []
            for v in (0, 2, 4, 6):
                okv = c["ok"][v] & (c["alpha"][v] > 0.5)
                py, px = torch.nonzero(okv, as_tuple=True)
                pv = torch.full_like(py, v)
                errs = {k: torch.zeros(H, W, device=dev) for k in ("model", "feat_cand", "rgb_cand", "feat_direct")}
                for c0 in range(0, py.numel(), 8192):
                    sl = slice(c0, c0 + 8192)
                    z0, zg = c["z0"][v, py[sl], px[sl]], c["d_gt"][v, py[sl], px[sl]]
                    zk = z0[:, None] * (1 + offs)[None]
                    errs["model"][py[sl], px[sl]] = (z0 - zg).abs() / zg
                    for variant in ("feat_cand", "rgb_cand"):
                        k = run_variant(nets[variant], variant, c, pv[sl], py[sl], px[sl], offs).argmax(-1)
                        errs[variant][py[sl], px[sl]] = (zk.gather(1, k[:, None])[:, 0] - zg).abs() / zg
                    errs["feat_direct"][py[sl], px[sl]] = (z0 * torch.exp(run_variant(nets["feat_direct"], "feat_direct", c, pv[sl], py[sl], px[sl], offs)) - zg).abs() / zg
                rows.append([label(map_img(c["vis"][v].float(), 0, 6, c["ok"][v]), f"target {v}: visible input views 0-6")] +
                            [label(map_img(errs[k], 0, 0.15, okv), f"{k} AbsRel {float(errs[k][okv].mean()):.3f}") for k in errs])
            grid_image(rows).save(ddir / f"E3_error_maps_{src}.png")
            # candidate panels for 6 pixels on target 0: 4 where the model is wrong (>8 %) but feat_cand right (<2 %), 2 others
            v = 0
            okv = c["ok"][v] & (c["alpha"][v] > 0.5) & ((c["d_gt"][v] / c["z0"][v] - 1).abs() <= 0.3)
            py, px = torch.nonzero(okv, as_tuple=True)
            pv = torch.full_like(py, v)
            z0, zg = c["z0"][v, py, px], c["d_gt"][v, py, px]
            zk = z0[:, None] * (1 + offs)[None]
            sc = torch.cat([run_variant(nets["feat_cand"], "feat_cand", c, pv[i:i + 8192], py[i:i + 8192], px[i:i + 8192], offs) for i in range(0, py.numel(), 8192)])
            kf = sc.argmax(-1)
            e_m = (z0 - zg).abs() / zg
            e_f = (zk.gather(1, kf[:, None])[:, 0] - zg).abs() / zg
            g1 = torch.nonzero((e_m > 0.08) & (e_f < 0.02)).squeeze(1)
            g2 = torch.nonzero((e_m > 0.08) & (e_f > 0.08)).squeeze(1)
            gsel = torch.cat([g1[torch.randperm(g1.numel(), generator=torch.Generator().manual_seed(1))[:4].to(dev)],
                              g2[torch.randperm(g2.numel(), generator=torch.Generator().manual_seed(2))[:2].to(dev)]]).tolist()
            from PIL import ImageDraw
            prow, pts3, col3 = [], [], []
            c2w = torch.linalg.inv(c["cam_view"][v].T)
            for j, i in enumerate(gsel):
                prob = sc[i].softmax(-1)
                kt = int((zk[i] - zg[i]).abs().argmin())
                tile = rgb_img(dm["images_output"][0, v]) if fam == "tokengs" else rgb_img(d1["images_output"][0, v])
                ImageDraw.Draw(tile).ellipse((int(px[i]) - 6, int(py[i]) - 6, int(px[i]) + 6, int(py[i]) + 6), outline=(255, 0, 255), width=2)
                kind = "model wrong, scorer right" if j < 4 else "both wrong"
                row = [label(tile.resize((W // 2, H // 2)), f"#{j} {kind}: model {float(e_m[i]) * 100:.0f}% scorer {float(e_f[i]) * 100:.0f}%")]
                grid, valid = cand_grid(c, pv[i:i + 1], py[i:i + 1], px[i:i + 1], z0[i:i + 1], offs)
                for vi in range(6):
                    im = rgb_img(c["images_input"][vi])
                    dr = ImageDraw.Draw(im)
                    uv = ((grid[vi, 0] + 1) / 2 * torch.tensor([W, H], device=dev)).cpu()
                    for k_ in range(len(OFFS)):
                        if not bool(valid[vi, 0, k_]):
                            continue
                        x_, y_ = uv[k_].tolist()
                        cc = tuple(int(255 * t) for t in heat(prob[k_:k_ + 1] / prob.max(), 0, 1)[0].tolist())
                        dr.ellipse((x_ - 3, y_ - 3, x_ + 3, y_ + 3), fill=cc)
                        if k_ == kt:
                            dr.ellipse((x_ - 7, y_ - 7, x_ + 7, y_ + 7), outline=(0, 255, 0), width=2)
                        if k_ == len(OFFS) // 2:
                            dr.rectangle((x_ - 6, y_ - 6, x_ + 6, y_ + 6), outline=(255, 255, 255), width=1)
                    row.append(label(im.resize((W // 2, H // 2)), f"input {vi}"))
                prow.append(row)
                fx, fy, cx, cy = c["intrinsics"][v].tolist()
                dc = torch.tensor([(float(px[i]) + 0.5 - cx) / fx, (float(py[i]) + 0.5 - cy) / fy, 1.0], device=dev)
                for k_ in range(len(OFFS)):
                    Pw = c2w[:3, :3] @ (dc * zk[i, k_]) + c2w[:3, 3]
                    pts3.append(ball(Pw, 0.003)); col3.append(heat(prob[k_:k_ + 1] / prob.max(), 0, 1).expand(60, 3))
                Pg = c2w[:3, :3] @ (dc * zg[i]) + c2w[:3, 3]
                pts3.append(ball(Pg, 0.007)); col3.append(torch.tensor([[0.0, 1.0, 0.0]]).expand(60, 3))
                pts3.append(segment(c2w[:3, 3], Pg, 150)); col3.append(torch.tensor([[1.0, 1.0, 1.0]]).expand(150, 3))
            grid_image(prow).save(ddir / f"E3_candidates_{src}.png")
            P3 = torch.cat(pts3)
            if fam == "locus":
                P3 = P3 @ Rm.T + tm
            write_points(ddir / f"E3_candidates_{src}.ply", P3, torch.cat(col3))
    (ddir / "demo_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("conflict", "recover", "extract"):
        s = sub.add_parser(name)
        s.add_argument("--shard", type=int, default=0)
        s.add_argument("--num-shards", type=int, default=1)
        if name == "conflict":
            s.add_argument("--models", nargs="+", default=list(MODELS))
        if name == "extract":
            s.add_argument("--family", choices=sorted(FAMILIES), required=True)
            s.add_argument("--split", choices=("train", "val"), required=True)
    s = sub.add_parser("fitpca")
    s.add_argument("--family", choices=sorted(FAMILIES), required=True)
    s = sub.add_parser("probe")
    s.add_argument("--family", choices=sorted(FAMILIES), required=True)
    s.add_argument("--source", choices=sorted(MODELS), required=True)
    s.add_argument("--steps", type=int, default=4000)
    s.add_argument("--pixels", type=int, default=4096)
    s.add_argument("--smoke", action="store_true", help="allow partial caches (code check only; results not used)")
    s = sub.add_parser("resolution")
    s.add_argument("--family", choices=sorted(FAMILIES), required=True)
    s.add_argument("--source", choices=sorted(MODELS), required=True)
    sub.add_parser("summarize")
    sub.add_parser("demo")
    a = ap.parse_args()
    {"conflict": cmd_conflict, "recover": cmd_recover, "fitpca": cmd_fitpca, "extract": cmd_extract, "probe": cmd_probe,
     "summarize": cmd_summarize, "demo": cmd_demo, "resolution": cmd_resolution}[a.cmd](a)


if __name__ == "__main__":
    main()
