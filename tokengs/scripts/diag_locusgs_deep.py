"""Deep diagnosis of LocusGS (frozen DL3DV finetune checkpoint, no training) on ScanNet++ (laser mesh G1-G4).

Mechanism-level questions, each with statistics over the 50 official validation scenes and a viewable demo on the demo
scene (`demo`):
  layers      anchor / radius trajectory over the 12 decoder layers, per-layer decoded Gaussians (PSNR, AbsRel, thickness),
              error locking (does a wrong anchor at layer 6 get corrected by layer 12?)
  attention   decomposition of the anchor-guided cross-attention: content-only vs geometric prior vs combined; how hard the
              geometric gate is; sigma floor (sigma^2 >= 1e-3 in point_to_plucker_ray_bias makes the radius irrelevant
              when 0.1 r < 0.0316); saturation (anchor far from every ray -> flat prior)
  ambiguity   the prior of the dominant view is invariant to moving the anchor along its ray (depth comes only from
              other views): anchor error vs #views / parallax
  allocation  the initial anchors are a learned scene-independent template; are the final anchors scene-adaptive or
              fixed "frustum slots"?  coverage of the laser surface by anchor supports vs image texture / depth
  selfattn    does anchor-aware self-attention connect the crowded / duplicated tokens?
  template    is the 64-child layout a fixed per-slot template (shape expressiveness)?
  interventions  inference-time ablations: no geometric bias (all / last 6 layers), no self-attention anchor PE,
              frozen anchors, fixed radius, sigma floor removed, jittered / permuted initial anchors
  views / ext 2 / 4 / 6 input views; extrapolated targets (ext_far, same 6 inputs) vs paper TokenGS 4v

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/diag_locusgs_deep.py stats --shard k --num-shards N
  PYTHONPATH=scripts:. python scripts/diag_locusgs_deep.py summarize
  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/diag_locusgs_deep.py demo
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
import torch.nn.functional as F
from scipy.spatial import cKDTree

import demo_ply_export as DPE
from demo_query_diagnosis import (ball, grid_image, heat, label, map_img, palette, rgb_img, rigid, segment,
                                  write_gaussians, write_points)
from diag_dl3dv import load_any
from diag_thinning_and_parallax import parallax_deg, spearman
from diagnose_latent_attention_geometry import ALIVE_OPACITY
from evaluate_scannetpp import ROOT, idle_gpu, set_extrapolation_targets
from evaluate_scannetpp_query_models import build_raycasting_scene_first_cam
from probe_gs_token_geometry import child_centre
from scannetpp_evaluation_protocol import (build_raycasting_scene, compute_scene_geometry_metrics,
                                           normalize_world_transform, raycast_gt_depth_and_normals,
                                           render_ray_thickness_map)
from tokengs.data.provider import Provider
from tokengs.options import config_defaults

OUT = ROOT / "EXP_LocusGS_Diagnosis"
CACHE = Path("/mnt/disk/xveyan/workspace_scannetpp/diag_locusgs_deep")
SCANNETPP_ROOT = "/home/jms/xveyan/data/4dgs/scannetpp"
DEMO_SCENE = "7b6477cb95"
S = 0.3
CH = 64
SIGMA_FLOOR = math.sqrt(1e-3)          # point_to_plucker_ray_bias: sigma^2.clamp_min(1e-3)
CAP_LAYERS = (2, 6, 12)
GEO_KEYS = ("depth_absrel", "depth_delta_105", "ray_thick_med_m", "ray_thick_p90_m", "normal_med_deg", "surf_acc_cm",
            "surf_comp_cm", "surf_fscore_5cm")
INTERVENTIONS = {
    "baseline": {}, "no_geo_bias_all": {"no_geo": tuple(range(1, 13))}, "no_geo_bias_last6": {"no_geo": tuple(range(7, 13))},
    "no_selfattn_anchor_pe": {"no_selfpos": True}, "frozen_anchors": {"freeze_anchor": True},
    "fixed_radius": {"fixed_radius": True}, "sigma_floor_removed": {"sigma_floor_off": True},
    "init_anchor_jitter_0.05": {"init": "jitter"}, "init_anchor_permuted": {"init": "permute"},
}
VIEW_SETS = {2: (0, 5), 4: (0, 2, 3, 5), 6: (0, 1, 2, 3, 4, 5)}


# ------------------------------------------------------------------------------------------------ instrumented forward
def _bias_nofloor(anchor_xyz, plucker_rays, sigma, plucker_order="md", moment_convention="o_cross_d", clamp_min=-20.0):
    m, d = plucker_rays[..., :3], plucker_rays[..., 3:]
    dn = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    d, m = d / dn, m / dn
    res = torch.cross(anchor_xyz[:, :, None, :], d[:, None], dim=-1) - m[:, None]
    sigma2 = torch.as_tensor(sigma, dtype=anchor_xyz.dtype, device=anchor_xyz.device).pow(2).clamp_min(1e-8)
    while sigma2.ndim < res.ndim - 1:
        sigma2 = sigma2.unsqueeze(-1)
    return (-res.pow(2).sum(-1) / (2.0 * sigma2)).clamp(min=clamp_min, max=0.0)


@torch.no_grad()
def locus_forward(model, lopt, mi, capture=(), capture_self=(), decode_layers=(), no_geo=(), no_selfpos=False,
                  freeze_anchor=False, fixed_radius=False, sigma_floor_off=False, init_anchor=None, perturb=None):
    """perturb: {layer: (token_idx, delta[n, 3])} added to the anchors fed INTO that decoder layer."""
    import locusgs.models.decoder_blocks as LDB

    enc = model.forward_encoder(mi.encoder)
    blocks = model.enc_dec_backbone.decoder_blocks
    head = model.activation_head
    tok = model.get_gs_tokens(1)
    anchor = (model.gs_anchor_xyz_raw if init_anchor is None else init_anchor).unsqueeze(0)
    rr = model.gs_anchor_radius_raw.unsqueeze(0)
    rad = model.activate_anchor_radius(rr)
    anchors, radii, caps, dec = [anchor[0]], [rad[0, :, 0]], {}, {}
    saved = [(blocks[li - 1].gs_cross_attn.geo_scale, blocks[li - 1].gs_cross_attn.geo_scale.data.clone()) for li in no_geo]
    orig_bias = LDB.point_to_plucker_ray_bias
    try:
        for prm, _v in saved:
            prm.data.fill_(-30.0)
        if sigma_floor_off:
            LDB.point_to_plucker_ray_bias = _bias_nofloor
        for li, layer in enumerate(blocks, 1):
            if perturb and li in perturb:
                idx, delta = perturb[li]
                anchor = anchor.clone()
                anchor[0, idx] = anchor[0, idx] + delta
            want = li in capture or li in capture_self
            orig_pe = layer._build_anchor_positional_features
            if no_selfpos:
                layer._build_anchor_positional_features = lambda _x: (None, None)
            out = layer(gs_tokens=tok, gs_anchor_normalized=model.normalize_anchor(anchor),
                        gs_anchor_radius=rad if lopt.anchor_radius_affects_bias else None, keys=enc.keys, values=enc.values,
                        plucker_emb_for_concat=enc.plucker_emb_for_concat, plucker_rays_patch_raw=enc.plucker_rays_patch_raw,
                        gs_anchor_raw=anchor, handcraft=want, return_attn=want)
            layer._build_anchor_positional_features = orig_pe
            if want:
                tok, info = out
                c = {"gamma": float(info["cross_gamma_g"])}
                if li in capture:
                    c.update(attn=info["attn"][0].float().mean(0), attn_c=info["attn_c"][0].float().mean(0),
                             attn_p=info["attn_p"][0, 0].float())
                if li in capture_self:
                    c["self"] = info["self_attn"][0].float().mean(0)
                caps[li] = c
                del info
            else:
                tok = out
            if not freeze_anchor:
                anchor = anchor + head.pos_act(model.xyz_embed(tok))
            if lopt.anchor_radius_refinement and not fixed_radius:
                rr = rr + model.radius_embed(tok)
                rad = model.activate_anchor_radius(rr)
            anchors.append(anchor[0])
            radii.append(rad[0, :, 0])
            if li in decode_layers:
                G = head(tok)
                dec[li] = torch.cat([model.compose_gaussian_xyz(G[..., :3], anchor, rad, head.num_gaussians_per_token), G[..., 3:]], -1)
    finally:
        for prm, v in saved:
            prm.data.copy_(v)
        LDB.point_to_plucker_ray_bias = orig_bias
    G = head(tok)
    G = torch.cat([model.compose_gaussian_xyz(G[..., :3], anchor, rad, head.num_gaussians_per_token), G[..., 3:]], -1)
    return {"G": G, "anchors": torch.stack(anchors), "radii": torch.stack(radii), "caps": caps, "dec": dec, "tok": tok[0]}


def init_anchor_variant(model, kind, seed=0):
    a = model.gs_anchor_xyz_raw.detach().clone()
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "jitter":
        return a + 0.05 * torch.randn(a.shape, generator=g).to(a.device)
    return a[torch.randperm(a.shape[0], generator=g).to(a.device)]


def subset_views(data, sel):
    V = data["cam_view_input"].shape[1]
    out = dict(data)
    idx = torch.tensor(sel, device=data["input"].device)
    out["input"] = torch.cat([data["input"][:, idx], data["input"][:, V:]], 1)
    for k in ("images_input", "intrinsics_input", "cam_view_input", "cam_to_world_input"):
        out[k] = data[k][:, idx]
    return out


# ------------------------------------------------------------------------------------------------ setup / evaluation
def scannetpp_opt(ext=False):
    return dataclasses.replace(
        config_defaults["eval_scannetpp_latent_6view_ssim"], camera_normalization_method="first_cam", evaluating=True,
        use_ttt_for_eval=False, ttt_n_steps=0, use_geo_losses=False, load_lidar_pc=False, use_input_supervision=False,
        mixed_precision="no", num_workers=0, val_depth_metric=False, dataset_kwargs={"window_size": 18, "root_path": SCANNETPP_ROOT})


class Evaluator:
    """PSNR + laser G1-G4 (same functions as evaluate_scannetpp.py) for Gaussians in the first-cam frame."""

    def __init__(self, model, data, rc, opt, dev):
        self.model, self.data, self.rc, self.opt, self.dev = model, data, rc, opt, dev
        self.d_gt, self.n_gt, self.ok = raycast_gt_depth_and_normals(rc, data["cam_view"][0], data["intrinsics"][0], opt.img_size)
        self.bg = model._background_color(torch.float32, dev)

    def __call__(self, G, full=True):
        d = self.data
        r = self.model.gs.render(G, d["cam_view"], bg_color=self.bg, intrinsics=d["intrinsics"])
        pred = r["images_pred"][0].clamp(0, 1)
        out = {"psnr": float((-10 * (pred - d["images_output"][0]).square().mean((1, 2, 3)).clamp_min(1e-12).log10()).mean())}
        sig, _ = render_ray_thickness_map(G, d["cam_view"], d["intrinsics"], self.opt.img_size, float(self.opt.znear),
                                          float(self.opt.zfar), S)
        _f, sc = compute_scene_geometry_metrics(r["depths_pred"][0], r["alphas_pred"][0], sig[0], self.d_gt, self.n_gt, self.ok,
                                                d["cam_view"][0], d["intrinsics"][0], scene_scale=S, nn_device=self.dev,
                                                gaussians=G[0], rc_scene=self.rc)
        out.update({k: float(sc[k]) for k in GEO_KEYS})
        if full:
            out["_render"], out["_sigma"] = r, sig
        return out


def surf_dist_fn(rc):
    import open3d as o3d

    def f(x):
        return torch.from_numpy(rc.compute_distance(o3d.core.Tensor(x.detach().cpu().numpy().astype(np.float32))).numpy()).to(x.device) / S
    return f


def scene_setup(prov, idx, dev):
    ds = prov.dataset
    scene = Path(ds.sample_list[idx]).name
    data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in prov[idx].items()}
    ctx, _tgt = ds.get_context_target_frames(idx)
    raw_c2w = torch.from_numpy(ds._scene_meta_cache[scene]["c2ws"][ctx].copy()).float()
    rc = build_raycasting_scene_first_cam(Path(SCANNETPP_ROOT) / "data" / scene / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
    return scene, data, raw_c2w, rc


def eff_n(p):
    return (-(p.clamp_min(1e-12).log() * p).sum(-1)).exp()


# ------------------------------------------------------------------------------------------------ per-scene analyses
def attention_stats(c, alive, V):
    A, Ac, Ap = c["attn"], c["attn_c"], c["attn_p"]
    srt, ix = Ap.sort(-1, descending=True)
    cum = srt.cumsum(-1)
    keep = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool), cum[:, :-1] < 0.9], 1)
    prior90 = torch.zeros_like(Ap, dtype=torch.bool).scatter_(1, ix, keep)
    Np = Ap.shape[1]
    sat = eff_n(Ap) > 0.25 * Np
    pv = Ap.reshape(Ap.shape[0], V, -1).sum(-1)
    return {"gamma": c["gamma"],
            "combined_mass_in_prior90": (A * prior90).sum(-1)[alive].cpu().half(),
            "argmax_combined_eq_prior": (A.argmax(-1) == Ap.argmax(-1))[alive].float().mean().item(),
            "argmax_combined_eq_content": (A.argmax(-1) == Ac.argmax(-1))[alive].float().mean().item(),
            "eff_patches_combined": eff_n(A)[alive].cpu().half(), "eff_patches_prior": eff_n(Ap)[alive].cpu().half(),
            "eff_patches_content": eff_n(Ac)[alive].cpu().half(), "prior_saturated_frac": sat[alive].float().mean().item(),
            "prior_views_gt_0.1": (pv > 0.1).sum(-1)[alive].cpu().short()}


def analyse_base(model, lopt, run, ev, data, rc, dev):
    G, An, Rn, caps = run["G"], run["anchors"], run["radii"], run["caps"]
    N = G.shape[1]
    ntok = N // CH
    g64 = G[0].reshape(ntok, CH, 14)
    op64 = g64[..., 3]
    walive = op64 * (op64 >= ALIVE_OPACITY)
    alive = walive.sum(-1) > 0.5
    V = data["cam_view_input"].shape[1]
    sd = surf_dist_fn(rc)
    cams = torch.stack([torch.linalg.inv(c.float().T)[:3, 3] for c in data["cam_view_input"][0]])
    A12 = caps[12]["attn"]
    vm = A12.reshape(ntok, V, -1).sum(-1)
    use = vm > 0.1
    dom = vm.argmax(-1)
    rec = {}
    # layers: anchor surface distance / displacement / radius / sigma floor per layer
    dist_l = torch.stack([sd(An[l]) for l in range(An.shape[0])])                        # [13, ntok] metres
    rec["layers"] = {"anchor_surface_dist_m": dist_l[:, alive].cpu().half(),
                     "anchor_step_m": ((An[1:] - An[:-1]).norm(dim=-1) / S)[:, alive].cpu().half(),
                     "radius_m": (Rn / S)[:, alive].cpu().half(),
                     "sigma_floor_active_frac": ((0.1 * Rn) < SIGMA_FLOOR)[:, alive].float().mean(-1).cpu()}
    rec["per_layer_decode"] = {li: ev(Gl, full=False) for li, Gl in run["dec"].items()}
    # attention decomposition
    rec["attention"] = {li: attention_stats(caps[li], alive, V) for li in caps if "attn" in caps[li]}
    # ambiguity: anchor error vs views / parallax
    Xp = child_centre(g64)
    par = parallax_deg(An[12], cams, use)
    rec["ambiguity"] = {"n_views": use.sum(-1)[alive].cpu().short(), "parallax_deg": par[alive].cpu().half(),
                        "anchor_dist_m": dist_l[12, alive].cpu().half(), "gauss_centre_dist_m": sd(Xp)[alive].cpu().half()}
    # allocation: final anchors in the first-cam frame (scene independent template?)
    w2c0 = data["cam_view_input"][0, 0].float().T
    xc = An[12] @ w2c0[:3, :3].T + w2c0[:3, 3]
    xi = An[0] @ w2c0[:3, :3].T + w2c0[:3, 3]
    fx, fy, cx, cy = data["intrinsics_input"][0, 0].tolist()
    rec["allocation"] = {"final_cam0": xc.cpu().half(), "init_cam0": xi.cpu().half(), "alive": alive.cpu(),
                         "uv_final": torch.stack([fx * xc[:, 0] / xc[:, 2] + cx, fy * xc[:, 1] / xc[:, 2] + cy], -1).cpu().half(),
                         "uv_init": torch.stack([fx * xi[:, 0] / xi[:, 2] + cx, fy * xi[:, 1] / xi[:, 2] + cy], -1).cpu().half()}
    # coverage of the laser surface (input-view depth back-projected at stride 4)
    d_in, _n, ok_in = raycast_gt_depth_and_normals(rc, data["cam_view_input"][0], data["intrinsics_input"][0], lopt.img_size)
    H, W = map(int, lopt.img_size)
    pts, tex, dep = [], [], []
    for v in range(V):
        img = data["images_input"][0, v].mean(0)
        gx = F.pad(img[:, 1:] - img[:, :-1], (0, 1)).abs()
        gy = F.pad(img[1:] - img[:-1], (0, 0, 0, 1)).abs()
        grad = F.avg_pool2d((gx + gy)[None, None], 9, 1, 4)[0, 0]
        yy, xx = torch.meshgrid(torch.arange(2, H, 4), torch.arange(2, W, 4), indexing="ij")
        m = ok_in[v][yy, xx]
        yy, xx = yy[m], xx[m]
        z = d_in[v][yy, xx].to(dev)
        fx, fy, cx, cy = data["intrinsics_input"][0, v].tolist()
        pc = torch.stack([(xx.to(dev) + 0.5 - cx) / fx * z, (yy.to(dev) + 0.5 - cy) / fy * z, z], -1)
        c2w = torch.linalg.inv(data["cam_view_input"][0, v].float().T)
        pts.append(pc @ c2w[:3, :3].T + c2w[:3, 3]); tex.append(grad[yy.to(dev), xx.to(dev)]); dep.append(z)
    P, tex, dep = torch.cat(pts), torch.cat(tex), torch.cat(dep)
    a, r = An[12][alive], Rn[12][alive]
    cnt = torch.zeros(P.shape[0], device=dev)
    nn_over_r = torch.full((P.shape[0],), float("inf"), device=dev)
    for c0 in range(0, P.shape[0], 8192):
        dd = torch.cdist(P[c0:c0 + 8192], a)
        cnt[c0:c0 + 8192] = (dd < r).sum(-1)
        nn_over_r[c0:c0 + 8192] = (dd / r).min(-1).values
    rec["coverage"] = {"count": cnt.cpu().short(), "nn_over_r": nn_over_r.cpu().half(), "texture": tex.cpu().half(), "depth": (dep / S).cpu().half()}
    # self-attention at layer 12
    Sm = caps[12]["self"]
    dd = torch.cdist(An[12], An[12])
    knn = dd.topk(17, largest=False).indices[:, 1:]
    crowd = dd < Rn[12].unsqueeze(1)
    crowd.fill_diagonal_(False)
    nn_mass = Sm.gather(1, knn).sum(-1)
    crowd_w = (Sm * crowd).sum(-1) / crowd.sum(-1).clamp_min(1)
    rec["selfattn"] = {"mass_on_16_nearest": nn_mass[alive].cpu().half(), "uniform_16": 16 / ntok,
                       "mean_weight_crowded_neighbours_x_ntok": (crowd_w * ntok)[alive & (crowd.sum(-1) > 0)].cpu().half(),
                       "self_weight": Sm.diagonal()[alive].cpu().half(),
                       "top8_partner_dist_over_r": (dd.gather(1, Sm.topk(9).indices[:, 1:]).mean(-1) / Rn[12])[alive].cpu().half()}
    # child template in the dominant camera frame, offsets in radius units
    w2c_dom = data["cam_view_input"][0][dom].float().transpose(1, 2)[:, :3, :3]          # [ntok, 3, 3]
    off = (g64[..., :3] - An[12].unsqueeze(1)) / Rn[12][:, None, None]
    offc = torch.einsum("nij,nkj->nki", w2c_dom, off)
    am = (op64 >= ALIVE_OPACITY) & alive.unsqueeze(1)
    slot_mean = torch.stack([offc[:, k][am[:, k]].mean(0) if am[:, k].any() else torch.zeros(3, device=dev) for k in range(CH)])
    res = ((offc - slot_mean.unsqueeze(0)).square().sum(-1) * am).sum()
    tot = ((offc - offc[am].mean(0)).square().sum(-1) * am).sum()
    rec["template"] = {"slot_mean_offset_r": slot_mean.cpu(), "r2_slot_template": float(1 - res / tot),
                       "slot_alive_frac": am.float().mean(0).cpu(), "slot_mean_opacity": op64.mean(0).cpu()}
    return rec


# ------------------------------------------------------------------------------------------------ stats command
@torch.no_grad()
def cmd_stats(a):
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")
    model, lopt, _d, _k = load_any("locusgs_finetune_4v", dev)
    paper, popt, _d2, _k2 = load_any("tokengs_paper_4v", dev)
    from locusgs.models.input_types import split_data as ls

    opt = scannetpp_opt()
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov_ext = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    set_extrapolation_targets(prov_ext.dataset, "ext_far")
    mine = list(range(len(prov)))[a.shard::a.num_shards][: a.limit or None]
    out, t0 = [], time.time()
    print(f"GPU {gpu['index']}: shard {a.shard}/{a.num_shards}: {len(mine)} scenes", flush=True)
    for n, idx in enumerate(mine):
        scene, data, raw_c2w, rc = scene_setup(prov, idx, dev)
        mi, _ = ls(data, lopt)
        ev = Evaluator(model, data, rc, opt, dev)
        run = locus_forward(model, lopt, mi, capture=CAP_LAYERS, capture_self=(12,), decode_layers=tuple(range(1, 13)))
        rec = {"scene_id": scene, **analyse_base(model, lopt, run, ev, data, rc, dev)}
        base_G = run["G"]
        del run
        interv = {"baseline": ev(base_G, full=False)}
        for name, cfg in INTERVENTIONS.items():
            if name == "baseline":
                continue
            kw = {k: v for k, v in cfg.items() if k != "init"}
            if "init" in cfg:
                kw["init_anchor"] = init_anchor_variant(model, cfg["init"], seed=idx)
            interv[name] = ev(locus_forward(model, lopt, mi, **kw)["G"], full=False)
        rec["interventions"] = interv
        pG = paper.forward_decoder(paper.forward_encoder(ls(data, popt)[0].encoder), ls(data, popt)[0].decoder)
        rec["paper_baseline"] = Evaluator(paper, data, rc, opt, dev)(pG, full=False)
        views = {}
        for nv, sel in VIEW_SETS.items():
            if nv == 6:
                views[nv] = {"locus": interv["baseline"], "paper": rec["paper_baseline"]}
                continue
            dsub = subset_views(data, sel)
            lo, po = dataclasses.replace(lopt, num_input_views=nv), dataclasses.replace(popt, num_input_views=nv)
            Gl = model.forward_decoder(model.forward_encoder(ls(dsub, lo)[0].encoder), ls(dsub, lo)[0].decoder)
            Gp = paper.forward_decoder(paper.forward_encoder(ls(dsub, po)[0].encoder), ls(dsub, po)[0].decoder)
            views[nv] = {"locus": ev(Gl, full=False), "paper": ev(Gp, full=False)}
        rec["views"] = views
        _s, dext, _r, _rc = scene_setup(prov_ext, idx, dev)
        if not torch.equal(dext["cam_view_input"], data["cam_view_input"]):
            raise RuntimeError("extrapolation provider changed the inputs")
        ev_ext = Evaluator(model, dext, rc, opt, dev)
        rec["ext_far"] = {"locus": ev_ext(base_G, full=False), "paper": Evaluator(paper, dext, rc, opt, dev)(pG, full=False)}
        out.append(rec)
        torch.cuda.empty_cache()
        print(f"[shard {a.shard}] {n + 1}/{len(mine)} {scene} {time.time() - t0:.0f}s psnr {interv['baseline']['psnr']:.2f} "
              f"(paper {rec['paper_baseline']['psnr']:.2f}) no_geo {interv['no_geo_bias_all']['psnr']:.2f} "
              f"ext_far {rec['ext_far']['locus']['psnr']:.2f}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    name = f"shard_{a.shard}" if not a.limit else f"smoke_{a.shard}"
    torch.save(out, CACHE / f"{name}.tmp")
    (CACHE / f"{name}.tmp").rename(CACHE / f"{name}.pt")
    print("wrote", CACHE / f"{name}.pt", flush=True)


# ------------------------------------------------------------------------------------------------ summarize
def q(x, ps=(0.1, 0.5, 0.9)):
    x = x[torch.isfinite(x)].double()
    return {f"p{int(p * 100)}": round(float(x.quantile(p)), 4) for p in ps} if x.numel() else {}


def mean_dict(ds):
    return {k: round(float(np.mean([d[k] for d in ds])), 4) for k in ds[0] if not k.startswith("_")}


def cmd_summarize(_a):
    recs = []
    for f in sorted(CACHE.glob("shard_*.pt")):
        recs.extend(torch.load(f, map_location="cpu", weights_only=False))
    if len(recs) != 50:
        raise RuntimeError(f"expected 50 scenes, got {len(recs)}")
    res = {"definitions": __doc__.strip().split("\n\n")[1], "scenes": len(recs), "frame": "first_cam, scale 0.3; metres"}
    L = {k: torch.cat([r["layers"][k].float() for r in recs], 1) for k in ("anchor_surface_dist_m", "radius_m")}
    step = torch.cat([r["layers"]["anchor_step_m"].float() for r in recs], 1)
    res["layers"] = {
        "anchor_surface_dist_m_median_by_state(0=init..12)": [round(float(x.median()), 4) for x in L["anchor_surface_dist_m"]],
        "anchor_surface_dist_m_p90_by_state": [round(float(x.quantile(0.9)), 4) for x in L["anchor_surface_dist_m"]],
        "anchor_step_m_median_by_layer(1..12)": [round(float(x.median()), 4) for x in step],
        "radius_m_median_by_state": [round(float(x.median()), 4) for x in L["radius_m"]],
        "sigma_floor_active_frac_by_state": [round(float(x), 4) for x in torch.stack([r["layers"]["sigma_floor_active_frac"] for r in recs]).mean(0)],
        "per_layer_decode_mean": {li: mean_dict([r["per_layer_decode"][li] for r in recs]) for li in range(1, 13)}}
    d6, d12 = L["anchor_surface_dist_m"][6], L["anchor_surface_dist_m"][12]
    bad6 = d6 > 0.10
    res["layers"]["error_locking"] = {"spearman_dist_layer6_vs_12": spearman(d6, d12),
                                      "frac_tokens_dist_gt_10cm_at_6": float(bad6.float().mean()),
                                      "of_those_still_gt_10cm_at_12": float((d12[bad6] > 0.10).float().mean()),
                                      "of_those_improved_by_half": float((d12[bad6] < 0.5 * d6[bad6]).float().mean()),
                                      "frac_good_at_6_becoming_gt_10cm_at_12": float((d12[~bad6] > 0.10).float().mean())}
    att = {}
    for li in CAP_LAYERS:
        rs = [r["attention"][li] for r in recs]
        att[li] = {"gamma_g": round(float(np.mean([x["gamma"] for x in rs])), 3),
                   **{k: round(float(np.mean([x[k] for x in rs])), 4) for k in ("argmax_combined_eq_prior", "argmax_combined_eq_content", "prior_saturated_frac")},
                   **{k: q(torch.cat([x[k].float() for x in rs])) for k in ("combined_mass_in_prior90", "eff_patches_combined", "eff_patches_prior", "eff_patches_content", "prior_views_gt_0.1")}}
    res["attention"] = att
    amb = {k: torch.cat([r["ambiguity"][k].float() for r in recs]) for k in recs[0]["ambiguity"]}
    bins = {}
    for nv in (1, 2, 3, 4):
        m = amb["n_views"] == nv if nv < 4 else amb["n_views"] >= 4
        bins[f"{nv}{'+' if nv == 4 else ''} views"] = {"token_share": float(m.float().mean()), "anchor_dist_m": q(amb["anchor_dist_m"][m], (0.5, 0.9)),
                                                      "gauss_centre_dist_m": q(amb["gauss_centre_dist_m"][m], (0.5, 0.9))}
    res["ambiguity"] = {"by_attended_views": bins, "spearman_anchor_dist_vs_-parallax": spearman(-amb["parallax_deg"], amb["anchor_dist_m"]),
                        "spearman_anchor_dist_vs_-views": spearman(-amb["n_views"], amb["anchor_dist_m"])}
    # allocation template: per-token spread of the final anchor across scenes (first-cam frame)
    fin = torch.stack([r["allocation"]["final_cam0"].float() for r in recs])                 # [50, ntok, 3]
    uv = torch.stack([r["allocation"]["uv_final"].float() for r in recs])
    alive = torch.stack([r["allocation"]["alive"] for r in recs])
    ini = recs[0]["allocation"]["init_cam0"].float()
    uvi = recs[0]["allocation"]["uv_init"].float()
    dirs = F.normalize(fin, dim=-1)
    mdir = F.normalize(dirs.mean(0), dim=-1)
    ang = torch.rad2deg(torch.acos((dirs * mdir).sum(-1).clamp(-1, 1)))                     # [50, ntok]
    zr = fin[..., 2]
    vis = (zr > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < 448) & (uv[..., 1] >= 0) & (uv[..., 1] < 256)
    res["allocation"] = {
        "per_token_median_angle_to_its_mean_direction_deg_across_scenes": q(ang.median(0).values),
        "per_token_depth_cv_across_scenes": q((zr.std(0) / zr.mean(0).abs().clamp_min(1e-6))),
        "per_token_uv_std_px_across_scenes": q(uv.std(0).norm(dim=-1)),
        "init_vs_final_direction_deg": q(torch.rad2deg(torch.acos((F.normalize(ini, dim=-1) * dirs).sum(-1).clamp(-1, 1))).median(0).values),
        "tokens_alive_in_all_50_scenes": float(alive.all(0).float().mean()), "tokens_dead_in_all_50_scenes": float((~alive).all(0).float().mean()),
        "final_anchor_in_view0_frustum_frac": float(vis.float().mean()), "init_anchor_in_view0_frustum_frac": float(
            ((ini[:, 2] > 0) & (uvi[:, 0] >= 0) & (uvi[:, 0] < 448) & (uvi[:, 1] >= 0) & (uvi[:, 1] < 256)).float().mean())}
    cov = {k: torch.cat([r["coverage"][k].float() for r in recs]) for k in recs[0]["coverage"]}
    tq = cov["texture"].quantile(torch.tensor([0.25, 0.5, 0.75]))
    dq = cov["depth"].quantile(torch.tensor([0.25, 0.5, 0.75]))
    def qbin(x, e):
        return [(x <= e[0]), (x > e[0]) & (x <= e[1]), (x > e[1]) & (x <= e[2]), (x > e[2])]
    res["coverage"] = {"surface_points": int(cov["count"].numel()), "uncovered_frac_(no_anchor_support)": float((cov["count"] == 0).float().mean()),
                       "anchors_supporting_each_point": q(cov["count"], (0.1, 0.5, 0.9)), "nearest_anchor_over_r": q(cov["nn_over_r"], (0.5, 0.9)),
                       "by_texture_quartile_(low->high)": [{"uncovered": float((cov["count"][m] == 0).float().mean()), "support_median": float(cov["count"][m].median())} for m in qbin(cov["texture"], tq)],
                       "by_depth_quartile_(near->far)": [{"uncovered": float((cov["count"][m] == 0).float().mean()), "support_median": float(cov["count"][m].median())} for m in qbin(cov["depth"], dq)],
                       "spearman_support_vs_texture": spearman(cov["texture"], cov["count"]), "spearman_support_vs_depth": spearman(cov["depth"], cov["count"])}
    sa = {k: torch.cat([r["selfattn"][k].float() for r in recs]) for k in ("mass_on_16_nearest", "mean_weight_crowded_neighbours_x_ntok", "self_weight", "top8_partner_dist_over_r")}
    res["selfattn_layer12"] = {"mass_on_16_nearest_anchors": q(sa["mass_on_16_nearest"]), "uniform_baseline": recs[0]["selfattn"]["uniform_16"],
                               "crowded_neighbour_weight_relative_to_uniform": q(sa["mean_weight_crowded_neighbours_x_ntok"]),
                               "self_weight": q(sa["self_weight"]), "top8_partner_mean_dist_over_r": q(sa["top8_partner_dist_over_r"])}
    res["child_template"] = {"r2_explained_by_slot_template": q(torch.tensor([r["template"]["r2_slot_template"] for r in recs])),
                             "slot_mean_offset_r_mean_over_scenes": torch.stack([r["template"]["slot_mean_offset_r"] for r in recs]).mean(0).round(decimals=3).tolist(),
                             "slot_alive_frac": torch.stack([r["template"]["slot_alive_frac"] for r in recs]).mean(0).round(decimals=3).tolist()}
    res["interventions_mean_50_scenes"] = {k: mean_dict([r["interventions"][k] for r in recs]) for k in INTERVENTIONS}
    res["paper_tokengs_4v_baseline"] = mean_dict([r["paper_baseline"] for r in recs])
    res["views_mean_50_scenes"] = {nv: {m: mean_dict([r["views"][nv][m] for r in recs]) for m in ("locus", "paper")} for nv in VIEW_SETS}
    res["ext_far_mean_50_scenes"] = {m: mean_dict([r["ext_far"][m] for r in recs]) for m in ("locus", "paper")}
    ref = ROOT / "EXP_ScanNetPP/eval_extrapolation/ext_far"
    res["ext_far_reference_release_models"] = {d.name: {k: json.loads((d / "summary.json").read_text())[k] for k in ("psnr", *GEO_KEYS)}
                                               for d in sorted(ref.iterdir()) if d.name in ("baseline_pretrained_6view", "sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20") and (d / "summary.json").is_file()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats_50_scenes.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("definitions", "child_template")}, indent=1))


# ------------------------------------------------------------------------------------------------ demo
def strip_row(ev_out, v, title):
    """Prediction, depth and ray thickness (0-30 cm) of target view v."""
    r, sig = ev_out["_render"], ev_out["_sigma"]
    img, dep = r["images_pred"][0, v].clamp(0, 1), r["depths_pred"][0, v, 0]
    return [label(rgb_img(img), f"{title} {ev_out['psnr']:.2f}dB"),
            label(map_img(dep, 0, 3.5 * S, r["alphas_pred"][0, v, 0] > 0.05), f"depth AbsRel {ev_out['depth_absrel']:.3f}"),
            label(map_img(sig[0, v, 0], 0, 0.3, r["alphas_pred"][0, v, 0] > 0.5), f"thick p90 {ev_out['ray_thick_p90_m'] * 100:.0f}cm")]


@torch.no_grad()
def cmd_demo(_a):
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")
    model, lopt, _d, _k = load_any("locusgs_finetune_4v", dev)
    paper, popt, _d2, _k2 = load_any("tokengs_paper_4v", dev)
    from locusgs.models.input_types import split_data as ls

    opt = scannetpp_opt()
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    idx = [Path(p).name for p in prov.dataset.sample_list].index(DEMO_SCENE)
    scene, data, raw_c2w, rc = scene_setup(prov, idx, dev)
    sdir = OUT / f"scene_{scene}"
    sdir.mkdir(parents=True, exist_ok=True)
    A = normalize_world_transform(raw_c2w) @ raw_c2w[0]
    Rm, tm = A[:3, :3].to(dev), (A[:3, 3] * S).to(dev)
    to_m = lambda x: x @ Rm.T + tm                                                   # first-cam -> mean-cam (shared mesh)  # noqa: E731
    # shared references in the mean-cam frame
    prov_m = Provider("scannetpp_val_scaled_0.3", dataclasses.replace(opt, camera_normalization_method="mean_cam"), training=False)
    dm = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in prov_m[idx].items()}
    mesh = Path(SCANNETPP_ROOT) / "data" / scene / "scans" / "mesh_aligned_0.05.ply"
    DPE.export_scannetpp_lidar_ply(sdir / "gt_lidar_mesh_frustum.ply", mesh, raw_c2w, dm["cam_view_input"][0], dm["intrinsics_input"][0], opt.img_size, S)
    DPE.export_camera_ply(sdir / "cameras.ply", dm["cam_to_world_input"][0], torch.linalg.inv(dm["cam_view"][0].float().transpose(1, 2)))
    mi, _ = ls(data, lopt)
    ev = Evaluator(model, data, rc, opt, dev)
    run = locus_forward(model, lopt, mi, capture=CAP_LAYERS, capture_self=(12,), decode_layers=(2, 4, 6, 8, 10, 12))
    G, An, Rn, caps = run["G"], run["anchors"], run["radii"], run["caps"]
    ntok = G.shape[1] // CH
    g64 = G[0].reshape(ntok, CH, 14)
    op64 = g64[..., 3]
    alive = (op64 * (op64 >= ALIVE_OPACITY)).sum(-1) > 0.5
    sd = surf_dist_fn(rc)
    V = data["cam_view_input"].shape[1]
    H, W = map(int, lopt.img_size)
    hp, wp = H // 8, W // 8
    cams = torch.stack([torch.linalg.inv(c.float().T)[:3, 3] for c in data["cam_view_input"][0]])
    vm = caps[12]["attn"].reshape(ntok, V, -1).sum(-1)
    dom = vm.argmax(-1)
    info = {"scene": scene, "gpu": gpu["index"], "frame": "all PLYs in the release mean_cam normalized frame (scale 0.3), "
            "overlay with gt_lidar_mesh_frustum.ply / cameras.ply; token_slots_* are in each scene's first-camera frame"}

    # D1 anchor trajectories over layers
    d1 = sdir / "D1_anchor_trajectories"
    d1.mkdir(exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    al_idx = torch.nonzero(alive).squeeze(1).cpu()
    pick = al_idx[torch.randperm(al_idx.numel(), generator=gen)[:150]].to(dev)
    traj, tcol = [], []
    for t_ in pick.tolist():
        for l in range(12):
            traj.append(segment(to_m(An[l, t_]), to_m(An[l + 1, t_]), 12))
            tcol.append(heat(torch.full((12,), l / 11.0), 0, 1))
    write_points(d1 / "trajectories_150tokens.ply", torch.cat(traj), torch.cat(tcol))
    for l in (0, 3, 6, 9, 12):
        dl = sd(An[l])
        write_points(d1 / f"anchors_state{l:02d}_surface_dist.ply", to_m(An[l][alive]), heat(dl[alive], 0, 0.3),
                     {"dist_m": dl[alive], "radius_m": (Rn[l] / S)[alive]})
    write_points(d1 / "anchors_init_template_all4096.ply", to_m(An[0]), heat(Rn[0] / S, 0, 0.5))
    # D2 per-layer decoded Gaussians
    d2 = sdir / "D2_per_layer_decoding"
    d2.mkdir(exist_ok=True)
    rows, pl = [], {}
    for li, Gl in run["dec"].items():
        e = ev(Gl)
        pl[li] = {k: v for k, v in e.items() if not k.startswith("_")}
        rows.append(strip_row(e, 0, f"layer {li}") + strip_row(e, 4, f"layer {li}"))
        if li in (2, 6, 12):
            write_gaussians(d2 / f"gaussians_layer{li:02d}.ply", rigid(Gl[0], Rm, tm), Gl[0, :, 11:14])
            DPE.export_pruned_ply(d2 / f"gaussians_layer{li:02d}.ply", d2 / f"gaussians_layer{li:02d}_prune0.05.ply", 0.05)
    grid_image(rows).save(d2 / "per_layer_strip.png")
    info["per_layer_decode"] = pl
    # D3 attention decomposition (content / geometric prior / combined) for 8 tokens at layers 2, 6, 12
    d3 = sdir / "D3_attention_decomposition"
    d3.mkdir(exist_ok=True)
    tsel = pick[:8].tolist()
    rows = []
    for t_ in tsel:
        v = int(dom[t_])
        base = data["images_input"][0, v].permute(1, 2, 0) * 0.4
        row = []
        for li in CAP_LAYERS:
            for key, ch, nm in (("attn_c", 2, "content"), ("attn_p", 1, "geo prior"), ("attn", 0, "combined")):
                m = caps[li][key][t_].reshape(V, hp, wp)[v]
                up = F.interpolate(m[None, None], size=(H, W), mode="nearest")[0, 0]
                im = base.clone()
                im[..., ch] += 0.6 * up / up.max().clamp_min(1e-12)
                row.append(label(rgb_img(im.permute(2, 0, 1)).resize((W // 2, H // 2)),
                                 f"L{li} {nm} eff{float(eff_n(caps[li][key][t_])):.0f}"))
        rows.append(row)
    grid_image(rows).save(d3 / "attention_content_prior_combined_8tokens.png")
    gam = {li: caps[li]["gamma"] for li in CAP_LAYERS}
    info["gamma_g"] = gam
    # D4 depth ambiguity: geometric prior when the anchor slides along its dominant-view ray
    d4 = sdir / "D4_depth_ambiguity"
    d4.mkdir(exist_ok=True)
    from locusgs.models.geometry import point_to_plucker_ray_bias
    enc = model.forward_encoder(mi.encoder)
    rays = enc.plucker_rays_patch_raw
    rows, pts, cols = [], [], []
    for k, t_ in enumerate(tsel[:6]):
        v = int(dom[t_])
        c = cams[v]
        a = An[11, t_]
        for s, scol in ((0.8, (0.2, 0.4, 1.0)), (1.0, (1.0, 1.0, 1.0)), (1.25, (1.0, 0.3, 0.2))):
            ash = c + (a - c) * s
            sig = (0.1 * Rn[11, t_]).clamp_min(SIGMA_FLOOR)
            prior = torch.softmax(caps[12]["gamma"] * point_to_plucker_ray_bias(ash[None, None], rays, sig)[0, 0], -1)
            row = []
            for vv in range(V):
                m = prior.reshape(V, hp, wp)[vv]
                up = F.interpolate(m[None, None], size=(H, W), mode="nearest")[0, 0]
                im = data["images_input"][0, vv].permute(1, 2, 0) * 0.45
                im[..., 1] += 0.6 * up / up.max().clamp_min(1e-12)
                row.append(label(rgb_img(im.permute(2, 0, 1)).resize((W // 3, H // 3)), f"tok{k} x{s} v{vv}{'*' if vv == v else ''}"))
            rows.append(row)
            pts.append(ball(to_m(ash), 0.006)); cols.append(torch.tensor([scol]).expand(60, 3))
        pts.append(segment(to_m(c), to_m(c + (a - c) * 1.4), 200)); cols.append(torch.tensor([[1.0, 1.0, 0.0]]).expand(200, 3))
    grid_image(rows).save(d4 / "prior_vs_anchor_depth_6tokens.png")
    write_points(d4 / "anchor_slides_along_dominant_ray.ply", torch.cat(pts), torch.cat(cols))
    amb_v = (vm > 0.1).sum(-1)
    write_points(d4 / "anchors_by_attended_views.ply", to_m(An[12][alive]), heat(amb_v[alive].float(), 1, 6),
                 {"attended_views": amb_v[alive].float(), "dist_m": sd(An[12])[alive]})
    # D5 sigma floor / radius irrelevance
    d5 = sdir / "D5_sigma_floor"
    d5.mkdir(exist_ok=True)
    floor = (0.1 * Rn[11]) < SIGMA_FLOOR
    write_points(d5 / "anchors_radius_ineffective.ply", to_m(An[12][alive]),
                 torch.where(floor[alive].unsqueeze(1), torch.tensor([1.0, 0.2, 0.2], device=dev), torch.tensor([0.2, 0.9, 0.3], device=dev)),
                 {"radius_m": (Rn[12] / S)[alive], "floor_active": floor[alive].float()})
    rows = []
    for k, t_ in enumerate(tsel[:4]):
        v = int(dom[t_])
        row = []
        for f_ in (0.25, 1.0, 4.0):
            sig = (0.1 * Rn[11, t_] * f_).clamp_min(SIGMA_FLOOR)
            prior = torch.softmax(caps[12]["gamma"] * point_to_plucker_ray_bias(An[11, t_][None, None], rays, sig)[0, 0], -1)
            m = prior.reshape(V, hp, wp)[v]
            up = F.interpolate(m[None, None], size=(H, W), mode="nearest")[0, 0]
            im = data["images_input"][0, v].permute(1, 2, 0) * 0.45
            im[..., 1] += 0.6 * up / up.max().clamp_min(1e-12)
            row.append(label(rgb_img(im.permute(2, 0, 1)).resize((W // 2, H // 2)), f"tok{k} radius x{f_} sigma {float(sig) / S * 100:.1f}cm"))
        rows.append(row)
    grid_image(rows).save(d5 / "prior_with_radius_x0.25_x1_x4.png")
    info["sigma_floor_active_frac_layer12_input"] = float(floor[alive].float().mean())
    # D6 allocation: token slots across the 50 scenes (needs the stats cache)
    d6 = sdir / "D6_allocation_template"
    d6.mkdir(exist_ok=True)
    recs = []
    for f in sorted(CACHE.glob("shard_*.pt")):
        recs.extend(torch.load(f, map_location="cpu", weights_only=False))
    if recs:
        fin = torch.stack([r["allocation"]["final_cam0"].float() for r in recs])
        uv = torch.stack([r["allocation"]["uv_final"].float() for r in recs])
        tk = torch.randperm(ntok, generator=torch.Generator().manual_seed(3))[:40]
        pal = palette(40, 5)
        write_points(d6 / "token_slots_40tokens_x50scenes_firstcam_frame.ply", fin[:, tk].reshape(-1, 3),
                     pal.repeat(fin.shape[0], 1), {"token_rank": torch.arange(40).repeat(fin.shape[0]).float()})
        ini = recs[0]["allocation"]["init_cam0"].float()
        write_points(d6 / "token_slots_40tokens_init_anchor_firstcam_frame.ply", torch.cat([ball(ini[t_], 0.01) for t_ in tk.tolist()]),
                     pal.repeat_interleave(60, 0))
        write_points(d6 / "camera0_origin_firstcam_frame.ply", ball(torch.zeros(3), 0.02), torch.tensor([[1.0, 0.0, 0.0]]).expand(60, 3))
        from PIL import Image, ImageDraw
        canvas = rgb_img(data["images_input"][0, 0] * 0.5)
        dr = ImageDraw.Draw(canvas)
        for j, t_ in enumerate(tk.tolist()):
            cc = tuple(int(x * 255) for x in pal[j])
            for s_ in range(uv.shape[0]):
                x, y = uv[s_, t_].tolist()
                if 0 <= x < W and 0 <= y < H:
                    dr.ellipse((x - 2, y - 2, x + 2, y + 2), fill=cc)
        canvas.save(d6 / "token_slots_projected_view0_50scenes.png")
    # D7 coverage of the laser surface
    d7 = sdir / "D7_surface_coverage"
    d7.mkdir(exist_ok=True)
    d_in, _n, ok_in = raycast_gt_depth_and_normals(rc, data["cam_view_input"][0], data["intrinsics_input"][0], lopt.img_size)
    P, pix = [], []
    for v in range(V):
        yy, xx = torch.meshgrid(torch.arange(1, H, 2), torch.arange(1, W, 2), indexing="ij")
        m = ok_in[v][yy, xx]
        yy, xx = yy[m], xx[m]
        z = d_in[v][yy, xx].to(dev)
        fx, fy, cx, cy = data["intrinsics_input"][0, v].tolist()
        pc = torch.stack([(xx.to(dev) + 0.5 - cx) / fx * z, (yy.to(dev) + 0.5 - cy) / fy * z, z], -1)
        c2w = torch.linalg.inv(data["cam_view_input"][0, v].float().T)
        P.append(pc @ c2w[:3, :3].T + c2w[:3, 3]); pix.append((v, yy, xx))
    Pc = torch.cat(P)
    a, r = An[12][alive], Rn[12][alive]
    cnt = torch.cat([(torch.cdist(Pc[c0:c0 + 8192], a) < r).sum(-1) for c0 in range(0, Pc.shape[0], 8192)]).float()
    write_points(d7 / "laser_surface_anchor_support.ply", to_m(Pc), heat(torch.log1p(cnt), 0, math.log1p(200)), {"anchors_supporting": cnt})
    write_points(d7 / "laser_surface_uncovered.ply", to_m(Pc[cnt == 0]), torch.tensor([[1.0, 0.1, 0.8]]).expand(int((cnt == 0).sum()), 3))
    v0 = pix[0]
    cmap = torch.full((H, W), -1.0)
    cmap[v0[1], v0[2]] = cnt[:v0[1].numel()].cpu()
    img0 = data["images_input"][0, 0].mean(0)
    grad = (F.pad(img0[:, 1:] - img0[:, :-1], (0, 1)).abs() + F.pad(img0[1:] - img0[:-1], (0, 0, 0, 1)).abs())
    cm = F.max_pool2d(cmap[None, None], 3, 1, 1)[0, 0]
    grid_image([[label(rgb_img(data["images_input"][0, 0]), "input view 0"),
                 label(map_img(F.avg_pool2d(grad[None, None], 9, 1, 4)[0, 0], 0, 0.15), "texture (gradient)"),
                 label(map_img(torch.log1p(cm.clamp_min(0)), 0, math.log1p(200), cm >= 0), "#anchors supporting (log)"),
                 label(map_img((cm == 0).float(), 0, 1, cm >= 0), "uncovered = red")]]).save(d7 / "coverage_vs_texture_view0.png")
    info["coverage_uncovered_frac"] = float((cnt == 0).float().mean())
    # D8 self-attention links
    d8 = sdir / "D8_self_attention"
    d8.mkdir(exist_ok=True)
    Sm = caps[12]["self"]
    lines, lcol, cen = [], [], []
    pal8 = palette(12, 9)
    for k, t_ in enumerate(pick[:12].tolist()):
        w, j = Sm[t_].topk(9)
        for wt, jj in zip(w.tolist(), j.tolist()):
            if jj == t_:
                continue
            lines.append(segment(to_m(An[12, t_]), to_m(An[12, jj]), 30)); lcol.append(heat(torch.full((30,), wt), 0, 0.2))
        cen.append(ball(to_m(An[12, t_]), 0.01)); lcol.append(pal8[k].expand(60, 3)); lines.append(cen[-1])
    write_points(d8 / "self_attention_top8_links_12tokens.ply", torch.cat(lines), torch.cat(lcol))
    # D9 child template
    d9 = sdir / "D9_child_template"
    d9.mkdir(exist_ok=True)
    w2c_dom = data["cam_view_input"][0][dom].float().transpose(1, 2)[:, :3, :3]
    off = (g64[..., :3] - An[12].unsqueeze(1)) / Rn[12][:, None, None]
    offc = torch.einsum("nij,nkj->nki", w2c_dom, off)
    am = (op64 >= ALIVE_OPACITY) & alive.unsqueeze(1)
    slot = torch.arange(CH, device=dev).expand(ntok, CH)
    sub = torch.rand(am.shape, generator=torch.Generator(device="cpu").manual_seed(1)).to(dev) < 0.15
    mm = am & sub
    write_points(d9 / "child_offsets_canonical_frame_radius_units.ply", offc[mm].clamp(-6, 6) * 0.1, heat(slot[mm].float(), 0, 63),
                 {"slot": slot[mm].float(), "opacity": op64[mm]})
    smean = torch.stack([offc[:, k][am[:, k]].mean(0) if am[:, k].any() else torch.zeros(3, device=dev) for k in range(CH)])
    write_points(d9 / "child_slot_mean_offsets.ply", torch.cat([ball(smean[k].clamp(-6, 6) * 0.1, 0.01) for k in range(CH)]),
                 heat(torch.arange(CH).repeat_interleave(60).float(), 0, 63))
    write_points(d9 / "unit_radius_sphere.ply", ball(torch.zeros(3), 0.1, 300), torch.tensor([[0.7, 0.7, 0.7]]).expand(300, 3))
    # D10 interventions
    d10 = sdir / "D10_interventions"
    d10.mkdir(exist_ok=True)
    rows, iv = [], {}
    for name, cfg in INTERVENTIONS.items():
        kw = {k: v for k, v in cfg.items() if k != "init"}
        if "init" in cfg:
            kw["init_anchor"] = init_anchor_variant(model, cfg["init"], seed=idx)
        Gi = G if name == "baseline" else locus_forward(model, lopt, mi, **kw)["G"]
        e = ev(Gi)
        iv[name] = {k: v for k, v in e.items() if not k.startswith("_")}
        rows.append(strip_row(e, 0, name) + strip_row(e, 4, name)[:1])
        write_gaussians(d10 / f"{name}.ply", rigid(Gi[0], Rm, tm), Gi[0, :, 11:14])
        DPE.export_pruned_ply(d10 / f"{name}.ply", d10 / f"{name}_prune0.05.ply", 0.05)
        (d10 / f"{name}.ply").unlink()
    grid_image(rows).save(d10 / "interventions_overview.png")
    info["interventions"] = iv
    # D11 view count and extrapolation
    d11 = sdir / "D11_views_and_extrapolation"
    d11.mkdir(exist_ok=True)
    rows = []
    for nv, sel in VIEW_SETS.items():
        dsub = subset_views(data, sel)
        for mdl, o, nm in ((model, lopt, "LocusGS"), (paper, popt, "paperTokenGS")):
            oo = dataclasses.replace(o, num_input_views=nv)
            m_ = ls(dsub, oo)[0]
            Gv = mdl.forward_decoder(mdl.forward_encoder(m_.encoder), m_.decoder)
            e = Evaluator(mdl, data, rc, opt, dev)(Gv)
            rows.append(strip_row(e, 0, f"{nm} {nv}v"))
    grid_image(rows).save(d11 / "views_2_4_6.png")
    prov_ext = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    set_extrapolation_targets(prov_ext.dataset, "ext_far")
    _s, dext, _r, _rc = scene_setup(prov_ext, idx, dev)
    rows = [[label(rgb_img(dext["images_output"][0, v]), f"GT ext_far t{v}") for v in (0, 2, 5, 7)]]
    pG = paper.forward_decoder(paper.forward_encoder(ls(data, popt)[0].encoder), ls(data, popt)[0].decoder)
    for mdl, Gx, nm in ((model, G, "LocusGS"), (paper, pG, "paperTokenGS")):
        e = Evaluator(mdl, dext, rc, opt, dev)(Gx)
        r_ = e["_render"]
        rows.append([label(rgb_img(r_["images_pred"][0, v].clamp(0, 1)), f"{nm} {e['psnr']:.2f}dB") for v in (0, 2, 5, 7)])
        rows.append([label(map_img(e["_sigma"][0, v, 0], 0, 0.3, r_["alphas_pred"][0, v, 0] > 0.5), f"{nm} thick") for v in (0, 2, 5, 7)])
    grid_image(rows).save(d11 / "ext_far_locus_vs_paper.png")
    DPE.export_camera_ply(d11 / "cameras_ext_far.ply", dm["cam_to_world_input"][0], rigid_cams(dext["cam_view"][0], Rm, tm))
    (sdir / "demo_summary.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "per_layer_decode"}, indent=1))


def rigid_cams(cam_view, R, t):
    c2w = torch.linalg.inv(cam_view.float().transpose(1, 2))
    out = c2w.clone()
    out[:, :3, :3] = R @ c2w[:, :3, :3]
    out[:, :3, 3] = c2w[:, :3, 3] @ R.T + t
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--limit", type=int, default=0)
    sub.add_parser("summarize")
    sub.add_parser("demo")
    a = ap.parse_args()
    {"stats": cmd_stats, "summarize": cmd_summarize, "demo": cmd_demo}[a.cmd](a)


if __name__ == "__main__":
    main()
