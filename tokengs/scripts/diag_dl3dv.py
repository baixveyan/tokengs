"""Re-check the ScanNet++ diagnoses on DL3DV-10K-Benchmark for frozen query-based models (no training of the models).

140 scenes, official 6-view evaluation json (6 context frames, 50 targets), FP32, TF32 off.  Models (--model):
  tokengs_release_latent_6v  official dl3dv_latent_6v_ssim (12-layer encoder, 4096 scene latents, 1-layer decoder; mean_cam)
  tokengs_paper_4v           official paper TokenGS dl3dv_4v (3-layer encoder, 12-layer decoder; first_cam) = the
                             TokenGS baseline of the LocusGS paper (loaded with the LocusGS code, model_type tokengs)
  locusgs_finetune_4v        LocusGS DL3DV finetune (paper architecture + per-token 3D anchor centre/radius; first_cam)
Token -> input-patch attention T: release = decoder cross-attn (head mean) @ layer-7 local-head latent attention; decoder
models = last decoder layer cross-attn (head mean; for LocusGS including its anchor-to-ray bias).
DL3DV has no laser mesh: the geometry reference is the COLMAP sparse cloud of each scene (points visible in a frame =
points whose track contains that image), transformed into each model's normalized frame (scale 0.15).  COLMAP is not
metric, sparse and biased to textured surfaces, so only scale-free / relative quantities are reported; lengths are
divided by the depth from the relevant camera.  Definitions follow the ScanNet++ scripts they mirror.

  extract --model M --shard k --num-shards N   per scene: PSNR (all 50 targets), depth vs COLMAP, thickness split
                                     (diag_thickness_split), render ownership + pixel token counts
                                     (diag_token_render_ownership), token / scene-latent attention overlap + D0 locality
                                     (probe_scene_latent_overlap / diagnose_latent_attention_geometry), token shape
                                     (diag_token_shape), LocusGS C_centroid, junk slots (diag_junk_slots), parallax vs
                                     error (diag_thinning_and_parallax parallax), voxel snap (diag_voxel_snap), LocusGS
                                     anchors, probe features
  summarize --model M                -> EXP_DL3DV/diag_dl3dv_summary[_M].json
  probe --model M                    light probes on frozen GS-token features (train even scenes, test odd scenes)
                                     -> EXP_DL3DV/diag_dl3dv_probe[_M].json

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/diag_dl3dv.py extract --model locusgs_finetune_4v --shard 0 --num-shards 7
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

import diag_thickness_split as TS
from diag_junk_slots import render_weight
from diag_thinning_and_parallax import parallax_deg, spearman
from diag_token_render_ownership import rasterize_channels, top_mass_set
from diagnose_latent_attention_geometry import ALIVE_OPACITY, capture_cross_attention
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, idle_gpu
from probe_gs_token_geometry import child_centre, decode, load_model
from safetensors.torch import load_file
from scannetpp_evaluation_protocol import sha256_file
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults

CACHE_ROOT = Path("/mnt/disk/xveyan/workspace_dl3dv/diag_dl3dv")
OUT = ROOT / "EXP_DL3DV"
LOCUS_ROOT = ROOT.parent / "LocusGS"
DEFAULT_MODEL = "tokengs_release_latent_6v"
MODELS = {
    "tokengs_release_latent_6v": {"kind": "release", "ckpt": CHECKPOINT, "sha": CHECKPOINT_SHA},
    "tokengs_paper_4v": {"kind": "tokengs_paper", "ckpt": ROOT / "checkpoints_hf/checkpoints/dl3dv_4v.safetensors",
                         "sha": "0d6012d133e2488be82da4c747aae9e7cf7f2320d4d72f6bc224387f65e11980"},
    "locusgs_finetune_4v": {"kind": "locusgs",
                            "ckpt": Path("/mnt/disk/xveyan/checkpoints/locusgs/checkpoints/dl3dv/finetune/model.safetensors"),
                            "sha": "a554f552505f97122ae7e745090d37cefd023b4d9c7f87de3ce22e5683e09ec0"},
}
ATTN_LAYERS = {"release": (3, 7, 11), "tokengs_paper": (4, 8, 12), "locusgs": (4, 8, 12)}  # latent idx (0-based) / decoder layer (1-based)


def cache_dir(model):
    """Per-model cache; the first release-only run (shard_*.pt directly in CACHE_ROOT) is kept untouched."""
    return CACHE_ROOT / model


def out_name(stem, model):
    """Per-model outputs; the first release-only run's diag_dl3dv_summary.json / diag_dl3dv_probe.json stay as they are."""
    return OUT / f"{stem}_{model}.json"
EVAL_JSON = ROOT / "assets/evaluation_idx_dl3dv_depthsplat_6v.json"
SCALE = 0.15
N_HEAVY = 8                      # novel target views per scene used for the expensive per-pixel diagnostics
VOXEL_NORM = (0.0, 0.006, 0.012, 0.024)   # = 0 / 2 / 4 / 8 cm on ScanNet++ (scale 0.3), same normalized units
FAR_NORM = 0.09                  # = 30 cm on ScanNet++, normalized units (junk-slot distance threshold)
PIX_SAMPLE, DEPTH_SAMPLE, PROBE_TOK, CH, CHUNK = 4000, 3000, 1024, 64, 512
REL_BINS = ((0, 0.015), (0.015, 0.05), (0.05, 0.15), (0.15, 1e9))   # thickness / depth  (~3/10/30 cm at 2 m)
SCANNETPP_REF = {   # official model on ScanNet++ (docs/DIAG_*.json, AGENT_MEMORY 15.13-15.14)
    "thickness_between_share_pooled": 0.90, "thickness_between_share_per_pixel_median": 0.72,
    "render_weight_in_attention90_median": 0.87, "render_weight_in_attention90_weighted": 0.83,
    "patches_per_latent_median": 17, "latents_covering_each_patch_median": 6, "near_duplicate_latent_frac": 0.19,
    "token_flatness_median": 0.58, "token_ray_over_lateral_median": 1.37, "junk_slot_share": 0.24,
    "junk_render_weight_share": 0.049, "dead_child_slots_all_tokens": [0, 7, 16, 19, 20],
    "parallax_lt2deg_token_share": 0.25, "spearman_error_vs_distance_over_parallax": 0.47,
    "spearman_error_vs_parallax": 0.40, "voxel_snap_between_share": 0.91, "depth_absrel_vs_laser": 0.174,
}


def make_opt(kind="release"):
    """Data / renderer options of the TokenGS repo matching the model's own evaluation preset."""
    preset = "eval_dl3dv_latent_6view_ssim" if kind == "release" else "eval_dl3dv_6view"
    extra = {"use_geo_losses": False, "load_lidar_pc": False, "val_depth_metric": False} if kind == "release" else {}
    return dataclasses.replace(
        config_defaults[preset], evaluating=True, use_ttt_for_eval=False, ttt_n_steps=0, use_input_supervision=False,
        mixed_precision="no", num_workers=0, dataset_kwargs={"evaluation_json": str(EVAL_JSON)}, **extra)


def colmap_norm(ds, idx, ctx, frames, method="mean_cam"):
    """COLMAP points in the provider's normalized frame and, per frame, the indices of the points it observes."""
    clip, _n, _intr, c2w_all, zh, js = ds.load_video_reader(idx)
    try:
        xyz, tracks = ds._get_colmap_points(zh, clip)
    finally:
        zh.close()
    pw = np.c_[xyz, np.ones(len(xyz))] @ ds._colmap_world_to_final_world_transform(js).T
    c = torch.from_numpy(c2w_all[ctx]).double()
    if method == "first_cam":
        w2n = torch.linalg.inv(c[0])
    else:
        fwd = F.normalize(c[:, :3, 2].mean(0), dim=0)
        down = c[:, :3, 1].mean(0)
        down = F.normalize(down - down.dot(fwd) * fwd, dim=0)
        M = torch.eye(4, dtype=torch.float64)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = torch.linalg.cross(down, fwd), down, fwd, c[:, :3, 3].mean(0)
        w2n = torch.linalg.inv(M)
    pn = (torch.from_numpy(pw[:, :3]) @ w2n[:3, :3].T + w2n[:3, 3]) * SCALE
    by_img: dict[int, list[int]] = {}
    for j, s in enumerate(tracks):
        for im in s:
            by_img.setdefault(im, []).append(j)
    vis = [torch.tensor(by_img.get(js["frames"][f].get("colmap_im_id"), []), dtype=torch.long) for f in frames]
    return pn.float(), vis


def project(P, w2c, intr, H, W):
    xc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = xc[:, 2]
    fx, fy, cx, cy = [float(x) for x in intr]
    u, v = fx * xc[:, 0] / z.clamp_min(1e-6) + cx, fy * xc[:, 1] / z.clamp_min(1e-6) + cy
    return u, v, z, (z > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def patch_points(pn, vis_in, cam_in, intr_in, H, W, p):
    """Per input-view 8x8 patch: the median-depth visible COLMAP point inside it (if any)."""
    hp, wp = H // p, W // p
    pp, pval = torch.zeros(len(vis_in) * hp * wp, 3), torch.zeros(len(vis_in) * hp * wp, dtype=torch.bool)
    for v, ids in enumerate(vis_in):
        if not len(ids):
            continue
        P = pn[ids]
        u, vv, z, ok = project(P, cam_in[v].T, intr_in[v], H, W)
        P, u, vv, z = P[ok], u[ok], vv[ok], z[ok]
        if not len(P):
            continue
        pid = ((vv // p).long() * wp + (u // p).long()).numpy()
        order = np.lexsort((z.numpy(), pid))
        uniq, start, cnt = np.unique(pid[order], return_index=True, return_counts=True)
        sel = torch.from_numpy(order[start + cnt // 2])
        pp[v * hp * wp + torch.from_numpy(uniq)] = P[sel]
        pval[v * hp * wp + torch.from_numpy(uniq)] = True
    return pp, pval


def kmat(intr, device):
    fx, fy, cx, cy = intr.float().tolist()
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=device)


def sample(x, n, gen):
    return x if x.shape[0] <= n else x[torch.randperm(x.shape[0], generator=gen)[:n]]


def thickness_views(g, cam_view, intr, dep, views, H, W, opt, gen):
    """Pooled variance sums and a pixel sample (var_tot, var_within, var_between, depth) over alpha > 0.5 pixels."""
    sums, rows = torch.zeros(3, dtype=torch.float64), []
    for v in views:
        alpha, vt, vw, vb, _ve = TS.split_view(g, cam_view[v].float().T.contiguous(), kmat(intr[v], g.device), H, W, opt)
        m = alpha > 0.5
        sums += torch.tensor([float(vt[m].sum()), float(vw[m].sum()), float(vb[m].sum())], dtype=torch.float64)
        rows.append(sample(torch.stack([vt[m], vw[m], vb[m], dep[v][m]], -1).cpu(), PIX_SAMPLE, gen))
    return sums, torch.cat(rows)


def head_stats(at):
    """Per-head effective number of attended patches (exp entropy) for attention [H, Nq, Npatch]."""
    eff = (-(at.clamp_min(1e-12).log() * at).sum(-1)).exp()
    loc = eff.median(-1).values < 30
    return {"n_local_heads": int(loc.sum()), "n_heads": int(loc.numel()),
            "eff_patches_local_median": float(eff[loc].median()) if loc.any() else float("nan"),
            "eff_patches_global_median": float(eff[~loc].median()) if (~loc).any() else float("nan")}, loc


def load_any(name, dev):
    """Return (model, model_opt, data_opt, kind)."""
    m = MODELS[name]
    if sha256_file(m["ckpt"]) != m["sha"]:
        raise RuntimeError(f"Unexpected checkpoint SHA256 for {name}")
    if m["kind"] == "release":
        opt = make_opt("release")
        return load_model(m["ckpt"], opt, dev), opt, opt, "release"
    if str(LOCUS_ROOT) not in sys.path:
        sys.path.append(str(LOCUS_ROOT))
    from locusgs.models import model_registry as locus_registry
    from locusgs.options import config_defaults as locus_defaults

    lopt = locus_defaults["eval_dl3dv_6view"].evolve(
        model_type="locusgs" if m["kind"] == "locusgs" else "tokengs", mixed_precision="no", num_workers=0,
        dataset_kwargs={"evaluation_json": str(EVAL_JSON)})
    net = locus_registry[lopt.model_type](lopt).eval().requires_grad_(False)
    res = net.load_state_dict(load_file(str(m["ckpt"]), device="cpu"), strict=False)
    # the released LocusGS checkpoint also stores anchor_head_cross_attn (only used by cross_attn_variant
    # "learned_positional"); it is trained with geometric_positional (geo_scale present), so these are unused
    if res.missing_keys or any("anchor_head_cross_attn" not in k for k in res.unexpected_keys):
        raise RuntimeError(f"checkpoint mismatch: missing {res.missing_keys[:5]} unexpected {res.unexpected_keys[:5]}")
    return net.to(device=dev, dtype=torch.float32), lopt, make_opt(m["kind"]), m["kind"]


def run_release(model, opt, data):
    mi, _ = split_data(data, opt)
    rec = model.forward_reconstruction(mi)
    g = rec.gaussians[0].float().clone()
    latents, attn = capture_cross_attention(model, mi, list(ATTN_LAYERS["release"]))
    d0, a_lat = {}, None
    for li, at in attn.items():
        d0[f"latent_layer_{li}"], loc = head_stats(at)
        if li == 7:
            a_lat = at[loc].mean(0)
    del attn
    A_dec, h, g64 = decode(model, latents, mi)
    bg = rec.background_color
    return {"g": g, "T": (A_dec @ a_lat).float(), "h": h, "mix": A_dec @ latents[0].float(), "d0": d0, "lat_attn": a_lat,
            "check": float((g64.reshape(-1, 14) - g).abs().max()), "anchor": None, "radius": None,
            "render": lambda G, cam, intr: model.gs.render(G, cam, bg_color=bg, intrinsics=intr)}


def run_decoder(model, lopt, data, kind):
    """Paper-architecture TokenGS / LocusGS: replay the decoder, capturing the cross-attention of selected layers."""
    from locusgs.models.input_types import split_data as locus_split

    mi, _ = locus_split(data, lopt)
    enc = model.forward_encoder(mi.encoder)
    ref = model.forward_decoder(enc, mi.decoder)
    blocks = model.enc_dec_backbone.decoder_blocks
    tok = model.get_gs_tokens(1)
    if kind == "locusgs":
        anchor = model.gs_anchor_xyz_raw.unsqueeze(0)
        rr = model.gs_anchor_radius_raw.unsqueeze(0)
        rad = model.activate_anchor_radius(rr)
    else:
        tok = model._apply_time_embedding_to_gs_tokens(tok, mi.decoder)
    d0, T = {}, None
    for li, layer in enumerate(blocks, 1):
        want = li in ATTN_LAYERS[kind]
        if kind == "locusgs":
            out = layer(gs_tokens=tok, gs_anchor_normalized=model.normalize_anchor(anchor),
                        gs_anchor_radius=rad if lopt.anchor_radius_affects_bias else None, keys=enc.keys, values=enc.values,
                        plucker_emb_for_concat=enc.plucker_emb_for_concat, plucker_rays_patch_raw=enc.plucker_rays_patch_raw,
                        gs_anchor_raw=anchor, handcraft=want, return_attn=want)
        else:
            out = layer(gs_tokens=tok, keys=enc.keys, values=enc.values, return_attn=want)
        if want:
            tok, info = out
            at = info["attn"][0].float()
            d0[f"decoder_layer_{li}"], _loc = head_stats(at)
            if li == len(blocks):
                T = at.mean(0)
            del info, at
        else:
            tok = out
        if kind == "locusgs":
            anchor = anchor + model.activation_head.pos_act(model.xyz_embed(tok))
            if lopt.anchor_radius_refinement:
                rr = rr + model.radius_embed(tok)
                rad = model.activate_anchor_radius(rr)
    G = model.activation_head(tok)
    if kind == "locusgs":
        xyz = model.compose_gaussian_xyz(G[..., :3], anchor, rad, model.activation_head.num_gaussians_per_token)
        G = torch.cat([xyz, G[..., 3:]], -1)
    else:
        G[..., 2] = G[..., 2] + lopt.gaussian_z_offset
    bg = model._background_color(torch.float32, G.device)
    return {"g": G[0].float(), "T": T, "h": tok[0].float(), "mix": None, "d0": d0, "lat_attn": None,
            "check": float((G - ref).abs().max()),
            "anchor": anchor[0].float() if kind == "locusgs" else None,
            "radius": rad[0, :, 0].float() if kind == "locusgs" else None,
            "render": lambda Gs, cam, intr: model.gs.render(Gs, cam, bg_color=bg, intrinsics=intr)}


def overlap_stats(A):
    """90 % top-64 patch sets of attention rows A [N, V*P]: set sizes, patch coverage, pairwise cosine duplicates."""
    tw, ti = A.topk(64, -1)
    cum = tw.cumsum(-1) / A.sum(-1, keepdim=True)
    keep = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool), cum[:, :-1] < 0.9], 1)
    Wm = torch.zeros_like(A).scatter_(1, ti, tw * keep)
    cover = (Wm > 0).sum(0)
    Wn = F.normalize(Wm, dim=-1)
    sim = Wn @ Wn.T
    sim.fill_diagonal_(0)
    res = {"n_patch": keep.sum(-1).cpu().short(), "cover": cover[cover > 0].cpu().short(),
           "n_sim": (sim > 0.5).sum(-1).cpu().short(), "max_sim": sim.max(-1).values.cpu().half(),
           "top64_mass": tw.sum(-1).cpu().half()}
    return res, Wm


@torch.no_grad()
def cmd_extract(a):
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")
    model, mopt, opt, kind = load_any(a.model, dev)
    prov = Provider("dl3dv_eval_scaled_0.15", opt, training=False)
    ds = prov.dataset
    H, W = map(int, opt.img_size)
    p = int(opt.patch_size)
    P = (H // p) * (W // p)
    mine = list(range(len(prov)))[a.shard::a.num_shards][: a.limit or None]
    print(f"GPU {gpu['index']}: {a.model} shard {a.shard}/{a.num_shards}: {len(mine)} scenes "
          f"(camera normalization {opt.camera_normalization_method})", flush=True)
    out, t0 = [], time.time()
    for n, idx in enumerate(mine):
        sid = Path(ds.sample_list[idx]).stem
        gen = torch.Generator().manual_seed(idx)
        ctx, tgt = ds.get_context_target_frames(idx)
        data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in prov[idx].items()}
        mr = run_release(model, mopt, data) if kind == "release" else run_decoder(model, mopt, data, kind)
        g, render = mr["g"], mr["render"]
        N, V = g.shape[0], len(ctx)
        cam_t, intr_t = data["cam_view"][0], data["intrinsics"][0]
        cam_in, intr_in = data["cam_view_input"][0], data["intrinsics_input"][0]
        r = render(g[None], data["cam_view"], data["intrinsics"])
        img, dep = r["images_pred"][0].clamp(0, 1), r["depths_pred"][0, :, 0]
        psnr = (-10 * (img - data["images_output"][0]).square().mean((1, 2, 3)).log10()).cpu()
        ctx_set = set(ctx)
        novel = [k for k, f in enumerate(tgt) if f not in ctx_set]
        heavy = sorted({novel[int(i)] for i in np.linspace(0, len(novel) - 1, N_HEAVY).round()})
        rec_ = {"scene_id": sid, "psnr_all": psnr, "novel": torch.tensor(novel), "heavy": torch.tensor(heavy)}

        # ---- COLMAP reference in the normalized frame
        pn, vis = colmap_norm(ds, idx, ctx, list(ctx) + list(tgt), opt.camera_normalization_method)
        pp, pval = patch_points(pn, vis[:V], cam_in.cpu(), intr_in.cpu(), H, W, p)
        rec_["colmap_points"], rec_["patch_coverage"] = int(pn.shape[0]), float(pval.float().mean())
        dp, dg = [], []
        for k in heavy:
            ids = vis[V + k]
            if not len(ids):
                continue
            u, vv, z, ok = project(pn[ids].to(dev), cam_t[k].T, intr_t[k], H, W)
            pr = dep[k][vv[ok].long(), u[ok].long()]
            dp.append(pr.cpu()); dg.append(z[ok].cpu())
        if dp:
            dd = sample(torch.stack([torch.cat(dp), torch.cat(dg)], -1), DEPTH_SAMPLE * len(heavy), gen)
            rec_["depth_pred_gt"] = dd

        # ---- attention locality (D0) and overlap of the attention 90 % patch sets
        rec_["d0"] = mr["d0"]
        pv, ppd = pval.to(dev).float(), pp.to(dev)
        cams = torch.stack([torch.linalg.inv(c.float().T)[:3, 3] for c in cam_in])

        def rms_rel(Wm):
            wv = Wm * pv
            ok_l = wv.sum(-1) > 0.5 * Wm.sum(-1)
            cen = wv @ ppd / wv.sum(-1, keepdim=True).clamp_min(1e-12)
            rms = ((wv * (ppd.unsqueeze(0) - cen.unsqueeze(1)).square().sum(-1)).sum(-1) / wv.sum(-1).clamp_min(1e-12)).sqrt()
            return (rms / (cen - cams.mean(0)).norm(dim=-1).clamp_min(1e-6))[ok_l].cpu().half()

        if mr["lat_attn"] is not None:   # release model: the scene latents' own attention (as the first DL3DV run)
            rec_["latent_overlap"], Wm = overlap_stats(mr["lat_attn"])
            rec_["latent_rms_rel"] = rms_rel(Wm)
            del Wm
        T = mr["T"]                                                           # [Ntok, V*P]
        rec_["token_overlap"], Wm = overlap_stats(T)
        rec_["token_attn_rms_rel"] = rms_rel(Wm)
        del Wm
        h = mr["h"]
        rec_["decode_vs_render_max_abs_diff"] = mr["check"]
        g64 = g.reshape(-1, CH, 14)
        ntok = T.shape[0]
        a90 = top_mass_set(T)
        vm = T.reshape(ntok, V, P).sum(-1)
        use = vm > 0.1
        dom = cams[vm.argmax(-1)]
        op64 = g64[..., 3]
        walive = op64 * (op64 >= ALIVE_OPACITY)
        alive = walive.sum(-1) > 0.5
        Xp = child_centre(g64)

        # ---- render ownership + pixel token counts (6 input views)
        token_id = torch.arange(N, device=dev) // CH
        R = torch.zeros((ntok, V * P), device=dev)
        pix_cnt, pix_need = [], []
        for v in range(V):
            vmv, K = cam_in[v].float().T.contiguous(), kmat(intr_in[v], dev)
            _img, alpha = rasterize_channels(g, torch.ones((N, 1), device=dev), vmv, K, H, W, opt)
            sel_pix = (alpha > 0.5).flatten().nonzero().squeeze(1)
            ap = alpha.flatten()[sel_pix]
            cnt = torch.zeros_like(ap)
            top = torch.zeros((sel_pix.numel(), 0), device=dev)
            for c0 in range(0, ntok, CHUNK):
                k = min(CHUNK, ntok - c0)
                col = torch.zeros((N, k), device=dev)
                s = (token_id >= c0) & (token_id < c0 + k)
                col[s, token_id[s] - c0] = 1.0
                im, _ = rasterize_channels(g, col, vmv, K, H, W, opt)
                R[c0:c0 + k, v * P:(v + 1) * P] = im.reshape(H // p, p, W // p, p, k).sum((1, 3)).reshape(P, k).T
                wpx = im.reshape(-1, k)[sel_pix]
                cnt += (wpx >= 0.01 * ap.unsqueeze(1)).sum(-1)
                top = torch.cat([top, wpx], 1).topk(min(32, top.shape[1] + k), dim=1).values
                del im, wpx, col
            need = (top.cumsum(1) < 0.9 * ap.unsqueeze(1)).sum(1) + 1
            sp = torch.randperm(sel_pix.numel(), generator=gen)[:PIX_SAMPLE].to(dev)
            pix_cnt.append(cnt[sp].cpu().short()); pix_need.append(need[sp].cpu().short())
        mass = R.sum(-1)
        rec_["ownership"] = {
            "mass": mass.cpu(), "r_in_a90": ((R * a90).sum(-1) / mass.clamp_min(1e-12)).cpu(),
            "a_in_r90": ((T * top_mass_set(R)).sum(-1) / T.sum(-1).clamp_min(1e-12)).cpu(),
            "same_view": (vm.argmax(-1) == R.reshape(ntok, V, P).sum(-1).argmax(-1)).cpu()}
        rec_["pixel_tokens_ge1pct"], rec_["pixel_tokens_for_90pct"] = torch.cat(pix_cnt), torch.cat(pix_need)
        del R

        # ---- token shape (opacity-weighted covariance of the 64 child centres, as diag_token_shape)
        ws = walive / walive.sum(-1, keepdim=True).clamp_min(1e-12)
        mu = (ws.unsqueeze(-1) * g64[..., :3]).sum(1)
        d = g64[..., :3] - mu.unsqueeze(1)
        cov = torch.einsum("nm,nmi,nmj->nij", ws, d, d) + 1e-12 * torch.eye(3, device=dev)
        ev = torch.linalg.eigvalsh(cov).clamp_min(0)
        ray = F.normalize(mu - dom, dim=-1)
        along = (ws * (d * ray.unsqueeze(1)).sum(-1).square()).sum(-1)
        total = (ws * d.square().sum(-1)).sum(-1)
        depth_tok = (mu - dom).norm(dim=-1).clamp_min(1e-6)
        rec_["shape"] = {k: v[alive].cpu().half() for k, v in {
            "flatness": (ev[:, 0] / ev[:, 1].clamp_min(1e-18)).sqrt(), "thin_rel": ev[:, 0].sqrt() / depth_tok,
            "long_rel": ev[:, 2].sqrt() / depth_tok, "ray_over_lateral": (along / ((total - along) / 2).clamp_min(1e-18)).sqrt()}.items()}
        rec_["child_dead_frac"] = (op64 < ALIVE_OPACITY).float().mean(0).cpu()
        # LocusGS Table 3: C_centroid = mean distance of ALL 64 child centres to their unweighted mean (normalized units)
        cen_u = g64[..., :3].mean(1, keepdim=True)
        rec_["c_centroid"] = {"all_children_norm": (g64[..., :3] - cen_u).norm(dim=-1).mean(1).cpu(),
                              "alive_weighted_rel": ((ws * d.norm(dim=-1)).sum(-1) / depth_tok)[alive].cpu().half()}

        # ---- junk slots (opacity < 0.05 and far from every COLMAP point)
        dist = torch.from_numpy(cKDTree(pn.numpy()).query(g[:, :3].cpu().numpy(), k=1, workers=8)[0]).float()
        op = g[:, 3].cpu()
        junk = (op < 0.05) & (dist > FAR_NORM)
        with torch.enable_grad():
            wgt = render_weight(g, cam_t[heavy], intr_t[heavy], H, W).cpu()
        rec_["junk"] = {"n": int(junk.sum()), "n_all": N, "n_low_opacity": int((op < 0.05).sum()),
                        "w_junk": float(wgt[junk].sum()), "w_all": float(wgt.sum()),
                        "dist_junk": sample(dist[junk], 20000, gen).half(), "dist_alive": sample(dist[op >= 0.05], 20000, gen).half(),
                        "op_junk": sample(op[junk], 20000, gen).half()}

        # ---- parallax and errors against the COLMAP region of each token
        wr = T * a90 * pv
        okr = (wr.sum(-1) > 0.5 * (T * a90).sum(-1)) & alive
        Xg = wr @ ppd / wr.sum(-1, keepdim=True).clamp_min(1e-12)
        dgt = (Xg - dom).norm(dim=-1).clamp_min(1e-6)
        rec_["parallax"] = {
            "alive": alive.cpu(), "valid": okr.cpu(), "nv": use.sum(-1).cpu().short(),
            "par_pred": parallax_deg(Xp, cams, use).cpu(), "par_gt": parallax_deg(Xg, cams, use).cpu(),
            "dist_rel_cam": dgt.cpu(), "e_depth_rel": (((Xp - dom).norm(dim=-1) - dgt).abs() / dgt).cpu(),
            "e_3d_rel": ((Xp - Xg).norm(dim=-1) / dgt).cpu()}

        # ---- LocusGS anchors: centre / radius of the final layer
        if mr["anchor"] is not None:
            c, rad = mr["anchor"], mr["radius"]
            dc = (c - dom).norm(dim=-1).clamp_min(1e-6)
            off = (g64[..., :3] - c.unsqueeze(1)).norm(dim=-1) / rad.unsqueeze(1)       # child offset in radius units
            wsum = walive.sum(-1).clamp_min(1e-12)
            dd_ = torch.cdist(c, c)
            dd_.fill_diagonal_(float("inf"))
            rec_["anchor"] = {
                "radius_rel": (rad / dc).cpu(), "radius_norm": rad.cpu(),
                "frac_alive_mass_within_1r": ((walive * (off <= 1)).sum(-1) / wsum).cpu().half(),
                "frac_alive_mass_within_2r": ((walive * (off <= 2)).sum(-1) / wsum).cpu().half(),
                "off_over_r_alive_sample": sample(off[op64 >= ALIVE_OPACITY].flatten(), 50000, gen).half().cpu(),
                "anchor_to_gauss_centre_rel": ((Xp - c).norm(dim=-1) / dc).cpu(),
                "anchor_e_depth_rel": ((dc - dgt).abs() / dgt).cpu(),
                "anchor_e_3d_rel": ((c - Xg).norm(dim=-1) / dgt).cpu(),
                "nn_anchor_over_r": (dd_.min(-1).values / rad).cpu(),
                "n_anchors_within_own_r": (dd_ < rad.unsqueeze(1)).sum(-1).cpu().short(),
                "n_support_overlaps": (dd_ < (rad.unsqueeze(1) + rad.unsqueeze(0))).sum(-1).cpu().short()}
            del dd_

        # ---- thickness split (heavy novel targets) and voxel snap
        sums, pix = thickness_views(g, cam_t, intr_t, dep, heavy, H, W, opt, gen)
        rec_["thickness"] = {"sums": sums, "pix": pix}
        vox = {}
        for eps in VOXEL_NORM:
            gs_ = TS_snap(g, eps)
            ri = render(gs_[None], data["cam_view"][:, heavy], data["intrinsics"][:, heavy])
            pt = -10 * (ri["images_pred"][0].clamp(0, 1) - data["images_output"][0][heavy]).square().mean((1, 2, 3)).log10()
            rin = render(gs_[None], data["cam_view_input"], data["intrinsics_input"])
            pin = -10 * (rin["images_pred"][0].clamp(0, 1) - data["images_input"][0]).square().mean((1, 2, 3)).log10()
            dv = ri["depths_pred"][0, :, 0]
            dep_h = torch.zeros_like(dep)
            dep_h[heavy] = dv
            s_, px_ = thickness_views(gs_, cam_t, intr_t, dep_h, heavy[::2], H, W, opt, gen)
            vox[eps] = {"psnr_t": pt.cpu(), "psnr_in": pin.cpu(), "sums": s_, "pix": px_[:, [0, 3]]}
        rec_["voxel"] = vox

        # ---- probe features (frozen): deconv input h and the latent mixture the token reads
        cand = torch.nonzero(okr).squeeze(1)
        cand = cand[torch.randperm(cand.numel(), generator=gen)[:PROBE_TOK].to(dev)]
        rec_["probe"] = {"h": h[cand].cpu().half(), "Xg": Xg[cand].cpu(),
                         "dom": dom[cand].cpu(), "Xp": Xp[cand].cpu(), "nv": use.sum(-1)[cand].cpu().short()}
        if mr["mix"] is not None:
            rec_["probe"]["mix"] = mr["mix"][cand].cpu().half()
        if mr["anchor"] is not None:
            rec_["probe"]["anchor"] = mr["anchor"][cand].cpu()
        out.append(rec_)
        del T, a90, h, g64, mr
        torch.cuda.empty_cache()
        print(f"[shard {a.shard}] {n + 1}/{len(mine)} {sid} {time.time() - t0:.0f}s psnr_all {float(psnr.mean()):.2f} "
              f"colmap pts {pn.shape[0]} patch cov {float(pval.float().mean()):.2f} decode diff {rec_['decode_vs_render_max_abs_diff']:.1e}", flush=True)
    cache = cache_dir(a.model)
    cache.mkdir(parents=True, exist_ok=True)
    name = f"shard_{a.shard}" if not a.limit else f"smoke_{a.shard}"
    torch.save(out, cache / f"{name}.tmp")
    (cache / f"{name}.tmp").rename(cache / f"{name}.pt")
    print("wrote", cache / f"{name}.pt", flush=True)


def TS_snap(g, eps):
    if eps <= 0:
        return g
    key = torch.floor(g[:, :3] / eps).long()
    _, inv = torch.unique(key, dim=0, return_inverse=True)
    w = g[:, 3:4].clamp_min(1e-6)
    num = torch.zeros((int(inv.max()) + 1, 3), device=g.device).index_add_(0, inv, w * g[:, :3])
    den = torch.zeros((int(inv.max()) + 1, 1), device=g.device).index_add_(0, inv, w)
    out = g.clone()
    out[:, :3] = (num / den)[inv]
    return out


def q(x, ps=(0.1, 0.5, 0.9)):
    x = x[torch.isfinite(x)].double()
    if not x.numel():
        return {}
    if x.numel() > 8_000_000:
        x = x[torch.randperm(x.numel())[:8_000_000]]
    return {f"p{int(p * 100)}": round(float(x.quantile(p)), 4) for p in ps}


def load_recs(model, prefix="shard_"):
    recs = []
    for f in sorted(cache_dir(model).glob(f"{prefix}*.pt")):
        recs.extend(torch.load(f, map_location="cpu", weights_only=False))
    return sorted(recs, key=lambda r: r["scene_id"])


def thickness_summary(sums, pix):
    vt, vb, dep = pix[:, 0].double(), pix[:, 2].double(), pix[:, 3].double().clamp_min(1e-6)
    rel = vt.clamp_min(0).sqrt() / dep
    ok = rel > 1e-4
    share = (vb[ok] / vt[ok].clamp_min(1e-18)).clamp(0, 1)
    bins = {}
    for lo, hi in REL_BINS:
        m = (rel[ok] >= lo) & (rel[ok] < hi)
        if m.any():
            bins[f"{lo:g}-{hi if hi < 1e9 else 'inf'}"] = {
                "pixel_share": float(m.double().mean()), "pooled_between_share": float(vb[ok][m].sum() / vt[ok][m].sum()),
                "median_pixel_between_share": float(share[m].median())}
    return {"between_share_pooled": float(sums[2] / sums[0]), "within_share_pooled": float(sums[1] / sums[0]),
            "between_share_per_pixel_median": float(share.median()), "thickness_over_depth": q(rel, (0.5, 0.9)),
            "by_thickness_over_depth": bins}


def overlap_summary(recs, key):
    lo = {k: torch.cat([r[key][k] for r in recs]).float() for k in ("n_patch", "cover", "n_sim", "max_sim", "top64_mass")
          if k in recs[0][key]}
    out = {"patches_in_90pct_set_(capped_64)": q(lo["n_patch"]), "rows_covering_each_patch": q(lo["cover"]),
           "other_rows_with_cosine_gt_0.5": q(lo["n_sim"]), "frac_rows_with_near_duplicate_cos_gt_0.8": float((lo["max_sim"] > 0.8).float().mean())}
    if "top64_mass" in lo:
        out["attention_mass_in_top64_patches"] = q(lo["top64_mass"])
    return out


def cmd_summarize(a):
    recs = load_recs(a.model)
    if len(recs) != 140 or len({r["scene_id"] for r in recs}) != 140:
        raise RuntimeError(f"expected 140 scenes, got {len(recs)}")
    res = {"model": a.model, "checkpoint": str(MODELS[a.model]["ckpt"]), "checkpoint_sha256": MODELS[a.model]["sha"],
           "definitions": __doc__.strip().split("\n\n")[0] + "\n" + __doc__.strip().split("\n\n")[1],
           "scenes": len(recs), "scannetpp_reference_official_model": SCANNETPP_REF}
    ps = torch.stack([r["psnr_all"].mean() for r in recs])
    pn_ = torch.stack([r["psnr_all"][r["novel"]].mean() for r in recs])
    res["psnr"] = {"all_50_targets_scene_mean": float(ps.mean()), "novel_44_targets_scene_mean": float(pn_.mean()),
                   "reference": {"tokengs_release_latent_6v": "earlier official eval 25.1935, release table 25.120",
                                 "tokengs_paper_4v": "LocusGS paper Table 1 (6 views) 24.16",
                                 "locusgs_finetune_4v": "LocusGS paper Table 1 (6 views) 25.78"}[a.model]}
    res["decode_vs_render_max_abs_diff"] = max(r["decode_vs_render_max_abs_diff"] for r in recs)
    res["colmap"] = {"points_per_scene": q(torch.tensor([float(r["colmap_points"]) for r in recs])),
                     "input_patch_coverage": q(torch.tensor([r["patch_coverage"] for r in recs]))}
    dd = torch.cat([r["depth_pred_gt"] for r in recs if "depth_pred_gt" in r]).double()
    pr, gt = dd[:, 0], dd[:, 1].clamp_min(1e-6)
    ratio = pr / gt
    per_scene = torch.tensor([float(((r["depth_pred_gt"][:, 0] - r["depth_pred_gt"][:, 1]).abs() / r["depth_pred_gt"][:, 1]).mean())
                              for r in recs if "depth_pred_gt" in r])
    res["depth_vs_colmap_heavy_targets"] = {
        "points": int(dd.shape[0]), "median_ratio_pred_over_colmap": float(ratio.median()),
        "absrel_pooled_mean": float(((pr - gt).abs() / gt).mean()), "absrel_pooled_median": float(((pr - gt).abs() / gt).median()),
        "absrel_scene_mean": float(per_scene.mean()),
        "delta_lt_1.05": float((torch.maximum(ratio, 1 / ratio) < 1.05).double().mean()),
        "delta_lt_1.10": float((torch.maximum(ratio, 1 / ratio) < 1.10).double().mean()),
        "note": "COLMAP points only (textured, sparse); ScanNet++ AbsRel 0.174 is dense laser depth - not directly comparable"}
    res["thickness_split_heavy_targets"] = thickness_summary(sum(r["thickness"]["sums"] for r in recs),
                                                             torch.cat([r["thickness"]["pix"] for r in recs]).float())
    m = torch.cat([r["ownership"]["mass"] for r in recs])
    live = m >= 1.0
    rin = torch.cat([r["ownership"]["r_in_a90"] for r in recs])
    res["render_ownership_input_views"] = {
        "live_token_frac": float(live.float().mean()), "r_in_a90": q(rin[live]),
        "r_in_a90_weighted_by_render": float((rin[live] * m[live]).sum() / m[live].sum()),
        "a_in_r90": q(torch.cat([r["ownership"]["a_in_r90"] for r in recs])[live]),
        "same_dominant_view_frac": float(torch.cat([r["ownership"]["same_view"] for r in recs])[live].float().mean())}
    res["pixel_token_counts_input_views"] = {
        "tokens_with_ge_1pct_of_pixel_weight": q(torch.cat([r["pixel_tokens_ge1pct"] for r in recs]).float()),
        "tokens_needed_for_90pct_weight_(capped_33)": q(torch.cat([r["pixel_tokens_for_90pct"] for r in recs]).float())}
    if "latent_overlap" in recs[0]:
        res["scene_latent_overlap_layer7"] = overlap_summary(recs, "latent_overlap") | {
            "latent_rms_dispersion_over_depth": q(torch.cat([r["latent_rms_rel"] for r in recs]).float())}
    res["gs_token_attention_overlap"] = overlap_summary(recs, "token_overlap") | {
        "token_attention_rms_dispersion_over_depth": q(torch.cat([r["token_attn_rms_rel"] for r in recs]).float())}
    res["d0_locality"] = {li: {k: float(np.nanmean([r["d0"][li][k] for r in recs])) for k in recs[0]["d0"][li]} for li in recs[0]["d0"]}
    sh = {k: torch.cat([r["shape"][k] for r in recs]).float() for k in recs[0]["shape"]}
    res["token_shape_alive_tokens"] = {k: q(v) for k, v in sh.items()} | {
        "share_disc_flatness_lt_0.3": float((sh["flatness"] < 0.3).float().mean()),
        "share_ray_spread_gt_lateral": float((sh["ray_over_lateral"] > 1).float().mean())}
    cc = torch.cat([r["c_centroid"]["all_children_norm"] for r in recs]).double()
    res["c_centroid"] = {"locusgs_table3_definition_mean_all_children_normalized": float(cc.mean()), "median": float(cc.median()),
                         "variance": float(cc.var()), "alive_weighted_over_depth": q(torch.cat([r["c_centroid"]["alive_weighted_rel"] for r in recs]).float()),
                         "locusgs_paper_dl3dv": {"TokenGS": {"mean": 0.5881, "median": 0.5531}, "LocusGS": {"mean": 0.0433, "median": 0.0383}}}
    dead = torch.stack([r["child_dead_frac"] for r in recs]).mean(0)
    res["child_slots"] = {"dead_in_ge99pct_tokens": torch.nonzero(dead >= 0.99).squeeze(1).tolist(),
                          "dead_in_100pct_tokens": torch.nonzero(dead >= 0.9999).squeeze(1).tolist(),
                          "alive_in_ge95pct_tokens": int((dead <= 0.05).sum()), "dead_frac_per_child": [round(float(x), 4) for x in dead]}
    J = {k: sum(r["junk"][k] for r in recs) for k in ("n", "n_all", "n_low_opacity", "w_junk", "w_all")}
    dj, da = torch.cat([r["junk"]["dist_junk"] for r in recs]).float(), torch.cat([r["junk"]["dist_alive"] for r in recs]).float()
    res["junk_slots"] = {"junk_share_opacity_lt_0.05_and_far": J["n"] / J["n_all"], "low_opacity_share": J["n_low_opacity"] / J["n_all"],
                         "junk_share_of_heavy_target_render_weight": J["w_junk"] / J["w_all"],
                         "dist_to_nearest_colmap_point_normalized": {"junk": q(dj, (0.5, 0.9)), "opacity_ge_0.05": q(da, (0.5, 0.9))},
                         "junk_opacity": q(torch.cat([r["junk"]["op_junk"] for r in recs]).float(), (0.5, 0.9)),
                         "far_threshold_normalized": FAR_NORM, "note": "nearest COLMAP point overestimates distance to the surface (sparse cloud)"}
    t = {k: torch.cat([r["parallax"][k] for r in recs]) for k in recs[0]["parallax"]}
    al, va = t["alive"].bool(), t["valid"].bool()
    par, parg, nv = t["par_pred"].float(), t["par_gt"].float(), t["nv"].float()
    zpar = t["dist_rel_cam"] / torch.deg2rad(parg).clamp_min(1e-3)
    bins = {}
    for lo_, hi_, nm in ((0, 1e-6, "single view (0 deg)"), (1e-6, 2, "0-2 deg"), (2, 5, "2-5 deg"), (5, 10, "5-10 deg"), (10, 20, "10-20 deg"), (20, 180, ">20 deg")):
        mm = ((parg >= lo_) & (parg < hi_) if lo_ > 0 else (parg < hi_)) & va
        bins[nm] = {"token_share_of_valid": float(mm.float().sum() / va.float().sum()),
                    "e_depth_rel_median": float(t["e_depth_rel"][mm].median()) if mm.any() else None,
                    "e_depth_rel_p90": float(t["e_depth_rel"][mm].float().quantile(0.9)) if mm.sum() > 1 else None}
    res["parallax"] = {
        "alive_tokens": int(al.sum()), "colmap_valid_tokens": int(va.sum()),
        "share_parallax_lt_2deg_alive_tokens_inference_time": float((par[al] < 2).float().mean()),
        "share_parallax_lt_2deg_valid_tokens_colmap_region": float((parg[va] < 2).float().mean()),
        "spearman_e_depth_rel": {"-#views": spearman(-nv[va], t["e_depth_rel"][va]), "-parallax": spearman(-parg[va], t["e_depth_rel"][va]),
                                 "-parallax_inference_time": spearman(-par[va], t["e_depth_rel"][va]),
                                 "distance/parallax_(relative)": spearman(1 / torch.deg2rad(parg[va]).clamp_min(1e-3), t["e_depth_rel"][va])},
        "spearman_e_3d_rel": {"-parallax": spearman(-parg[va], t["e_3d_rel"][va]), "-#views": spearman(-nv[va], t["e_3d_rel"][va])},
        "e_depth_rel_all_valid": q(t["e_depth_rel"][va].float()), "binned_by_colmap_region_parallax": bins,
        "note": "errors are relative to depth (COLMAP region centroid vs the token's opacity-weighted Gaussian centre); "
                "on a relative scale distance/parallax reduces to 1/parallax"}
    vx = {}
    for eps in VOXEL_NORM:
        s_ = sum(r["voxel"][eps]["sums"] for r in recs)
        px = torch.cat([r["voxel"][eps]["pix"] for r in recs]).float().double()
        vx[f"eps_{eps:g}_normalized"] = {
            "between_share_pooled": float(s_[2] / s_[0]), "thickness_over_depth": q(px[:, 0].clamp_min(0).sqrt() / px[:, 1].clamp_min(1e-6), (0.5, 0.9)),
            "psnr_heavy_targets": float(torch.stack([r["voxel"][eps]["psnr_t"].mean() for r in recs]).mean()),
            "psnr_input_views": float(torch.stack([r["voxel"][eps]["psnr_in"].mean() for r in recs]).mean())}
    res["voxel_snap"] = vx
    if "anchor" in recs[0]:
        A = {k: torch.cat([r["anchor"][k] for r in recs]).float() for k in recs[0]["anchor"]}
        tok_alive = torch.cat([r["parallax"]["alive"] for r in recs]).bool()
        tok_valid = torch.cat([r["parallax"]["valid"] for r in recs]).bool()
        e_tok = torch.cat([r["parallax"]["e_depth_rel"] for r in recs]).float()
        rr = A["radius_rel"]
        res["locusgs_anchor"] = {
            "radius_over_depth_alive_tokens": q(rr[tok_alive]),
            "radius_normalized_alive_tokens": q(A["radius_norm"][tok_alive]),
            "alive_child_mass_within_1r": q(A["frac_alive_mass_within_1r"][tok_alive], (0.1, 0.5)),
            "alive_child_mass_within_2r": q(A["frac_alive_mass_within_2r"][tok_alive], (0.1, 0.5)),
            "alive_child_offset_over_r": q(A["off_over_r_alive_sample"], (0.5, 0.9, 0.99)),
            "anchor_to_gaussian_centre_over_depth": q(A["anchor_to_gauss_centre_rel"][tok_alive]),
            "e_depth_rel_valid_tokens": {"anchor": q(A["anchor_e_depth_rel"][tok_valid]), "gaussian_centre": q(e_tok[tok_valid])},
            "e_3d_rel_valid_tokens_anchor": q(A["anchor_e_3d_rel"][tok_valid]),
            "nearest_other_anchor_over_own_radius": q(A["nn_anchor_over_r"][tok_alive]),
            "other_anchors_within_own_radius": q(A["n_anchors_within_own_r"][tok_alive]),
            "other_anchors_with_overlapping_support": q(A["n_support_overlaps"][tok_alive]),
            "spearman_radius_over_depth_vs": {
                "-parallax_colmap_region": spearman(-parg[tok_valid], rr[tok_valid]),
                "e_depth_rel_gaussian_centre": spearman(rr[tok_valid], e_tok[tok_valid]),
                "-#views": spearman(-nv[tok_valid], rr[tok_valid])},
            "note": "anchor / radius of the final decoder layer, normalized units; child offset = |mu_k - anchor| / radius"}
    OUT.mkdir(parents=True, exist_ok=True)
    out_name("diag_dl3dv_summary", a.model).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("definitions", "child_slots")}, indent=2))
    print("dead child slots (>=99% tokens):", res["child_slots"]["dead_in_ge99pct_tokens"])


def cmd_probe(a):
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda:0")
    recs = load_recs(a.model)
    tr, te = recs[0::2], recs[1::2]
    cat = lambda rs, k: torch.cat([r["probe"][k] for r in rs]).float().to(dev)  # noqa: E731

    def err(pred, rs):
        Xg, dom = cat(rs, "Xg"), cat(rs, "dom")
        dgt = (Xg - dom).norm(dim=-1).clamp_min(1e-6)
        return ((pred - dom).norm(dim=-1) - dgt).abs() / dgt, (pred - Xg).norm(dim=-1) / dgt

    def report(pred, rs):
        e1, e3 = err(pred, rs)
        nv = cat(rs, "nv")
        return {"e_depth_rel": q(e1.cpu(), (0.5, 0.9)), "e_3d_rel": q(e3.cpu(), (0.5, 0.9)),
                "e_depth_rel_1view": q(e1[nv <= 1].cpu(), (0.5, 0.9)), "e_depth_rel_4plus_views": q(e1[nv >= 4].cpu(), (0.5, 0.9))}

    res = {"model": a.model,
           "definitions": "frozen model; features of COLMAP-valid alive tokens (<=1024 per scene); target = "
                          "attention-weighted COLMAP centroid of the token's attention-90% patch set; train on the 70 even "
                          "scenes, test on the 70 odd scenes; errors relative to target depth from the dominant camera",
           "gpu": gpu["index"], "train_tokens": int(cat(tr, "Xg").shape[0]), "test_tokens": int(cat(te, "Xg").shape[0]),
           "official_output_gaussian_centre": report(cat(te, "Xp"), te)}
    if "anchor" in recs[0]["probe"]:
        res["locusgs_anchor"] = report(cat(te, "anchor"), te)
    ytr = cat(tr, "Xg")
    for feat in [f for f in ("h", "mix") if f in recs[0]["probe"]]:
        xtr, xte = cat(tr, feat), cat(te, feat)
        mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        Xa = torch.cat([xtr, torch.ones_like(xtr[:, :1])], 1).double()
        best = None
        for lam in (1e-1, 1e1, 1e3):
            w = torch.linalg.solve(Xa.T @ Xa + lam * torch.eye(Xa.shape[1], device=dev, dtype=torch.float64), Xa.T @ ytr.double())
            pred = (torch.cat([xte, torch.ones_like(xte[:, :1])], 1).double() @ w).float()
            r_ = report(pred, te)
            if best is None or r_["e_depth_rel"]["p50"] < best[1]["e_depth_rel"]["p50"]:
                best = (lam, r_)
        res[f"linear_ridge_{feat}"] = {"lambda": best[0], **best[1]}
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 1024), torch.nn.GELU(), torch.nn.Linear(1024, 1024),
                                  torch.nn.GELU(), torch.nn.Linear(1024, 3)).to(dev)
        optim = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 40
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, 1e-3, total_steps=epochs * ((xtr.shape[0] + 4095) // 4096))
        for _ep in range(epochs):
            perm = torch.randperm(xtr.shape[0], device=dev)
            for b in range(0, xtr.shape[0], 4096):
                i = perm[b:b + 4096]
                loss = F.smooth_l1_loss(net(xtr[i]), ytr[i], beta=0.01)
                optim.zero_grad()
                loss.backward()
                optim.step()
                sched.step()
        with torch.no_grad():
            res[f"mlp_{feat}"] = {**report(net(xte), te), "train_fit": report(net(xtr), tr)}
    OUT.mkdir(parents=True, exist_ok=True)
    out_name("diag_dl3dv_probe", a.model).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1)
    e.add_argument("--limit", type=int, default=0, help="smoke test: first N scenes of the shard")
    s = sub.add_parser("summarize")
    pr = sub.add_parser("probe")
    for p_ in (e, s, pr):
        p_.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    a = ap.parse_args()
    {"extract": cmd_extract, "summarize": cmd_summarize, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
