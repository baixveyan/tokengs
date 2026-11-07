"""Region-based re-evaluation of the scene-latent position probe (no model training).

Instead of collapsing a latent's attention into one (possibly floating) centroid, each latent is mapped to the
REGION of laser (GT mesh) points it attends to: its top attended input patches (local-head mean attention,
cumulative mass 90%, at most --max-patches) and, for each patch, all 8x8 full-resolution raycast mesh points.

  fitpred  retrain the two strongest probes from the cached features (large MLP, set Transformer) and save
           their validation predictions (cache of scripts/probe_scene_latent_limit.py)
  region   for the 50 validation windows: per latent, distance of the probe prediction (and of the old
           centroid) to the nearest region point and the attention-weighted mean distance; region extent and a
           multi-surface flag (depth spread along the viewing ray > 20 cm); demo-scene PLYs

    PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/probe_scene_latent_region.py fitpred
    PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/probe_scene_latent_region.py region
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from diagnose_latent_attention_geometry import capture_cross_attention
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, idle_gpu
from probe_scene_latent_limit import CACHE, S, make_opt, train_pointwise, train_set_probe
from probe_scene_latent_position import stats, write_points
from highlight_selected_tokens_ply import PALETTE, hex_rgb
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals, sha256_file
from tokengs.data.provider import Provider
from tokengs.models import model_registry
from tokengs.models.input_types import split_data

PRED_FILE = CACHE / "val_preds.pt"
DEMO_SCENE = "7b6477cb95"
DEMO_DIR = ROOT / "EXP_ScanNetPP" / "demo" / "scene_7b6477cb95" / "baseline_pretrained_6view"


def load_cache():
    recs = []
    for p in sorted(CACHE.glob("shard_*.pt")):
        recs.extend(torch.load(p, map_location="cpu", weights_only=False))
    return [r for r in recs if r["split"] == "train"], [r for r in recs if r["split"] == "val"]


def cmd_fitpred(args):
    idle_gpu()
    device = torch.device("cuda:0")
    tr, te = load_cache()
    X = torch.cat([w["latent"][w["valid"]] for w in tr])
    Y = torch.cat([w["target"][w["valid"]] for w in tr])
    t0 = time.time()
    f = train_pointwise(X, Y, device, 2048, 4, args.epochs)
    del X, Y
    out = {"large_mlp": {}, "set_transformer": {}}
    for w in te:
        out["large_mlp"][w["scene_id"]] = f(w["latent"][w["valid"]])
    print(f"large MLP done ({time.time() - t0:.0f}s)", flush=True)
    fset = train_set_probe(tr, device, args.set_epochs)
    for w in te:
        out["set_transformer"][w["scene_id"]] = fset(w)[w["valid"]]
    torch.save(out, PRED_FILE)
    print("wrote", PRED_FILE)


def full_res_points(rc, cam_in, intr_in, img_size):
    """World-space mesh hit point per pixel [V, H, W, 3], validity [V, H, W], camera centres [V, 3]."""
    d_gt, _n, ok = raycast_gt_depth_and_normals(rc, cam_in, intr_in, img_size)
    V, H, W = d_gt.shape
    ys, xs = torch.meshgrid(torch.arange(H).float() + 0.5, torch.arange(W).float() + 0.5, indexing="ij")
    pts, cams = [], []
    for v in range(V):
        c2w = torch.linalg.inv(cam_in[v].float().T)
        fx, fy, cx, cy = intr_in[v].float().tolist()
        d_cam = torch.stack([(xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)], -1)
        pts.append((d_cam * d_gt[v].unsqueeze(-1)) @ c2w[:3, :3].T + c2w[:3, 3])
        cams.append(c2w[:3, 3])
    return torch.stack(pts), ok, torch.stack(cams)


def cmd_region(args):
    gpu = idle_gpu()
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected checkpoint SHA256")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    _tr, te = load_cache()
    te = {w["scene_id"]: w for w in te}
    preds = torch.load(PRED_FILE, map_location="cpu", weights_only=False)
    opt = make_opt("val")
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov.dataset._offset_frac = None
    model = model_registry[opt.model_type](opt).eval().requires_grad_(False)
    model.load_state_dict(load_file(str(CHECKPOINT), device="cpu"), strict=True)
    model = model.to(device=device, dtype=torch.float32)
    print(f"GPU {gpu['index']}: region evaluation on {len(prov)} validation windows", flush=True)

    keys = ["old_pred_to_centroid", "centroid_to_region_nearest"] + [
        f"{p}_{m}" for p in preds for m in ("to_region_nearest", "to_region_wmean", "to_centroid")]
    acc = {k: [] for k in keys}
    acc_along = {p: [] for p in preds}
    region_rms, multi_surface, n_patches = [], [], []
    t0 = time.time()
    for idx in range(len(prov)):
        ds = prov.dataset
        sid = Path(ds.sample_list[idx]).name
        w = te[sid]
        context, _ = ds.get_context_target_frames(idx)
        raw = prov[idx]
        data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
        model_input, _ = split_data(data, opt)
        with torch.inference_mode():
            _lat, attn = capture_cross_attention(model, model_input, [args.layer])
        a = attn[args.layer][0] if attn[args.layer].dim() == 4 else attn[args.layer]
        a = a.to(device)
        cam_in, intr_in = data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu()
        raw_c2w = torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][context].copy())
        rc = build_raycasting_scene(ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
        pix, ok, cams = full_res_points(rc, cam_in, intr_in, opt.img_size)
        V, H, Wd, _ = pix.shape
        p = opt.patch_size
        hp, wp = H // p, Wd // p
        # [V*P, 64, 3] laser points under each patch, validity [V*P, 64]
        patch_pts = pix.reshape(V, hp, p, wp, p, 3).permute(0, 1, 3, 2, 4, 5).reshape(V * hp * wp, p * p, 3).to(device)
        patch_ok = ok.reshape(V, hp, p, wp, p).permute(0, 1, 3, 2, 4).reshape(V * hp * wp, p * p).to(device)
        eff = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
        local = [h for h in range(a.shape[0]) if float(eff[h].median()) < 30]
        am = a[local].mean(0)  # [N, V*P]
        valid = w["valid"].to(device)
        lat_idx = torch.nonzero(valid).squeeze(1)
        am = am[lat_idx]
        top_w, top_i = am.topk(args.max_patches, dim=-1)
        cum = top_w.cumsum(-1) / am.sum(-1, keepdim=True)
        keep = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool), cum[:, :-1] < args.mass], 1)
        top_w = top_w * keep
        n_patches.append(keep.sum(-1).cpu())
        centroid = w["target"].to(device)[lat_idx]
        cam = w["cam"].to(device)[lat_idx]
        ray = torch.nn.functional.normalize(centroid - cam, dim=-1)
        P_all = {k: v[sid].to(device) for k, v in preds.items()}
        acc["old_pred_to_centroid"].append(((P_all["set_transformer"] - centroid).norm(dim=-1) / S).cpu())
        dmin = {k: torch.full((len(lat_idx),), 1e9, device=device) for k in list(P_all) + ["centroid"]}
        vec_min = {k: torch.zeros(len(lat_idx), 3, device=device) for k in P_all}
        wsum = {k: torch.zeros(len(lat_idx), device=device) for k in P_all}
        wtot = torch.zeros(len(lat_idx), device=device)
        r_mean = torch.zeros(len(lat_idx), 3, device=device)
        r_sq = torch.zeros(len(lat_idx), device=device)
        t_lo = torch.full((len(lat_idx),), 1e9, device=device)
        t_hi = torch.full((len(lat_idx),), -1e9, device=device)
        for c in range(0, len(lat_idx), args.chunk):
            sl = slice(c, c + args.chunk)
            pts = patch_pts[top_i[sl]]  # [n, K, 64, 3]
            okm = patch_ok[top_i[sl]] & (top_w[sl] > 0).unsqueeze(-1)
            wpix = (top_w[sl].unsqueeze(-1) / (p * p)).expand_as(okm) * okm
            pts = pts.reshape(pts.shape[0], -1, 3)
            okm = okm.reshape(okm.shape[0], -1)
            wpix = wpix.reshape(wpix.shape[0], -1)
            big = torch.where(okm, 0.0, 1e9)
            wtot[sl] = wpix.sum(-1)
            mu = (wpix.unsqueeze(-1) * pts).sum(1) / wtot[sl].clamp_min(1e-12).unsqueeze(-1)
            r_mean[sl] = mu
            r_sq[sl] = (wpix * ((pts - mu.unsqueeze(1)) ** 2).sum(-1)).sum(-1) / wtot[sl].clamp_min(1e-12)
            t = ((pts - cam[sl].unsqueeze(1)) * ray[sl].unsqueeze(1)).sum(-1)
            t_lo[sl] = torch.where(okm, t, 1e9).amin(-1)
            t_hi[sl] = torch.where(okm, t, -1e9).amax(-1)
            for k, P in list(P_all.items()) + [("centroid", centroid)]:
                d = (pts - P[sl].unsqueeze(1)).norm(dim=-1) + big
                m, j = d.min(-1)
                dmin[k][sl] = m
                if k != "centroid":
                    vec_min[k][sl] = pts[torch.arange(len(j), device=device), j] - P[sl]
                    wsum[k][sl] = (wpix * (pts - P[sl].unsqueeze(1)).norm(dim=-1)).sum(-1)
        has = wtot > 0
        acc["centroid_to_region_nearest"].append((dmin["centroid"][has] / S).cpu())
        for k, P in P_all.items():
            acc[f"{k}_to_region_nearest"].append((dmin[k][has] / S).cpu())
            acc[f"{k}_to_region_wmean"].append((wsum[k][has] / wtot[has] / S).cpu())
            acc[f"{k}_to_centroid"].append(((P - centroid).norm(dim=-1)[has] / S).cpu())
            acc_along[k].append(((vec_min[k] * ray).sum(-1).abs()[has] / S).cpu())
        region_rms.append((r_sq[has].sqrt() / S).cpu())
        multi_surface.append(((t_hi - t_lo)[has] / S > 0.20).cpu())

        if sid == DEMO_SCENE:
            export_demo(patch_pts, patch_ok, top_i, top_w, has, centroid, P_all["set_transformer"], am, device)
        torch.cuda.empty_cache()
        if (idx + 1) % 10 == 0:
            print(f"[region] {idx + 1}/{len(prov)} windows, {time.time() - t0:.0f}s", flush=True)

    res = {k: stats(torch.cat(v)) for k, v in acc.items()}
    for k, v in acc_along.items():
        res[f"{k}_to_region_nearest_along_ray"] = stats(torch.cat(v))
    ms = torch.cat(multi_surface)
    rr = torch.cat(region_rms)
    res["region_rms_extent"] = stats(rr)
    res["n_patches_per_latent_median"] = float(torch.cat(n_patches).float().median())
    res["frac_multi_surface_latents"] = float(ms.float().mean())
    old = torch.cat(acc["set_transformer_to_centroid"])
    new = torch.cat(acc["set_transformer_to_region_nearest"])
    cfl = torch.cat(acc["centroid_to_region_nearest"])
    for name, m in (("single_surface", ~ms), ("multi_surface", ms)):
        res[f"set_transformer_{name}"] = {"to_centroid": stats(old[m]), "to_region_nearest": stats(new[m]),
                                          "centroid_floating": stats(cfl[m])}
    tail = old > torch.quantile(old, 0.75)
    res["old_tail_p75_cm"] = float(torch.quantile(old, 0.75) * 100)
    res["old_tail_frac_multi_surface"] = float(ms[tail].float().mean())
    res["old_tail_new_metric"] = stats(new[tail])
    out = ROOT / "docs" / "DIAG_scene_latent_region.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    fmt = lambda s: f"{s['median_cm']:5.1f}/{s['p90_cm']:5.1f}"
    print("metric (median/p90 cm):")
    for k in keys:
        print(f"  {k:40s} {fmt(res[k])}")
    for k in preds:
        print(f"  {k + '_to_region_nearest_along_ray':40s} {fmt(res[k + '_to_region_nearest_along_ray'])}")
    print(f"region extent RMS {fmt(res['region_rms_extent'])}; patches/latent {res['n_patches_per_latent_median']:.0f}; "
          f"multi-surface latents {res['frac_multi_surface_latents']:.3f}")
    for name in ("single_surface", "multi_surface"):
        r = res[f"set_transformer_{name}"]
        print(f"  {name:15s} to_centroid {fmt(r['to_centroid'])}  to_region {fmt(r['to_region_nearest'])}  centroid floating {fmt(r['centroid_floating'])}")
    print(f"old tail (> {res['old_tail_p75_cm']:.1f} cm): multi-surface frac {res['old_tail_frac_multi_surface']:.3f}; "
          f"new metric {fmt(res['old_tail_new_metric'])}")
    print("wrote", out)


def export_demo(patch_pts, patch_ok, top_i, top_w, has, centroid, pred, am, device):
    """Laser points of 20 latents' attention regions (palette colours) + their centroid (white) and prediction;
    plus an ownership map: every laser pixel coloured by the latent with the most attention on its patch."""
    g = torch.Generator().manual_seed(0)
    cand = torch.nonzero(has).squeeze(1).cpu()
    sel = cand[torch.randperm(len(cand), generator=g)[:20]]
    xyz, rgb, err = [], [], []
    for r, li in enumerate(sel.tolist()):
        col = hex_rgb(PALETTE[r][1])
        m = top_w[li] > 0
        pts = patch_pts[top_i[li][m]].reshape(-1, 3)
        ok = patch_ok[top_i[li][m]].reshape(-1)
        pts = pts[ok][::2].cpu().numpy()
        xyz.append(pts)
        rgb.append(np.tile(col, (len(pts), 1)))
        err.append(np.full(len(pts), r, dtype=np.float32))
    write_points(DEMO_DIR / "baseline_pretrained_6view_scene_latent_region20_laser.ply", np.concatenate(xyz),
                 np.concatenate(rgb), np.concatenate(err), radius=0.0012)
    cent = centroid[sel.to(device)].cpu().numpy()
    pr = pred[sel.to(device)].cpu().numpy()
    write_points(DEMO_DIR / "baseline_pretrained_6view_scene_latent_region20_centroid.ply", cent,
                 np.ones((20, 3)), np.arange(20, dtype=np.float32), radius=0.006)
    write_points(DEMO_DIR / "baseline_pretrained_6view_scene_latent_region20_pred.ply", pr,
                 np.stack([hex_rgb(PALETTE[r][1]) for r in range(20)]), np.arange(20, dtype=np.float32), radius=0.008)
    owner = am.argmax(0)  # [V*P] latent (valid-subset index) with most attention per patch
    rng = np.random.default_rng(0)
    cols = rng.uniform(0.15, 1.0, size=(am.shape[0], 3)).astype(np.float32)
    pts = patch_pts.reshape(-1, 3)
    ok = patch_ok.reshape(-1)
    own = owner.unsqueeze(1).expand(-1, patch_pts.shape[1]).reshape(-1)
    pts, own = pts[ok][::4].cpu().numpy(), own[ok][::4].cpu().numpy()
    write_points(DEMO_DIR / "baseline_pretrained_6view_scene_latent_ownership_laser.ply", pts, cols[own],
                 own.astype(np.float32), radius=0.0012)


def cmd_surface(args):
    """Laser-surface-only metrics (no attention centroid as reference).

    For every valid latent and every predictor (probes; the old centroid only as a GT-assisted reference):
      surface_dist   unsigned distance from the predicted point to the GT mesh surface (anywhere in the scene)
      depth_err      along the ray dominant-camera -> predicted point: |range(pred) - range(first mesh hit)|,
                     signed version also kept (negative = in front of the surface, i.e. floating in free space)
      lateral        distance from that mesh hit point to the nearest laser point of the latent's OWN attention
                     region (is the predicted direction inside the region the latent looks at?)
      region_nearest distance from the predicted point to the nearest laser point of its attention region
    """
    import open3d as o3d

    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    _tr, te = load_cache()
    te = {w["scene_id"]: w for w in te}
    preds = torch.load(args.pred_file or PRED_FILE, map_location="cpu", weights_only=False)
    opt = make_opt("val")
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov.dataset._offset_frac = None
    model = model_registry[opt.model_type](opt).eval().requires_grad_(False)
    model.load_state_dict(load_file(str(CHECKPOINT), device="cpu"), strict=True)
    model = model.to(device=device, dtype=torch.float32)
    print(f"GPU {gpu['index']}: laser-surface metrics on {len(prov)} validation windows", flush=True)
    names = list(preds) + ["centroid_reference"]
    acc = {n: {k: [] for k in ("surface_dist", "depth_err", "depth_signed", "lateral", "region_nearest", "miss")} for n in names}
    nviews_all, dist_all = [], []
    t0 = time.time()
    for idx in range(len(prov)):
        ds = prov.dataset
        sid = Path(ds.sample_list[idx]).name
        w = te[sid]
        context, _ = ds.get_context_target_frames(idx)
        raw = prov[idx]
        data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
        model_input, _ = split_data(data, opt)
        with torch.inference_mode():
            _l, attn = capture_cross_attention(model, model_input, [args.layer])
        a = attn[args.layer].to(device)
        cam_in, intr_in = data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu()
        raw_c2w = torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][context].copy())
        rc = build_raycasting_scene(ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
        pix, ok, _c = full_res_points(rc, cam_in, intr_in, opt.img_size)
        V, H, Wd, _ = pix.shape
        p = opt.patch_size
        hp, wp = H // p, Wd // p
        ppts = pix.reshape(V, hp, p, wp, p, 3).permute(0, 1, 3, 2, 4, 5).reshape(V * hp * wp, p * p, 3).to(device)
        pok = ok.reshape(V, hp, p, wp, p).permute(0, 1, 3, 2, 4).reshape(V * hp * wp, p * p).to(device)
        eff = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
        local = [h for h in range(a.shape[0]) if float(eff[h].median()) < 30]
        lat_idx = torch.nonzero(w["valid"]).squeeze(1).to(device)
        am = a[local].mean(0)[lat_idx]
        tw, ti = am.topk(args.max_patches, -1)
        cum = tw.cumsum(-1) / am.sum(-1, keepdim=True)
        keep = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool), cum[:, :-1] < args.mass], 1)
        tw = tw * keep
        cam = w["cam"][w["valid"]]
        nviews_all.append(w["n_views"][w["valid"]].float())
        P_all = {k: v[sid] for k, v in preds.items()}
        P_all["centroid_reference"] = w["target"][w["valid"]]
        for n, P in P_all.items():
            P = P.float()
            sd = rc.compute_distance(o3d.core.Tensor(P.numpy().astype(np.float32))).numpy()
            dvec = P - cam
            rng = dvec.norm(dim=-1)
            d = dvec / rng.clamp_min(1e-9).unsqueeze(-1)
            rays = o3d.core.Tensor(torch.cat([cam, d], 1).numpy().astype(np.float32))
            t_hit = torch.from_numpy(rc.cast_rays(rays)["t_hit"].numpy())
            hit_ok = torch.isfinite(t_hit)
            hitp = cam + t_hit.clamp(max=1e3).unsqueeze(-1) * d
            lat = torch.full((len(P),), float("nan"))
            rn = torch.full((len(P),), float("nan"))
            Pd, Hd = P.to(device), hitp.to(device)
            for c in range(0, len(P), args.chunk):
                sl = slice(c, c + args.chunk)
                pts = ppts[ti[sl]].reshape(len(ti[sl]), -1, 3)
                big = torch.where((pok[ti[sl]] & (tw[sl] > 0).unsqueeze(-1)).reshape(len(ti[sl]), -1), 0.0, 1e9)
                rn[c:c + args.chunk] = ((pts - Pd[sl].unsqueeze(1)).norm(dim=-1) + big).amin(-1).cpu()
                lat[c:c + args.chunk] = ((pts - Hd[sl].unsqueeze(1)).norm(dim=-1) + big).amin(-1).cpu()
            has_region = rn < 1e8
            A = acc[n]
            A["surface_dist"].append(torch.from_numpy(sd) / S)
            A["depth_err"].append(torch.where(hit_ok, (rng - t_hit).abs(), torch.nan) / S)
            A["depth_signed"].append(torch.where(hit_ok, rng - t_hit, torch.nan) / S)
            A["lateral"].append(torch.where(hit_ok & has_region, lat, torch.nan) / S)
            A["region_nearest"].append(torch.where(has_region, rn, torch.nan) / S)
            A["miss"].append((~hit_ok).float())
        dist_all.append((w["target"][w["valid"]] - cam).norm(dim=-1) / S)
        torch.cuda.empty_cache()
        if (idx + 1) % 10 == 0:
            print(f"[surface] {idx + 1}/{len(prov)} windows, {time.time() - t0:.0f}s", flush=True)

    nviews, dist = torch.cat(nviews_all), torch.cat(dist_all)
    res = {"definition": cmd_surface.__doc__, "n_latents": int(len(nviews))}
    for n in names:
        A = {k: torch.cat(v) for k, v in acc[n].items()}
        r = {}
        for k in ("surface_dist", "depth_err", "lateral", "region_nearest"):
            x = A[k][torch.isfinite(A[k])]
            r[k] = stats(x)
        s = A["depth_signed"][torch.isfinite(A["depth_signed"])]
        r["frac_in_front_of_surface_gt_5cm"] = float((s < -0.05).float().mean())
        r["frac_behind_surface_gt_5cm"] = float((s > 0.05).float().mean())
        r["ray_miss_frac"] = float(A["miss"].mean())
        de = A["depth_err"]
        r["depth_err_by_n_views"] = {}
        for lo, hi, nm in ((0, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 3.5, "3"), (3.5, 99, "4+")):
            m = (nviews >= lo) & (nviews < hi) & torch.isfinite(de)
            r["depth_err_by_n_views"][nm] = {"n_frac": float(m.float().mean()), **stats(de[m])}
        r["depth_err_by_distance_m"] = {}
        for lo, hi, nm in ((0, 1.5, "<1.5"), (1.5, 2.5, "1.5-2.5"), (2.5, 3.5, "2.5-3.5"), (3.5, 99, ">3.5")):
            m = (dist >= lo) & (dist < hi) & torch.isfinite(de)
            r["depth_err_by_distance_m"][nm] = {"n_frac": float(m.float().mean()), **stats(de[m])}
        res[n] = r
    out = ROOT / "docs" / f"DIAG_scene_latent_surface_metrics{args.out_suffix}.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    fmt = lambda s: f"{s['median_cm']:5.1f}/{s['p90_cm']:5.1f}"
    for n in names:
        r = res[n]
        print(f"{n:20s} surface {fmt(r['surface_dist'])}  depth {fmt(r['depth_err'])}  lateral {fmt(r['lateral'])}  "
              f"region {fmt(r['region_nearest'])}  front>5cm {r['frac_in_front_of_surface_gt_5cm']:.3f} "
              f"behind>5cm {r['frac_behind_surface_gt_5cm']:.3f} miss {r['ray_miss_frac']:.3f}")
        print("   depth by views: " + "  ".join(f"{k}:{v['n_frac']:.2f} {v['median_cm']:.1f}/{v['p90_cm']:.1f}" for k, v in r["depth_err_by_n_views"].items()))
        print("   depth by dist : " + "  ".join(f"{k}:{v['n_frac']:.2f} {v['median_cm']:.1f}/{v['p90_cm']:.1f}" for k, v in r["depth_err_by_distance_m"].items()))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["fitpred", "region", "surface"])
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--set-epochs", type=int, default=12)
    ap.add_argument("--max-patches", type=int, default=64)
    ap.add_argument("--mass", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--pred-file", type=Path, default=None)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    {"fitpred": cmd_fitpred, "region": cmd_region, "surface": cmd_surface}[args.mode](args)


if __name__ == "__main__":
    main()
