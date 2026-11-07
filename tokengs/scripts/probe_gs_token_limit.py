"""GS-token geometry test, matched 1:1 to the scene-latent test (probe_scene_latent_limit + lasertarget + surface).

Same data (853 training scenes x 3 windows, all valid units; all 50 validation windows), same best probe
(set Transformer over all 4096 units of a window), same laser target (first GT-mesh hit along the unit's
attention ray from its dominant input camera), same laser-surface metrics and the same demo exports.
Only the unit changes: GS token feature h (decoder output = input of deconv) instead of the scene latent.
GS-token attention over input patches = decoder cross-attention (head mean) x layer-7 local-head latent attention.

  extract --shard k --num-shards N ; fit ; eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import probe_scene_latent_limit as L  # noqa: F401  (window offsets patch)
from diagnose_latent_attention_geometry import capture_cross_attention
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, idle_gpu
from highlight_selected_tokens_ply import PALETTE, hex_rgb
from probe_gs_token_geometry import child_centre, decode, load_model
from probe_scene_latent_limit import S, SPLITS, TRAIN_OFFSETS, make_opt, patch_geometry, train_set_probe
from probe_scene_latent_position import err_colour, stats, write_points
from probe_scene_latent_region import full_res_points
from scannetpp_evaluation_protocol import build_raycasting_scene, sha256_file
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data

CACHE = Path("/mnt/disk/xveyan/workspace_scannetpp/probe_gs_token_limit")
DEMO_SCENE = "7b6477cb95"
DEMO_DIR = ROOT / "EXP_ScanNetPP" / "demo" / "scene_7b6477cb95" / "baseline_pretrained_6view"


def cmd_extract(args):
    import open3d as o3d

    gpu = idle_gpu()
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected checkpoint SHA256")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    prov = {sp: Provider("scannetpp_val_scaled_0.3", make_opt(sp), training=False) for sp in SPLITS}
    tasks = [("val", None, i) for i in range(len(prov["val"]))]
    tasks += [("train", f, i) for f in TRAIN_OFFSETS for i in range(len(prov["train"]))]
    mine = tasks[args.shard::args.num_shards]
    opt = make_opt("val")
    model = load_model(CHECKPOINT, opt, device)
    print(f"GPU {gpu['index']}: shard {args.shard}: {len(mine)} windows", flush=True)
    out, t0 = [], time.time()
    for n, (sp, frac, idx) in enumerate(mine):
        ds = prov[sp].dataset
        ds._offset_frac = frac
        sid = Path(ds.sample_list[idx]).name
        context, _ = ds.get_context_target_frames(idx)
        raw = prov[sp][idx]
        data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
        mi, _ = split_data(data, make_opt(sp))
        with torch.inference_mode():
            latents, attn = capture_cross_attention(model, mi, [7])
        a = attn[7]
        eff = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
        a_lat = a[[h for h in range(a.shape[0]) if float(eff[h].median()) < 30]].mean(0)
        A_dec, h, g = decode(model, latents, mi)
        T = A_dec @ a_lat
        cam_in, intr_in = data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu()
        raw_c2w = torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][context].copy())
        rc = build_raycasting_scene(ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
        _p, _n, val, dirs, cams = patch_geometry(rc, cam_in, intr_in, opt.img_size, opt.patch_size)
        V = cams.shape[0]
        Tc = T.cpu()
        P = Tc.shape[1] // V
        vm = Tc.reshape(-1, V, P).sum(-1)
        dom = vm.argmax(-1)
        wdom = Tc.reshape(-1, V, P)[torch.arange(len(dom)), dom]
        ray = torch.nn.functional.normalize((wdom.unsqueeze(-1) * dirs.reshape(V, P, 3)[dom]).sum(1), dim=-1)
        cam = cams[dom]
        th = torch.from_numpy(rc.cast_rays(o3d.core.Tensor(torch.cat([cam, ray], 1).numpy().astype(np.float32)))["t_hit"].numpy())
        valid = ((Tc * val.float()).sum(-1) > 0.5) & torch.isfinite(th)
        target = cam + torch.where(torch.isfinite(th), th, torch.zeros_like(th)).unsqueeze(-1) * ray
        rec = {"split": sp, "scene_id": sid, "offset": -1.0 if frac is None else float(frac), "latent": h.half().cpu(),
               "target": target, "valid": valid, "cam": cam}
        if sp == "val":
            tw, ti = T.topk(64, -1)
            cum = tw.cumsum(-1) / T.sum(-1, keepdim=True)
            keep = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool), cum[:, :-1] < 0.9], 1)
            pix, ok, _c = full_res_points(rc, cam_in, intr_in, opt.img_size)
            rec.update({"n_views": (vm > 0.1).sum(-1).to(torch.int8), "top_i": ti.int().cpu(), "top_w": (tw * keep).half().cpu(),
                        "pix": pix.half(), "pix_ok": ok, "context": context, "out_base": child_centre(g).cpu()})
        out.append(rec)
        torch.cuda.empty_cache()
        if (n + 1) % 20 == 0 or n + 1 == len(mine):
            print(f"[shard {args.shard}] {n + 1}/{len(mine)} {time.time() - t0:.0f}s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"shard_{args.shard}.tmp"
    torch.save(out, tmp)
    tmp.rename(CACHE / f"shard_{args.shard}.pt")
    print("wrote", CACHE / f"shard_{args.shard}.pt")


def load_all():
    recs = []
    for p in sorted(CACHE.glob("shard_*.pt")):
        recs.extend(torch.load(p, map_location="cpu", weights_only=False))
    return [r for r in recs if r["split"] == "train"], [r for r in recs if r["split"] == "val"]


def cmd_fit(args):
    idle_gpu()
    device = torch.device("cuda:0")
    tr, te = load_all()
    print(f"train windows {len(tr)}, tokens {sum(int(r['valid'].sum()) for r in tr)}", flush=True)
    f = train_set_probe(tr, device, args.set_epochs)
    torch.save({r["scene_id"]: f(r)[r["valid"]] for r in te}, CACHE / "val_preds.pt")
    print("wrote", CACHE / "val_preds.pt")


def cmd_eval(args):
    import open3d as o3d

    from tokengs.data.registry import dataset_registry
    from tokengs.data.static.scannetpp import ScanNetPPVal

    _tr, te = load_all()
    preds = torch.load(CACHE / "val_preds.pt", weights_only=False)
    ds = ScanNetPPVal(**{**dataset_registry["scannetpp_val"]["kwargs"], "window_size": 18})
    names = ["gs_token_best_probe", "deconv_output_base"]
    keys = ("surface_dist", "depth_err", "depth_signed", "lateral", "region_nearest", "miss")
    acc = {n: {k: [] for k in keys} for n in names}
    nv_all, dist_all = [], []
    for r in te:
        sid = r["scene_id"]
        rc = build_raycasting_scene(ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply",
                                    torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][r["context"]].copy()), S)
        pix, ok = r["pix"].float(), r["pix_ok"]
        V, H, W, _ = pix.shape
        p = 8
        ppts = pix.reshape(V, H // p, p, W // p, p, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, p * p, 3)
        pok = ok.reshape(V, H // p, p, W // p, p).permute(0, 1, 3, 2, 4).reshape(-1, p * p)
        v = r["valid"]
        cam, tgt = r["cam"][v], r["target"][v]
        ti, tw = r["top_i"][v].long(), r["top_w"][v].float()
        nv_all.append(r["n_views"][v].float())
        dist_all.append((tgt - cam).norm(dim=-1) / S)
        P_all = {"gs_token_best_probe": preds[sid].float(), "deconv_output_base": r["out_base"][v].float()}
        for n, Pp in P_all.items():
            sd = torch.from_numpy(rc.compute_distance(o3d.core.Tensor(Pp.numpy().astype(np.float32))).numpy())
            d = Pp - cam
            rng = d.norm(dim=-1)
            dn = d / rng.clamp_min(1e-9).unsqueeze(-1)
            th = torch.from_numpy(rc.cast_rays(o3d.core.Tensor(torch.cat([cam, dn], 1).numpy().astype(np.float32)))["t_hit"].numpy())
            okh = torch.isfinite(th)
            hp = cam + th.clamp(max=1e3).unsqueeze(-1) * dn
            lat, rn = torch.empty(len(Pp)), torch.empty(len(Pp))
            for c in range(0, len(Pp), 512):
                sl = slice(c, c + 512)
                pts = ppts[ti[sl]].reshape(len(ti[sl]), -1, 3)
                big = torch.where((pok[ti[sl]] & (tw[sl] > 0).unsqueeze(-1)).reshape(len(ti[sl]), -1), 0.0, 1e9)
                lat[sl] = ((pts - hp[sl].unsqueeze(1)).norm(dim=-1) + big).amin(-1)
                rn[sl] = ((pts - Pp[sl].unsqueeze(1)).norm(dim=-1) + big).amin(-1)
            A = acc[n]
            A["surface_dist"].append(sd / S)
            A["depth_err"].append(torch.where(okh, (rng - th).abs(), torch.nan) / S)
            A["depth_signed"].append(torch.where(okh, rng - th, torch.nan) / S)
            A["lateral"].append(torch.where(okh & (lat < 1e8), lat, torch.nan) / S)
            A["region_nearest"].append(torch.where(rn < 1e8, rn, torch.nan) / S)
            A["miss"].append((~okh).float())
        if sid == DEMO_SCENE:
            export_demo(r, preds[sid].float(), ppts, pok)
    nv, dist = torch.cat(nv_all), torch.cat(dist_all)
    res = {"n_tokens": int(len(nv))}
    fmt = lambda s: f"{s['median_cm']:5.1f}/{s['p90_cm']:5.1f}"
    for n in names:
        A = {k: torch.cat(x) for k, x in acc[n].items()}
        fin = lambda x: x[torch.isfinite(x)]
        rr = {k: stats(fin(A[k])) for k in ("surface_dist", "depth_err", "lateral", "region_nearest")}
        sg = fin(A["depth_signed"])
        rr["frac_in_front_of_surface_gt_5cm"] = float((sg < -0.05).float().mean())
        rr["frac_behind_surface_gt_5cm"] = float((sg > 0.05).float().mean())
        rr["ray_miss_frac"] = float(A["miss"].mean())
        de = A["depth_err"]
        rr["depth_err_by_n_views"], rr["depth_err_by_distance_m"] = {}, {}
        for lo, hi, nm in ((0, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 3.5, "3"), (3.5, 99, "4+")):
            m = (nv >= lo) & (nv < hi) & torch.isfinite(de)
            rr["depth_err_by_n_views"][nm] = {"n_frac": float(m.float().mean()), **stats(de[m])}
        for lo, hi, nm in ((0, 1.5, "<1.5"), (1.5, 2.5, "1.5-2.5"), (2.5, 3.5, "2.5-3.5"), (3.5, 99, ">3.5")):
            m = (dist >= lo) & (dist < hi) & torch.isfinite(de)
            rr["depth_err_by_distance_m"][nm] = {"n_frac": float(m.float().mean()), **stats(de[m])}
        res[n] = rr
        print(f"{n:22s} surface {fmt(rr['surface_dist'])}  depth {fmt(rr['depth_err'])}  lateral {fmt(rr['lateral'])}  "
              f"region {fmt(rr['region_nearest'])}  front {rr['frac_in_front_of_surface_gt_5cm']:.3f} "
              f"behind {rr['frac_behind_surface_gt_5cm']:.3f} miss {rr['ray_miss_frac']:.3f}")
        print("   by views: " + "  ".join(f"{k}:{x['n_frac']:.2f} {x['median_cm']:.1f}/{x['p90_cm']:.1f}" for k, x in rr["depth_err_by_n_views"].items()))
        print("   by dist : " + "  ".join(f"{k}:{x['n_frac']:.2f} {x['median_cm']:.1f}/{x['p90_cm']:.1f}" for k, x in rr["depth_err_by_distance_m"].items()))
    out = ROOT / "docs" / "DIAG_gs_token_surface_metrics.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


def export_demo(r, pred, ppts, pok):
    v = r["valid"]
    tgt, cam = r["target"][v], r["cam"][v]
    d = pred - cam
    e = (d.norm(dim=-1) - ((tgt - cam) * (d / d.norm(dim=-1, keepdim=True))).sum(-1)).abs() / S  # depth error proxy
    rgb = np.array([err_colour(float(x)) for x in e])
    write_points(DEMO_DIR / "baseline_pretrained_6view_gs_token_probe_pred.ply", pred.numpy(), rgb, (e * 100).numpy())
    write_points(DEMO_DIR / "baseline_pretrained_6view_gs_token_laser_target.ply", tgt.numpy(), np.ones((len(tgt), 3)),
                 np.zeros(len(tgt)), radius=0.003)
    ts = torch.linspace(0, 1, 7).view(1, -1, 1)
    seg = (pred.unsqueeze(1) * (1 - ts) + tgt.unsqueeze(1) * ts).reshape(-1, 3)
    write_points(DEMO_DIR / "baseline_pretrained_6view_gs_token_probe_links.ply", seg.numpy(), np.repeat(rgb, 7, 0),
                 np.repeat((e * 100).numpy(), 7), radius=0.0015)
    ti, tw = r["top_i"][v].long(), r["top_w"][v].float()
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(len(tgt), generator=g)[:20]
    xyz, col, lab = [], [], []
    for k, i in enumerate(sel.tolist()):
        m = tw[i] > 0
        pts = ppts[ti[i][m]].reshape(-1, 3)[pok[ti[i][m]].reshape(-1)][::2].numpy()
        xyz.append(pts)
        col.append(np.tile(hex_rgb(PALETTE[k][1]), (len(pts), 1)))
        lab.append(np.full(len(pts), k, dtype=np.float32))
    write_points(DEMO_DIR / "baseline_pretrained_6view_gs_token_region20_laser.ply", np.concatenate(xyz), np.concatenate(col),
                 np.concatenate(lab), radius=0.0012)
    write_points(DEMO_DIR / "baseline_pretrained_6view_gs_token_region20_target.ply", tgt[sel].numpy(), np.ones((20, 3)),
                 np.arange(20, dtype=np.float32), radius=0.006)
    write_points(DEMO_DIR / "baseline_pretrained_6view_gs_token_region20_pred.ply", pred[sel].numpy(),
                 np.stack([hex_rgb(PALETTE[k][1]) for k in range(20)]), np.arange(20, dtype=np.float32), radius=0.008)
    W = torch.zeros(len(tgt), ppts.shape[0]).scatter_(1, ti, tw)
    owner = W.argmax(0)
    has = W.sum(0) > 0
    cols = np.random.default_rng(0).uniform(0.15, 1.0, size=(len(tgt), 3)).astype(np.float32)
    pts = ppts[has].reshape(-1, 3)
    okm = pok[has].reshape(-1)
    own = owner[has].unsqueeze(1).expand(-1, ppts.shape[1]).reshape(-1)
    pts, own = pts[okm][::4].numpy(), own[okm][::4].numpy()
    write_points(DEMO_DIR / "baseline_pretrained_6view_gs_token_ownership_laser.ply", pts, cols[own], own.astype(np.float32),
                 radius=0.0012)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["extract", "fit", "eval"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--set-epochs", type=int, default=12)
    args = ap.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit, "eval": cmd_eval}[args.mode](args)


if __name__ == "__main__":
    main()
