"""Two no-training hypothesis checks on existing caches.

thinning_cost  "fog hedges depth uncertainty"?  Where does sft_readout_rgb_depth_thick (thin) lose image quality
    relative to sft_readout_rgb_depth (depth-only, foggy)?
    Official interpolation targets (eval_by_visibility cache, interp), GT-mesh pixels.
    Pixels binned by (a) the depth-only model's own depth error at the pixel (proxy of depth uncertainty; uncovered = own bin),
    (b) number of input views that see the surface point, (c) proximity to a GT depth discontinuity.
    Per bin: pixel share, pooled PSNR of both models, delta, and the bin's share of the total SSE increase.
    The fog-hedging hypothesis predicts the loss concentrates where the depth-only model's depth is wrong / few views / near depth edges.

parallax  "TokenGS geometry is multi-view-matching-like (triangulation limit)": per GS token (official model, 50 validation windows, readout-capacity probe cache),
    parallax = largest angle between the rays from the input cameras that attend to the token (attention mass > 0.1
    per view) to the token's laser region centroid; also an inference-time version using the token's own Gaussian
    centre instead of the laser centroid.  Errors: (E1) official output, per-token median child -> own laser region
    distance; (E2) information ceiling, set-Transformer probe distance-from-camera error (probe_gs_token_limit).
    Spearman correlations with #views, parallax, distance and distance/parallax (triangulation error ~ z / angle),
    overall and within fixed #views; binned medians.

  PYTHONPATH=scripts:. python scripts/diag_thinning_and_parallax.py thinning_cost
  PYTHONPATH=scripts:. python scripts/diag_thinning_and_parallax.py parallax
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BD = Path("/mnt/disk/xveyan/workspace_scannetpp/eval_by_visibility")
S = 0.3


# ------------------------------------------------------------------------------------------------ thinning cost
def edge_mask(d, valid, rel=0.05, dilate=2):
    """GT depth discontinuity: relative jump > rel to a 4-neighbour, dilated by `dilate` px."""
    m = np.zeros_like(valid)
    for ax in (1, 2):
        a, b = np.take(d, range(d.shape[ax] - 1), ax), np.take(d, range(1, d.shape[ax]), ax)
        va, vb = np.take(valid, range(d.shape[ax] - 1), ax), np.take(valid, range(1, d.shape[ax]), ax)
        j = (np.abs(a - b) / np.maximum(np.minimum(a, b), 1e-6) > rel) | (va != vb)
        pad = [(0, 0)] * 3
        pad[ax] = (0, 1)
        m |= np.pad(j, pad)
        pad[ax] = (1, 0)
        m |= np.pad(j, pad)
    t = torch.from_numpy(m).float().unsqueeze(1)
    t = torch.nn.functional.max_pool2d(t, 2 * dilate + 1, 1, dilate)
    return t.squeeze(1).numpy() > 0


def cmd_thinning_cost(_a):
    mode, a_exp, b_exp = "interp", "sft_readout_rgb_depth_thick", "sft_readout_rgb_depth"   # A = thin, B = depth-only (foggy)
    groups = {"depth-only model depth error": {}, "input views": {}, "depth edge": {}}
    tot = [0.0, 0.0, 0.0]
    err_bins = (("<2%", 0, 0.02), ("2-5%", 0.02, 0.05), ("5-10%", 0.05, 0.10), ("10-20%", 0.10, 0.20), (">20%", 0.20, 1e9))
    view_bins = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4+", 4, 6))
    for f in sorted((BD / mode / "_gt").glob("*.npz")):
        g = np.load(f)
        pa, pb = np.load(BD / mode / a_exp / f.name), np.load(BD / mode / b_exp / f.name)
        v = g["valid"]
        sa, sb = pa["sq"].astype(np.float64), pb["sq"].astype(np.float64)
        db, ab = pb["depth"], pb["alpha"].astype(np.float32)
        cov_b = v & (ab > 0.5) & (db > 0)
        rel = np.where(cov_b, np.abs(db - g["depth"]) / np.maximum(g["depth"], 1e-6), np.inf)
        edge = edge_mask(g["depth"], v)
        masks = {"depth-only model depth error": {n: v & (rel >= lo) & (rel < hi) for n, lo, hi in err_bins} | {"depth-only model uncovered": v & ~cov_b},
                 "input views": {n: v & (g["n_vis"] >= lo) & (g["n_vis"] <= hi) for n, lo, hi in view_bins},
                 "depth edge": {"near edge (<=2px)": v & edge, "away from edges": v & ~edge}}
        tot[0] += float(v.sum())
        tot[1] += float(sa[v].sum())
        tot[2] += float(sb[v].sum())
        for gname, ms in masks.items():
            for n, m in ms.items():
                c = groups[gname].setdefault(n, [0.0, 0.0, 0.0])
                c[0] += float(m.sum())
                c[1] += float(sa[m].sum())
                c[2] += float(sb[m].sum())
    psnr = lambda s, n: -10 * math.log10(max(s / max(n, 1), 1e-12))  # noqa: E731
    d_tot = tot[1] - tot[2]
    out = {"setup": {"mode": mode, "A": a_exp, "B": b_exp, "pixels": "GT-mesh pixels of the 400 official targets",
                     "delta_psnr_pooled_A_minus_B": psnr(tot[1], tot[0]) - psnr(tot[2], tot[0])}}
    for gname, cells in groups.items():
        out[gname] = {n: {"pixel_share": c[0] / tot[0], "psnr_thick": psnr(c[1], c[0]), "psnr_depth_only": psnr(c[2], c[0]),
                          "delta_psnr": psnr(c[1], c[0]) - psnr(c[2], c[0]), "share_of_sse_increase": (c[1] - c[2]) / d_tot}
                      for n, c in cells.items()}
    (ROOT / "docs" / "DIAG_thinning_image_cost.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"pooled PSNR thick - depth-only on GT pixels: {out['setup']['delta_psnr_pooled_A_minus_B']:+.3f} dB")
    for gname in groups:
        print(f"\n## {gname}\n| bin | pixels | PSNR depth-only | PSNR thick | delta | share of SSE increase |\n|---|---:|---:|---:|---:|---:|")
        for n, r in out[gname].items():
            print(f"| {n} | {r['pixel_share']:.1%} | {r['psnr_depth_only']:.2f} | {r['psnr_thick']:.2f} | {r['delta_psnr']:+.2f} | {r['share_of_sse_increase']:+.1%} |")


# ------------------------------------------------------------------------------------------------ parallax
def spearman(x, y):
    m = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[m].double(), y[m].double()
    if x.numel() < 10:
        return float("nan")
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


def parallax_deg(X, cams, use):
    """Max pairwise angle between rays cam_v -> X over views with use[:, v]; 0 if < 2 views."""
    r = torch.nn.functional.normalize(X.unsqueeze(1) - cams.unsqueeze(0), dim=-1)          # [N, V, 3]
    cos = (r.unsqueeze(2) * r.unsqueeze(1)).sum(-1).clamp(-1, 1)                            # [N, V, V]
    pair = use.unsqueeze(2) & use.unsqueeze(1)
    ang = torch.rad2deg(torch.acos(cos)).masked_fill(~pair, 0.0)
    return ang.flatten(1).max(-1).values


def cmd_parallax(_a):
    import probe_readout_capacity as RC
    from probe_scene_latent_limit import make_opt
    from tokengs.data.provider import Provider

    _tr, v1 = RC.load_cache()
    v1 = {r["scene_id"]: r for r in v1}
    pc = Path("/mnt/disk/xveyan/workspace_scannetpp/probe_gs_token_limit")
    preds = torch.load(pc / "val_preds.pt", weights_only=False)
    pval = {}
    for p in sorted(pc.glob("shard_*.pt")):
        for r in torch.load(p, map_location="cpu", weights_only=False):
            if r["split"] == "val":
                pval[r["scene_id"]] = {k: r[k] for k in ("target", "valid", "cam")}
    prov = Provider("scannetpp_val_scaled_0.3", make_opt("val"), training=False)
    prov.dataset._offset_frac = None
    rows = {k: [] for k in ("nv", "par", "par_pred", "dist", "e1", "e1_rec5", "e2")}
    for i in range(len(prov)):
        sid = Path(prov.dataset.sample_list[i]).name
        r, q = v1[sid], pval[sid]
        cams = torch.stack([torch.linalg.inv(c.float().T)[:3, 3] for c in prov[i]["cam_view_input"]])
        V = cams.shape[0]
        P = r["pp"].shape[0] // V
        ti, tw = r["top_i"].long(), r["top_w"].float()
        vm = torch.zeros((ti.shape[0], V)).scatter_add_(1, ti // P, tw)
        use = vm > 0.1
        pts, ok, w = RC.region(r, "cpu")
        X = (w.unsqueeze(-1) * pts).sum(1) / w.sum(-1, keepdim=True).clamp_min(1e-12)
        op = r["opacity"].float()
        wc = op * (op >= 1 / 255)
        Xp = (r["pos0"] * wc.unsqueeze(-1)).sum(1) / wc.sum(-1, keepdim=True).clamp_min(1e-8)
        dom = cams[vm.argmax(-1)]
        c2r, r2c = RC.chamfer_terms(r["pos0"], pts, ok, w)
        rec5 = ((r2c < 0.05).float() * w).sum(-1) / w.sum(-1).clamp_min(1e-12)
        pv = q["valid"]
        e2 = torch.full((ti.shape[0],), float("nan"))
        pr = preds[sid].float()
        e2[pv] = ((pr - q["cam"][pv]).norm(dim=-1) - (q["target"][pv] - q["cam"][pv]).norm(dim=-1)).abs() / S
        v = r["valid"]
        rows["nv"].append(use.sum(-1)[v].float())
        rows["par"].append(parallax_deg(X, cams, use)[v])
        rows["par_pred"].append(parallax_deg(Xp, cams, use)[v])
        rows["dist"].append(((X - dom).norm(dim=-1) / S)[v])
        rows["e1"].append((c2r.median(-1).values * 100)[v])
        rows["e1_rec5"].append(rec5[v])
        rows["e2"].append((e2 * 100)[v])
    t = {k: torch.cat(x) for k, x in rows.items()}
    zpar = t["dist"] / torch.deg2rad(t["par"]).clamp_min(1e-3)
    preds_ = {"#views": t["nv"], "parallax_deg (higher=better)": -t["par"], "parallax_deg_inference_time": -t["par_pred"],
              "distance_m": t["dist"], "distance/parallax": zpar}
    res = {"tokens": int(t["nv"].numel()), "definitions": __doc__.split("parallax  ")[1].strip(),
           "spearman": {e: {p: spearman(x, t[e]) for p, x in preds_.items()} for e in ("e1", "e2")},
           "spearman_recall5_vs_parallax": spearman(t["par"], t["e1_rec5"]),
           "spearman_within_views": {}, "binned": {}}
    res["spearman"]["e1"]["#views"] = spearman(-t["nv"], t["e1"])
    res["spearman"]["e2"]["#views"] = spearman(-t["nv"], t["e2"])
    for nv in (2, 3, 4):
        m = (t["nv"] == nv) if nv < 4 else (t["nv"] >= 4)
        res["spearman_within_views"][f"{nv}{'+' if nv == 4 else ''} views"] = {
            "tokens": int(m.sum()), "e1_vs_-parallax": spearman(-t["par"][m], t["e1"][m]),
            "e2_vs_-parallax": spearman(-t["par"][m], t["e2"][m]),
            "e1_vs_distance/parallax": spearman(zpar[m], t["e1"][m]), "e2_vs_distance/parallax": spearman(zpar[m], t["e2"][m])}
    edges = ((0, 1e-6, "single view (0 deg)"), (1e-6, 2, "0-2 deg"), (2, 5, "2-5 deg"), (5, 10, "5-10 deg"),
             (10, 20, "10-20 deg"), (20, 180, ">20 deg"))
    for lo, hi, n in edges:
        m = (t["par"] >= lo) & (t["par"] < hi) if lo > 0 else (t["par"] < hi)
        e2m = t["e2"][m & torch.isfinite(t["e2"])]
        res["binned"][n] = {"token_share": float(m.float().mean()),
                            "e1_median_cm": float(t["e1"][m].median()) if m.any() else None,
                            "e1_recall5_mean": float(t["e1_rec5"][m].mean()) if m.any() else None,
                            "e2_median_cm": float(e2m.median()) if e2m.numel() else None,
                            "e2_p90_cm": float(e2m.quantile(0.9)) if e2m.numel() else None}
    (ROOT / "docs" / "DIAG_parallax_vs_geometry_error.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("tokens", "spearman", "spearman_recall5_vs_parallax",
                                          "spearman_within_views", "binned")}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["thinning_cost", "parallax"])
    a = ap.parse_args()
    cmd_thinning_cost(a) if a.cmd == "thinning_cost" else cmd_parallax(a)


if __name__ == "__main__":
    main()
