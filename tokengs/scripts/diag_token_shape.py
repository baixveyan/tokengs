"""Is a token's group of 64 Gaussians a disc on the surface, or a blob?  (no training; 50 validation windows)

Per GS token (tokens with a valid laser region, see probe_readout_capacity.py), opacity-weighted covariance of its
64 child centres -> principal axes l1 >= l2 >= l3:
  flatness    sqrt(l3 / l2)            0 = disc, 1 = blob
  thickness   sqrt(l3)  (cm)           extent along the thinnest axis
  normal_cos  |cos(thinnest axis, laser surface normal)|, the normal = thinnest axis of the token's laser region
  ray/lateral spread along the dominant camera ray / mean spread across it
Reference: the same statistics for the token's laser region (attention-weighted laser points; the official model's
attention defines the region for every model).  Models: official, sft_readout_rgb_depth, sft_readout_rgb_depth_thick.

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/diag_token_shape.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import probe_readout_capacity as RC
from evaluate_scannetpp import ROOT, idle_gpu
from probe_gs_token_geometry import load_model
from probe_scene_latent_limit import S, make_opt
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data

MODELS = {"official": None,
          "sft_readout_rgb_depth": ROOT / "EXP_ScanNetPP/sft_readout_rgb_depth/model.safetensors",
          "sft_readout_rgb_depth_thick": ROOT / "EXP_ScanNetPP/sft_readout_rgb_depth_thick/model.safetensors"}
ALIVE = 1.0 / 255.0


def wcov(x, w):
    """x [N,M,3], w [N,M] -> mean [N,3], eigenvalues ascending [N,3], eigenvectors [N,3,3]."""
    ws = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    mu = (ws.unsqueeze(-1) * x).sum(1)
    d = x - mu.unsqueeze(1)
    c = torch.einsum("nm,nmi,nmj->nij", ws, d, d) + 1e-12 * torch.eye(3, device=x.device)
    ev, evec = torch.linalg.eigh(c)
    return mu, ev.clamp_min(0), evec, d, ws


def shape_stats(x, w, ray_from, n_ref=None):
    mu, ev, evec, d, ws = wcov(x, w)
    out = {"flatness": (ev[:, 0] / ev[:, 1].clamp_min(1e-18)).sqrt(), "thickness_cm": ev[:, 0].sqrt() / S * 100,
           "extent_cm": ev[:, 2].sqrt() / S * 100, "normal": evec[:, :, 0]}
    ray = torch.nn.functional.normalize(mu - ray_from, dim=-1)
    along = (ws * (d * ray.unsqueeze(1)).sum(-1).square()).sum(-1)
    total = (ws * d.square().sum(-1)).sum(-1)
    out["ray_over_lateral"] = (along / ((total - along) / 2).clamp_min(1e-18)).sqrt()
    if n_ref is not None:
        out["normal_cos"] = (evec[:, :, 0] * n_ref).sum(-1).abs()
    return out


def q(x, ps=(0.1, 0.5, 0.9)):
    x = x[torch.isfinite(x)].float()
    return {f"p{int(p * 100)}": round(float(x.quantile(p)), 4) for p in ps}


@torch.no_grad()
def main():
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    _tr, val = RC.load_cache()
    val = {r["scene_id"]: r for r in val}
    opt = make_opt("val")
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov.dataset._offset_frac = None
    nets = {k: load_model(p, opt, device) for k, p in MODELS.items() if p is not None}
    acc = {}
    for i in range(len(prov)):
        sid = Path(prov.dataset.sample_list[i]).name
        r = val[sid]
        raw = prov[i]
        data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
        mi, _ = split_data(data, opt)
        v = r["valid"].to(device)
        pts, ok, w = RC.region(r, device)
        jump = RC.depth_jump(pts, ok, w, r["dom_cam"].to(device))
        cam = r["dom_cam"].to(device)
        gt = shape_stats(pts, w, cam)
        groups = {"official": (r["pos0"].to(device), r["opacity"].to(device).float())}
        for k, net in nets.items():
            g = net.forward_reconstruction(mi).gaussians[0].float().reshape(-1, 64, 14)
            groups[k] = (g[..., :3], g[..., 3])
        sel = v & (gt["flatness"] < 2)
        rows = {"laser_region": {**gt, "jump": jump}}
        for k, (x, op) in groups.items():
            wk = op * (op >= ALIVE)
            alive_mass = wk.sum(-1)
            st = shape_stats(x, wk, cam, gt["normal"])
            st["jump"] = jump
            st["alive_mass"] = alive_mass
            rows[k] = st
        for k, st in rows.items():
            a = acc.setdefault(k, {})
            m = sel & (rows["official"]["alive_mass"] > 0.5) if k == "laser_region" else sel & (st["alive_mass"] > 0.5)
            for name, val_ in st.items():
                if name in ("normal",):
                    continue
                a.setdefault(name, []).append(val_[m].cpu())
        print(f"[{i + 1:02d}/50] {sid}", flush=True)
    res = {"definitions": __doc__.strip().split("\n\n")[0], "gpu": gpu["index"], "models": {}}
    for k, a in acc.items():
        t = {n: torch.cat(x) for n, x in a.items()}
        jump = t["jump"].bool()
        res["models"][k] = {"tokens": int(jump.numel()),
                            **{n: q(t[n]) for n in ("flatness", "thickness_cm", "extent_cm", "ray_over_lateral", "normal_cos") if n in t},
                            "no_depth_jump": {n: q(t[n][~jump]) for n in ("flatness", "thickness_cm", "normal_cos") if n in t},
                            "share_disc_flatness_lt_0.3": float((t["flatness"][~jump] < 0.3).float().mean()),
                            "share_ray_spread_gt_lateral": float((t["ray_over_lateral"] > 1).float().mean())}
    out = ROOT / "docs" / "DIAG_token_shape.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["models"], indent=2))


if __name__ == "__main__":
    main()
