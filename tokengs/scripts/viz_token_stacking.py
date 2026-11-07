"""Top-down slice of the official model's Gaussians on the demo scene: are different tokens stacked at different depths?

One input view of scene 7b6477cb95; a thin horizontal band of image rows; every Gaussian (opacity >= 0.05) whose centre
projects into the band is drawn at (image column, depth), coloured by its token; black = laser-mesh depth of the band.

  ALLOW_SHARED_GPU=1 PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/viz_token_stacking.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import diag_thickness_split as TS  # noqa: E402
from evaluate_scannetpp import ROOT, idle_gpu  # noqa: E402
from probe_scene_latent_limit import S, make_opt  # noqa: E402
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals  # noqa: E402
from tokengs.data.provider import Provider  # noqa: E402
from tokengs.models.input_types import split_data  # noqa: E402

SCENE, VIEW, BAND = "7b6477cb95", 0, 3
OUT = ROOT / "EXP_ScanNetPP/report/TokenGS分析汇报/fig_token_stacking_slice.png"


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
    cv, K = data["cam_view_input"][0, VIEW].float().T, data["intrinsics_input"][0, VIEW].float()
    pc = g[:, :3] @ cv[:3, :3].T + cv[:3, 3]
    z = pc[:, 2]
    u = K[0] * pc[:, 0] / z.clamp_min(1e-6) + K[2]
    v = K[1] * pc[:, 1] / z.clamp_min(1e-6) + K[3]
    ctx = prov.dataset.get_context_target_frames(i)[0]
    rc = build_raycasting_scene(prov.dataset.root_path / "data" / SCENE / "scans" / "mesh_aligned_0.05.ply",
                                torch.from_numpy(prov.dataset._scene_meta_cache[SCENE]["c2ws"][ctx].copy()), S)
    dgt, _n, ok = raycast_gt_depth_and_normals(rc, data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu(), opt.img_size)
    row = int(H * 0.45)
    sel = (v - row).abs() < BAND
    sel &= (g[:, 3] >= 0.05) & (z > 0) & (u >= 0) & (u < W)
    tok = (torch.arange(g.shape[0], device=dev) // 64)[sel].cpu().numpy()
    rng = np.random.default_rng(0)
    cols = rng.random((4096, 3)) * 0.85
    uu, zz, aa = u[sel].cpu().numpy(), (z[sel] / S).cpu().numpy(), g[sel, 3].cpu().numpy()
    gt_u = np.arange(W)
    gt_z = (dgt[VIEW, row] / S).numpy()
    gt_ok = ok[VIEW, row].numpy()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2, 1]})
    a = ax[0]
    a.scatter(uu, zz, s=8 + 40 * aa, c=cols[tok], alpha=0.85, linewidths=0)
    a.plot(gt_u[gt_ok], gt_z[gt_ok], "k.", ms=2.5, label="laser surface")
    a.set_xlim(0, W); a.invert_yaxis(); a.grid(alpha=.25); a.legend(loc="lower right", fontsize=8)
    a.set_xlabel("image column (px)"); a.set_ylabel("depth from camera (m)")
    a.set_title(f"official model, input view {VIEW}, rows {row}+-{BAND}: Gaussians coloured by token (opacity>=0.05)", fontsize=9)
    best, lo = -1, 0
    for c0 in range(0, W - 50, 5):  # smooth, fully valid 50-px wall segment with the most Gaussians
        seg = slice(c0, c0 + 50)
        if gt_ok[seg].all() and np.abs(np.diff(gt_z[seg])).max() < 0.03:
            mm = (uu >= c0) & (uu < c0 + 50)
            n = int(mm.sum())
            r = np.abs(zz[mm] - gt_z[np.clip(np.round(uu[mm]).astype(int), 0, W - 1)]) if n else np.array([1.0])
            if n > best and np.median(r) < 0.08:  # the segment's Gaussians belong to this wall, not to an occluder
                best, lo = n, c0
    m = (uu >= lo) & (uu < lo + 50)
    res = (zz[m] - gt_z[np.clip(np.round(uu[m]).astype(int), 0, W - 1)]) * 100
    b = ax[1]
    b.scatter(uu[m], res, s=10 + 50 * aa[m], c=cols[tok[m]], alpha=0.9, linewidths=0)
    b.axhline(0, color="k", lw=1.2, label="laser surface")
    b.set_ylim(-30, 30); b.invert_yaxis(); b.grid(alpha=.25); b.legend(loc="lower right", fontsize=8)
    b.set_xlabel("image column (px)"); b.set_ylabel("depth minus laser depth (cm)")
    b.set_title(f"zoom on a flat wall (columns {lo}-{lo + 50}): {len(set(tok[m].tolist()))} tokens, each colour = one token", fontsize=9)
    ax[0].axvspan(lo, lo + 50, color="gray", alpha=0.15)
    fig.tight_layout(); fig.savefig(OUT, dpi=150)
    print("saved", OUT, "gaussians in band", int(sel.sum()), "tokens", len(set(tok.tolist())))


if __name__ == "__main__":
    main()
