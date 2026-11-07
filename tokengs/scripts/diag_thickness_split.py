"""Where does the ray thickness come from?  (no training; official interpolation targets of the 50 validation scenes)

The evaluated "ray thickness" is the rendering-weighted spread of Gaussian CENTRE depths along a pixel ray.  By the law
of total variance it splits exactly into
  within-token   spread of the centres of the SAME token's 64 Gaussians along the ray
  between-token  spread between the depth centroids of DIFFERENT tokens on the same ray
(computed exactly by rasterising one-hot token channels times 1, z, z^2 with the model's renderer settings).
Also reported, but not part of the metric: the intrinsic depth extent of each Gaussian along the ray,
1 / (d^T Sigma^-1 d), rendering-weighted.  Pixels: GT-mesh hit and rendered alpha > 0.5.
Side measurement: PSNR on the 6 INPUT views (views seen by the model) next to the official target-view PSNR.

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/diag_thickness_split.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from diag_token_render_ownership import rasterize_channels
from evaluate_scannetpp import ROOT, idle_gpu
from probe_gs_token_geometry import load_model
from probe_scene_latent_limit import S, make_opt
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data

MODELS = {"official": ROOT / "checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors",
          "sft_readout_rgb_depth": ROOT / "EXP_ScanNetPP/sft_readout_rgb_depth/model.safetensors",
          "sft_readout_rgb_depth_thick": ROOT / "EXP_ScanNetPP/sft_readout_rgb_depth_thick/model.safetensors"}
CH, CHUNK = 64, 512


def quat_to_rot(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


@torch.no_grad()
def split_view(g, viewmat, K, H, W, opt):
    """Per-pixel alpha, total / within-token / between-token centre-depth variance and intrinsic extent variance."""
    N = g.shape[0]
    tok = torch.arange(N, device=g.device) // CH
    z = (g[:, :3] @ viewmat[:3, :3].T + viewmat[:3, 3])[:, 2].clamp_min(0)
    cam = -viewmat[:3, :3].T @ viewmat[:3, 3]
    d = torch.nn.functional.normalize(g[:, :3] - cam, dim=-1)
    R = quat_to_rot(torch.nn.functional.normalize(g[:, 7:11], dim=-1))
    local = (R.transpose(-1, -2) @ d.unsqueeze(-1)).squeeze(-1)
    ext = 1.0 / (local.square() / g[:, 4:7].square().clamp_min(1e-12)).sum(-1).clamp_min(1e-12)
    tot, alpha = rasterize_channels(g, torch.stack([z, z * z, ext], -1).contiguous(), viewmat, K, H, W, opt)
    S1 = torch.zeros((H, W), device=g.device)
    for c0 in range(0, (N + CH - 1) // CH, CHUNK):
        k = min(CHUNK, N // CH - c0)
        sel = (tok >= c0) & (tok < c0 + k)
        oh = torch.zeros((N, k), device=g.device)
        oh[sel, tok[sel] - c0] = 1.0
        Wt, _ = rasterize_channels(g, oh, viewmat, K, H, W, opt)
        At, _ = rasterize_channels(g, oh * z.unsqueeze(-1), viewmat, K, H, W, opt)
        S1 += (At.square() / Wt.clamp_min(1e-12)).sum(-1)
    a = alpha.clamp_min(1e-6)
    A, B = tot[..., 0], tot[..., 1]
    var_tot = (B / a - (A / a).square()).clamp_min(0)
    var_within = ((B - S1) / a).clamp_min(0)
    var_between = ((S1 - A * A / a) / a).clamp_min(0)
    return alpha, var_tot, var_within, var_between, tot[..., 2] / a


def q(x):
    x = x.float().flatten()
    k = lambda p: float(x.kthvalue(max(1, int(p * x.numel()))).values) / S * 100  # noqa: E731
    return {"p50_cm": round(k(0.5), 2), "p90_cm": round(k(0.9), 2)}


@torch.no_grad()
def main():
    import sys
    global MODELS
    tag = ""
    if len(sys.argv) > 1:  # optional experiment names, e.g. new runs
        official = MODELS["official"]
        MODELS = {n: official if n == "official" else ROOT / "EXP_ScanNetPP" / n / "model.safetensors" for n in sys.argv[1:]}
        tag = "_" + "_".join(sorted(MODELS))[:80]
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    opt = make_opt("val")
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov.dataset._offset_frac = None
    H, W = map(int, opt.img_size)
    res = {"definitions": __doc__.strip().split("\n\n")[0], "gpu": gpu["index"], "models": {}}
    for name, ckpt in MODELS.items():
        import dataclasses
        from safetensors.torch import load_file
        st = load_file(str(ckpt), device="cpu")
        am = st.get("activation_head.attr_mlp.5.weight")
        o = dataclasses.replace(opt, readout_pos_mlp=any(k.startswith("activation_head.pos_mlp.") for k in st),
                                readout_attr_mlp=am is not None,
                                readout_attr_mlp_channels="opacity" if am is not None and am.shape[0] == 64 else "all")
        model = load_model(ckpt, o, device)
        acc = {k: [] for k in ("tot", "within", "between", "ext")}
        sums = torch.zeros(3, dtype=torch.float64)
        psnr_in, psnr_tg = [], []
        for i in range(len(prov)):
            sid = Path(prov.dataset.sample_list[i]).name
            raw = prov[i]
            data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
            mi, _ = split_data(data, opt)
            rec = model.forward_reconstruction(mi)
            g = rec.gaussians[0].float()
            tg = model.render_reconstruction(rec, mi.decoder)["images_pred"][0].clamp(0, 1)
            psnr_tg.append(-10 * (tg - data["images_output"][0]).square().mean((1, 2, 3)).log10())
            inp = model.gs.render(rec.gaussians, data["cam_view_input"], bg_color=rec.background_color,
                                  intrinsics=data["intrinsics_input"])["images_pred"][0].clamp(0, 1)
            psnr_in.append(-10 * (inp - data["images_input"][0]).square().mean((1, 2, 3)).log10())
            ctx = prov.dataset.get_context_target_frames(i)[0]
            raw_c2w = torch.from_numpy(prov.dataset._scene_meta_cache[sid]["c2ws"][ctx].copy())
            rc = build_raycasting_scene(prov.dataset.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
            cam_view, intr = data["cam_view"][0], data["intrinsics"][0]
            _d, _n, gt_ok = raycast_gt_depth_and_normals(rc, cam_view.cpu(), intr.cpu(), opt.img_size)
            for v in range(cam_view.shape[0]):
                viewmat = cam_view[v].float().T.contiguous()
                fx, fy, cx, cy = intr[v].float().tolist()
                K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=device)
                alpha, vt, vw, vb, ve = split_view(g, viewmat, K, H, W, opt)
                m = gt_ok[v].to(device) & (alpha > 0.5)
                acc["tot"].append(vt[m].sqrt().cpu()); acc["within"].append(vw[m].sqrt().cpu())
                acc["between"].append(vb[m].sqrt().cpu()); acc["ext"].append(ve[m].sqrt().cpu())
                sums += torch.tensor([float(vt[m].sum()), float(vw[m].sum()), float(vb[m].sum())], dtype=torch.float64)
            print(f"[{name}] {i + 1:02d}/50 {sid}", flush=True)
        t = {k: torch.cat(x) for k, x in acc.items()}
        tot_cm = t["tot"] / S * 100
        ok_ = tot_cm > 0.05
        share = (t["between"][ok_].square() / t["tot"][ok_].square()).clamp(0, 1)
        bins = {}
        for lo, hi in ((0, 3), (3, 10), (10, 30), (30, 1e9)):
            m = (tot_cm[ok_] >= lo) & (tot_cm[ok_] < hi)
            if m.any():
                vt, vb = t["tot"][ok_][m].square(), t["between"][ok_][m].square()
                bins[f"{lo}-{hi if hi < 1e9 else 'inf'}cm"] = {"pixel_share": float(m.float().mean()),
                    "pooled_between_share": float(vb.sum() / vt.sum()), "median_pixel_between_share": float(share[m].median())}
        res["models"].setdefault(name, {})
        res["models"][name]["per_pixel_between_share_median"] = float(share.median())
        res["models"][name]["by_thickness"] = bins
        res["models"][name].update({
            "ray_thickness_total": q(t["tot"]), "within_token": q(t["within"]), "between_token": q(t["between"]),
            "variance_share_within_token": float(sums[1] / sums[0]), "variance_share_between_token": float(sums[2] / sums[0]),
            "intrinsic_gaussian_extent_along_ray": q(t["ext"]),
            "psnr_target_views": round(float(torch.cat(psnr_tg).mean()), 3), "psnr_input_views": round(float(torch.cat(psnr_in).mean()), 3)})
        print(json.dumps(res["models"][name]), flush=True)
        del model
        torch.cuda.empty_cache()
    (ROOT / "docs" / f"DIAG_thickness_split{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
