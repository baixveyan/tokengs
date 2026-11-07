"""If Gaussians that share a small voxel shared ONE position, how much thinner and how much worse would it get?
(no training; 50 validation windows, official interpolation targets + the 6 input views)

Snap: every Gaussian's centre is moved to the opacity-weighted mean centre of all Gaussians in its voxel (size eps);
all other attributes unchanged.  Reports ray thickness (total / between-token share, as diag_thickness_split.py) and PSNR.

  ALLOW_SHARED_GPU=1 PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/diag_voxel_snap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import diag_thickness_split as TS
from evaluate_scannetpp import ROOT, idle_gpu
from probe_scene_latent_limit import S, make_opt
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data

MODELS = {"official": TS.MODELS["official"],
          "sft_latenttail_readout_rgb_depth_surface3d_posmlp": ROOT / "EXP_ScanNetPP/sft_latenttail_readout_rgb_depth_surface3d_posmlp/model.safetensors"}
EPS_CM = (0, 2, 4, 8)


def snap(g, eps):
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


@torch.no_grad()
def main():
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")
    opt = make_opt("val")
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov.dataset._offset_frac = None
    H, W = map(int, opt.img_size)
    res = {"definitions": __doc__.strip().split("\n\n")[0], "gpu": gpu["index"], "models": {}}
    for name, ckpt in MODELS.items():
        import dataclasses
        from safetensors.torch import load_file
        o = dataclasses.replace(opt, readout_pos_mlp=any(k.startswith("activation_head.pos_mlp.") for k in load_file(str(ckpt), device="cpu")))
        model = TS.load_model(ckpt, o, dev)
        acc = {e: {"thick": [], "between": 0.0, "total": 0.0, "psnr_tg": [], "psnr_in": []} for e in EPS_CM}
        for i in range(len(prov)):
            sid = Path(prov.dataset.sample_list[i]).name
            data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in prov[i].items()}
            mi, _ = split_data(data, o)
            rec = model.forward_reconstruction(mi)
            ctx = prov.dataset.get_context_target_frames(i)[0]
            rc = build_raycasting_scene(prov.dataset.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply",
                                        torch.from_numpy(prov.dataset._scene_meta_cache[sid]["c2ws"][ctx].copy()), S)
            cam_view, intr = data["cam_view"][0], data["intrinsics"][0]
            _d, _n, gt_ok = raycast_gt_depth_and_normals(rc, cam_view.cpu(), intr.cpu(), o.img_size)
            for e in EPS_CM:
                g = snap(rec.gaussians[0].float(), e / 100 * S)
                gb = g.unsqueeze(0)
                tg = model.gs.render(gb, data["cam_view"], bg_color=rec.background_color, intrinsics=data["intrinsics"])["images_pred"][0].clamp(0, 1)
                acc[e]["psnr_tg"].append(-10 * (tg - data["images_output"][0]).square().mean((1, 2, 3)).log10())
                ip = model.gs.render(gb, data["cam_view_input"], bg_color=rec.background_color, intrinsics=data["intrinsics_input"])["images_pred"][0].clamp(0, 1)
                acc[e]["psnr_in"].append(-10 * (ip - data["images_input"][0]).square().mean((1, 2, 3)).log10())
                for v in range(0, cam_view.shape[0], 2):  # every other target view keeps the cost moderate
                    viewmat = cam_view[v].float().T.contiguous()
                    fx, fy, cx, cy = intr[v].float().tolist()
                    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=dev)
                    alpha, vt, _vw, vb, _ve = TS.split_view(g, viewmat, K, H, W, o)
                    m = gt_ok[v].to(dev) & (alpha > 0.5)
                    acc[e]["thick"].append(vt[m].sqrt().cpu())
                    acc[e]["between"] += float(vb[m].sum()); acc[e]["total"] += float(vt[m].sum())
            print(f"[{name}] {i + 1:02d}/50", flush=True)
        res["models"][name] = {f"eps_{e}cm": {"ray_thickness": TS.q(torch.cat(a["thick"])),
                                              "between_token_share": a["between"] / max(a["total"], 1e-12),
                                              "psnr_target": round(float(torch.cat(a["psnr_tg"]).mean()), 3),
                                              "psnr_input": round(float(torch.cat(a["psnr_in"]).mean()), 3)} for e, a in acc.items()}
        print(json.dumps(res["models"][name]), flush=True)
        del model
        torch.cuda.empty_cache()
    (ROOT / "docs" / "DIAG_voxel_snap.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
