"""Did the far, transparent Gaussians that the 3D surface loss pulled back become useful?  (no training; 50 val windows)

Gaussian slots (token t, child k) correspond across models (all start from the official weights, new modules zero-init).
"Junk" slots = in the OFFICIAL model: opacity < 0.05 and centre > 30 cm from the laser mesh.  For each model we report, for
the junk slots and for all slots: distance to the mesh, opacity, and their share of the total rendering weight in the 4
official target views (per-Gaussian weight = d(sum of image rendered with colour 1)/d(colour), i.e. exact sum of
compositing weights over pixels).

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/diag_junk_slots.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from gsplat.rendering import rasterization
from safetensors.torch import load_file

from evaluate_scannetpp import ROOT, idle_gpu
from probe_gs_token_geometry import load_model
from probe_scene_latent_limit import S, make_opt
from scannetpp_evaluation_protocol import build_raycasting_scene
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data

MODELS = {"official": ROOT / "checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors",
          "sft_readout_rgb_depth": ROOT / "EXP_ScanNetPP/sft_readout_rgb_depth/model.safetensors",
          "sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20": ROOT / "EXP_ScanNetPP/sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20/model.safetensors"}


def render_weight(g, cam_view, intr, H, W):
    """Per-Gaussian sum of compositing weights over all pixels of the given views."""
    c = torch.ones((g.shape[0], 1), device=g.device, requires_grad=True)
    tot = 0.0
    for v in range(cam_view.shape[0]):
        fx, fy, cx, cy = intr[v].float().tolist()
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=g.device)
        img, _a, _m = rasterization(g[:, :3], torch.nn.functional.normalize(g[:, 7:11], dim=-1), g[:, 4:7], g[:, 3], c,
                                    cam_view[v].float().T.contiguous()[None], K[None], W, H, render_mode="RGB", packed=False)
        tot = tot + img.sum()
    (w,) = torch.autograd.grad(tot, c)
    return w[:, 0].detach()


def main():
    import sys
    for n in sys.argv[1:]:  # extra experiments to compare
        MODELS[n] = ROOT / "EXP_ScanNetPP" / n / "model.safetensors"
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")
    opt = make_opt("val")
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov.dataset._offset_frac = None
    H, W = map(int, opt.img_size)
    nets = {}
    for k, ck in MODELS.items():
        st = load_file(str(ck), device="cpu")
        am = st.get("activation_head.attr_mlp.5.weight")
        o = dataclasses.replace(opt, readout_pos_mlp=any(n.startswith("activation_head.pos_mlp.") for n in st),
                                readout_attr_mlp=am is not None,
                                readout_attr_mlp_channels="opacity" if am is not None and am.shape[0] == 64 else "all")
        nets[k] = (load_model(ck, o, dev), o)
    acc = {k: {"junk_dist": [], "junk_op": [], "junk_w": 0.0, "all_w": 0.0, "junk_n": 0, "all_n": 0,
               "junk_on_surface_opaque": 0, "junk_on_surface": 0, "junk_opaque": 0} for k in MODELS}
    for i in range(len(prov)):
        sid = Path(prov.dataset.sample_list[i]).name
        data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in prov[i].items()}
        ctx = prov.dataset.get_context_target_frames(i)[0]
        rc = build_raycasting_scene(prov.dataset.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply",
                                    torch.from_numpy(prov.dataset._scene_meta_cache[sid]["c2ws"][ctx].copy()), S)
        junk = None
        for k, (net, o) in nets.items():
            with torch.no_grad():
                g = net.forward_reconstruction(split_data(data, o)[0]).gaussians[0].float()
            dist = torch.from_numpy(rc.compute_distance(o3d.core.Tensor(g[:, :3].cpu().numpy().astype(np.float32))).numpy()).to(dev) / S
            op = g[:, 3]
            w = render_weight(g, data["cam_view"][0], data["intrinsics"][0], H, W)
            if junk is None:
                junk = (op < 0.05) & (dist > 0.30)
            a = acc[k]
            a["junk_dist"].append(dist[junk].cpu()); a["junk_op"].append(op[junk].cpu())
            a["junk_w"] += float(w[junk].sum()); a["all_w"] += float(w.sum())
            a["junk_n"] += int(junk.sum()); a["all_n"] += int(junk.numel())
            on = dist[junk] < 0.05
            a["junk_on_surface"] += int(on.sum()); a["junk_opaque"] += int((op[junk] >= 0.5).sum())
            a["junk_on_surface_opaque"] += int((on & (op[junk] >= 0.5)).sum())
        print(f"[{i + 1:02d}/50] {sid}", flush=True)
    res = {"definitions": __doc__.strip().split("\n\n")[0], "gpu": gpu["index"], "models": {}}
    for k, a in acc.items():
        d, op = torch.cat(a["junk_dist"]), torch.cat(a["junk_op"])
        res["models"][k] = {
            "junk_slot_share_of_all_slots": a["junk_n"] / a["all_n"],
            "junk_dist_to_mesh_cm_p50_p90": [round(float(d.median()) * 100, 1), round(float(d.quantile(0.9) if d.numel() < 16_000_000 else d.kthvalue(int(0.9 * d.numel())).values) * 100, 1)],
            "junk_opacity_p50_p90": [round(float(op.median()), 4), round(float(op.kthvalue(max(1, int(0.9 * op.numel()))).values), 4)],
            "junk_share_on_surface_lt5cm": a["junk_on_surface"] / max(a["junk_n"], 1),
            "junk_share_opaque_ge0.5": a["junk_opaque"] / max(a["junk_n"], 1),
            "junk_share_on_surface_and_opaque": a["junk_on_surface_opaque"] / max(a["junk_n"], 1),
            "junk_share_of_target_view_render_weight": a["junk_w"] / max(a["all_w"], 1e-12)}
        print(k, json.dumps(res["models"][k]), flush=True)
    (ROOT / "docs" / ("DIAG_junk_slots.json" if len(MODELS) == 3 else "DIAG_junk_slots_extra.json")).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
