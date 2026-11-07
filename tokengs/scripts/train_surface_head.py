"""Shared-surface layer, step 1: can a dedicated surface head predict better input-view depth than the Gaussians do?

Frozen backbone = best model (sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20).  Surface head: every input patch
feature (multi-scale encoder output, 4096-d) cross-attends the 4096 scene latents (2 blocks, width 512) and predicts
8x8 sub-pixel log-depths + Laplace log-scales; trained on laser z-depth of the 6 input views (853 training scenes,
3 deterministic windows each).  Validation (50 official windows): depth error on laser pixels vs. the frozen model's own
rendered input-view depth, overall / by number of input views seeing the point / by distance.  RGB + cameras at inference.

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/train_surface_head.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from diag_token_render_ownership import rasterize_channels
from evaluate_scannetpp import ROOT, idle_gpu
from probe_gs_token_geometry import load_model
from probe_scene_latent_limit import S, make_opt
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data

CKPT = ROOT / "EXP_ScanNetPP/sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20/model.safetensors"
OUT = Path("/mnt/disk/xveyan/workspace_scannetpp/surface_head")
OFFSETS, EPOCHS, LR = (0.15, 0.5, 0.85), 3, 3e-4
SMOKE = int(os.environ.get("SMOKE", "0"))  # >0: only that many windows, 1 epoch, 1 offset, tiny val


class SurfaceHead(nn.Module):
    def __init__(self, din=4096, dlat=1024, d=512, heads=8, layers=2, sub=64):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, d))
        self.lat = nn.Sequential(nn.LayerNorm(dlat), nn.Linear(dlat, d))
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(
            n1=nn.LayerNorm(d), att=nn.MultiheadAttention(d, heads, batch_first=True), n2=nn.LayerNorm(d),
            mlp=nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)))) for _ in range(layers)])
        self.out = nn.Linear(d, 2 * sub)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias[:sub] = math.log(2.5 * S)   # ~2.5 m
            self.out.bias[sub:] = -2.0

    def forward(self, f, lat):
        x, kv = self.inp(f), self.lat(lat)
        for b in self.blocks:
            x = x + b["att"](b["n1"](x), kv, kv, need_weights=False)[0]
            x = x + b["mlp"](b["n2"](x))
        o = self.out(x)
        return o[..., :o.shape[-1] // 2], o[..., o.shape[-1] // 2:].clamp(-6, 3)


def to_map(x, V, hp, wp, p=8):
    """[V*hp*wp, p*p] -> [V, hp*p, wp*p] (patch order view-major, row-major; sub-pixel row-major)."""
    return x.reshape(V, hp, wp, p, p).permute(0, 1, 3, 2, 4).reshape(V, hp * p, wp * p)


def n_views_seeing(depth, ok, cam_view, intr):
    """For every laser pixel of every input view: how many input views see the same 3D point (depth-consistent)."""
    V, H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    cnt = torch.zeros_like(depth, dtype=torch.long)
    c2w = torch.linalg.inv(cam_view.transpose(-1, -2))
    for v in range(V):
        fx, fy, cx, cy = intr[v]
        pc = torch.stack([(xs + 0.5 - cx) / fx * depth[v], (ys + 0.5 - cy) / fy * depth[v], depth[v]], -1)
        pw = pc @ c2w[v, :3, :3].T + c2w[v, :3, 3]
        for u in range(V):
            w2c = cam_view[u].T
            q = pw @ w2c[:3, :3].T + w2c[:3, 3]
            fx2, fy2, cx2, cy2 = intr[u]
            px, py = fx2 * q[..., 0] / q[..., 2].clamp_min(1e-6) + cx2, fy2 * q[..., 1] / q[..., 2].clamp_min(1e-6) + cy2
            inside = (q[..., 2] > 1e-3) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            xi, yi = px.clamp(0, W - 1).long(), py.clamp(0, H - 1).long()
            dz = depth[u][yi, xi]
            cnt[v] += (inside & ok[u][yi, xi] & ((dz - q[..., 2]).abs() < 0.05 * q[..., 2])).long()
    return cnt


@torch.no_grad()
def backbone(model, data, opt):
    mi, _ = split_data(data, opt)
    x = model._embed_encoder_input(mi.encoder)
    feats = model.enc_dec_backbone._encode_features(x)
    lat = model.enc_dec_backbone._features_to_scene_latents(feats)
    return mi, feats[0], lat[0]


def gt_depth(prov, i, data, opt):
    sid = Path(prov.dataset.sample_list[i]).name
    ctx = prov.dataset.get_context_target_frames(i)[0]
    rc = build_raycasting_scene(prov.dataset.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply",
                                torch.from_numpy(prov.dataset._scene_meta_cache[sid]["c2ws"][ctx].copy()), S)
    d, _n, ok = raycast_gt_depth_and_normals(rc, data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu(), opt.img_size)
    return d, ok


def main():
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    opt_v = make_opt("val")
    o = dataclasses.replace(opt_v, readout_pos_mlp=any(k.startswith("activation_head.pos_mlp.") for k in load_file(str(CKPT), device="cpu")))
    model = load_model(CKPT, o, dev)
    H, W = map(int, o.img_size)
    hp, wp, V = H // 8, W // 8, int(o.num_input_views)
    head = SurfaceHead().to(dev)
    optim = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-2)
    tr = Provider("scannetpp_val_scaled_0.3", dataclasses.replace(o, dataset_kwargs={"window_size": 18, "split": make_opt("train").dataset_kwargs["split"]}), training=False)
    n_tr = SMOKE or len(tr)
    eps, offs = (1, OFFSETS[:1]) if SMOKE else (EPOCHS, OFFSETS)
    total = eps * len(offs) * n_tr
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, LR, total_steps=total, pct_start=0.05)
    gt_cache, step, t0 = {}, 0, time.time()
    for ep in range(eps):
        for off in offs:
            tr.dataset._offset_frac = off
            for i in torch.randperm(len(tr)).tolist()[:n_tr]:
                data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in tr[i].items()}
                key = (i, off)
                if key not in gt_cache:
                    d, ok = gt_depth(tr, i, data, o)
                    gt_cache[key] = (d.half(), ok)
                d, ok = gt_cache[key][0].float().to(dev), gt_cache[key][1].to(dev) & (gt_cache[key][0].to(dev) > 1e-3)
                _mi, feats, lat = backbone(model, data, o)
                ld, ls = head(feats.unsqueeze(0), lat.unsqueeze(0))
                ld, ls = to_map(ld[0], V, hp, wp), to_map(ls[0], V, hp, wp)
                err = (ld - torch.log(d.clamp_min(1e-4))).abs()
                loss = (err * torch.exp(-ls) + ls)[ok].mean()
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optim.step(); sched.step(); step += 1
                if step % 200 == 0:
                    print(f"[surface] step {step}/{total} loss {loss.item():.4f} logdepth-L1 {err[ok].mean().item():.4f} {time.time() - t0:.0f}s", flush=True)
        torch.save(head.state_dict(), OUT / "head.pt")
    # ---- validation
    head.eval()
    va = Provider("scannetpp_val_scaled_0.3", opt_v, training=False)
    va.dataset._offset_frac = None
    acc = {"head": [], "gaussians": [], "nvis": [], "dist": [], "conf": []}
    with torch.no_grad():
        for i in range(SMOKE and 2 or len(va)):
            data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in va[i].items()}
            d, ok = gt_depth(va, i, data, o)
            d, ok = d.to(dev), ok.to(dev) & (d.to(dev) > 1e-3)
            mi, feats, lat = backbone(model, data, o)
            ld, ls = head(feats.unsqueeze(0), lat.unsqueeze(0))
            dh, sh = to_map(ld[0], V, hp, wp).exp(), to_map(ls[0], V, hp, wp)
            g = model.decode_gaussians(model.encoder_latent_from_scene_latents(lat.unsqueeze(0)), mi)[0] if hasattr(model, "decode_gaussians") else model.forward_reconstruction(mi).gaussians[0]
            g = g.float()
            dg = torch.zeros_like(d)
            for v in range(V):
                viewmat = data["cam_view_input"][0, v].float().T.contiguous()
                fx, fy, cx, cy = data["intrinsics_input"][0, v].float().tolist()
                K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=dev)
                z = (g[:, :3] @ viewmat[:3, :3].T + viewmat[:3, 3])[:, 2:3].clamp_min(0)
                A, a = rasterize_channels(g, z.contiguous(), viewmat, K, H, W, o)
                dg[v] = A[..., 0] / a.clamp_min(1e-6)
            nv = n_views_seeing(d, ok, data["cam_view_input"][0].float(), data["intrinsics_input"][0].float())
            acc["head"].append(((dh - d).abs() / S * 100)[ok].cpu()); acc["gaussians"].append(((dg - d).abs() / S * 100)[ok].cpu())
            acc["nvis"].append(nv[ok].cpu()); acc["dist"].append((d / S)[ok].cpu()); acc["conf"].append(sh[ok].cpu())
            acc.setdefault("rel_head", []).append(((dh - d).abs() / d)[ok].cpu()); acc.setdefault("rel_g", []).append(((dg - d).abs() / d)[ok].cpu())
    t = {k: torch.cat(v) for k, v in acc.items()}
    kq = lambda x, p: float(x.float().kthvalue(max(1, int(p * x.numel()))).values)  # noqa: E731
    st = lambda m: {"pixel_share": float(m.float().mean()),
                    "head_cm_p50_p90": [round(kq(t["head"][m], .5), 2), round(kq(t["head"][m], .9), 2)],
                    "gaussians_cm_p50_p90": [round(kq(t["gaussians"][m], .5), 2), round(kq(t["gaussians"][m], .9), 2)]}  # noqa: E731
    all_ = torch.ones_like(t["nvis"], dtype=torch.bool)
    res = {"definitions": __doc__.strip().split("\n\n")[0], "gpu": gpu["index"], "backbone": str(CKPT.relative_to(ROOT)),
           "absrel_head": float(t["rel_head"].mean()), "absrel_gaussians_rendered": float(t["rel_g"].mean()), "all": st(all_),
           "by_views_seeing": {k: st(m) for k, m in (("1 (only this view)", t["nvis"] <= 1), ("2", t["nvis"] == 2), ("3", t["nvis"] == 3), ("4+", t["nvis"] >= 4)) if m.any()},
           "by_distance_m": {k: st(m) for k, m in (("<1.5", t["dist"] < 1.5), ("1.5-2.5", (t["dist"] >= 1.5) & (t["dist"] < 2.5)), ("2.5-3.5", (t["dist"] >= 2.5) & (t["dist"] < 3.5)), (">3.5", t["dist"] >= 3.5)) if m.any()},
           "confidence_vs_error_spearman_proxy": float(torch.corrcoef(torch.stack([t["conf"].float(), torch.log(t["head"].float() + 0.1)]))[0, 1])}
    (ROOT / "docs" / ("DIAG_surface_head_smoke.json" if SMOKE else "DIAG_surface_head.json")).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
