"""How much geometry survives from the scene latents to the GS tokens, and how much reaches the Gaussians?

Per GS token (frozen official model; laser-surface-only target and metrics, no attention centroid):
  token -> patch attention  T = A_dec (decoder cross-attn, head mean) @ A_lat (layer-7 local-head mean)
  target = first GT-mesh hit along the token's attention ray (dominant input view, attention-weighted patch rays)

Compared levels (same tokens, same target, same metrics):
  latent_mix  : A_dec @ scene_latents  (what the token reads from the scene latents)      -> probes
  gs_token    : decoder output feature h (exact input of the linear deconv head)            -> probes
  output_base : opacity-weighted centre of the token's 64 Gaussians, official model         (no probe)
  output_R1   : same, R1 model (readout chain trained with one rendered depth loss)         (no probe)
Probes: ridge (linear, like deconv) and a 4x2048 MLP, trained on 853 training scenes (1 window, 1024 tokens each).

  extract --shard k --num-shards N   (one process per idle GPU)
  fit                                  probes -> validation predictions
  eval                                 laser-surface metrics (depth along ray, lateral, surface distance, by #views)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from safetensors.torch import load_file

from diagnose_latent_attention_geometry import ALIVE_OPACITY, capture_cross_attention
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, idle_gpu
from probe_scene_latent_limit import S, SPLITS, make_opt, patch_geometry, train_pointwise
from probe_scene_latent_position import fit_linear, stats
from probe_scene_latent_region import full_res_points
from scannetpp_evaluation_protocol import build_raycasting_scene, sha256_file
from tokengs.data.provider import Provider
from tokengs.models import model_registry
from tokengs.models.input_types import EncoderLatent, split_data

CACHE = Path("/mnt/disk/xveyan/workspace_scannetpp/probe_gs_token")
R1_CKPT = ROOT / "EXP_ScanNetPP" / "sft_readout_rgb_depth" / "model.safetensors"


def load_model(path, opt, device):
    m = model_registry[opt.model_type](opt).eval().requires_grad_(False)
    m.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return m.to(device=device, dtype=torch.float32)


def decode(model, latents, model_input):
    """Return decoder cross-attn head mean [Ntok, Nlat], deconv input h [Ntok, C], Gaussians [Ntok, 64, 14]."""
    cap = {}
    blk = model.enc_dec_backbone.decoder_blocks[-1].gs_cross_attn

    def hook(_m, inputs, _o):
        gs, keys, _v = inputs
        q = blk.q_norm(rearrange(blk.q_proj(blk.gs_token_norm(gs)), "b n (h d) -> b h n d", h=blk.num_heads))
        cap["a"] = torch.softmax(torch.einsum("bhnd,bhmd->bhnm", q.float(), keys.float()) / math.sqrt(q.shape[-1]), -1)[0].mean(0)

    def pre(_m, inputs):
        cap["h"] = inputs[0][0].float()

    h1 = blk.register_forward_hook(hook)
    h2 = model.activation_head.register_forward_pre_hook(pre)
    with torch.inference_mode():
        k, v = model.enc_dec_backbone._latents_to_kv(latents)
        g = model.forward_decoder(EncoderLatent(keys=k, values=v), model_input.decoder)[0]
    h1.remove()
    h2.remove()
    return cap["a"], cap["h"], g.float().reshape(-1, 64, 14)


def child_centre(g):
    w = g[..., 3].clamp_min(0) * (g[..., 3] >= ALIVE_OPACITY).float()
    return (g[..., :3] * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-8)


def cmd_extract(args):
    import open3d as o3d

    gpu = idle_gpu()
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected checkpoint SHA256")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    prov = {sp: Provider("scannetpp_val_scaled_0.3", make_opt(sp), training=False) for sp in SPLITS}
    for p in prov.values():
        p.dataset._offset_frac = None
    tasks = [("val", i) for i in range(len(prov["val"]))] + [("train", i) for i in range(len(prov["train"]))]
    mine = tasks[args.shard::args.num_shards]
    opt = make_opt("val")
    base = load_model(CHECKPOINT, opt, device)
    r1 = load_model(R1_CKPT, opt, device)
    print(f"GPU {gpu['index']}: shard {args.shard}: {len(mine)} windows", flush=True)
    out, t0 = [], time.time()
    for n, (sp, idx) in enumerate(mine):
        ds = prov[sp].dataset
        sid = Path(ds.sample_list[idx]).name
        context, _ = ds.get_context_target_frames(idx)
        raw = prov[sp][idx]
        data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
        mi, _ = split_data(data, make_opt(sp))
        with torch.inference_mode():
            latents, attn = capture_cross_attention(base, mi, [args.layer])
        a = attn[args.layer]
        eff = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
        a_lat = a[[h for h in range(a.shape[0]) if float(eff[h].median()) < 30]].mean(0)  # [Nlat, VP]
        A_dec, h, g = decode(base, latents, mi)
        mix = A_dec @ latents[0].float()
        T = A_dec @ a_lat  # [Ntok, VP]
        cam_in, intr_in = data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu()
        raw_c2w = torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][context].copy())
        rc = build_raycasting_scene(ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
        _pts, _nrm, val, dirs, cams = patch_geometry(rc, cam_in, intr_in, make_opt(sp).img_size, make_opt(sp).patch_size)
        V = cams.shape[0]
        Tc = T.cpu()
        P = Tc.shape[1] // V
        vm = Tc.reshape(-1, V, P).sum(-1)
        dom = vm.argmax(-1)
        wdom = Tc.reshape(-1, V, P)[torch.arange(len(dom)), dom]
        ray = torch.nn.functional.normalize((wdom.unsqueeze(-1) * dirs.reshape(V, P, 3)[dom]).sum(1), dim=-1)
        cam = cams[dom]
        t_hit = torch.from_numpy(rc.cast_rays(o3d.core.Tensor(torch.cat([cam, ray], 1).numpy().astype(np.float32)))["t_hit"].numpy())
        valid = ((Tc * val.float()).sum(-1) > 0.5) & torch.isfinite(t_hit)
        target = cam + torch.where(torch.isfinite(t_hit), t_hit, torch.zeros_like(t_hit)).unsqueeze(-1) * ray
        rec = {"split": sp, "scene_id": sid}
        if sp == "train":
            keep = torch.nonzero(valid).squeeze(1)
            gsel = torch.Generator().manual_seed(int(sid, 16) % (2 ** 31))
            keep = keep[torch.randperm(len(keep), generator=gsel)[:args.train_tokens]]
            rec.update({"h": h[keep.to(device)].half().cpu(), "mix": mix[keep.to(device)].half().cpu(), "target": target[keep]})
        else:
            _a1, _h1, g1 = decode(r1, latents, mi)
            tw, ti = T.topk(64, -1)
            cum = tw.cumsum(-1) / T.sum(-1, keepdim=True)
            keepm = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool), cum[:, :-1] < 0.9], 1)
            rec.update({"h": h.half().cpu(), "mix": mix.half().cpu(), "target": target, "valid": valid, "cam": cam,
                        "n_views": (vm > 0.1).sum(-1).to(torch.int8), "out_base": child_centre(g).cpu(),
                        "out_r1": child_centre(g1).cpu(), "top_i": ti.to(torch.int32).cpu(),
                        "top_w": (tw * keepm).half().cpu(), "context": context})
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
    Y = torch.cat([r["target"] for r in tr])
    preds = {}
    for feat in ("mix", "h"):
        X = torch.cat([r[feat] for r in tr])
        print(f"{feat}: {len(X)} training tokens", flush=True)
        lin = fit_linear(X.float(), Y)
        mlp = train_pointwise(X, Y, device, 2048, 4, args.epochs)
        for name, f in ((f"{feat}_linear", lambda Z: lin(Z.float())), (f"{feat}_mlp", mlp)):
            preds[name] = {r["scene_id"]: f(r[feat]) for r in te}
        del X
    torch.save(preds, CACHE / "val_preds.pt")
    print("wrote", CACHE / "val_preds.pt")


def cmd_eval(args):
    import open3d as o3d

    from tokengs.data.registry import dataset_registry
    from tokengs.data.static.scannetpp import ScanNetPPVal

    _tr, te = load_all()
    preds = torch.load(CACHE / "val_preds.pt", weights_only=False)
    ds = ScanNetPPVal(**{**dataset_registry["scannetpp_val"]["kwargs"], "window_size": 18})
    opt = make_opt("val")
    names = list(preds) + ["output_base", "output_R1", "target_reference"]
    acc = {n: {k: [] for k in ("depth", "signed", "lateral", "surface")} for n in names}
    nv_all = []
    for r in te:
        sid = r["scene_id"]
        idx = [Path(p).name for p in ds.sample_list].index(sid)
        raw_c2w = torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][r["context"]].copy())
        rc = build_raycasting_scene(ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
        # input-view cameras in normalized frame are needed for the laser region: rebuild via provider-free path
        prov_sample = _cams_for(ds, idx, opt)
        pix, ok, _c = full_res_points(rc, prov_sample[0], prov_sample[1], opt.img_size)
        V, H, Wd, _ = pix.shape
        p = opt.patch_size
        ppts = pix.reshape(V, H // p, p, Wd // p, p, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, p * p, 3)
        pok = ok.reshape(V, H // p, p, Wd // p, p).permute(0, 1, 3, 2, 4).reshape(-1, p * p)
        v = r["valid"]
        cam = r["cam"][v]
        ti, tw = r["top_i"][v].long(), r["top_w"][v].float()
        nv_all.append(r["n_views"][v].float())
        P_all = {k: preds[k][sid][v] for k in preds}
        P_all["output_base"], P_all["output_R1"], P_all["target_reference"] = r["out_base"][v], r["out_r1"][v], r["target"][v]
        for n, Pp in P_all.items():
            Pp = Pp.float()
            d = Pp - cam
            rng = d.norm(dim=-1)
            dn = d / rng.clamp_min(1e-9).unsqueeze(-1)
            th = torch.from_numpy(rc.cast_rays(o3d.core.Tensor(torch.cat([cam, dn], 1).numpy().astype(np.float32)))["t_hit"].numpy())
            okh = torch.isfinite(th)
            hp = cam + th.clamp(max=1e3).unsqueeze(-1) * dn
            lat = torch.empty(len(Pp))
            for c in range(0, len(Pp), 512):
                sl = slice(c, c + 512)
                pts = ppts[ti[sl]].reshape(len(ti[sl]), -1, 3)
                big = torch.where((pok[ti[sl]] & (tw[sl] > 0).unsqueeze(-1)).reshape(len(ti[sl]), -1), 0.0, 1e9)
                lat[sl] = ((pts - hp[sl].unsqueeze(1)).norm(dim=-1) + big).amin(-1)
            A = acc[n]
            A["depth"].append(torch.where(okh, (rng - th).abs(), torch.nan) / S)
            A["signed"].append(torch.where(okh, rng - th, torch.nan) / S)
            A["lateral"].append(torch.where(okh & (lat < 1e8), lat, torch.nan) / S)
            A["surface"].append(torch.from_numpy(rc.compute_distance(o3d.core.Tensor(Pp.numpy().astype(np.float32))).numpy()) / S)
        print(f"[eval] {sid}", flush=True)
    nv = torch.cat(nv_all)
    res = {"n_val_tokens": int(len(nv))}
    fmt = lambda s: f"{s['median_cm']:5.1f}/{s['p90_cm']:5.1f}"
    for n in names:
        A = {k: torch.cat(x) for k, x in acc[n].items()}
        f = lambda x: x[torch.isfinite(x)]
        sg = f(A["signed"])
        rr = {k: stats(f(A[k])) for k in ("depth", "lateral", "surface")}
        rr["front_gt5cm"] = float((sg < -0.05).float().mean())
        rr["behind_gt5cm"] = float((sg > 0.05).float().mean())
        rr["depth_by_views"] = {}
        for lo, hi, nm in ((0, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 3.5, "3"), (3.5, 99, "4+")):
            m = (nv >= lo) & (nv < hi) & torch.isfinite(A["depth"])
            rr["depth_by_views"][nm] = {"n_frac": float(m.float().mean()), **stats(A["depth"][m])}
        res[n] = rr
        print(f"{n:16s} depth {fmt(rr['depth'])}  lateral {fmt(rr['lateral'])}  surface {fmt(rr['surface'])}  "
              f"front {rr['front_gt5cm']:.3f} behind {rr['behind_gt5cm']:.3f}  | by views "
              + "  ".join(f"{k}:{x['median_cm']:.1f}/{x['p90_cm']:.1f}" for k, x in rr["depth_by_views"].items()))
    out = ROOT / "docs" / "DIAG_gs_token_geometry.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


def _cams_for(ds, idx, opt):
    """Normalized input-view cam_view / intrinsics exactly as the Provider builds them (no images loaded)."""
    global _PROV
    if "_PROV" not in globals():
        _PROV = Provider("scannetpp_val_scaled_0.3", opt, training=False)
        _PROV.dataset._offset_frac = None
    s = _PROV[idx]
    return s["cam_view_input"], s["intrinsics_input"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["extract", "fit", "eval"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--train-tokens", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit, "eval": cmd_eval}[args.mode](args)


if __name__ == "__main__":
    main()
