"""Overlap between scene latents and depth-discontinuity ("cross-object" proxy) of each latent's laser region.

For the 50 validation windows (frozen official model, layer-7 local-head mean attention, 90 % mass patch sets):
  * patch coverage: how many latents include each patch in their 90 % set
  * pairwise overlap: cosine similarity of the latents' patch-attention vectors; per latent, the number of other
    latents with cosine > 0.5 and the max cosine
  * depth discontinuity: laser points of the latent's region projected on its viewing ray; a latent "spans a
    depth jump" if the sorted depths contain a gap > --gap metres with >= 10 % of the attention mass on each side
    (unlike the earlier 20 cm depth-range flag, a slanted continuous surface does not trigger this)
  * relation to the probe error (set Transformer predictions from probe_scene_latent_region.py fitpred)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from diagnose_latent_attention_geometry import capture_cross_attention
from evaluate_scannetpp import CHECKPOINT, ROOT, idle_gpu
from probe_scene_latent_limit import S, make_opt
from probe_scene_latent_position import stats
from probe_scene_latent_region import PRED_FILE, full_res_points, load_cache
from scannetpp_evaluation_protocol import build_raycasting_scene
from tokengs.data.provider import Provider
from tokengs.models import model_registry
from tokengs.models.input_types import split_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=0.15)
    ap.add_argument("--mass", type=float, default=0.9)
    ap.add_argument("--max-patches", type=int, default=64)
    args = ap.parse_args()
    idle_gpu()
    device = torch.device("cuda:0")
    _tr, te = load_cache()
    te = {w["scene_id"]: w for w in te}
    preds = torch.load(PRED_FILE, map_location="cpu", weights_only=False)["set_transformer"]
    opt = make_opt("val")
    prov = Provider("scannetpp_val_scaled_0.3", opt, training=False)
    prov.dataset._offset_frac = None
    model = model_registry[opt.model_type](opt).eval().requires_grad_(False)
    model.load_state_dict(load_file(str(CHECKPOINT), device="cpu"), strict=True)
    model = model.to(device=device, dtype=torch.float32)

    cover, n_sim, max_sim, jump, err, n_patch = [], [], [], [], [], []
    for idx in range(len(prov)):
        ds = prov.dataset
        sid = Path(ds.sample_list[idx]).name
        w = te[sid]
        context, _ = ds.get_context_target_frames(idx)
        raw = prov[idx]
        data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
        model_input, _ = split_data(data, opt)
        with torch.inference_mode():
            _l, attn = capture_cross_attention(model, model_input, [7])
        a = attn[7].to(device)
        eff = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
        local = [h for h in range(a.shape[0]) if float(eff[h].median()) < 30]
        lat_idx = torch.nonzero(w["valid"]).squeeze(1).to(device)
        am = a[local].mean(0)[lat_idx]  # [N, VP]
        tw, ti = am.topk(args.max_patches, -1)
        cum = tw.cumsum(-1) / am.sum(-1, keepdim=True)
        keep = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool), cum[:, :-1] < args.mass], 1)
        tw = tw * keep
        n_patch.append(keep.sum(-1).cpu())
        Wm = torch.zeros_like(am).scatter_(1, ti, tw)  # sparse-ish patch weights of the 90 % sets
        cover.append((Wm > 0).sum(0)[(Wm > 0).sum(0) > 0].cpu())
        Wn = torch.nn.functional.normalize(Wm, dim=-1)
        sim = Wn @ Wn.T
        sim.fill_diagonal_(0)
        n_sim.append((sim > 0.5).sum(-1).cpu())
        max_sim.append(sim.max(-1).values.cpu())
        del sim, Wn

        cam_in, intr_in = data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu()
        raw_c2w = torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][context].copy())
        rc = build_raycasting_scene(ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
        pix, ok, _c = full_res_points(rc, cam_in, intr_in, opt.img_size)
        V, H, Wd, _ = pix.shape
        p = opt.patch_size
        hp, wp = H // p, Wd // p
        ppts = pix.reshape(V, hp, p, wp, p, 3).permute(0, 1, 3, 2, 4, 5).reshape(V * hp * wp, p * p, 3).to(device)
        pok = ok.reshape(V, hp, p, wp, p).permute(0, 1, 3, 2, 4).reshape(V * hp * wp, p * p).to(device)
        cent = w["target"].to(device)[lat_idx]
        cam = w["cam"].to(device)[lat_idx]
        ray = torch.nn.functional.normalize(cent - cam, dim=-1)
        jm = torch.zeros(len(lat_idx), dtype=torch.bool)
        for c in range(0, len(lat_idx), 256):
            sl = slice(c, c + 256)
            pts = ppts[ti[sl]].reshape(len(ti[sl]), -1, 3)
            okm = (pok[ti[sl]] & (tw[sl] > 0).unsqueeze(-1)).reshape(len(ti[sl]), -1)
            wp_ = (tw[sl].unsqueeze(-1).expand(-1, -1, p * p)).reshape(len(ti[sl]), -1) * okm
            t = ((pts - cam[sl].unsqueeze(1)) * ray[sl].unsqueeze(1)).sum(-1)
            t = torch.where(okm, t, float("inf"))
            ts, order = t.sort(-1)
            ws = torch.gather(wp_, 1, order)
            cw = ws.cumsum(-1) / ws.sum(-1, keepdim=True).clamp_min(1e-12)
            gaps = (ts[:, 1:] - ts[:, :-1]) / S
            fin = torch.isfinite(gaps)
            ok_split = (cw[:, :-1] >= 0.1) & (cw[:, :-1] <= 0.9) & fin
            jm[c:c + 256] = ((gaps > args.gap) & ok_split).any(-1).cpu()
        jump.append(jm)
        err.append(((preds[sid] - w["target"][w["valid"]]).norm(dim=-1) / S))
        torch.cuda.empty_cache()
        if (idx + 1) % 10 == 0:
            print(f"[overlap] {idx + 1}/{len(prov)}", flush=True)

    cover, n_sim, max_sim = torch.cat(cover).float(), torch.cat(n_sim).float(), torch.cat(max_sim)
    jump, err, n_patch = torch.cat(jump), torch.cat(err), torch.cat(n_patch).float()
    tail = err > torch.quantile(err, 0.75)
    q = lambda x: {"median": float(x.median()), "p10": float(torch.quantile(x, 0.1)), "p90": float(torch.quantile(x, 0.9))}
    res = {
        "patches_per_latent": q(n_patch),
        "latents_covering_each_patch": q(cover),
        "other_latents_with_cosine_gt_0.5": q(n_sim),
        "frac_latents_with_a_near_duplicate_cos_gt_0.8": float((max_sim > 0.8).float().mean()),
        "max_cosine_to_another_latent": q(max_sim),
        "frac_latents_spanning_depth_jump": float(jump.float().mean()),
        "depth_jump_gap_m": args.gap,
        "probe_error_no_jump": stats(err[~jump]),
        "probe_error_with_jump": stats(err[jump]),
        "frac_tail_spanning_depth_jump": float(jump[tail].float().mean()),
    }
    out = ROOT / "docs" / "DIAG_scene_latent_overlap.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
