"""Readout-capacity probe ("readout bottleneck" hypothesis): can a per-token readout lay the 64 child Gaussians of a GS token onto the laser surface the token is
responsible for?  (frozen official model; offline on cached decoder features; no rendering)

Responsibility region (validated by diag_token_render_ownership.py: 87 % of a token's render weight lies in it):
  token -> input-patch attention T = A_dec @ A_lat(layer-7 local heads); the patches holding 90 % of T (at most 64);
  per patch 4 laser points (GT mesh hits of the pixels at (2,2) (2,6) (6,2) (6,6) inside the 8x8 patch), weighted by
  the token's attention to the patch.
Readouts (all keep the official position parametrisation  pos = sign(r) * expm1(|r|),  r = raw position channels):
  deconv0  official deconv position rows, untouched                                   (reference, no training)
  linear   deconv position rows trained (warm start)                                   ~ same structure as now
  mlp_s    linear + zero-init residual MLP 1024-1024-1024-192
  mlp_l    linear + zero-init residual MLP 1024-4096-4096-4096-192
  moe      linear + zero-init residual mixture of 8 MLP experts (1024-1024-192), top-2 gating
  set      linear + zero-init residual from a 2-layer Transformer over all 4096 tokens of the window (upper reference:
           tokens may exchange information)
  oracle   no readout: 64 farthest-point samples of the GT region itself (coverage ceiling of a 64-point layout)
Region = top-64 patches capped at 90 % attention mass (the retained mass is reported; tokens are never dropped for it).
Loss (direct 3D, metres): smooth-L1 of [mean over children of the distance to the nearest region point] +
  [attention-weighted mean over region points of the distance to the nearest child].  Opacities are not trained.
Validation (50 windows, tokens with > 50 % of their attention on patches with a mesh hit) and a fixed 50-window
  training subset (train-fit, to separate under-fitting from generalisation).  Primary endpoints are opacity-free and
  permutation-invariant like the loss: child -> own region, attention-weighted region -> nearest child (recall),
  child -> any visible laser point; split by #attended views and by depth-jump tokens (> 15 cm gap along the dominant
  camera ray, >= 10 % weight on both sides).  Official-opacity-weighted numbers are secondary (slot-dependent).
Conclusions hold for the stated optimisation budget (8 epochs, one LR per group; LR sweep for `linear` via --tag):
  improvements are evidence of benefit, not proof of necessity.

  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/probe_readout_capacity.py extract --shard k --num-shards N
  PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/probe_readout_capacity.py fit --head mlp_s
  PYTHONPATH=scripts:. python scripts/probe_readout_capacity.py report
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from diagnose_latent_attention_geometry import capture_cross_attention
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, idle_gpu
from probe_gs_token_geometry import decode, load_model
from probe_scene_latent_limit import S, SPLITS, make_opt
from probe_scene_latent_region import full_res_points
from scannetpp_evaluation_protocol import build_raycasting_scene, sha256_file
from tokengs.data.provider import Provider
from tokengs.models.input_types import split_data

CACHE = Path("/mnt/disk/xveyan/workspace_scannetpp/probe_readout_capacity")
OUT_JSON = ROOT / "docs" / "DIAG_readout_capacity.json"
CH, TOPK, SUB = 64, 64, ((2, 2), (2, 6), (6, 2), (6, 6))
JUMP_M, JUMP_W = 0.15, 0.10
HEADS = ("oracle", "deconv0", "linear", "mlp_s", "mlp_l", "moe", "set")


# ----------------------------------------------------------------------------------------------- extraction
def cmd_extract(args):
    gpu = idle_gpu()
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected checkpoint SHA256")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    prov = {sp: Provider("scannetpp_val_scaled_0.3", make_opt(sp), training=False) for sp in SPLITS}
    for p in prov.values():
        p.dataset._offset_frac = None  # one deterministic window per scene (official validation rule)
    tasks = [("val", i) for i in range(len(prov["val"]))] + [("train", i) for i in range(len(prov["train"]))]
    mine = tasks[args.shard::args.num_shards]
    opt = make_opt("val")
    model = load_model(CHECKPOINT, opt, device)
    H, W = map(int, opt.img_size)
    pz = int(opt.patch_size)
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
            latents, attn = capture_cross_attention(model, mi, [7])
        a = attn[7]
        eff = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
        a_lat = a[[h for h in range(a.shape[0]) if float(eff[h].median()) < 30]].mean(0)
        A_dec, h, g64 = decode(model, latents, mi)
        T = (A_dec @ a_lat).float().cpu()  # [Ntok, V*P]
        cam_in, intr_in = data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu()
        raw_c2w = torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][context].copy())
        rc = build_raycasting_scene(ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
        pix, ok, cams = full_res_points(rc, cam_in, intr_in, opt.img_size)  # [V,H,W,3], [V,H,W], [V,3]
        V = pix.shape[0]
        pp = torch.stack([pix[:, oy::pz, ox::pz] for oy, ox in SUB], 3).reshape(V * (H // pz) * (W // pz), 4, 3)
        pok = torch.stack([ok[:, oy::pz, ox::pz] for oy, ox in SUB], 3).reshape(-1, 4)
        tw, ti = T.topk(TOPK, -1)
        cum = tw.cumsum(-1) / T.sum(-1, keepdim=True)
        keep = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool), cum[:, :-1] < 0.9], 1)
        tw = tw * keep
        has = pok.any(-1)[ti].float()
        valid = (tw * has).sum(-1) > 0.5 * tw.sum(-1)
        vm = T.reshape(T.shape[0], V, -1).sum(-1)
        rec = {"split": sp, "scene_id": sid, "h": h.half().cpu(), "top_i": ti.short(), "top_w": tw.half(),
               "pp": pp.half(), "pok": pok, "valid": valid, "kept_mass": (tw.sum(-1) / T.sum(-1)).half(),
               "n_views": (vm > 0.1).sum(-1).to(torch.int8), "dom_cam": cams[vm.argmax(-1)],
               "opacity": g64[..., 3].half().cpu()}
        if sp == "val":
            rec["pos0"] = g64[..., :3].float().cpu()
        out.append(rec)
        torch.cuda.empty_cache()
        if (n + 1) % 20 == 0 or n + 1 == len(mine):
            print(f"[shard {args.shard}] {n + 1}/{len(mine)} {time.time() - t0:.0f}s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"shard_{args.shard}.tmp"
    torch.save(out, tmp)
    tmp.rename(CACHE / f"shard_{args.shard}.pt")
    print("wrote", CACHE / f"shard_{args.shard}.pt")


def load_cache():
    recs = []
    for p in sorted(CACHE.glob("shard_*.pt")):
        recs.extend(torch.load(p, map_location="cpu", weights_only=False))
    return [r for r in recs if r["split"] == "train"], [r for r in recs if r["split"] == "val"]


# ----------------------------------------------------------------------------------------------- readouts
def pos_act(r):
    return torch.sign(r) * torch.expm1(r.abs())


def mlp(dims, zero_last=True):
    layers = [nn.LayerNorm(dims[0])]
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.GELU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if zero_last:
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


class MoE(nn.Module):
    def __init__(self, d=1024, hidden=1024, out=3 * CH, experts=8, k=2):
        super().__init__()
        self.norm, self.gate, self.k = nn.LayerNorm(d), nn.Linear(d, experts), k
        self.experts = nn.ModuleList([mlp((d, hidden, out)) for _ in range(experts)])
        self.aux = torch.tensor(0.0)

    def forward(self, h):
        logits = self.gate(self.norm(h))
        topv, topi = logits.topk(self.k, -1)
        w = torch.softmax(topv, -1)
        out = torch.zeros(h.shape[0], self.experts[0][-1].out_features, device=h.device)
        for e, ex in enumerate(self.experts):
            m = (topi == e)
            rows = m.any(-1)
            if rows.any():
                out[rows] += (w * m)[rows].sum(-1, keepdim=True) * ex(h[rows])
        probs = torch.softmax(logits, -1)
        load = F.one_hot(topi[:, 0], len(self.experts)).float().mean(0)
        self.aux = len(self.experts) * (load * probs.mean(0)).sum()  # Switch-style load balancing
        return out


class SetHead(nn.Module):
    def __init__(self, d=1024, dm=512, layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dm))
        enc = nn.TransformerEncoderLayer(dm, 8, 4 * dm, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = mlp((dm + d, 1024, 3 * CH))

    def forward(self, h):
        z = self.tr(self.inp(h).unsqueeze(0))[0]
        return self.out(torch.cat([z, h], -1))


class Readout(nn.Module):
    def __init__(self, kind, W0, b0):
        super().__init__()
        self.kind = kind
        self.register_buffer("W0", W0)
        self.register_buffer("b0", b0)
        self.dW = nn.Parameter(torch.zeros_like(W0)) if kind != "deconv0" else None
        self.db = nn.Parameter(torch.zeros_like(b0)) if kind != "deconv0" else None
        self.res = {"mlp_s": lambda: mlp((1024, 1024, 1024, 3 * CH)),
                    "mlp_l": lambda: mlp((1024, 4096, 4096, 4096, 3 * CH)),
                    "moe": lambda: MoE(), "set": lambda: SetHead()}.get(kind, lambda: None)()

    def forward(self, h):
        W = self.W0 if self.dW is None else self.W0 + self.dW
        b = self.b0 if self.db is None else self.b0 + self.db
        r = h @ W.T + b
        if self.res is not None:
            r = r + self.res(h)
        return pos_act(r.view(-1, CH, 3))


def official_pos_rows():
    sd = load_file(str(CHECKPOINT), device="cpu")
    W, b = sd["activation_head.deconv.weight"].float(), sd["activation_head.deconv.bias"].float()
    rows = torch.tensor([p * 14 + c for p in range(CH) for c in range(3)])  # deconv output layout (p c), pos = c 0..2
    return W[rows].contiguous(), b[rows].contiguous()


# ----------------------------------------------------------------------------------------------- loss / metrics
def region(rec, device):
    ti = rec["top_i"].long().to(device)
    pts = rec["pp"].to(device).float()[ti].reshape(ti.shape[0], -1, 3)  # [N, 64*4, 3]
    tw = rec["top_w"].to(device).float()
    # only patches kept by the cumulative-90 % cut are part of the region (both distance directions)
    ok = (rec["pok"].to(device)[ti] & (tw > 0).unsqueeze(-1)).reshape(ti.shape[0], -1)
    w = tw.unsqueeze(-1).expand(-1, -1, 4).reshape(ti.shape[0], -1) * ok
    return pts, ok, w


@torch.no_grad()
def fps_oracle(pts, ok, k=CH):
    """Farthest-point sampling of k region points per token: a GT-only layout of 64 points (k-centre approximation),
    i.e. the coverage a perfect 64-child readout could reach on this region definition."""
    N, M, _ = pts.shape
    d_min = torch.full((N, M), float("inf"), device=pts.device).masked_fill(~ok, -1.0)
    first = ok.float().argmax(-1)
    sel = torch.empty((N, k, 3), device=pts.device)
    cur = pts[torch.arange(N), first]
    for j in range(k):
        sel[:, j] = cur
        d_min = torch.minimum(d_min, (pts - cur.unsqueeze(1)).norm(dim=-1).masked_fill(~ok, -1.0))
        cur = pts[torch.arange(N), d_min.argmax(-1)]
    return sel


def chamfer_terms(x, pts, ok, w):
    """x [N,64,3], pts [N,M,3] (normalised); returns child->region [N,64] and region->child [N,M] distances (m)."""
    d = torch.cdist(x, pts) / S
    c2r = d.masked_fill(~ok.unsqueeze(1), float("inf")).min(-1).values
    r2c = d.min(1).values
    return c2r, r2c


def smooth(x, beta=0.02):
    return torch.where(x < beta, 0.5 * x * x / beta, x - 0.5 * beta)


def depth_jump(pts, ok, w, cam):
    dist = ((pts - cam.unsqueeze(1)).norm(dim=-1) / S).masked_fill(~ok, float("inf"))
    order = dist.argsort(-1)
    ds, ws = dist.gather(-1, order), w.gather(-1, order)
    tot = ws.sum(-1, keepdim=True).clamp_min(1e-12)
    before = ws.cumsum(-1)[:, :-1] / tot
    gap = (ds[:, 1:] - ds[:, :-1]).nan_to_num(posinf=0.0)
    return ((gap > JUMP_M) & (before >= JUMP_W) & (before <= 1 - JUMP_W)).any(-1)


@torch.no_grad()
def evaluate(model, val, device, kind=None):
    """kind: model.kind, or "oracle" (model=None: 64 farthest-point samples of the GT region itself)."""
    kind = kind or model.kind
    acc = {k: [] for k in ("c2r", "op", "r2c", "r2c_w", "any", "views", "jump", "kept")}
    max_dev0 = 0.0
    for rec in val:
        v = rec["valid"].to(device)
        pts, ok, w = region(rec, device)
        if kind == "oracle":
            x = torch.zeros((v.shape[0], CH, 3), device=device)
            x[v] = fps_oracle(pts[v], ok[v])
        else:
            x = model(rec["h"].to(device).float())
        if kind == "deconv0" and "pos0" in rec:
            max_dev0 = max(max_dev0, float((x.cpu() - rec["pos0"]).abs().max()))
        acc["kept"].append(rec["kept_mass"][v.cpu()].float())
        c2r, r2c = chamfer_terms(x[v], pts[v], ok[v], w[v])
        op = rec["opacity"].to(device).float()[v]
        allp = rec["pp"].to(device).float().reshape(-1, 3)[rec["pok"].to(device).reshape(-1)]
        xs = x[v].reshape(-1, 3)
        dany = torch.cat([(torch.cdist(xs[i:i + 8192], allp) / S).min(-1).values for i in range(0, xs.shape[0], 8192)])
        acc["c2r"].append(c2r.cpu())
        acc["op"].append(op.cpu())
        acc["any"].append(dany.view(-1, CH).cpu())
        acc["r2c"].append(r2c.cpu())
        acc["r2c_w"].append(w[v].cpu())
        acc["views"].append(rec["n_views"][v.cpu()])
        acc["jump"].append(depth_jump(pts[v], ok[v], w[v], rec["dom_cam"].to(device)[v]).cpu())
    c2r, op, dany = torch.cat(acc["c2r"]), torch.cat(acc["op"]), torch.cat(acc["any"])
    views, jump = torch.cat(acc["views"]).long(), torch.cat(acc["jump"])
    r2c = torch.cat([x.flatten() for x in acc["r2c"]])
    r2w = torch.cat([x.flatten() for x in acc["r2c_w"]])
    tok_rec5 = torch.cat([((r < 0.05).float() * ww).sum(-1) / ww.sum(-1).clamp_min(1e-12) for r, ww in zip(acc["r2c"], acc["r2c_w"])])
    tok_c2r = c2r.median(-1).values

    def wshare(d, w, thr):
        return float(((d < thr).float() * w).sum() / w.sum())

    def wq(d, w, p):
        o = d.flatten().argsort()
        cw = w.flatten()[o].double().cumsum(0) / w.double().sum()
        i = min(int(torch.searchsorted(cw, torch.tensor(p, dtype=torch.float64))), o.numel() - 1)
        return float(d.flatten()[o][i]) * 100

    def qnt(x, p):
        x = x.flatten()
        return float(x.kthvalue(max(1, int(math.ceil(p * x.numel())))).values)

    def block(mask):
        if int(mask.sum()) == 0:
            return {}
        return {"tokens": int(mask.sum()),
                "child_to_region_cm_p50": qnt(c2r[mask], 0.5) * 100,
                "child_to_region_cm_p90": qnt(c2r[mask], 0.9) * 100,
                "token_median_child_to_region_cm_p50": float(tok_c2r[mask].median()) * 100,
                "region_recall_5cm_token_mean": float(tok_rec5[mask].mean())}

    kept = torch.cat(acc["kept"])
    res = {
        "tokens": int(c2r.shape[0]),
        # primary (opacity-free, permutation-invariant like the loss)
        "child_to_region_cm": {"p50": qnt(c2r, 0.5) * 100, "p90": qnt(c2r, 0.9) * 100},
        "child_to_any_visible_share": {"<3cm": float((dany < 0.03).float().mean()), "<10cm": float((dany < 0.10).float().mean())},
        "region_to_child_cm_attention_weighted": {"p50": wq(r2c, r2w, 0.5), "p90": wq(r2c, r2w, 0.9)},
        "region_recall_3cm": wshare(r2c, r2w, 0.03), "region_recall_5cm": wshare(r2c, r2w, 0.05),
        "by_views": {str(nv): block(views == nv) for nv in (1, 2, 3)} | {"4+": block(views >= 4)},
        "depth_jump": block(jump), "no_depth_jump": block(~jump), "depth_jump_token_share": float(jump.float().mean()),
        "region_kept_attention_mass": {"p10": qnt(kept, 0.1), "p50": qnt(kept, 0.5), "share_below_0.9": float((kept < 0.899).float().mean())},
        # secondary: weighted by the OFFICIAL per-slot opacities (meaningful for deconv0; slot-dependent for trained heads)
        "secondary_child_to_region_opacity_weighted_cm": {"p50": wq(c2r, op, 0.5), "p90": wq(c2r, op, 0.9)},
        "secondary_child_to_any_visible_share_opacity_weighted": {"<3cm": wshare(dany, op, 0.03), "<10cm": wshare(dany, op, 0.10)},
    }
    if kind == "deconv0":
        res["check_max_abs_dev_vs_model_output"] = max_dev0
    return res


# ----------------------------------------------------------------------------------------------- fit / report
def cmd_fit(args):
    gpu = idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    tr, val = load_cache()
    if len(val) != 50:
        raise RuntimeError(f"expected 50 validation windows, got {len(val)}")
    W0, b0 = official_pos_rows()
    fit_subset = [tr[i] for i in torch.randperm(len(tr), generator=torch.Generator().manual_seed(1))[:50].tolist()]
    log = {"head": args.head, "tag": args.tag, "gpu": gpu["index"], "train_windows": len(tr), "epochs": args.epochs,
           "lr": args.lr, "lr_linear": args.lr_linear, "history": []}
    if args.head == "oracle":
        log["history"].append({"epoch": -1, "val": evaluate(None, val, device, "oracle"),
                               "train_fit": evaluate(None, fit_subset, device, "oracle")})
        return write_log(args, log)
    model = Readout(args.head, W0, b0).to(device)
    if args.head != "deconv0":
        lin = [model.dW, model.db]
        res = [p for n, p in model.named_parameters() if n.startswith("res.")]
        groups = [{"params": lin, "lr": args.lr_linear}] + ([{"params": res, "lr": args.lr}] if res else [])
        optim = torch.optim.AdamW(groups, weight_decay=0.0)
        steps = args.epochs * len(tr)
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                    pct_start=min(200 / steps, 0.3), final_div_factor=100)
        g = torch.Generator().manual_seed(0)
        t0 = time.time()
        for ep in range(args.epochs):
            run, n = 0.0, 0
            for wi in torch.randperm(len(tr), generator=g).tolist():
                rec = tr[wi]
                v = rec["valid"].to(device)
                h = rec["h"].to(device).float()[v] if args.head != "set" else rec["h"].to(device).float()
                x = model(h)
                x = x if args.head != "set" else x[v]
                pts, ok, w = region(rec, device)
                c2r, r2c = chamfer_terms(x, pts[v], ok[v], w[v])
                ww = w[v]
                loss = smooth(c2r).mean() + ((smooth(r2c) * ww).sum(-1) / ww.sum(-1).clamp_min(1e-12)).mean()
                if args.head == "moe":
                    loss = loss + 0.01 * model.res.aux
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                run, n = run + float(loss), n + 1
                if n % 200 == 0:
                    print(f"[{args.head}] ep {ep} step {n}/{len(tr)} loss {run / n:.4f} {time.time() - t0:.0f}s", flush=True)
            model.eval()
            ev, ft = evaluate(model, val, device), evaluate(model, fit_subset, device)
            model.train()
            log["history"].append({"epoch": ep, "train_loss": run / n, "val": ev, "train_fit": ft})
            print(f"[{args.head}{args.tag}] epoch {ep}: loss {run / n:.4f} | val child->region p50 "
                  f"{ev['child_to_region_cm']['p50']:.2f} cm, recall@5cm {ev['region_recall_5cm']:.3f} | train-fit "
                  f"{ft['child_to_region_cm']['p50']:.2f} cm, {ft['region_recall_5cm']:.3f}", flush=True)
    else:
        log["history"].append({"epoch": -1, "val": evaluate(model.eval(), val, device),
                               "train_fit": evaluate(model, fit_subset, device)})
    write_log(args, log)


def write_log(args, log):
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"fit_{args.head}{args.tag}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log["history"][-1]["val"], indent=2))


def cmd_report(_args):
    out = {}
    for f in sorted(CACHE.glob("fit_*.json")):
        log = json.loads(f.read_text())
        out[f.stem[4:]] = {"val": log["history"][-1]["val"], "train_fit": log["history"][-1]["train_fit"],
                           "lr": log.get("lr"), "lr_linear": log.get("lr_linear"), "epochs": log.get("epochs"),
                           "val_curve_recall5": [e["val"]["region_recall_5cm"] for e in log["history"]]}
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out_v = {k: v["val"] for k, v in out.items()}
    rows = [("child->region p50/p90 (cm)", lambda r: f"{r['child_to_region_cm']['p50']:.1f} / {r['child_to_region_cm']['p90']:.1f}"),
            ("child <3/<10cm of any visible", lambda r: f"{r['child_to_any_visible_share']['<3cm']:.3f} / {r['child_to_any_visible_share']['<10cm']:.3f}"),
            ("region->child p50/p90 (cm)", lambda r: f"{r['region_to_child_cm_attention_weighted']['p50']:.1f} / {r['region_to_child_cm_attention_weighted']['p90']:.1f}"),
            ("region recall@3/5cm", lambda r: f"{r['region_recall_3cm']:.3f} / {r['region_recall_5cm']:.3f}"),
            ("1-view token child->region p50", lambda r: f"{r['by_views']['1'].get('child_to_region_cm_p50', math.nan):.1f}"),
            ("4+-view token child->region p50", lambda r: f"{r['by_views']['4+'].get('child_to_region_cm_p50', math.nan):.1f}"),
            ("depth-jump tokens recall@5cm", lambda r: f"{r['depth_jump'].get('region_recall_5cm_token_mean', math.nan):.3f}"),
            ("no-jump tokens recall@5cm", lambda r: f"{r['no_depth_jump'].get('region_recall_5cm_token_mean', math.nan):.3f}"),
            ("[secondary] op-weighted child->region p50/p90", lambda r: f"{r['secondary_child_to_region_opacity_weighted_cm']['p50']:.1f} / {r['secondary_child_to_region_opacity_weighted_cm']['p90']:.1f}")]
    print("| metric (validation) | " + " | ".join(out_v) + " |")
    print("|---|" + "---:|" * len(out_v))
    for name, f in rows:
        print(f"| {name} | " + " | ".join(f(r) for r in out_v.values()) + " |")
    print("| train-fit recall@5cm | " + " | ".join(f"{v['train_fit']['region_recall_5cm']:.3f}" for v in out.values()) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1)
    f = sub.add_parser("fit")
    f.add_argument("--head", choices=HEADS, required=True)
    f.add_argument("--tag", default="", help="suffix of the output file, e.g. _lr1e-4 for a learning-rate sweep")
    f.add_argument("--epochs", type=int, default=8)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--lr-linear", type=float, default=3e-5)
    sub.add_parser("report")
    a = ap.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
