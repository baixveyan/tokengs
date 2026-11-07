"""A: information ceiling of the frozen scene latents (no model training).

Target (same as the earlier probes): per scene latent, the layer-7 local-head GT attention surface point.
New in this script:
  * more data: every valid latent, 3 windows per training scene (offset fractions 0, 1/3, 2/3), all 50 val windows
  * probe-capacity / data curve: small MLP vs large MLP, 1 window x 1024 latents vs all data
  * set-level probe: a small Transformer that sees all 4096 latents of a window jointly
  * depth-only probe: the viewing ray comes from attention (GT-free); the probe predicts only the distance
  * target noise floor: half-split of the local heads (distance between the two half centroids / 2)
  * tail attribution on validation: error binned by surface planarity, distance, #views, mesh-hole mass, head noise

    PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/probe_scene_latent_limit.py extract --shard k --num-shards N
    PYTHONPATH=scripts:. CUDA_VISIBLE_DEVICES=<g> python scripts/probe_scene_latent_limit.py fit
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from diagnose_latent_attention_geometry import capture_cross_attention
from evaluate_scannetpp import CHECKPOINT, CHECKPOINT_SHA, ROOT, idle_gpu
from probe_scene_latent_position import err_colour, stats, write_points
from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals, sha256_file
from tokengs.data.provider import Provider
from tokengs.data.static.scannetpp import ScanNetPPVal
from tokengs.models import model_registry
from tokengs.models.input_types import split_data
from tokengs.options import config_defaults

CACHE = Path("/mnt/disk/xveyan/workspace_scannetpp/probe_scene_latent_limit")
SPLITS = {"train": "nvs_sem_train.txt", "val": "nvs_sem_val.txt"}
TRAIN_OFFSETS = (1 / 3, 0.0, 2 / 3)
DEMO_SCENE = "7b6477cb95"
S = 0.3

_orig_ctx = ScanNetPPVal.get_context_target_frames


def _ctx_with_offset(self, idx):
    frac = getattr(self, "_offset_frac", None)
    if frac is None:
        return _orig_ctx(self, idx)
    total = self.count_frames(idx)
    win = min(self.window_size, total)
    start = int(round((total - win) * frac))
    end = start + win - 1
    ctx = np.linspace(start, end, self.num_input_views, dtype=int).tolist()
    rem = [f for f in range(start, end + 1) if f not in set(ctx)]
    pick = np.linspace(0, len(rem) - 1, self.num_target_views, dtype=int)
    return ctx, [rem[i] for i in pick]


ScanNetPPVal.get_context_target_frames = _ctx_with_offset


def make_opt(split):
    return dataclasses.replace(
        config_defaults["eval_scannetpp_latent_6view_ssim"],
        evaluating=True, use_ttt_for_eval=False, ttt_n_steps=0, use_geo_losses=False, load_lidar_pc=False,
        use_input_supervision=False, mixed_precision="no", num_workers=0,
        dataset_kwargs={"window_size": 18, "split": SPLITS[split]},
    )


def patch_geometry(rc_scene, cam_in, intr_in, img_size, patch):
    """Per input-view patch centre: GT point, GT world normal, validity, unit ray dir; plus camera centres."""
    d_gt, n_gt, ok = raycast_gt_depth_and_normals(rc_scene, cam_in, intr_in, img_size)
    H, W = int(img_size[0]), int(img_size[1])
    hp, wp = H // patch, W // patch
    ys = torch.arange(hp) * patch + patch // 2
    xs = torch.arange(wp) * patch + patch // 2
    pts, nrm, val, dirs, cams = [], [], [], [], []
    for v in range(cam_in.shape[0]):
        c2w = torch.linalg.inv(cam_in[v].float().T)
        fx, fy, cx, cy = intr_in[v].float().tolist()
        u = (xs.float() + 0.5).view(1, wp).expand(hp, wp)
        vv = (ys.float() + 0.5).view(hp, 1).expand(hp, wp)
        d_cam = torch.stack([(u - cx) / fx, (vv - cy) / fy, torch.ones_like(u)], -1)
        d = d_gt[v][ys][:, xs]
        p_w = (d_cam * d.unsqueeze(-1)) @ c2w[:3, :3].T + c2w[:3, 3]
        n_w = n_gt[v][ys][:, xs] @ c2w[:3, :3].T
        pts.append(p_w.reshape(-1, 3))
        nrm.append(torch.nn.functional.normalize(n_w.reshape(-1, 3), dim=-1))
        val.append(ok[v][ys][:, xs].reshape(-1))
        dirs.append(torch.nn.functional.normalize((d_cam @ c2w[:3, :3].T).reshape(-1, 3), dim=-1))
        cams.append(c2w[:3, 3])
    return torch.cat(pts), torch.cat(nrm), torch.cat(val), torch.cat(dirs), torch.stack(cams)


def extract_window(model, provider, idx, opt, device, layer):
    ds = provider.dataset
    sid = Path(ds.sample_list[idx]).name
    context, _ = ds.get_context_target_frames(idx)
    raw = provider[idx]
    data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
    model_input, _ = split_data(data, opt)
    with torch.inference_mode():
        latents, attn = capture_cross_attention(model, model_input, [layer])
    a = attn[layer].cpu()  # [H, N, P]
    cam_in, intr_in = data["cam_view_input"][0].cpu(), data["intrinsics_input"][0].cpu()
    raw_c2w = torch.from_numpy(ds._scene_meta_cache[sid]["c2ws"][context].copy())
    rc = build_raycasting_scene(ds.root_path / "data" / sid / "scans" / "mesh_aligned_0.05.ply", raw_c2w, S)
    pts, nrm, val, dirs, cams = patch_geometry(rc, cam_in, intr_in, opt.img_size, opt.patch_size)
    V = cams.shape[0]
    P = pts.shape[0] // V
    eff = (-(a.clamp_min(1e-12).log() * a).sum(-1)).exp()
    local = [h for h in range(a.shape[0]) if float(eff[h].median()) < 30]
    al = a[local]  # [L, N, P]
    pv = al * val.float()
    mass = pv.sum(-1)
    cent = (pv / mass.unsqueeze(-1).clamp_min(1e-12)) @ pts  # [L, N, 3]
    target = cent.mean(0)
    spread = cent.std(0).norm(dim=-1) / S
    valid = (mass.min(0).values > 0.5) & (spread < 0.30)
    half_a, half_b = cent[0::2].mean(0), cent[1::2].mean(0)
    head_noise = (half_a - half_b).norm(dim=-1) / 2 / S
    am = al.mean(0)  # [N, V*P]
    view_mass = am.reshape(-1, V, P).sum(-1)
    dom = view_mass.argmax(-1)
    n_views = (view_mass > 0.1).sum(-1)
    w_dom = am.reshape(-1, V, P)[torch.arange(am.shape[0]), dom]  # [N, P]
    dirs_v = dirs.reshape(V, P, 3)[dom]  # [N, P, 3]
    ray = torch.nn.functional.normalize((w_dom.unsqueeze(-1) * dirs_v).sum(1), dim=-1)
    cam = cams[dom]
    w_val = am * val.float()
    w_n = w_val / w_val.sum(-1, keepdim=True).clamp_min(1e-12)
    n_mean = torch.nn.functional.normalize(w_n @ nrm, dim=-1)
    cosd = (nrm.unsqueeze(0) * n_mean.unsqueeze(1)).sum(-1).abs().clamp(max=1.0)  # [N, P]
    planarity_deg = (w_n * torch.rad2deg(torch.acos(cosd))).sum(-1)
    hole_mass = 1 - (am * val.float()).sum(-1) / am.sum(-1).clamp_min(1e-12)
    return {
        "scene_id": sid, "latent": latents[0].half().cpu(), "target": target, "valid": valid,
        "ray": ray, "cam": cam, "head_noise": head_noise.half(), "n_views": n_views.to(torch.int8),
        "planarity_deg": planarity_deg.half(), "hole_mass": hole_mass.half(),
    }


def cmd_extract(args):
    gpu = idle_gpu()
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected checkpoint SHA256")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    prov = {sp: Provider("scannetpp_val_scaled_0.3", make_opt(sp), training=False) for sp in SPLITS}
    tasks = [("val", None, i) for i in range(len(prov["val"]))]
    tasks += [("train", f, i) for f in TRAIN_OFFSETS for i in range(len(prov["train"]))]
    mine = tasks[args.shard::args.num_shards]
    opt = make_opt("val")
    model = model_registry[opt.model_type](opt).eval().requires_grad_(False)
    model.load_state_dict(load_file(str(CHECKPOINT), device="cpu"), strict=True)
    model = model.to(device=device, dtype=torch.float32)
    print(f"GPU {gpu['index']}: shard {args.shard}/{args.num_shards}: {len(mine)} windows", flush=True)
    out, t0 = [], time.time()
    for n, (sp, frac, i) in enumerate(mine):
        prov[sp].dataset._offset_frac = frac
        rec = extract_window(model, prov[sp], i, make_opt(sp), device, args.layer)
        rec.update({"split": sp, "offset": -1.0 if frac is None else float(frac)})
        out.append(rec)
        torch.cuda.empty_cache()
        if (n + 1) % 20 == 0 or n + 1 == len(mine):
            print(f"[shard {args.shard}] {n + 1}/{len(mine)} windows, {time.time() - t0:.0f}s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"shard_{args.shard}.tmp"
    torch.save(out, tmp)
    tmp.rename(CACHE / f"shard_{args.shard}.pt")
    print("wrote", CACHE / f"shard_{args.shard}.pt")


# ----------------------------------------------------------------------------------------- fit


class MLP(nn.Module):
    def __init__(self, din, dout, hidden, layers):
        super().__init__()
        mods, d = [], din
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.05)]
            d = hidden
        mods.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class SetProbe(nn.Module):
    def __init__(self, din=1024, d=384, layers=6, heads=8):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.05, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))

    def forward(self, x):
        return self.out(self.enc(self.inp(x)))


def train_pointwise(X, Y, device, hidden, layers, epochs, batch=8192, lr=1e-3, extra=None):
    """X: fp16 CPU [M,1024]; Y: [M, k] float; extra: optional float CPU [M, e] concatenated after normalisation."""
    mu = X[:200000].float().mean(0, keepdim=True)
    sd = X[:200000].float().std(0, keepdim=True) + 1e-6
    ym, ys = Y.mean(0, keepdim=True), Y.std(0, keepdim=True) + 1e-6
    din = X.shape[1] + (extra.shape[1] if extra is not None else 0)
    torch.manual_seed(0)
    net = MLP(din, Y.shape[1], hidden, layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * (len(X) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    mu_d, sd_d, ym_d, ys_d = mu.to(device), sd.to(device), ym.to(device), ys.to(device)

    def prep(idx_or_x, e=None):
        x = (idx_or_x.to(device, non_blocking=True).float() - mu_d) / sd_d
        if e is not None:
            x = torch.cat([x, e.to(device)], 1)
        return x

    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X) - batch + 1, batch):
            b = perm[i:i + batch]
            x = prep(X[b], extra[b] if extra is not None else None)
            loss = nn.functional.smooth_l1_loss(net(x), (Y[b].to(device) - ym_d) / ys_d, beta=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    net.eval()

    def predict(Xq, Eq=None):
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xq), 65536):
                x = prep(Xq[i:i + 65536], Eq[i:i + 65536] if Eq is not None else None)
                outs.append((net(x) * ys_d + ym_d).cpu())
        return torch.cat(outs)
    return predict


def train_set_probe(train_w, device, epochs, batch=4):
    Ys = torch.cat([w["target"][w["valid"]] for w in train_w[:300]])
    ym, ys = Ys.mean(0).to(device), Ys.std(0).to(device) + 1e-6
    torch.manual_seed(0)
    net = SetProbe().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    steps = epochs * (len(train_w) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    for ep in range(epochs):
        perm = torch.randperm(len(train_w))
        for i in range(0, len(train_w) - batch + 1, batch):
            ws = [train_w[j] for j in perm[i:i + batch].tolist()]
            x = torch.stack([w["latent"] for w in ws]).to(device).float()
            y = (torch.stack([w["target"] for w in ws]).to(device) - ym) / ys
            m = torch.stack([w["valid"] for w in ws]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(x)
            loss = nn.functional.smooth_l1_loss(pred.float()[m], y[m], beta=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        print(f"  set-probe epoch {ep + 1}/{epochs} loss {loss.item():.4f}", flush=True)
    net.eval()

    def predict(w):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return (net(w["latent"].unsqueeze(0).to(device).float())[0].float() * ys + ym).cpu()
    return predict


def err_split(pred, target, cam):
    ray = torch.nn.functional.normalize(target - cam, dim=-1)
    e = pred - target
    along = (e * ray).sum(-1)
    lat = (e - along.unsqueeze(-1) * ray).norm(dim=-1)
    return e.norm(dim=-1) / S, along.abs() / S, lat / S


def summarize(pred, target, cam):
    t, a, l = err_split(pred, target, cam)
    return {"total": stats(t), "depth_along_ray": stats(a), "lateral": stats(l)}


def bin_report(err, key, edges, names):
    out = {}
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = (key >= lo) & (key < hi)
        if int(m.sum()) < 50:
            continue
        e = err[m].numpy()
        out[nm] = {"n_frac": float(m.float().mean()), "depth_median_cm": float(np.median(e) * 100),
                   "depth_p90_cm": float(np.percentile(e, 90) * 100)}
    return out


def cmd_fit(args):
    idle_gpu()
    device = torch.device("cuda:0")
    recs = []
    for p in sorted(CACHE.glob("shard_*.pt")):
        recs.extend(torch.load(p, map_location="cpu", weights_only=False))
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "val"]
    assert len({r["scene_id"] for r in te}) == 50, "need all 50 val windows"
    print(f"train windows {len(tr)} ({len({r['scene_id'] for r in tr})} scenes), val windows {len(te)}", flush=True)

    def flat(ws, key, sub=None):
        parts = []
        for w in ws:
            v = torch.nonzero(w["valid"]).squeeze(1)
            if sub is not None and len(v) > sub:
                g = torch.Generator().manual_seed(int(w["scene_id"], 16) % (2 ** 31))
                v = v[torch.randperm(len(v), generator=g)[:sub]]
            parts.append(w[key][v])
        return torch.cat(parts)

    te_lat, te_tgt, te_cam = flat(te, "latent"), flat(te, "target"), flat(te, "cam")
    te_ray = flat(te, "ray")
    res = {"checkpoint_sha256": CHECKPOINT_SHA, "n_val_latents": int(len(te_lat))}

    # noise floor of the target itself
    hn = flat(te, "head_noise").float()
    res["target_noise_floor"] = stats(hn)

    tr1 = [w for w in tr if abs(w["offset"] - 1 / 3) < 1e-6]
    data_cfgs = {
        "1win_1024": (flat(tr1, "latent", 1024), flat(tr1, "target", 1024)),
        "3win_all": (flat(tr, "latent"), flat(tr, "target")),
    }
    print({k: len(v[0]) for k, v in data_cfgs.items()}, flush=True)
    res["n_train_latents"] = {k: int(len(v[0])) for k, v in data_cfgs.items()}
    capacity = {"small_mlp": (512, 2), "large_mlp": (2048, 4)}
    preds = {}
    for dname, (X, Y) in data_cfgs.items():
        for cname, (h, L) in capacity.items():
            ep = args.epochs if dname == "3win_all" else args.epochs * 3
            t0 = time.time()
            f = train_pointwise(X, Y, device, h, L, ep)
            P = f(te_lat)
            name = f"{cname}@{dname}"
            preds[name] = P
            res[name] = summarize(P, te_tgt, te_cam)
            print(f"{name:24s} total {res[name]['total']['median_cm']:5.1f}/{res[name]['total']['p90_cm']:5.1f}  "
                  f"depth {res[name]['depth_along_ray']['median_cm']:5.1f}/{res[name]['depth_along_ray']['p90_cm']:5.1f}  "
                  f"lat {res[name]['lateral']['median_cm']:4.1f}/{res[name]['lateral']['p90_cm']:4.1f}  ({time.time() - t0:.0f}s)", flush=True)

    # depth-only probe along the attention ray (ray + camera given, GT-free)
    X, _ = data_cfgs["3win_all"]
    t_tr = ((flat(tr, "target") - flat(tr, "cam")) * flat(tr, "ray")).sum(-1, keepdim=True)
    geo_tr = torch.cat([flat(tr, "ray"), flat(tr, "cam")], 1)
    f = train_pointwise(X, t_tr, device, 2048, 4, args.epochs, extra=geo_tr)
    t_hat = f(te_lat, torch.cat([te_ray, te_cam], 1)).squeeze(1)
    t_true = ((te_tgt - te_cam) * te_ray).sum(-1)
    p_depth = te_cam + t_hat.unsqueeze(-1) * te_ray
    res["depth_only_probe"] = {"along_ray_abs": stats((t_hat - t_true).abs() / S),
                               "ray_direction_lateral": stats(((te_cam + t_true.unsqueeze(-1) * te_ray) - te_tgt).norm(dim=-1) / S),
                               **{f"reconstructed_{k}": v for k, v in summarize(p_depth, te_tgt, te_cam).items()}}
    print(f"depth-only: along-ray {res['depth_only_probe']['along_ray_abs']['median_cm']:.1f}/"
          f"{res['depth_only_probe']['along_ray_abs']['p90_cm']:.1f} cm", flush=True)

    # set-level transformer probe
    t0 = time.time()
    fset = train_set_probe(tr, device, args.set_epochs)
    P = torch.cat([fset(w)[w["valid"]] for w in te])
    preds["set_transformer@3win_all"] = P
    res["set_transformer@3win_all"] = summarize(P, te_tgt, te_cam)
    print(f"set-transformer total {res['set_transformer@3win_all']['total']['median_cm']:.1f}/"
          f"{res['set_transformer@3win_all']['total']['p90_cm']:.1f} depth "
          f"{res['set_transformer@3win_all']['depth_along_ray']['median_cm']:.1f}/"
          f"{res['set_transformer@3win_all']['depth_along_ray']['p90_cm']:.1f} ({time.time() - t0:.0f}s)", flush=True)

    # best probe -> tail attribution
    best = min((k for k in preds), key=lambda k: res[k]["depth_along_ray"]["median_cm"] + res[k]["depth_along_ray"]["p90_cm"] / 4)
    res["best_probe"] = best
    _, d_err, _ = err_split(preds[best], te_tgt, te_cam)
    dist = (te_tgt - te_cam).norm(dim=-1) / S
    res["tail_attribution"] = {
        "planarity_deg": bin_report(d_err, flat(te, "planarity_deg").float(), [0, 5, 12, 25, 1e9], ["<5", "5-12", "12-25", ">25"]),
        "distance_m": bin_report(d_err, dist, [0, 1.5, 2.5, 3.5, 1e9], ["<1.5", "1.5-2.5", "2.5-3.5", ">3.5"]),
        "n_views": bin_report(d_err, flat(te, "n_views").float(), [0, 1.5, 2.5, 3.5, 99], ["1", "2", "3", "4+"]),
        "hole_mass": bin_report(d_err, flat(te, "hole_mass").float(), [0, 0.05, 0.2, 1.01], ["<0.05", "0.05-0.2", ">0.2"]),
        "head_noise_m": bin_report(d_err, hn, [0, 0.02, 0.05, 0.10, 1e9], ["<2cm", "2-5cm", "5-10cm", ">10cm"]),
    }
    thr = float(np.percentile(d_err.numpy(), 75))
    tail = d_err > thr
    res["tail_definition"] = f"depth error > p75 = {thr * 100:.1f} cm"
    res["tail_composition"] = {
        "frac_planar_lt12deg": float((flat(te, "planarity_deg").float()[tail] < 12).float().mean()),
        "frac_planar_lt12deg_overall": float((flat(te, "planarity_deg").float() < 12).float().mean()),
        "frac_far_gt3.5m": float((dist[tail] > 3.5).float().mean()),
        "frac_far_gt3.5m_overall": float((dist > 3.5).float().mean()),
        "frac_single_view": float((flat(te, "n_views")[tail] <= 1).float().mean()),
        "frac_single_view_overall": float((flat(te, "n_views") <= 1).float().mean()),
        "frac_head_noise_gt5cm": float((hn[tail] > 0.05).float().mean()),
        "frac_head_noise_gt5cm_overall": float((hn > 0.05).float().mean()),
    }

    # demo PLY for the best probe
    off = 0
    for w in te:
        n = int(w["valid"].sum())
        if w["scene_id"] == DEMO_SCENE:
            P0, T0 = preds[best][off:off + n], te_tgt[off:off + n]
            e0 = (P0 - T0).norm(dim=-1) / S
            rgb = np.array([err_colour(float(e)) for e in e0])
            d = ROOT / "EXP_ScanNetPP" / "demo" / "scene_7b6477cb95" / "baseline_pretrained_6view"
            write_points(d / "baseline_pretrained_6view_scene_latent_probe_best_pred.ply", P0.numpy(), rgb, (e0 * 100).numpy())
            ts = torch.linspace(0, 1, 7).view(1, -1, 1)
            seg = (P0.unsqueeze(1) * (1 - ts) + T0.unsqueeze(1) * ts).reshape(-1, 3)
            write_points(d / "baseline_pretrained_6view_scene_latent_probe_best_links.ply", seg.numpy(),
                         np.repeat(rgb, 7, axis=0), np.repeat((e0 * 100).numpy(), 7), radius=0.0015)
            res["demo_scene_best"] = summarize(P0, T0, te_cam[off:off + n])
        off += n

    out = ROOT / "docs" / "DIAG_scene_latent_limit.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("best:", best)
    print("noise floor:", json.dumps({k: round(v, 3) for k, v in res["target_noise_floor"].items()}))
    print("tail:", res["tail_definition"], json.dumps({k: round(v, 3) for k, v in res["tail_composition"].items()}))
    for k, v in res["tail_attribution"].items():
        print(f"  {k}: " + "  ".join(f"{b}: n={x['n_frac']:.2f} med={x['depth_median_cm']:.1f} p90={x['depth_p90_cm']:.1f}" for b, x in v.items()))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["extract", "fit"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--set-epochs", type=int, default=12)
    args = ap.parse_args()
    cmd_extract(args) if args.mode == "extract" else cmd_fit(args)


if __name__ == "__main__":
    main()
