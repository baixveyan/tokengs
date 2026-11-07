"""Scene-latent probe trained WITHOUT the attention centroid: the target is a real laser (GT mesh) surface point.

Per latent the target is the first GT-mesh hit along its attention ray (dominant input camera, attention-weighted
patch-ray direction; both cached by probe_scene_latent_limit.py). No averaging over patches, so the target always
lies on the laser surface.

  targets  raycast the targets for every cached window (CPU, spawn pool)
  fit      train the large MLP and the set Transformer on these targets; save validation predictions to
           <cache>/val_preds_laser.pt; then evaluate with
           probe_scene_latent_region.py surface --pred-file <cache>/val_preds_laser.pt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from probe_scene_latent_limit import CACHE, S, SPLITS, train_pointwise, train_set_probe
from probe_scene_latent_region import load_cache

TARGET_FILE = CACHE / "laser_targets.pt"
PRED_FILE = CACHE / "val_preds_laser.pt"


def _cast(job):
    import open3d as o3d
    from scannetpp_evaluation_protocol import build_raycasting_scene

    key, mesh, raw_c2w, cam, ray = job
    rc = build_raycasting_scene(Path(mesh), torch.from_numpy(raw_c2w), S)
    rays = o3d.core.Tensor(np.concatenate([cam, ray], 1).astype(np.float32))
    return key, rc.cast_rays(rays)["t_hit"].numpy()


def cmd_targets(args):
    from tokengs.data.registry import dataset_registry
    from tokengs.data.static.scannetpp import ScanNetPPVal

    import probe_scene_latent_limit  # noqa: F401  (patches get_context_target_frames with offsets)

    tr, te = load_cache()
    kw = dict(dataset_registry["scannetpp_val"]["kwargs"])
    ds = {sp: ScanNetPPVal(**{**kw, "split": f, "window_size": 18}) for sp, f in SPLITS.items()}
    index = {sp: {Path(p).name: i for i, p in enumerate(d.sample_list)} for sp, d in ds.items()}
    jobs = []
    for w in tr + te:
        d = ds[w["split"]]
        d._offset_frac = None if w["offset"] < 0 else w["offset"]
        ctx, _ = d.get_context_target_frames(index[w["split"]][w["scene_id"]])
        raw = d._scene_meta_cache[w["scene_id"]]["c2ws"][ctx].copy()
        mesh = str(d.root_path / "data" / w["scene_id"] / "scans" / "mesh_aligned_0.05.ply")
        jobs.append(((w["split"], w["scene_id"], round(w["offset"], 4)), mesh, raw, w["cam"].numpy(), w["ray"].numpy()))
    print(f"{len(jobs)} windows", flush=True)
    out, t0 = {}, time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, (key, t) in enumerate(pool.imap_unordered(_cast, jobs, chunksize=4)):
            out[key] = torch.from_numpy(t)
            if (n + 1) % 200 == 0:
                print(f"  {n + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    torch.save(out, TARGET_FILE)
    print("wrote", TARGET_FILE)


def with_laser_target(ws, tg):
    out = []
    for w in ws:
        t = tg[(w["split"], w["scene_id"], round(w["offset"], 4))]
        ok = torch.isfinite(t) & w["valid"]
        tgt = w["cam"] + torch.where(torch.isfinite(t), t, torch.zeros_like(t)).unsqueeze(-1) * w["ray"]
        out.append({**w, "target": tgt, "valid": ok, "orig_valid": w["valid"]})
    return out


def cmd_fit(args):
    from evaluate_scannetpp import idle_gpu

    idle_gpu()
    device = torch.device("cuda:0")
    tr, te = load_cache()
    tg = torch.load(TARGET_FILE, weights_only=False)
    tr, te = with_laser_target(tr, tg), with_laser_target(te, tg)
    X = torch.cat([w["latent"][w["valid"]] for w in tr])
    Y = torch.cat([w["target"][w["valid"]] for w in tr])
    print(f"train latents {len(X)}", flush=True)
    f = train_pointwise(X, Y, device, 2048, 4, args.epochs)
    del X, Y
    preds = {"large_mlp_laser": {}, "set_transformer_laser": {}}
    for w in te:
        preds["large_mlp_laser"][w["scene_id"]] = f(w["latent"][w["orig_valid"]])
    fset = train_set_probe(tr, device, args.set_epochs)
    for w in te:
        preds["set_transformer_laser"][w["scene_id"]] = fset(w)[w["orig_valid"]]
    torch.save(preds, PRED_FILE)
    print("wrote", PRED_FILE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["targets", "fit"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--set-epochs", type=int, default=12)
    args = ap.parse_args()
    cmd_targets(args) if args.mode == "targets" else cmd_fit(args)


if __name__ == "__main__":
    main()
