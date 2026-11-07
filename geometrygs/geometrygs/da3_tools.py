"""Shared DA3 helpers: load the official DA3 checkout and run one ScanNet++ validation window.

DA3 is pose-conditioned on the six input cameras; depth is put on the input-camera scale by DA3's own Umeyama alignment
of its predicted cameras to the given ones (no laser). The laser is ray-cast on DA3's pixel grid for evaluation only.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

# DA3 benchmark scenes (Depth-Anything-3 src/depth_anything_3/utils/constants.py SCANNETPP_SCENES), excluded from DA3
# training according to the DA3 paper; the paper lists 230 other ScanNet++ training scenes without naming them.
DA3_HELD_OUT = {"09c1414f1b", "1ada7a0617", "40aec5fffa", "3e8bba0176", "acd95847c5", "578511c8a9", "5f99900f09",
                "c4c04e6d6c", "f3d64c30f8", "7bc286c1b6", "c5439f4607", "286b55a2bf", "fb5a96b1a2", "7831862f02",
                "38d58a7a31", "bde1e479ad", "9071e139d9", "21d970d8de", "bcd2436daf", "cc5237fd77"}


def load_da3(da3_root: Path, checkpoint: Path, device):
    sys.path.insert(0, str(da3_root / "src"))
    import depth_anything_3.api as da3_api
    da3_file = Path(da3_api.__file__).resolve()
    if da3_root.resolve() not in da3_file.parents:
        raise RuntimeError(f"imported DA3 from {da3_file}, expected the official checkout {da3_root}")
    return da3_api.DepthAnything3.from_pretrained(str(checkpoint)).to(device).eval(), da3_file


def da3_window(model, provider, index, labels_dir: Path, process_res: int):
    """DA3 depth for one validation window plus the laser depth on the same pixels and the cached label grid."""
    from scannetpp_evaluation_protocol import build_raycasting_scene, raycast_gt_depth_and_normals
    raw = provider[index]
    sid = raw["scene_id"]
    rec = torch.load(labels_dir / f"{sid}.pt", weights_only=False)
    ctx = raw["frame_ids"][:6].tolist()
    if rec["role"] != "test" or rec["input_frames"] != ctx:
        raise RuntimeError(f"{sid}: label grid does not match this window")
    c2w = raw["cam_to_world_input"].double()
    images = [(im.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy() for im in raw["images_sfm"]]
    k = raw["intrinsics_sfm"].double()
    K = torch.zeros(6, 3, 3, dtype=torch.float64)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = k[:, 0], k[:, 1], k[:, 2], k[:, 3], 1
    w2c = torch.linalg.inv(c2w)
    t0 = time.perf_counter()
    pred = model.inference(images, extrinsics=w2c.numpy().astype(np.float32), intrinsics=K.numpy().astype(np.float32),
                           align_to_input_ext_scale=True, process_res=process_res,
                           process_res_method="upper_bound_resize")
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    depth = torch.from_numpy(np.asarray(pred.depth)).float()
    Kp = torch.from_numpy(np.asarray(pred.intrinsics)).float()
    intr = torch.stack((Kp[:, 0, 0], Kp[:, 1, 1], Kp[:, 0, 2], Kp[:, 1, 2]), -1)
    if not np.allclose(np.asarray(pred.extrinsics)[:, :3], w2c.numpy()[:, :3], atol=1e-4):
        raise RuntimeError("DA3 did not keep the input extrinsics")
    h, w = depth.shape[-2:]
    ctx_raw = torch.from_numpy(provider.dataset._scene_meta_cache[sid]["c2ws"][ctx].copy())
    rc = build_raycasting_scene(provider.dataset.root_path / "data" / sid / "scans/mesh_aligned_0.05.ply", ctx_raw, .3)
    laser, _, ok = raycast_gt_depth_and_normals(rc, torch.linalg.inv(c2w.float()).transpose(-1, -2), intr, (h, w))
    return dict(sid=sid, raw=raw, rec=rec, ctx=ctx, c2w=c2w, images=images, pred=pred, secs=secs, depth=depth,
                conf=torch.from_numpy(np.asarray(pred.conf)).float(), intr=intr, h=h, w=w, laser=laser, ok=ok)
