"""Small grid diagnostics. Laser information creates labels, never predictor inputs."""

import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_grid(resolution=16, extent_m=6., scale=.3, device="cpu"):
    lower = torch.tensor([-extent_m / 2, -extent_m / 2, 0.], device=device) * scale
    step = extent_m * scale / resolution
    xyz = torch.stack(torch.meshgrid(*[torch.arange(resolution, device=device)] * 3, indexing="ij"), -1)
    return lower + (xyz.reshape(-1, 3) + .5) * step, lower, step


def bin_points(points, lower, step, resolution):
    ijk = torch.floor((points - lower) / step).long()
    valid = ((ijk >= 0) & (ijk < resolution)).all(-1) & torch.isfinite(points).all(-1)
    ids = (ijk[..., 0] * resolution + ijk[..., 1]) * resolution + ijk[..., 2]
    return ids, valid


def project(points, c2w, intr, size):
    """Pixel-edge coordinates: pixel centres are (u+.5,v+.5), align_corners=False."""
    w2c = torch.linalg.inv(c2w)
    cam = torch.einsum("vij,nj->vni", w2c[:, :3, :3], points) + w2c[:, None, :3, 3]
    z = cam[..., 2]
    uv = cam[..., :2] / z.clamp_min(1e-6)[..., None]
    uv = uv * intr[:, None, :2] + intr[:, None, 2:]
    h, w = size
    grid = uv / uv.new_tensor([w, h]) * 2 - 1
    valid = (z > 0) & (grid.abs() < 1).all(-1)
    return grid, z, valid


def backproject(depth, intr, c2w):
    v, h, w = depth.shape
    kw = dict(device=depth.device, dtype=depth.dtype)
    yy, xx = torch.meshgrid(torch.arange(h, **kw) + .5, torch.arange(w, **kw) + .5, indexing="ij")
    uv = torch.stack((xx, yy), -1)
    xy = (uv[None] - intr[:, None, None, 2:]) / intr[:, None, None, :2]
    cam = torch.cat((xy, torch.ones(v, h, w, 1, **kw)), -1) * depth[..., None]
    return torch.einsum("vij,vhwj->vhwi", c2w[:, :3, :3], cam) + c2w[:, None, None, :3, 3]


def grid_labels(centers, lower, step, resolution, depth, valid_depth, normals, intr, c2w, mesh_dist):
    """Surface=observed hit in cell; free=observed before surface AND conservative mesh-distance test.

    The circumsphere test excludes any mesh triangle within the voxel. Occluded,
    unseen, and unconfirmed near-surface cells stay unknown (-1).
    """
    n = len(centers)
    points = backproject(depth, intr, c2w)
    ids, inside = bin_points(points, lower, step, resolution)
    hits = valid_depth & inside
    count = torch.bincount(ids[hits], minlength=n)
    uv, z, in_frame = project(centers, c2w, intr, depth.shape[-2:])
    sampled = F.grid_sample(depth[:, None], uv[:, None], mode="nearest", align_corners=False)[:, 0, 0]
    radius = step * (3 ** .5) / 2
    free_votes = in_frame & (sampled > 0) & (z + radius + .003 < sampled)
    labels = torch.full((n,), -1, dtype=torch.long)
    labels[(free_votes.sum(0) >= 2) & (mesh_dist > radius + .003)] = 0
    labels[count > 0] = 1
    # Normal sign-invariant dispersion; zero on a perfectly planar surface.
    world_n = torch.einsum("vij,vhwj->vhwi", c2w[:, :3, :3], normals)
    outer = (world_n[..., :, None] * world_n[..., None, :]).reshape(-1, 9)
    moments = torch.zeros(n, 9)
    moments.index_add_(0, ids[hits], outer[hits.reshape(-1)])
    moments = moments.reshape(n, 3, 3) / count.clamp_min(1)[:, None, None]
    dispersion = (1 - torch.linalg.eigvalsh(moments)[:, -1]).clamp(0, 1)
    return labels, count, dispersion, points, hits, ids


@torch.no_grad()
def sample_grid_features(features, centers, lower, step, c2w, intr, size):
    """Nine samples/cell, frozen image features, camera-only valid masks. No depth/mesh inputs."""
    offsets = centers.new_tensor([[0, 0, 0]] + list(itertools.product((-.4, .4), repeat=3)))
    points = (centers[:, None] + offsets[None] * step).reshape(-1, 3)
    uv, z, valid = project(points, c2w, intr, size)
    sampled = F.grid_sample(features, uv[:, None], align_corners=False)[:, :, 0].permute(2, 0, 1)
    # [cells, samples, views, channels], pool appearance and cross-view dispersion.
    n, v, d = len(centers), len(c2w), features.shape[1]
    sampled = sampled.reshape(n, 9, v, d)
    mask = valid.T.reshape(n, 9, v, 1)
    denom = mask.sum(2).clamp_min(1)
    mean = (sampled * mask).sum(2) / denom
    variance = ((sampled - mean[:, :, None]).square() * mask).sum(2) / denom
    evidence = torch.cat((mean.mean(1), variance.mean(1).sqrt()), -1)
    uv0, z0, ok0 = project(centers, c2w, intr, size)
    geometric = torch.cat(((centers - lower) / (step * round(n ** (1 / 3))),
                           uv0.clamp(-2, 2).permute(1, 0, 2).flatten(1),
                           z0.T / 1.8, ok0.T.float(), mask.float().mean((1, 2))), -1)
    return torch.cat((geometric, evidence), -1), geometric.shape[1]


class GridClassifier(nn.Module):
    def __init__(self, mean, std, hidden=(128, 64)):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(.05))
        dims = [len(mean), *hidden]
        layers = [m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.GELU())]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], 1))

    def forward(self, x):
        return self.net(((x - self.mean) / self.std).clamp(-20, 20)).squeeze(-1)


def bbox_grid(points, cell, margin=1):
    """Axis-aligned grid with fixed metric cell size around the given (input-visible) points."""
    lo = points.min(0).values - margin * cell
    dims = torch.ceil((points.max(0).values + margin * cell - lo) / cell).long()
    axes = [torch.arange(int(n), device=points.device) for n in dims]
    ijk = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return lo + (ijk + .5) * cell, lo, dims


def grid_index(points, lo, cell, dims):
    ijk = torch.floor((points - lo) / cell).long()
    valid = ((ijk >= 0) & (ijk < dims)).all(-1) & torch.isfinite(points).all(-1)
    ijk = torch.minimum(ijk.clamp_min(0), dims - 1)
    return (ijk[..., 0] * dims[1] + ijk[..., 1]) * dims[2] + ijk[..., 2], valid


def cell_counts(points, lo, cell, dims):
    ids, valid = grid_index(points, lo, cell, dims)
    return torch.bincount(ids[valid], minlength=int(dims.prod()))


def dilate(mask, dims, radius):
    volume = mask.reshape(*[int(n) for n in dims])[None, None].float()
    return F.max_pool3d(volume, 2 * radius + 1, 1, radius)[0, 0].reshape(-1) > 0


def ray_crossed_cells(origins, dirs, lo, cell, dims, near, chunk=8192):
    """Cells crossed by rays (inside the grid box, beyond `near`), sampled every half cell."""
    hi = lo + dims * cell
    dirs = F.normalize(dirs, dim=-1)
    safe = torch.where(dirs.abs() < 1e-9, torch.full_like(dirs, 1e-9), dirs)
    t_a, t_b = (lo - origins) / safe, (hi - origins) / safe
    t0 = torch.minimum(t_a, t_b).max(-1).values.clamp_min(near)
    t1 = torch.maximum(t_a, t_b).min(-1).values
    keep = t1 > t0
    origins, dirs, t0, t1 = origins[keep], dirs[keep], t0[keep], t1[keep]
    crossed = torch.zeros(int(dims.prod()), dtype=torch.bool, device=lo.device)
    step = cell / 2
    for i in range(0, len(t0), chunk):
        o, d, a, b = origins[i:i + chunk], dirs[i:i + chunk], t0[i:i + chunk], t1[i:i + chunk]
        k = torch.arange(int(torch.ceil((b - a).max() / step)) + 1, device=lo.device)
        t = a[:, None] + (k[None] + .5) * step
        ok = t < b[:, None]
        ids, inside = grid_index(o[:, None] + t[..., None] * d[:, None], lo, cell, dims)
        crossed[ids[ok & inside]] = True
    return crossed


@torch.no_grad()
def occupancy_features(feature_maps, centers, cell, c2w, intr, size, znear, scale, chunk=16384):
    """Per-cell inputs from RGB features + cameras only.

    Geometry (28): mean-camera position in metres, per-view (u, v, depth, in-view), visible-sample fraction.
    Evidence: mask-weighted view mean and view std of frozen features at 9 points (centre + 8 inner corners).
    """
    offsets = centers.new_tensor([[0, 0, 0]] + list(itertools.product((-.4, .4), repeat=3))) * cell
    out = []
    for c in centers.split(chunk):
        n, v = len(c), len(c2w)
        uv, z, valid = project((c[:, None] + offsets).reshape(-1, 3), c2w, intr, size)
        valid = valid & (z > znear)
        feats = F.grid_sample(feature_maps, uv[:, None], align_corners=False)[:, :, 0].permute(2, 0, 1)
        feats = feats.reshape(n, 9, v, -1)
        mask = valid.T.reshape(n, 9, v, 1).float()
        denom = mask.sum(2).clamp_min(1)
        mean = (feats * mask).sum(2) / denom
        std = (((feats - mean[:, :, None]).square() * mask).sum(2) / denom).sqrt()
        seen = (mask.sum(2) > 0).float()
        w = seen / seen.sum(1, keepdim=True).clamp_min(1)
        evidence = torch.cat(((mean * w).sum(1), (std * w).sum(1)), -1)
        uv0, z0, ok0 = project(c, c2w, intr, size)
        ok0 = ok0 & (z0 > znear)
        geo = torch.cat((c / scale, uv0.clamp(-2, 2).permute(1, 0, 2).flatten(1),
                         (z0.T / scale / 5).clamp(-2, 2), ok0.T.float(), mask.mean((1, 2))), -1)
        out.append(torch.cat((geo, evidence), -1))
    return torch.cat(out)


def recall_threshold(scores, labels, recall):
    positive = np.sort(np.asarray(scores)[np.asarray(labels) == 1])
    if not len(positive):
        raise ValueError("calibration split has no surface cells")
    return float(positive[int(np.floor((1 - recall) * len(positive)))])


def classification_metrics(scores, labels, threshold):
    scores, labels = np.asarray(scores), np.asarray(labels)
    keep = scores >= threshold
    pos, neg, unknown = labels == 1, labels == 0, labels < 0
    ratio = lambda mask, denom: float(mask.sum() / max(1, denom.sum()))
    return dict(surface_recall=ratio(keep & pos, pos), free_closure=ratio(~keep & neg, neg),
                surface_precision=float((keep & pos).sum() / max(1, (keep & (pos | neg)).sum())),
                known_kept_fraction=ratio(keep & (pos | neg), pos | neg),
                unknown_closed_fraction=ratio(~keep & unknown, unknown),
                surface_cells=int(pos.sum()), free_cells=int(neg.sum()), unknown_cells=int(unknown.sum()))
