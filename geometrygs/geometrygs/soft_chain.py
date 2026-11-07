"""Training-only same-surface links. Laser maps define detached edges, never model inputs."""

import math

import torch
from torch.nn import functional as F


def planar_map(depth, normal, valid, intr, tolerance, cos_angle):
    """World-independent local plane test, including slanted surfaces; reject image/depth boundaries."""
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h, device=depth.device) + 0.5,
                          torch.arange(w, device=depth.device) + 0.5, indexing="ij")
    rays = torch.stack([(x-intr[2])/intr[0], (y-intr[3])/intr[1], torch.ones_like(x)], -1)
    points = rays * depth[..., None]
    ok = valid.clone()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = points.roll((dy, dx), (0, 1))
        nn = normal.roll((dy, dx), (0, 1))
        vv = valid.roll((dy, dx), (0, 1))
        ok &= vv & (((neighbor-points)*normal).sum(-1).abs() < tolerance)
        ok &= (normal*nn).sum(-1).abs() > cos_angle
    ok[0] = False
    ok[-1] = False
    ok[:, 0] = False
    ok[:, -1] = False
    return points, ok


def surface_pair_loss(representatives, surface, normals, valid, opt):
    """Neighbors selected on detached laser surfaces, both tangent tests required.

    The common normal displacement is a null direction: this is relative consistency,
    not absolute surface supervision. Tangential spacing is unpenalized on a plane.
    """
    ids = torch.where(valid)[0]
    zero = representatives.sum() * 0
    if len(ids) < 2:
        return zero, 0
    p, q, n = representatives[ids], surface[ids].detach(), normals[ids].detach()
    with torch.no_grad():
        dist = torch.cdist(q, q)
        dist.fill_diagonal_(float("inf"))
        d, j = dist.topk(min(opt.chain_neighbors, len(ids)-1), largest=False)
        delta = q[:, None] - q[j]
        dot = (n[:, None] * n[j]).sum(-1)
        edges = (d < opt.chain_radius_m * opt.scene_unit_scale)
        edges &= dot.abs() > math.cos(math.radians(opt.chain_normal_angle_deg))
        tol = opt.chain_plane_tolerance_m * opt.scene_unit_scale
        edges &= (delta * n[:, None]).sum(-1).abs() < tol
        edges &= (delta * n[j]).sum(-1).abs() < tol
        # Undirected edges once only.
        edges &= ids[:, None] < ids[j]
        mean_n = F.normalize(n[:, None] + n[j] * dot.sign()[..., None], dim=-1)
    if not edges.any():
        return zero, 0
    residual = ((p[:, None] - p[j]) * mean_n).sum(-1)[edges] / opt.scene_unit_scale
    loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=opt.chain_huber_m, reduction="sum")
    return loss, int(edges.sum())


def soft_chain_loss(gaussians, data, opt):
    """Build a local representative from actual children around each matched laser point.

    Per view, the highest-opacity child proposes a pixel; nearby children form the
    representative. All matching, masks and opacity weights are detached to prevent
    closing links by lowering opacity. No GT is consulted during reconstruction.
    """
    for key in ("depths_gt_sup", "normals_gt_sup", "valid_gt_sup"):
        if key not in data:
            raise ValueError(f"soft chain requires training supervision: {key}")
    b, _, _ = gaussians.shape
    children = opt.dec_patch_size ** 2
    g = gaussians.reshape(b, opt.num_gs_tokens, children, 14)
    views = data["cam_view_all"] if opt.use_input_supervision else data["cam_view"]
    intr = data["intrinsics_all"] if opt.use_input_supervision else data["intrinsics"]
    total, count = gaussians.sum() * 0, 0
    cos_angle = math.cos(math.radians(opt.chain_normal_angle_deg))
    for ib in range(b):
        xyz, alpha = g[ib, ..., :3], g[ib, ..., 3].detach()
        with torch.no_grad():
            tip = xyz.detach()[torch.arange(opt.num_gs_tokens, device=g.device), alpha.argmax(-1)]
        for iv in range(views.shape[1]):
            with torch.no_grad():
                depth = data["depths_gt_sup"][ib, iv].float()
                normal = data["normals_gt_sup"][ib, iv].float()
                valid = data["valid_gt_sup"][ib, iv].bool() & (depth > 0)
                h, w = depth.shape
                w2c = views[ib, iv].T.float()
                cam = tip @ w2c[:3, :3].T + w2c[:3, 3]
                k = intr[ib, iv]
                uv = cam[:, :2] / cam[:, 2:].clamp_min(1e-6) * k[:2] + k[2:]
                pixels = uv.floor().long()
                inside = (cam[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
                inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
                px, py = pixels[:, 0].clamp(0, w-1), pixels[:, 1].clamp(0, h-1)
                surface_cam, plane = planar_map(depth, normal, valid, k,
                                                opt.chain_plane_tolerance_m * opt.scene_unit_scale, cos_angle)
                c2w = torch.linalg.inv(w2c)
                q = surface_cam[py, px] @ c2w[:3, :3].T + c2w[:3, 3]
                n = F.normalize(normal[py, px] @ c2w[:3, :3].T, dim=-1)
                close = (xyz.detach() - q[:, None]).norm(dim=-1) < opt.chain_match_m * opt.scene_unit_scale
                weight = alpha * close
                weight_sum = weight.sum(-1, keepdim=True)
                keep = inside & plane[py, px] & (weight_sum[:, 0] > 1e-5)
                weight = weight / weight_sum.clamp_min(1e-5)
            representative = (xyz * weight[..., None]).sum(1)
            loss, edges = surface_pair_loss(representative, q, n, keep, opt)
            total = total + loss
            count += edges
    return total / max(count, 1), count
