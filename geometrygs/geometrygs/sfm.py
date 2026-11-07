"""Known-pose structure from motion on the input views, entirely on the GPU.

Only the selected input views are used: SIFT features are extracted per view, matched pairwise with mutual nearest
neighbours plus a ratio test, filtered by the epipolar constraint of the known poses, chained into tracks and
triangulated with DLT. Tracks are kept only if they are in front of every observing camera, reproject within a
pixel threshold and have a minimum triangulation angle. Camera convention follows TokenGS: OpenCV camera-to-world
matrices and pinhole intrinsics (fx, fy, cx, cy) whose pixel centres sit at integer + 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class SfmConfig:
    num_features: int = 4096
    match_ratio: float = 0.9
    max_epipolar_px: float = 2.0
    max_reproj_px: float = 2.0
    min_tri_angle_deg: float = 1.5
    min_track: int = 2
    z_near: float = 0.02
    z_far: float = 6.0


def _rgb_to_gray(images: torch.Tensor) -> torch.Tensor:
    weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (images * weights).sum(1, keepdim=True)


def _skew(t: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(t[..., 0])
    return torch.stack([
        torch.stack([z, -t[..., 2], t[..., 1]], -1),
        torch.stack([t[..., 2], z, -t[..., 0]], -1),
        torch.stack([-t[..., 1], t[..., 0], z], -1),
    ], -2)


def intrinsics_matrix(intr: torch.Tensor) -> torch.Tensor:
    K = torch.zeros(*intr.shape[:-1], 3, 3, dtype=intr.dtype, device=intr.device)
    K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2], K[..., 2, 2] = intr[..., 0], intr[..., 1], intr[..., 2], intr[..., 3], 1.0
    return K


def fundamental_matrix(w2c_i: torch.Tensor, w2c_j: torch.Tensor, K_i: torch.Tensor, K_j: torch.Tensor) -> torch.Tensor:
    """F such that x_j^T F x_i = 0 for homogeneous pixels of the same 3D point."""
    R_ij = w2c_j[:3, :3] @ w2c_i[:3, :3].T
    t_ij = w2c_j[:3, 3] - R_ij @ w2c_i[:3, 3]
    return torch.linalg.inv(K_j).T @ _skew(t_ij) @ R_ij @ torch.linalg.inv(K_i)


def sampson_distance(F_mat: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
    """First-order geometric epipolar error (pixels) for pixel arrays [M, 2]."""
    hi = F.pad(x_i, (0, 1), value=1.0)
    hj = F.pad(x_j, (0, 1), value=1.0)
    Fx = hi @ F_mat.T
    Ftx = hj @ F_mat
    num = (hj * Fx).sum(-1).square()
    den = Fx[:, 0].square() + Fx[:, 1].square() + Ftx[:, 0].square() + Ftx[:, 1].square()
    return (num / den.clamp_min(1e-12)).sqrt()


def triangulate_dlt(obs_uv: torch.Tensor, obs_mask: torch.Tensor, w2c: torch.Tensor, intr: torch.Tensor) -> torch.Tensor:
    """Linear triangulation of T tracks observed in V views (rows of unobserved views are zeroed).

    Args: obs_uv [T, V, 2] pixels, obs_mask [T, V] bool, w2c [V, 4, 4], intr [V, 4]. Returns points [T, 3].
    """
    xn = (obs_uv[..., 0] - intr[:, 2]) / intr[:, 0]
    yn = (obs_uv[..., 1] - intr[:, 3]) / intr[:, 1]
    P = w2c[:, :3, :]  # normalized camera projection [V, 3, 4]
    rows_x = xn[..., None] * P[:, 2] - P[:, 0]
    rows_y = yn[..., None] * P[:, 2] - P[:, 1]
    A = torch.stack([rows_x, rows_y], 2) * obs_mask[..., None, None]  # [T, V, 2, 4]
    A = A.reshape(A.shape[0], -1, 4).double()
    X = torch.linalg.svd(A).Vh[:, -1]
    return (X[:, :3] / X[:, 3:]).float()


def project(points: torch.Tensor, w2c: torch.Tensor, intr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project [T, 3] world points into V views. Returns uv [T, V, 2] and camera depth [T, V]."""
    pc = torch.einsum("vij,tj->tvi", w2c[:, :3, :3], points) + w2c[:, :3, 3]
    z = pc[..., 2]
    uv = torch.stack([intr[:, 0] * pc[..., 0] / z.clamp_min(1e-9) + intr[:, 2],
                      intr[:, 1] * pc[..., 1] / z.clamp_min(1e-9) + intr[:, 3]], -1)
    return uv, z


def max_triangulation_angle_deg(points: torch.Tensor, obs_mask: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    rays = F.normalize(points[:, None, :] - c2w[:, :3, 3][None], dim=-1)  # [T, V, 3]
    cos = torch.einsum("tvi,twi->tvw", rays, rays).clamp(-1, 1)
    pair = obs_mask[:, :, None] & obs_mask[:, None, :]
    cos = torch.where(pair, cos, torch.ones_like(cos))
    return torch.rad2deg(torch.acos(cos.amin(dim=(1, 2))))


def farthest_point_sample(points: torch.Tensor, k: int, start: int = 0) -> torch.Tensor:
    """Indices of k points chosen greedily to maximize the minimum mutual distance (spatial coverage)."""
    n = points.shape[0]
    k = min(int(k), n)
    idx = torch.empty(k, dtype=torch.long, device=points.device)
    if k == 0:
        return idx
    dist = torch.full((n,), float("inf"), device=points.device)
    cur = torch.tensor(start, device=points.device)
    for i in range(k):
        idx[i] = cur
        dist = torch.minimum(dist, (points - points[cur]).square().sum(-1))
        cur = dist.argmax()
    return idx


class KnownPoseSfM:
    """Plain (non-nn.Module) helper so the SIFT detector never enters a model state dict."""

    def __init__(self, cfg: SfmConfig):
        self.cfg = cfg
        self._sift = {}

    def _detector(self, device: torch.device):
        key = str(device)
        if key not in self._sift:
            import kornia.feature as KF

            self._sift[key] = KF.SIFTFeature(num_features=self.cfg.num_features, upright=False, device=device).eval()
        return self._sift[key]

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        import kornia.feature as KF

        sift = self._detector(images.device)
        gray = _rgb_to_gray(images.float())
        feats = []
        for v in range(gray.shape[0]):
            lafs, _, desc = sift(gray[v:v + 1])
            feats.append((KF.get_laf_center(lafs)[0] + 0.5, desc[0]))  # +0.5: kornia pixel centres are integers
        return feats

    @torch.no_grad()
    def match_pairs(self, feats, c2w: torch.Tensor, intr: torch.Tensor) -> list[tuple[int, int, torch.Tensor]]:
        import kornia.feature as KF

        w2c = torch.linalg.inv(c2w)
        K = intrinsics_matrix(intr)
        pairs = []
        V = len(feats)
        for i in range(V):
            for j in range(i + 1, V):
                (kp_i, d_i), (kp_j, d_j) = feats[i], feats[j]
                if len(kp_i) == 0 or len(kp_j) == 0:
                    continue
                _, idx = KF.match_smnn(d_i, d_j, self.cfg.match_ratio)
                if len(idx) == 0:
                    continue
                Fm = fundamental_matrix(w2c[i].double(), w2c[j].double(), K[i].double(), K[j].double()).float()
                ok = sampson_distance(Fm, kp_i[idx[:, 0]], kp_j[idx[:, 1]]) < self.cfg.max_epipolar_px
                if bool(ok.any()):
                    pairs.append((i, j, idx[ok]))
        return pairs

    @torch.no_grad()
    def build_tracks(self, feats, pairs, num_views: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Union of pairwise matches into tracks; tracks seeing a view twice are discarded as ambiguous."""
        device = feats[0][0].device if feats else torch.device("cpu")
        counts = [len(kp) for kp, _ in feats]
        offsets = np.concatenate([[0], np.cumsum(counts)])
        n_nodes = int(offsets[-1])
        empty = (torch.zeros(0, num_views, 2, device=device), torch.zeros(0, num_views, dtype=torch.bool, device=device))
        if not pairs or n_nodes == 0:
            return empty
        a = np.concatenate([offsets[i] + p[:, 0].cpu().numpy() for i, _, p in pairs])
        b = np.concatenate([offsets[j] + p[:, 1].cpu().numpy() for _, j, p in pairs])
        graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n_nodes, n_nodes))
        _, labels = connected_components(graph, directed=False)
        node_view = np.repeat(np.arange(len(counts)), counts)
        node_kp = np.concatenate([np.arange(c) for c in counts])
        size = np.bincount(labels)
        distinct_views = np.bincount(np.unique(labels.astype(np.int64) * num_views + node_view) // num_views,
                                     minlength=len(size))
        dup = size != distinct_views
        keep_track = (size >= max(2, self.cfg.min_track)) & ~dup
        keep_node = keep_track[labels]
        if not keep_node.any():
            return empty
        track_ids, remap = np.unique(labels[keep_node], return_inverse=True)
        T = len(track_ids)
        obs_uv = torch.zeros(T, num_views, 2, device=device)
        obs_mask = torch.zeros(T, num_views, dtype=torch.bool, device=device)
        nv, nk = node_view[keep_node], node_kp[keep_node]
        kp_all = torch.cat([kp for kp, _ in feats])
        flat = torch.from_numpy(offsets[nv] + nk).to(device)
        t_idx = torch.from_numpy(remap).to(device)
        v_idx = torch.from_numpy(nv).to(device)
        obs_uv[t_idx, v_idx] = kp_all[flat]
        obs_mask[t_idx, v_idx] = True
        return obs_uv, obs_mask

    @torch.no_grad()
    def triangulate(self, obs_uv, obs_mask, c2w, intr) -> dict[str, torch.Tensor]:
        w2c = torch.linalg.inv(c2w)
        cfg = self.cfg
        pts = triangulate_dlt(obs_uv, obs_mask, w2c, intr)
        for _ in range(2):  # drop gross outlier observations once, then re-triangulate
            uv, z = project(pts, w2c, intr)
            err = (uv - obs_uv).norm(dim=-1)
            obs_mask = obs_mask & (err < 2.0 * cfg.max_reproj_px) & (z > cfg.z_near)
            good = obs_mask.sum(1) >= max(2, cfg.min_track)
            obs_uv, obs_mask, pts = obs_uv[good], obs_mask[good], pts[good]
            if pts.shape[0] == 0:
                break
            pts = triangulate_dlt(obs_uv, obs_mask, w2c, intr)
        uv, z = project(pts, w2c, intr)
        err = torch.where(obs_mask, (uv - obs_uv).norm(dim=-1), torch.zeros_like(z))
        in_front = torch.where(obs_mask, (z > cfg.z_near) & (z < cfg.z_far), torch.ones_like(obs_mask))
        angle = max_triangulation_angle_deg(pts, obs_mask, c2w) if pts.shape[0] else pts.new_zeros(0)
        keep = (in_front.all(1) & (err.amax(1) < cfg.max_reproj_px) & (angle > cfg.min_tri_angle_deg)
                & torch.isfinite(pts).all(1))
        return {
            "points": pts[keep],
            "obs_uv": obs_uv[keep],
            "obs_mask": obs_mask[keep],
            "track_len": obs_mask[keep].sum(1),
            "reproj_px": err[keep].amax(1) if keep.any() else pts.new_zeros(0),
            "angle_deg": angle[keep] if keep.any() else pts.new_zeros(0),
        }

    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor, c2w: torch.Tensor, intr: torch.Tensor) -> dict[str, torch.Tensor]:
        """images [V, 3, H, W] in [0, 1]; c2w [V, 4, 4]; intr [V, 4] for the image resolution."""
        c2w, intr = c2w.float(), intr.float()
        feats = self.extract(images)
        pairs = self.match_pairs(feats, c2w, intr)
        obs_uv, obs_mask = self.build_tracks(feats, pairs, images.shape[0])
        out = self.triangulate(obs_uv, obs_mask, c2w, intr) if obs_uv.shape[0] else {
            "points": obs_uv.new_zeros(0, 3), "obs_uv": obs_uv, "obs_mask": obs_mask,
            "track_len": obs_mask.sum(1), "reproj_px": obs_uv.new_zeros(0), "angle_deg": obs_uv.new_zeros(0),
        }
        if out["points"].shape[0]:
            first = out["obs_mask"].float().argmax(1)
            uv = out["obs_uv"][torch.arange(len(first), device=first.device), first]
            out["color"] = self._sample_colors(images, uv, first)
        else:
            out["color"] = out["points"].new_zeros(0, 3)
        out["num_matches"] = torch.tensor(sum(len(p) for _, _, p in pairs))
        return out

    @staticmethod
    def _sample_colors(images: torch.Tensor, uv: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
        H, W = images.shape[-2:]
        grid = torch.stack([uv[:, 0] / W * 2 - 1, uv[:, 1] / H * 2 - 1], -1)
        colors = torch.zeros(len(uv), 3, device=uv.device)
        for v in view.unique().tolist():
            sel = view == v
            colors[sel] = F.grid_sample(images[v:v + 1].float(), grid[sel][None, None], align_corners=False)[0, :, 0].T
        return colors


def anchor_quality_components(sfm: dict[str, torch.Tensor], strong_track: int = 3, strong_angle_deg: float = 3.0,
                              knn: int = 5, max_knn_dist: float = 0.03) -> tuple[torch.Tensor, torch.Tensor]:
    """(strong, dense): seen by >= strong_track views or triangulated with >= strong_angle_deg; and not isolated
    (mean distance to the knn nearest other points < max_knn_dist)."""
    pts = sfm["points"]
    n = pts.shape[0]
    strong = (sfm["track_len"] >= strong_track) | (sfm["angle_deg"] >= strong_angle_deg) if n else \
        torch.zeros(0, dtype=torch.bool, device=pts.device)
    kk = min(knn, n - 1)
    if kk <= 0:
        return strong, torch.zeros(n, dtype=torch.bool, device=pts.device)
    dense = torch.cdist(pts, pts).topk(kk + 1, largest=False).values[:, 1:].mean(1) < max_knn_dist
    return strong, dense


def anchor_quality_mask(sfm: dict[str, torch.Tensor], **quality) -> torch.Tensor:
    """Reliable points (strong and dense). On the ScanNet++ validation windows isolated points are mostly wrong
    triangulations, and farthest point sampling would otherwise pick them first."""
    strong, dense = anchor_quality_components(sfm, **quality)
    return strong & dense


def select_anchors(sfm: dict[str, torch.Tensor], k: int, **quality) -> torch.Tensor:
    """Pick at most k reliable SfM points as fixed token positions: quality filter, then farthest point sampling
    for spatial coverage seeded at the best-observed point (longest track, then smallest reprojection error)."""
    pts = sfm["points"]
    if pts.shape[0] == 0 or k <= 0:
        return torch.zeros(0, dtype=torch.long, device=pts.device)
    cand = anchor_quality_mask(sfm, **quality).nonzero().squeeze(1)
    if cand.numel() == 0:
        return cand
    score = sfm["track_len"][cand].float() - 0.01 * sfm["reproj_px"][cand].float()
    return cand[farthest_point_sample(pts[cand], k, start=int(score.argmax()))]
