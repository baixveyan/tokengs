# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry losses that organize the children emitted by each Gaussian token."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_EPS = 1e-6
_MASS_WEIGHTINGS = ("opacity", "opacity_area")
_INTER_FORMS = ("squared", "log_smooth_l1")


def _validate_inputs(
    gs_centers: torch.Tensor,
    gs_scales: torch.Tensor,
    gs_opacities: torch.Tensor,
    gs_rotations: torch.Tensor | None,
) -> None:
    if gs_centers.ndim != 4 or gs_centers.shape[-1] != 3:
        raise ValueError(f"gs_centers must have shape (B, S, C, 3), got {tuple(gs_centers.shape)}")
    if gs_scales.shape != gs_centers.shape:
        raise ValueError(
            f"gs_scales must match gs_centers shape {tuple(gs_centers.shape)}, "
            f"got {tuple(gs_scales.shape)}"
        )
    if gs_opacities.shape != (*gs_centers.shape[:-1], 1):
        raise ValueError(
            f"gs_opacities must have shape {(*gs_centers.shape[:-1], 1)}, "
            f"got {tuple(gs_opacities.shape)}"
        )
    if gs_rotations is not None and gs_rotations.shape != (*gs_centers.shape[:-1], 4):
        raise ValueError(
            f"gs_rotations must have shape {(*gs_centers.shape[:-1], 4)}, "
            f"got {tuple(gs_rotations.shape)}"
        )
    if gs_centers.shape[2] < 2:
        raise ValueError("at least two children per slot are required")
    tensors = [gs_centers, gs_scales, gs_opacities]
    if gs_rotations is not None:
        tensors.append(gs_rotations)
    if not torch.stack([torch.isfinite(tensor).all() for tensor in tensors]).all():
        raise ValueError("geometry inputs must be finite (no NaN/Inf)")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_values = torch.where(mask, values, torch.zeros_like(values))
    denominator = mask.to(dtype=values.dtype).sum().clamp_min(1.0)
    return masked_values.sum() / denominator


def _opacity_mass_weights(opacities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize detached activated opacity mass and identify active slots.

    Slots with total mass at or below ``_EPS`` receive all-zero weights. Their
    geometry terms are skipped rather than assigning transparent children a
    synthetic uniform mass.
    """
    opacity_mass = opacities.detach().squeeze(-1).clamp_min(0.0)
    total_mass = opacity_mass.sum(dim=2, keepdim=True)
    active_slots = total_mass.squeeze(-1) > _EPS
    weights = torch.where(
        active_slots.unsqueeze(-1),
        opacity_mass / total_mass.clamp_min(_EPS),
        torch.zeros_like(opacity_mass),
    )
    return weights, active_slots


def _opacity_area_mass_weights(
    opacities: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FFRGS 质量 ``opacity * (prod scale)^(2/3)``，detach 后按 slot 归一化。

    active 判定仍按 opacity 质量（与 ``_opacity_mass_weights`` 相同），避免
    体积项很小时把正常 slot 误判为透明；透明 child 的质量严格为 0。
    """
    _, active_slots = _opacity_mass_weights(opacities)
    opacity_mass = opacities.detach().squeeze(-1).clamp_min(0.0)
    area = scales.detach().prod(dim=-1).clamp_min(1e-18).pow(2.0 / 3.0)
    mass = opacity_mass * area
    total_mass = mass.sum(dim=2, keepdim=True)
    weights = torch.where(
        active_slots.unsqueeze(-1) & (total_mass > 0),
        mass / total_mass.clamp_min(torch.finfo(mass.dtype).tiny),
        torch.zeros_like(mass),
    )
    return weights, active_slots & (total_mass.squeeze(-1) > 0)


def _quaternion_wxyz_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert normalized Hamilton-order ``[w, x, y, z]`` quaternions to matrices."""
    norms = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    identity_quaternion = torch.zeros_like(quaternions)
    identity_quaternion[..., 0] = 1.0
    quaternions = torch.where(
        norms > _EPS,
        quaternions / norms.clamp_min(_EPS),
        identity_quaternion,
    )
    w, x, y, z = quaternions.unbind(dim=-1)
    matrix = torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x.square() + y.square()),
        ),
        dim=-1,
    )
    return matrix.reshape(*quaternions.shape[:-1], 3, 3)


def _world_covariances(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    rotation_matrices = _quaternion_wxyz_to_rotation_matrix(rotations)
    local_covariances = torch.diag_embed(scales.square())
    return rotation_matrices @ local_covariances @ rotation_matrices.transpose(-1, -2)


def _aggregate_parent_covariance(
    center_covariance: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    child_covariances = _world_covariances(scales, rotations)
    weighted_child_covariance = (
        weights[..., None, None] * child_covariances
    ).sum(dim=2)
    return center_covariance + weighted_child_covariance


def _fixed_sphere_directions(
    count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic approximately uniform directions on the unit sphere."""
    if count <= 0:
        raise ValueError("num_sw_directions must be positive")

    indices = torch.arange(count, dtype=dtype, device=device)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    azimuth = indices * (math.pi * (3.0 - math.sqrt(5.0)))
    return torch.stack(
        (radius * torch.cos(azimuth), radius * torch.sin(azimuth), z),
        dim=-1,
    )


def _weighted_slot_moments(
    centers: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weighted_mean = (weights.unsqueeze(-1) * centers).sum(dim=2)
    centered = torch.where(
        weights.unsqueeze(-1) > 0,
        centers - weighted_mean.unsqueeze(2),
        torch.zeros_like(centers),
    )
    covariance = torch.einsum(
        "bsci,bscj,bsc->bsij",
        centered,
        centered,
        weights,
    )
    return weighted_mean, centered, covariance


def _stable_cholesky(covariance: torch.Tensor) -> torch.Tensor:
    """对半正定矩阵使用绝对及尺度相关的对角正则，不掩盖非有限值。"""
    if not torch.isfinite(covariance).all():
        raise ValueError("geometry covariance must be finite (no NaN/Inf)")
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    eye = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    magnitude = covariance.detach().abs().amax(dim=(-2, -1), keepdim=True)
    jitter = _EPS + 8.0 * torch.finfo(covariance.dtype).eps * magnitude
    cholesky, info = torch.linalg.cholesky_ex(covariance + jitter * eye)
    if not info.any():
        return cholesky

    # 特征值仅决定数值修复量；避免重复特征值处的特征向量反向不稳定。
    with torch.no_grad():
        minimum = torch.linalg.eigvalsh(covariance.detach()).amin(dim=-1)
        repair = (-minimum).clamp_min(0.0)[..., None, None] + 10.0 * jitter
        jitter = torch.where((info != 0)[..., None, None], repair, jitter)
    return torch.linalg.cholesky(covariance + jitter * eye)


def _weighted_normal_wasserstein(
    projections: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """逐方向精确计算离散加权分布到 N(0,1) 的 W2 平方。

    输入分别为 (B,S,C,D) 投影与 (B,S,C) detached 质量；零质量点
    不进入排序支持集。以累计质量区间积分，而非等权分位点近似。
    """
    mass = weights.detach().unsqueeze(-1).expand_as(projections)
    positive = mass > 0
    sortable = torch.where(positive, projections, torch.full_like(projections, float("inf")))
    sorted_values, order = sortable.sort(dim=2)
    sorted_mass = mass.gather(2, order)
    sorted_values = torch.where(sorted_mass > 0, sorted_values, torch.zeros_like(sorted_values))
    # 两端分别累加原始质量，直接取较小尾概率；即便质量小于 FP64 eps，
    # 也不计算会舍入成 1 的 1-small。排序、投影、loss 及其反向保持输入 dtype。
    total_mass = weights.detach().double().sum(dim=2, keepdim=True).unsqueeze(-1)
    denominator = torch.where(total_mass > 0, total_mass, torch.ones_like(total_mass))
    prefix = sorted_mass.cumsum(dim=2, dtype=torch.float64)
    suffix = sorted_mass.flip((2,)).cumsum(dim=2, dtype=torch.float64).flip((2,))
    tail_probability = torch.minimum(prefix[:, :, :-1, :], suffix[:, :, 1:, :]) / denominator
    del prefix, suffix

    # φ(Φ⁻¹(p)) = φ(Φ⁻¹(1-p))。只求 C-1 个内部边界，两个无限尾端
    # 使用解析极限 φ(±∞)=0；零尾概率替换为 1/2 后掩码，绝不向 ndtri 传 0/1。
    positive_tail = tail_probability > 0
    safe_tail = torch.where(positive_tail, tail_probability, 0.5)
    quantiles = torch.special.ndtri(safe_tail)
    density = torch.exp(-0.5 * quantiles.square()) / math.sqrt(2.0 * math.pi)
    density = torch.where(positive_tail, density, 0.0)
    endpoint = torch.zeros_like(density[:, :, :1, :])
    boundary_density = torch.cat((endpoint, density, endpoint), dim=2)

    # ∫_a^b Φ⁻¹(u)du = φ(Φ⁻¹(a)) - φ(Φ⁻¹(b)); ∫_0^1 Φ⁻¹(u)²du = 1。
    normal_first_moment = (boundary_density[:, :, :-1, :] - boundary_density[:, :, 1:, :]).to(projections.dtype)
    # 原子二阶项必须使用真实归一化质量，不能从相邻累计边界做差反推。
    normalized_mass = (sorted_mass.double() / denominator).to(projections.dtype)
    per_direction = 1.0 + (
        normalized_mass * sorted_values.square()
        - 2.0 * sorted_values * normal_first_moment
    ).sum(dim=2)
    # 无 clamp 截尾偏差。令 γ_n=nε/(1-nε)，非负 FP64 累加相对误差
    # ≤ γ_C（C=64 时 <1.43e-14），归一化尾概率相对误差 ≤ γ_(2C+2)。
    # 密度误差由区间内 sup|Φ⁻¹(p)|·|δp| 界定，另计 ndtri/exp 库函数舍入。
    # 无溢出/下溢时，FP32 算术绝对误差 ≤ γ_(C+8)·(1+Σ(w*x²+2|x*I|))，
    # 另计 FP64 系数误差；累计/积分全部 detached，不建立 FP64 反向图。
    return torch.where(total_mass.squeeze(2) > 0, per_direction.clamp_min(0.0), torch.zeros_like(per_direction))


def _sliced_wasserstein_loss(
    centered: torch.Tensor,
    covariance: torch.Tensor,
    active_slots: torch.Tensor,
    num_directions: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    centered = torch.where(
        active_slots[..., None, None], centered, torch.zeros_like(centered)
    )
    covariance = torch.where(
        active_slots[..., None, None], covariance, torch.zeros_like(covariance)
    )
    cholesky = _stable_cholesky(covariance)
    whitened = torch.linalg.solve_triangular(
        cholesky,
        centered.transpose(-1, -2),
        upper=False,
    ).transpose(-1, -2)

    directions = _fixed_sphere_directions(
        num_directions,
        dtype=centered.dtype,
        device=centered.device,
    )
    projections = torch.einsum("bsci,di->bscd", whitened, directions)
    per_slot_loss = _weighted_normal_wasserstein(projections, weights).mean(dim=-1)
    return _masked_mean(per_slot_loss, active_slots)


def _packing_loss(
    covariance: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    active_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    center_variance = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    local_scale_variance = (
        weights * scales.detach().square().mean(dim=-1)
    ).sum(dim=2)
    raw_packing_ratio = center_variance.clamp_min(_EPS) / local_scale_variance.clamp_min(_EPS)
    effective_children = torch.where(
        active_slots,
        weights.square().sum(dim=2).clamp_min(_EPS).reciprocal(),
        torch.zeros_like(center_variance),
    )
    packing_ratio = torch.where(active_slots, raw_packing_ratio, torch.ones_like(raw_packing_ratio))
    packing_target = torch.where(
        active_slots,
        effective_children.pow(2.0 / 3.0),
        torch.ones_like(effective_children),
    )
    per_slot_loss = (
        torch.log(packing_ratio.clamp_min(_EPS))
        - torch.log(packing_target.clamp_min(_EPS))
    ).square()
    loss = _masked_mean(per_slot_loss, active_slots)
    return (
        loss,
        packing_ratio,
        packing_target,
        center_variance,
        local_scale_variance,
        effective_children,
    )


def _inter_slot_loss(
    parent_means: torch.Tensor,
    parent_covariance: torch.Tensor,
    active_slots: torch.Tensor,
    k_neighbors: int,
    target_distance: float,
    inter_form: str = "log_smooth_l1",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, num_slots, _ = parent_means.shape
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    if target_distance <= 0:
        raise ValueError("target_distance must be positive")
    if inter_form not in _INTER_FORMS:
        raise ValueError(f"inter_form must be one of {_INTER_FORMS}, got {inter_form!r}")
    if num_slots < 2:
        zero = parent_means.sum() * 0.0
        stats = {key: zero.detach() for key in ("mean", "median", "too_close", "too_far")}
        return zero, stats

    knn_count = min(k_neighbors, num_slots - 1)
    with torch.no_grad():
        pairwise_distance = torch.cdist(parent_means.detach(), parent_means.detach())
        diagonal = torch.eye(num_slots, dtype=torch.bool, device=parent_means.device)
        valid_edges = (
            active_slots.unsqueeze(-1)
            & active_slots.unsqueeze(-2)
            & ~diagonal.unsqueeze(0)
        )
        pairwise_distance.masked_fill_(~valid_edges, float("inf"))
        neighbor_indices = pairwise_distance.topk(knn_count, dim=-1, largest=False).indices
        valid_neighbors = torch.gather(
            torch.isfinite(pairwise_distance), 2, neighbor_indices
        )
        del pairwise_distance

    batch_indices = torch.arange(batch_size, device=parent_means.device).view(-1, 1, 1)
    neighbor_means = parent_means[batch_indices, neighbor_indices]
    neighbor_covariance = parent_covariance[batch_indices, neighbor_indices]

    delta = parent_means.unsqueeze(2) - neighbor_means
    pair_covariance = 0.5 * (
        parent_covariance.unsqueeze(2) + neighbor_covariance
    )
    solved_delta = torch.cholesky_solve(
        delta.unsqueeze(-1), _stable_cholesky(pair_covariance)
    ).squeeze(-1)
    mahalanobis_sq = (delta * solved_delta).sum(dim=-1).clamp_min(_EPS)
    mahalanobis_distance = torch.sqrt(mahalanobis_sq)
    if inter_form == "squared":
        per_pair_loss = (mahalanobis_distance - target_distance).square()
    else:
        # FFRGS models/losses.py::inter_slot_loss：对数比的 smooth-L1，远近对称、远处不爆炸。
        log_ratio = torch.log(mahalanobis_distance / target_distance)
        per_pair_loss = F.smooth_l1_loss(
            log_ratio, torch.zeros_like(log_ratio), reduction="none", beta=1.0
        )
    loss = _masked_mean(per_pair_loss, valid_neighbors)
    with torch.no_grad():
        distance = mahalanobis_distance.detach()
        valid_distance = distance[valid_neighbors]
        stats = {
            "mean": _masked_mean(distance, valid_neighbors),
            "median": (
                valid_distance.median() if valid_distance.numel() else distance.sum() * 0.0
            ),
            "too_close": _masked_mean(
                (distance < 0.75 * target_distance).to(distance.dtype), valid_neighbors
            ),
            "too_far": _masked_mean(
                (distance > 1.25 * target_distance).to(distance.dtype), valid_neighbors
            ),
        }
    return loss, stats


def compute_geo_losses(
    gs_centers: torch.Tensor,
    gs_scales: torch.Tensor,
    gs_opacities: torch.Tensor,
    lambda_sw: float = 0.01,
    lambda_packing: float = 0.01,
    lambda_inter: float = 0.01,
    k_neighbors: int = 4,
    target_distance: float = 2.0,
    num_sw_directions: int = 8,
    gs_rotations: torch.Tensor | None = None,
    mass_weighting: str = "opacity_area",
    inter_form: str = "log_smooth_l1",
) -> dict:
    """Compute slot-local SW/packing losses and an inter-slot potential.

    Inputs have shape ``(batch, slots, children, channels)``. Opacity and scale
    inputs are already activated. Rotation inputs use normalized Hamilton-order
    ``[w, x, y, z]`` quaternions and are normalized again before covariance
    construction. Detached mass supplies one shared set of weights for parent
    centers, SW moments, packing, and child-covariance aggregation:
    ``mass_weighting="opacity_area"`` (default, FFRGS) uses
    ``opacity * (prod scale)^(2/3)``; ``"opacity"`` uses opacity only.
    ``inter_form="log_smooth_l1"`` (default, FFRGS) penalizes
    ``smooth_l1(log(d / target))``; ``"squared"`` is the legacy ``(d - target)^2``.
    Near-transparent slots contribute zero to every geometry term. The whole
    geometry region disables autocast for the input device and uses float32.
    Outputs retain the input dtype except float16, which keeps float32 scalars
    to avoid overflow of geometry metrics when added to mixed-precision losses.
    """
    _validate_inputs(gs_centers, gs_scales, gs_opacities, gs_rotations)
    if min(lambda_sw, lambda_packing, lambda_inter) < 0:
        raise ValueError("geometry loss weights must be non-negative")
    if mass_weighting not in _MASS_WEIGHTINGS:
        raise ValueError(
            f"mass_weighting must be one of {_MASS_WEIGHTINGS}, got {mass_weighting!r}"
        )

    output_dtype = torch.float32 if gs_centers.dtype == torch.float16 else gs_centers.dtype
    with torch.autocast(device_type=gs_centers.device.type, enabled=False):
        centers = gs_centers.float()
        scales = gs_scales.float()
        opacities = gs_opacities.float()
        if gs_rotations is None:
            rotations = torch.zeros(
                (*centers.shape[:-1], 4), dtype=torch.float32, device=centers.device
            )
            rotations[..., 0] = 1.0
        else:
            rotations = gs_rotations.float()

        if mass_weighting == "opacity":
            weights, active_slots = _opacity_mass_weights(opacities)
        else:
            weights, active_slots = _opacity_area_mass_weights(opacities, scales)
        parent_means, centered, center_covariance = _weighted_slot_moments(centers, weights)
        parent_covariance = _aggregate_parent_covariance(
            center_covariance,
            scales,
            rotations,
            weights,
        )

        sw_loss = _sliced_wasserstein_loss(
            centered,
            center_covariance,
            active_slots,
            num_directions=num_sw_directions,
            weights=weights,
        )
        (
            packing_loss,
            packing_ratio,
            packing_target,
            center_variance,
            local_scale_variance,
            effective_children,
        ) = _packing_loss(center_covariance, scales, weights, active_slots)
        inter_loss, inter_stats = _inter_slot_loss(
            parent_means,
            parent_covariance,
            active_slots,
            k_neighbors=k_neighbors,
            target_distance=target_distance,
            inter_form=inter_form,
        )

        total_geo_loss = (
            float(lambda_sw) * sw_loss
            + float(lambda_packing) * packing_loss
            + float(lambda_inter) * inter_loss
        )

        def output(value: torch.Tensor) -> torch.Tensor:
            return value if value.dtype == output_dtype else value.to(dtype=output_dtype)

        return {
            "sw_loss": output(sw_loss),
            "packing_loss": output(packing_loss),
            "inter_loss": output(inter_loss),
            "total_geo_loss": output(total_geo_loss),
            "stats": {
                "packing_ratio": output(_masked_mean(packing_ratio.detach(), active_slots)),
                "packing_target": output(_masked_mean(packing_target.detach(), active_slots)),
                "effective_children": output(
                    _masked_mean(effective_children.detach(), active_slots)
                ),
                "center_variance": output(
                    _masked_mean(center_variance.detach(), active_slots)
                ),
                "local_scale_variance": output(
                    _masked_mean(local_scale_variance.detach(), active_slots)
                ),
                "inter_distance": output(inter_stats["mean"]),
                "inter_distance_median": output(inter_stats["median"]),
                "inter_too_close": output(inter_stats["too_close"]),
                "inter_too_far": output(inter_stats["too_far"]),
                "active_slot_ratio": output(
                    active_slots.to(dtype=torch.float32).mean()
                ),
            },
        }
