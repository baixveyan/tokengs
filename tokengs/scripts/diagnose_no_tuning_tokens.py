"""CPU-only per-token geometry and opacity diagnosis for a no_tuning demo scene.

The decoder emits consecutive children for each Gaussian token. This script reads the saved activated
``gaussians.pt`` tensor, writes one CSV row per token, and summarizes whether each token is spatially compact
relative to its own Gaussian scales, its geometry-loss target, and neighbouring token centres. It also summarizes
token opacity mass and optionally relates the detailed demo to the existing 140-scene aggregate metrics.

Example:
    python scripts/diagnose_no_tuning_tokens.py \
      --gaussians EXP_TokenGS/demo/scene_2f3e1c0f688c/no_tuning/gaussians.pt \
      --scene-metrics EXP_TokenGS/no_tuning/scene_metrics.jsonl \
      --output-dir EXP_TokenGS/demo/scene_2f3e1c0f688c/no_tuning
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

ALIVE_OPACITY = 1.0 / 255.0
EPS = 1e-12
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
CSV_FIELDS = (
    "token_id", "opacity_sum", "opacity_area_sum", "opacity_mean", "opacity_median", "opacity_max",
    "alive_children", "opacity_gt_0_1_children", "opacity_effective_children", "effective_children",
    "top8_opacity_fraction", "center_x", "center_y", "center_z", "spread_rms", "spread_p90",
    "spread_max", "diameter", "unweighted_spread_rms", "gaussian_scale_rms", "packing_ratio",
    "packing_target", "packing_excess", "spread_scale_excess", "nearest_token_id",
    "nearest_center_distance", "spread_to_nearest", "p90_spread_to_nearest",
)


def load_gaussians(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"{path} must contain a tensor, got {type(data).__name__}")
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2 or data.shape[1] < 11:
        raise ValueError(f"expected [N, >=11] Gaussians, got {tuple(data.shape)}")
    data = data.detach().float().cpu()
    if not torch.isfinite(data).all():
        raise ValueError("gaussians contain NaN/Inf")
    if (data[:, 3] < 0).any() or (data[:, 3] > 1).any():
        raise ValueError("opacity must already be activated in [0,1]")
    if (data[:, 4:7] <= 0).any():
        raise ValueError("scales must already be activated and positive")
    return data


def _weighted_quantile_rows(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    cdf = np.cumsum(sorted_weights, axis=1)
    threshold = q * cdf[:, -1:]
    index = np.minimum((cdf < threshold).sum(axis=1), values.shape[1] - 1)
    return sorted_values[np.arange(values.shape[0]), index]


def _nearest_centres(centres: torch.Tensor, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    count = centres.shape[0]
    nearest_distance = torch.empty(count, dtype=torch.float32)
    nearest_index = torch.empty(count, dtype=torch.int64)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        # The matrix-multiplication cdist path can round very close float32 centres to distance 0.
        distance = torch.cdist(centres[start:stop], centres, compute_mode="donot_use_mm_for_euclid_dist")
        local = torch.arange(stop - start)
        distance[local, torch.arange(start, stop)] = float("inf")
        nearest_distance[start:stop], nearest_index[start:stop] = distance.min(dim=1)
    return nearest_distance.numpy().astype(np.float64), nearest_index.numpy()


def _diameters(xyz: torch.Tensor, chunk: int = 256) -> np.ndarray:
    out = torch.empty(xyz.shape[0], dtype=torch.float32)
    for start in range(0, xyz.shape[0], chunk):
        stop = min(start + chunk, xyz.shape[0])
        out[start:stop] = torch.cdist(xyz[start:stop], xyz[start:stop]).amax(dim=(1, 2))
    return out.numpy().astype(np.float64)


def diagnose(gaussians: torch.Tensor, children: int) -> tuple[list[dict], dict]:
    if children < 2 or gaussians.shape[0] % children:
        raise ValueError(f"Gaussian count {gaussians.shape[0]} is not divisible by children={children}")
    slots = gaussians.reshape(-1, children, gaussians.shape[-1])
    xyz_t, opacity_t, scales_t = slots[..., 0:3], slots[..., 3], slots[..., 4:7]
    xyz = xyz_t.double().numpy()
    opacity = opacity_t.double().numpy()
    scales = scales_t.double().numpy()

    opacity_sum = opacity.sum(axis=1)
    opacity_weights = opacity / np.maximum(opacity_sum[:, None], EPS)
    area = np.maximum(scales.prod(axis=2), 1e-18) ** (2.0 / 3.0)
    opacity_area = opacity * area
    opacity_area_sum = opacity_area.sum(axis=1)
    weights = opacity_area / np.maximum(opacity_area_sum[:, None], EPS)

    centres = (weights[..., None] * xyz).sum(axis=1)
    delta = xyz - centres[:, None, :]
    radius = np.linalg.norm(delta, axis=2)
    spread_rms = np.sqrt((weights * radius**2).sum(axis=1))
    spread_p90 = _weighted_quantile_rows(radius, weights, 0.9)
    spread_max = radius.max(axis=1)
    unweighted_centres = xyz.mean(axis=1)
    unweighted_spread = np.sqrt(((xyz - unweighted_centres[:, None, :]) ** 2).sum(axis=2).mean(axis=1))

    center_variance = (weights[..., None] * delta**2).sum(axis=(1, 2)) / 3.0
    local_scale_variance = (weights * (scales**2).mean(axis=2)).sum(axis=1)
    packing_ratio = np.maximum(center_variance, 1e-6) / np.maximum(local_scale_variance, 1e-6)
    effective_children = 1.0 / np.maximum((weights**2).sum(axis=1), 1e-6)
    packing_target = effective_children ** (2.0 / 3.0)
    packing_excess = packing_ratio / packing_target
    spread_scale_excess = np.sqrt(packing_excess)
    gaussian_scale_rms = np.sqrt(3.0 * local_scale_variance)

    opacity_effective = 1.0 / np.maximum((opacity_weights**2).sum(axis=1), 1e-6)
    top8_opacity = np.sort(opacity, axis=1)[:, -min(8, children):].sum(axis=1) / np.maximum(opacity_sum, EPS)
    nearest_distance, nearest_index = _nearest_centres(torch.from_numpy(centres).float())
    diameter = _diameters(xyz_t.float())
    spread_to_nearest = spread_rms / np.maximum(nearest_distance, EPS)
    p90_to_nearest = spread_p90 / np.maximum(nearest_distance, EPS)

    rows = []
    for token in range(slots.shape[0]):
        rows.append({
            "token_id": token,
            "opacity_sum": opacity_sum[token],
            "opacity_area_sum": opacity_area_sum[token],
            "opacity_mean": opacity[token].mean(),
            "opacity_median": np.median(opacity[token]),
            "opacity_max": opacity[token].max(),
            "alive_children": int((opacity[token] > ALIVE_OPACITY).sum()),
            "opacity_gt_0_1_children": int((opacity[token] > 0.1).sum()),
            "opacity_effective_children": opacity_effective[token],
            "effective_children": effective_children[token],
            "top8_opacity_fraction": top8_opacity[token],
            "center_x": centres[token, 0], "center_y": centres[token, 1], "center_z": centres[token, 2],
            "spread_rms": spread_rms[token], "spread_p90": spread_p90[token],
            "spread_max": spread_max[token], "diameter": diameter[token],
            "unweighted_spread_rms": unweighted_spread[token], "gaussian_scale_rms": gaussian_scale_rms[token],
            "packing_ratio": packing_ratio[token], "packing_target": packing_target[token],
            "packing_excess": packing_excess[token], "spread_scale_excess": spread_scale_excess[token],
            "nearest_token_id": int(nearest_index[token]), "nearest_center_distance": nearest_distance[token],
            "spread_to_nearest": spread_to_nearest[token], "p90_spread_to_nearest": p90_to_nearest[token],
        })

    arrays = {field: np.asarray([row[field] for row in rows], dtype=np.float64)
              for field in CSV_FIELDS if field not in ("token_id", "nearest_token_id")}
    total_opacity = float(opacity_sum.sum())
    order = np.argsort(opacity_sum)[::-1]
    cumulative = np.cumsum(opacity_sum[order]) / max(total_opacity, EPS)

    def tokens_for(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction, side="left") + 1)

    n = len(rows)
    top_tenth = max(1, math.ceil(0.1 * n))
    strong = packing_excess > 4.0
    opacity_quartile = opacity_sum >= np.quantile(opacity_sum, 0.75)
    sorted_opacity = np.sort(opacity_sum)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ sorted_opacity / (n * max(total_opacity, EPS)))
    summary = {
        "token_count": n,
        "children_per_token": children,
        "gaussian_count": int(gaussians.shape[0]),
        "definitions": {
            "mass": "activated opacity * (product of activated scales)^(2/3), matching geometry loss",
            "spread_rms": "mass-weighted RMS 3D distance of child centres from the token centre",
            "gaussian_scale_rms": "mass-weighted RMS 3D Gaussian scale",
            "packing_ratio": "mean per-axis child-centre variance / mean per-axis Gaussian scale variance",
            "packing_target": "effective_children^(2/3)",
            "packing_excess": "packing_ratio / packing_target",
            "spread_scale_excess": "sqrt(packing_excess); observed spread-to-scale divided by target",
            "strongly_scattered": "packing_excess > 4, i.e. spread-to-scale > 2x the target",
            "opacity_sum": "sum of 64 activated opacities; not a rendered contribution because projection/occlusion are absent",
            "alive": f"activated opacity > {ALIVE_OPACITY}",
        },
        "metric_distributions": {name: distribution(values) for name, values in arrays.items()},
        "counts": {
            "packing_above_target": int((packing_excess > 1).sum()),
            "strongly_scattered": int(strong.sum()),
            "spread_rms_larger_than_nearest_center_distance": int((spread_to_nearest > 1).sum()),
            "p90_radius_larger_than_nearest_center_distance": int((p90_to_nearest > 1).sum()),
            "all_64_children_alive": int(((opacity > ALIVE_OPACITY).sum(axis=1) == children).sum()),
            "no_alive_children": int(((opacity > ALIVE_OPACITY).sum(axis=1) == 0).sum()),
        },
        "fractions": {
            "packing_above_target": float((packing_excess > 1).mean()),
            "strongly_scattered": float(strong.mean()),
            "opacity_mass_in_strongly_scattered_tokens": float(opacity_sum[strong].sum() / max(total_opacity, EPS)),
            "top_opacity_quartile_strongly_scattered": float(strong[opacity_quartile].mean()),
            "spread_rms_larger_than_nearest_center_distance": float((spread_to_nearest > 1).mean()),
            "p90_radius_larger_than_nearest_center_distance": float((p90_to_nearest > 1).mean()),
            "alive_children": float((opacity > ALIVE_OPACITY).mean()),
        },
        "opacity_concentration": {
            "total_opacity": total_opacity,
            "gini": gini,
            "top_10_percent_token_share": float(opacity_sum[order[:top_tenth]].sum() / max(total_opacity, EPS)),
            "tokens_for_50_percent": tokens_for(0.5),
            "tokens_for_90_percent": tokens_for(0.9),
        },
        "correlations": {
            "spearman_log_opacity_sum_vs_log_packing_excess": spearman(
                np.log10(np.maximum(opacity_sum, EPS)), np.log10(np.maximum(packing_excess, EPS))
            )
        },
        "top_opacity_tokens": [int(i) for i in order[:20]],
        "most_scattered_opacity_rich_tokens": [
            int(i) for i in np.flatnonzero(opacity_sum >= np.median(opacity_sum))[
                np.argsort(packing_excess[opacity_sum >= np.median(opacity_sum)])[::-1][:20]
            ]
        ],
    }
    return rows, summary


def diagnose_opacity_filtered(
    gaussians: torch.Tensor, children: int, threshold: float = ALIVE_OPACITY
) -> tuple[list[dict], dict]:
    """Recompute geometry after excluding children whose activated opacity is at most ``threshold``."""
    if children < 2 or gaussians.shape[0] % children:
        raise ValueError(f"Gaussian count {gaussians.shape[0]} is not divisible by children={children}")
    slots = gaussians.reshape(-1, children, gaussians.shape[-1]).double().numpy()
    rows: list[dict] = []
    centres: list[np.ndarray] = []
    excluded: list[int] = []
    retained_gaussians = 0

    for token, slot in enumerate(slots):
        selected = slot[:, 3] > threshold
        if not selected.any():
            excluded.append(token)
            continue
        retained_gaussians += int(selected.sum())
        xyz, opacity, scales = slot[selected, 0:3], slot[selected, 3], slot[selected, 4:7]
        opacity_sum = float(opacity.sum())
        opacity_weights = opacity / max(opacity_sum, EPS)
        area = np.maximum(scales.prod(axis=1), 1e-18) ** (2.0 / 3.0)
        opacity_area = opacity * area
        opacity_area_sum = float(opacity_area.sum())
        weights = opacity_area / max(opacity_area_sum, EPS)
        centre = (weights[:, None] * xyz).sum(axis=0)
        delta = xyz - centre
        radius = np.linalg.norm(delta, axis=1)
        order = np.argsort(radius)
        cdf = np.cumsum(weights[order])
        p90 = float(radius[order[min(int((cdf < 0.9).sum()), len(radius) - 1)]])
        center_variance = float((weights[:, None] * delta**2).sum() / 3.0)
        local_scale_variance = float((weights * (scales**2).mean(axis=1)).sum())
        packing_ratio = max(center_variance, 1e-6) / max(local_scale_variance, 1e-6)
        effective_children = 1.0 / max(float((weights**2).sum()), 1e-6)
        packing_target = effective_children ** (2.0 / 3.0)
        packing_excess = packing_ratio / packing_target
        opacity_effective = 1.0 / max(float((opacity_weights**2).sum()), 1e-6)
        unweighted_centre = xyz.mean(axis=0)
        unweighted_spread = float(np.sqrt(((xyz - unweighted_centre) ** 2).sum(axis=1).mean()))
        diameter = float(np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2).max())
        centres.append(centre)
        rows.append({
            "token_id": token,
            "opacity_sum": opacity_sum,
            "opacity_area_sum": opacity_area_sum,
            "opacity_mean": float(opacity.mean()),
            "opacity_median": float(np.median(opacity)),
            "opacity_max": float(opacity.max()),
            "alive_children": int(selected.sum()),
            "opacity_gt_0_1_children": int((opacity > 0.1).sum()),
            "opacity_effective_children": opacity_effective,
            "effective_children": effective_children,
            "top8_opacity_fraction": float(np.sort(opacity)[-min(8, len(opacity)):].sum() / max(opacity_sum, EPS)),
            "center_x": float(centre[0]), "center_y": float(centre[1]), "center_z": float(centre[2]),
            "spread_rms": float(np.sqrt((weights * radius**2).sum())),
            "spread_p90": p90,
            "spread_max": float(radius.max()),
            "diameter": diameter,
            "unweighted_spread_rms": unweighted_spread,
            "gaussian_scale_rms": math.sqrt(3.0 * local_scale_variance),
            "packing_ratio": packing_ratio,
            "packing_target": packing_target,
            "packing_excess": packing_excess,
            "spread_scale_excess": math.sqrt(packing_excess),
            "nearest_token_id": -1,
            "nearest_center_distance": float("inf"),
            "spread_to_nearest": 0.0,
            "p90_spread_to_nearest": 0.0,
        })

    if len(rows) >= 2:
        nearest_distance, nearest_local_index = _nearest_centres(torch.from_numpy(np.asarray(centres)).float())
        for i, row in enumerate(rows):
            row["nearest_token_id"] = int(rows[int(nearest_local_index[i])]["token_id"])
            row["nearest_center_distance"] = float(nearest_distance[i])
            row["spread_to_nearest"] = row["spread_rms"] / max(float(nearest_distance[i]), EPS)
            row["p90_spread_to_nearest"] = row["spread_p90"] / max(float(nearest_distance[i]), EPS)

    arrays = {field: np.asarray([row[field] for row in rows], dtype=np.float64)
              for field in CSV_FIELDS if field not in ("token_id", "nearest_token_id")}
    excess = arrays["packing_excess"]
    strong = excess > 4.0
    overlap = arrays["spread_to_nearest"] > 1.0
    p90_overlap = arrays["p90_spread_to_nearest"] > 1.0
    opacity_sum = arrays["opacity_sum"]
    total_opacity = float(opacity_sum.sum())
    summary = {
        "input_token_count": int(slots.shape[0]),
        "token_count": len(rows),
        "children_per_token_before_filter": children,
        "original_gaussian_count": int(gaussians.shape[0]),
        "gaussian_count": retained_gaussians,
        "opacity_threshold": threshold,
        "filter_rule": f"retain activated opacity > {threshold}",
        "excluded_token_ids": excluded,
        "metric_distributions": {name: distribution(values) for name, values in arrays.items()},
        "counts": {
            "tokens_with_selected_children": len(rows),
            "tokens_without_selected_children": len(excluded),
            "packing_above_target": int((excess > 1).sum()),
            "strongly_scattered": int(strong.sum()),
            "spread_rms_larger_than_nearest_center_distance": int(overlap.sum()),
            "p90_radius_larger_than_nearest_center_distance": int(p90_overlap.sum()),
        },
        "fractions": {
            "retained_gaussians": retained_gaussians / gaussians.shape[0],
            "packing_above_target": float((excess > 1).mean()),
            "strongly_scattered": float(strong.mean()),
            "opacity_mass_in_strongly_scattered_tokens": float(opacity_sum[strong].sum() / max(total_opacity, EPS)),
            "spread_rms_larger_than_nearest_center_distance": float(overlap.mean()),
            "p90_radius_larger_than_nearest_center_distance": float(p90_overlap.mean()),
        },
    }
    return rows, summary


def distribution(values: np.ndarray) -> dict:
    out = {"mean": float(np.mean(values))}
    for q, value in zip(QUANTILES, np.quantile(values, QUANTILES)):
        out[f"p{int(q * 100):02d}"] = float(value)
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        rank = np.empty(values.size, dtype=np.float64)
        rank[order] = np.arange(values.size, dtype=np.float64)
        return rank

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def load_scene_population(path: Path | None) -> dict | None:
    if path is None:
        return None
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ratio = np.asarray([r["geo_packing_ratio"] for r in records], dtype=np.float64)
    target = np.asarray([r["geo_packing_target"] for r in records], dtype=np.float64)
    excess = ratio / target
    inter = np.asarray([r["geo_inter_distance"] for r in records], dtype=np.float64)
    close = np.asarray([r["geo_inter_too_close"] for r in records], dtype=np.float64)
    return {
        "scene_count": len(records),
        "packing_ratio": distribution(ratio),
        "packing_target": distribution(target),
        "packing_excess": distribution(excess),
        "spread_scale_excess": distribution(np.sqrt(excess)),
        "packing_above_target_scenes": int((excess > 1).sum()),
        "strongly_scattered_scenes": int((excess > 4).sum()),
        "inter_distance": distribution(inter),
        "inter_too_close": distribution(close),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _q(summary: dict, name: str, q: str = "p50") -> float:
    return summary["metric_distributions"][name][q]


def write_markdown(
    path: Path, source: Path, summary: dict, population: dict | None, filtered: dict | None = None
) -> None:
    n, children = summary["token_count"], summary["children_per_token"]
    counts, fractions = summary["counts"], summary["fractions"]
    opacity = summary["opacity_concentration"]
    top_rows = summary["top_opacity_tokens"][:10]
    lines = [
        "# no_tuning Gaussian token 详细诊断", "",
        f"- 数据：`{source}`，{n} 个 token × {children} 个 child = {summary['gaussian_count']:,} 个 Gaussian。",
        "- 本报告逐 token 数字来自当前预注册 demo 场景；140 场景部分来自正式 `scene_metrics.jsonl` 的场景级聚合，"
        "140 场景运行没有保存每个场景的 4096-token 明细。", "",
        "## 结论", "",
    ]
    if filtered:
        filtered_counts, filtered_fractions = filtered["counts"], filtered["fractions"]
        lines += [
            f"1. **若把全部 64 个 child 都纳入，确实会得到显著分散的结论：** {fractions['packing_above_target']:.1%} "
            f"高于 packing 目标，{fractions['strongly_scattered']:.1%} 达到强分散标准。",
            f"2. **但排除 opacity ≤ 1/255 的低 opacity child 后，结论大幅减弱：** 高于目标降至 "
            f"{filtered_fractions['packing_above_target']:.1%}，强分散降至 {filtered_fractions['strongly_scattered']:.1%}；"
            f"packing excess 中位数从 {_q(summary, 'packing_excess'):.2f} 降至 "
            f"{filtered['metric_distributions']['packing_excess']['p50']:.2f}（低于目标 1）。",
            "3. 因此不能再概括为“一个 token 的有效 Gaussian 通常非常散”。更准确的说法是："
            "低 opacity 远端 child 显著扩大了完整输出的空间包络；过滤后多数 token 内部并不过度分散，但仍有一部分异常分散。",
            f"4. token 云之间仍高度交叠：过滤后仍有 "
            f"{filtered_fractions['spread_rms_larger_than_nearest_center_distance']:.1%} 的 token，其 RMS 半径大于最近 token 中心距。"
            "这说明不同 token 的中心非常接近，但不等同于单个 token 自身超过 packing 目标。",
            "5. opacity 阈值只是诊断口径，不代表 feed-forward 清理，也不等同于实际渲染贡献。"
            "因此，当前几何损失把所有 child 一起收紧的优化假设缺少充分支持。", "",
        ]
    else:
        lines += [
            f"1. {counts['packing_above_target']}/{n} ({fractions['packing_above_target']:.1%}) 个 token 的 packing ratio 高于几何目标；"
            f"{counts['strongly_scattered']}/{n} ({fractions['strongly_scattered']:.1%}) 达到强分散标准。",
            f"2. 中位 token 的质量加权 RMS 散开半径为 {_q(summary, 'spread_rms'):.4g}，Gaussian RMS 尺寸为 "
            f"{_q(summary, 'gaussian_scale_rms'):.4g}。", "",
        ]
    if filtered:
        filtered_counts, filtered_fractions = filtered["counts"], filtered["fractions"]
        lines += [
            "## 排除低 opacity child 后重新统计", "",
            f"- 此处只在分析中保留 `opacity > 1/255` 的 child；共保留 {filtered['gaussian_count']:,}/"
            f"{filtered['original_gaussian_count']:,} ({filtered_fractions['retained_gaussians']:.1%}) 个 Gaussian。"
            f"有 {filtered_counts['tokens_without_selected_children']} 个 token 完全没有超过阈值的 child，故不参与几何复算。",
            "- 每个 token 的中心、Gaussian 尺寸、packing target 和最近 token 中心都用过滤后的 child 从头计算。", "",
            "| 指标 | 原始 64 child | 过滤后 |", "|---|---:|---:|",
            f"| packing excess 中位数 | {_q(summary, 'packing_excess'):.4g} | "
            f"{filtered['metric_distributions']['packing_excess']['p50']:.4g} |",
            f"| packing 高于目标的 token | {fractions['packing_above_target']:.1%} | "
            f"{filtered_fractions['packing_above_target']:.1%} |",
            f"| 强分散 token | {fractions['strongly_scattered']:.1%} | "
            f"{filtered_fractions['strongly_scattered']:.1%} |",
            f"| RMS 散开半径中位数 | {_q(summary, 'spread_rms'):.4g} | "
            f"{filtered['metric_distributions']['spread_rms']['p50']:.4g} |",
            f"| Gaussian RMS 尺寸中位数 | {_q(summary, 'gaussian_scale_rms'):.4g} | "
            f"{filtered['metric_distributions']['gaussian_scale_rms']['p50']:.4g} |",
            f"| RMS 半径大于最近 token 中心距 | {fractions['spread_rms_larger_than_nearest_center_distance']:.1%} | "
            f"{filtered_fractions['spread_rms_larger_than_nearest_center_distance']:.1%} |", "",
            "过滤后结果用于检验结论是否由低 opacity 远端点驱动；该阈值是分析口径，不代表 feed-forward 或渲染器执行了清理。", "",
        ]
    lines += [
        "## “散”的量化口径", "",
        "- token 中心及 spread 均按 `opacity × (scale_x scale_y scale_z)^(2/3)` 加权，与现有 geometry loss 完全一致。",
        "- `packing_ratio = 子中心每轴方差 / Gaussian 每轴尺度方差`；目标是 `effective_children^(2/3)`。",
        "- `packing_excess = packing_ratio / target`。大于 1 表示比目标散；大于 4 表示 spread/scale 超目标 2 倍，"
        "本报告把它称为“强分散”。这是一条透明、可复算的工程阈值，不是论文给出的自然定律。",
        "- opacity 总量只是 64 个 activated opacity 的和；未考虑投影面积、遮挡和视角，因此不能直接当作最终渲染贡献。", "",
        "## 4096 个 token 的分布", "",
        "| 指标 | p10 | p50 | p90 | p99 | 均值 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("opacity_sum", "每 token opacity 总量"), ("alive_children", "存活 child 数"),
        ("effective_children", "有效 child 数（opacity-area）"), ("spread_rms", "RMS 散开半径"),
        ("diameter", "child 中心最大直径"), ("gaussian_scale_rms", "Gaussian RMS 尺寸"),
        ("packing_ratio", "packing ratio"), ("packing_target", "packing target"),
        ("packing_excess", "packing excess"), ("spread_scale_excess", "spread/scale 超目标倍数"),
        ("nearest_center_distance", "最近 token 中心距离"), ("spread_to_nearest", "RMS 半径 / 最近中心距"),
    ):
        d = summary["metric_distributions"][key]
        lines.append(f"| {label} | {d['p10']:.4g} | {d['p50']:.4g} | {d['p90']:.4g} | {d['p99']:.4g} | {d['mean']:.4g} |")
    lines += ["", "## opacity 在 4096 个 token 中如何分配", "",
              f"- 全部 child opacity 总和：{opacity['total_opacity']:.3f}；token opacity Gini：{opacity['gini']:.3f}。",
              f"- opacity 最高的前 10% token 承担 {opacity['top_10_percent_token_share']:.1%} 的总 opacity。",
              f"- 累计 50% opacity 需要 {opacity['tokens_for_50_percent']}/{n} 个 token；"
              f"累计 90% 需要 {opacity['tokens_for_90_percent']}/{n} 个 token。",
              f"- 全体 child 存活率（opacity > 1/255）：{fractions['alive_children']:.1%}；"
              f"64 个 child 全部存活的 token 为 {counts['all_64_children_alive']}，无存活 child 的 token 为 {counts['no_alive_children']}。",
              f"- log(opacity 总量) 与 log(packing excess) 的 Spearman 相关："
              f"{summary['correlations']['spearman_log_opacity_sum_vs_log_packing_excess']:.3f}。", "",
              "## opacity 总量最高的 10 个 token", "",
              "| token | opacity 总量 | 存活 child | packing excess | spread/scale 超目标倍数 | RMS 半径 / 最近中心距 |",
              "|---:|---:|---:|---:|---:|---:|"]
    row_by_id = {int(row["token_id"]): row for row in summary.pop("_rows", [])}
    for token in top_rows:
        row = row_by_id[token]
        lines.append(f"| {token} | {row['opacity_sum']:.4g} | {row['alive_children']} | {row['packing_excess']:.4g} | "
                     f"{row['spread_scale_excess']:.3f} | {row['spread_to_nearest']:.3f} |")
    if population:
        lines += ["", "## 140 场景总体证据（场景级聚合）", "",
                  f"- {population['scene_count']} 个场景中，packing ratio/target 全部高于 1："
                  f"{population['packing_above_target_scenes']}/{population['scene_count']}；"
                  f"全部高于强分散阈值 4：{population['strongly_scattered_scenes']}/{population['scene_count']}。",
                  f"- 场景级 packing ratio：p10 {population['packing_ratio']['p10']:.2f} / "
                  f"p50 {population['packing_ratio']['p50']:.2f} / p90 {population['packing_ratio']['p90']:.2f}；"
                  f"目标中位数仅 {population['packing_target']['p50']:.2f}。",
                  f"- 场景级 packing excess 中位数 {population['packing_excess']['p50']:.1f}；"
                  f"对应 spread/scale 超目标中位数 {population['spread_scale_excess']['p50']:.2f} 倍。",
                  f"- inter 最近邻 Mahalanobis 距离均值的场景中位数为 {population['inter_distance']['p50']:.3f}（目标 2.0）；"
                  f"过近边比例的场景中位数为 {population['inter_too_close']['p50']:.1%}。",
                  "- 这说明 demo 不是孤例；但 140 场景只保留每场景聚合，不能声称每个场景的每个 token 都满足同一结论。"]
    lines += ["", "## 文件", "", "- `token_diagnostics.csv`：4096 行逐 token 完整明细。",
              "- `token_diagnostics_alive_filtered.csv`：排除 opacity ≤ 1/255 后的逐 token 明细。",
              "- `token_diagnostics.json`：定义、原始/过滤后全局统计和 140 场景摘要。",
              "- `token_diagnostics.png`：packing、opacity、重叠和二者关系的四联图。", ""]
    path.write_text("\n".join(lines))


def write_plot(path: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    opacity = np.asarray([r["opacity_sum"] for r in rows])
    excess = np.asarray([r["packing_excess"] for r in rows])
    overlap = np.asarray([r["spread_to_nearest"] for r in rows])
    order = np.argsort(opacity)[::-1]
    cumulative = np.cumsum(opacity[order]) / opacity.sum()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(np.log10(np.maximum(excess, EPS)), bins=60)
    axes[0, 0].axvline(0, color="black", linestyle="--", label="target")
    axes[0, 0].axvline(math.log10(4), color="red", linestyle="--", label="strong")
    axes[0, 0].set(xlabel="log10(packing excess)", ylabel="tokens", title="Within-token spread")
    axes[0, 0].legend()
    axes[0, 1].plot(np.arange(1, len(rows) + 1), cumulative)
    axes[0, 1].axhline(0.5, color="black", linestyle="--")
    axes[0, 1].axhline(0.9, color="black", linestyle=":")
    axes[0, 1].set(xlabel="tokens sorted by opacity", ylabel="cumulative opacity", title="Opacity concentration")
    axes[1, 0].hist(overlap, bins=np.linspace(0, min(np.quantile(overlap, 0.99), 10), 60))
    axes[1, 0].axvline(1, color="red", linestyle="--")
    axes[1, 0].set(xlabel="spread RMS / nearest token-centre distance", ylabel="tokens", title="Token-cloud overlap")
    axes[1, 1].scatter(opacity, excess, s=5, alpha=0.35)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].axhline(4, color="red", linestyle="--")
    axes[1, 1].set(xlabel="token opacity sum", ylabel="packing excess", title="Opacity vs spread")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--scene-metrics", type=Path, default=None,
                        help="optional 140-scene no_tuning scene_metrics.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--children", type=int, default=64)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    gaussians = load_gaussians(args.gaussians)
    rows, summary = diagnose(gaussians, args.children)
    filtered_rows, filtered_summary = diagnose_opacity_filtered(gaussians, args.children)
    population = load_scene_population(args.scene_metrics)
    result = {"source": str(args.gaussians), "demo": summary, "demo_alive_filtered": filtered_summary,
              "population_140_scenes": population}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / "token_diagnostics"
    write_csv(base.with_suffix(".csv"), rows)
    write_csv(args.output_dir / "token_diagnostics_alive_filtered.csv", filtered_rows)
    base.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    summary["_rows"] = rows
    write_markdown(base.with_suffix(".md"), args.gaussians, summary, population, filtered_summary)
    if not args.no_plot:
        write_plot(base.with_suffix(".png"), rows)
    print(f"wrote {base}.csv/.json/.md" + ("" if args.no_plot else "/.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
