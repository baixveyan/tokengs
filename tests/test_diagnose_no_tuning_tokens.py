"""CPU checks for per-token geometry and opacity diagnostics."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import diagnose_no_tuning_tokens as diag  # noqa: E402


def synthetic_gaussians() -> torch.Tensor:
    # Three tokens, four children each. Token 0/1 have identical compact shapes at different centres;
    # token 2 is wider. All scales and opacities are activated.
    offsets = torch.tensor([[-1.0, 0, 0], [1.0, 0, 0], [0, -1.0, 0], [0, 1.0, 0]])
    g = torch.zeros(12, 14)
    for token, (centre, factor) in enumerate(((0.0, 1.0), (10.0, 1.0), (30.0, 3.0))):
        part = slice(token * 4, (token + 1) * 4)
        g[part, 0:3] = offsets * factor + torch.tensor([centre, 0.0, 0.0])
        g[part, 3] = torch.tensor([0.0, 0.25, 0.5, 0.75]) + 0.01
        g[part, 4:7] = 0.5
        g[part, 7] = 1.0
    return g


def test_diagnose_formulas():
    rows, summary = diag.diagnose(synthetic_gaussians(), children=4)
    assert len(rows) == 3
    assert summary["token_count"] == 3
    assert [row["nearest_token_id"] for row in rows] == [1, 0, 1]
    assert np.isclose(rows[0]["opacity_sum"], 1.54)
    assert rows[0]["alive_children"] == 4
    assert np.isclose(rows[0]["packing_ratio"], rows[1]["packing_ratio"])
    assert np.isclose(rows[2]["packing_ratio"], 9 * rows[0]["packing_ratio"])
    assert np.isclose(rows[2]["spread_rms"], 3 * rows[0]["spread_rms"])
    assert rows[0]["nearest_center_distance"] < rows[2]["nearest_center_distance"]


def test_opacity_filtered_recomputes_geometry_and_excludes_empty_tokens():
    gaussians = synthetic_gaussians()
    gaussians[0, 0:3] = torch.tensor([100.0, 0.0, 0.0])
    gaussians[0, 3] = 0.001
    gaussians[8:12, 3] = 0.001
    raw_rows, _ = diag.diagnose(gaussians, children=4)
    rows, summary = diag.diagnose_opacity_filtered(gaussians, children=4)
    assert [row["token_id"] for row in rows] == [0, 1]
    assert [row["alive_children"] for row in rows] == [3, 4]
    assert [row["nearest_token_id"] for row in rows] == [1, 0]
    assert summary["gaussian_count"] == 7
    assert summary["excluded_token_ids"] == [2]
    assert rows[0]["spread_rms"] < raw_rows[0]["spread_rms"]


def test_main_writes_complete_token_table(tmp_path):
    gaussian_path = tmp_path / "gaussians.pt"
    torch.save(synthetic_gaussians().unsqueeze(0), gaussian_path)
    metrics = tmp_path / "scene_metrics.jsonl"
    metrics.write_text(json.dumps({"geo_packing_ratio": 12.0, "geo_packing_target": 3.0,
                                   "geo_inter_distance": 0.5, "geo_inter_too_close": 0.9}) + "\n")
    out = tmp_path / "out"
    assert diag.main(["--gaussians", str(gaussian_path), "--scene-metrics", str(metrics),
                      "--output-dir", str(out), "--children", "4", "--no-plot"]) == 0
    with (out / "token_diagnostics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    report = json.loads((out / "token_diagnostics.json").read_text())
    assert len(rows) == 3
    assert set(rows[0]) == set(diag.CSV_FIELDS)
    assert report["population_140_scenes"]["scene_count"] == 1
    assert report["population_140_scenes"]["strongly_scattered_scenes"] == 0
    assert report["demo_alive_filtered"]["token_count"] == 3
    with (out / "token_diagnostics_alive_filtered.csv").open() as handle:
        filtered_rows = list(csv.DictReader(handle))
    assert len(filtered_rows) == 3
    assert "排除 opacity ≤ 1/255 的低 opacity child 后" in (out / "token_diagnostics.md").read_text()
