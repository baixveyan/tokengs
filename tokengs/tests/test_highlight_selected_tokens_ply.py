"""CPU check of scripts/highlight_selected_tokens_ply.py on synthetic PLY files and sidecars.

Two methods on one scene, written with the real exporter (demo_ply_export.export_gaussians_ply): token
selection only takes active tokens of the reference, is shared by every method and nested (the k-token set is the
first k of the larger set); selected Gaussians keep everything but colour. The context outputs make other points
grey with scaled opacity; the selected-only outputs remove them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import demo_ply_export  # noqa: E402
import highlight_selected_tokens_ply as hl  # noqa: E402

TOKENS, CHILDREN = 40, 8
SCENE = "scene_abcdef012345"  # folder name: scene_<id[:12]>


def _gaussians(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    g = torch.zeros(1, TOKENS * CHILDREN, 14)
    g[0, :, 0:3] = torch.randn(TOKENS * CHILDREN, 3, generator=gen)
    # tokens 0..19 almost transparent, 20..39 clearly visible: only the latter are active
    g[0, :, 3] = torch.where(torch.arange(TOKENS * CHILDREN) // CHILDREN < TOKENS // 2, 0.01, 0.6)
    g[0, :, 4:7] = 0.1
    g[0, :, 7] = 1.0
    g[0, :, 11:14] = torch.rand(TOKENS * CHILDREN, 3, generator=gen)
    return g


def _write_method(root: Path, name: str, g: torch.Tensor) -> None:
    d = root / SCENE / name
    d.mkdir(parents=True)
    n = g.shape[1]
    per = {"aniso": torch.ones(n), "cos_ray": torch.zeros(n), "parallax_deg": torch.zeros(n),
           "in_frustum_count": torch.ones(n)}
    demo_ply_export.export_gaussians_ply(d / f"{name}.ply", g, CHILDREN, per)
    (d / f"{name}.json").write_text(json.dumps({"method": name, "files": {"ply": f"{name}.ply"},
                                                "children_per_slot": CHILDREN}))


def test_highlight(tmp_path, monkeypatch):
    _write_method(tmp_path, "no_tuning", _gaussians(0))
    _write_method(tmp_path, "other", _gaussians(1))
    monkeypatch.setattr(sys, "argv", ["x", "--demo-dir", str(tmp_path), "--counts", "3", "6"])
    assert hl.main() == 0

    legend = json.loads((tmp_path / SCENE / "highlighted_tokens.json").read_text())
    assert legend["files"]["other"] == [
        "other/other_highlight3tokens.ply", "other/other_only3tokens.ply",
        "other/other_highlight6tokens.ply", "other/other_only6tokens.ply",
    ]
    tokens = [t["token"] for t in legend["tokens"]]
    assert len(tokens) == 6 and len(set(tokens)) == 6 and all(t >= TOKENS // 2 for t in tokens)
    for m in ("no_tuning", "other"):
        orig = hl.read_vertices(tmp_path / SCENE / m / f"{m}.ply")
        for k in (3, 6):
            h = hl.read_vertices(tmp_path / SCENE / m / f"{m}_highlight{k}tokens.ply")
            rank = h["selected_token_rank"].astype(int)
            slot = np.arange(h.shape[0]) // CHILDREN
            assert set(slot[rank >= 0]) == set(tokens[:k])
            assert all((rank[slot == t] == i).all() for i, t in enumerate(tokens[:k]))
            sel = rank >= 0
            for f in ("x", "y", "z", "scale_0", "rot_0"):
                assert np.array_equal(h[f], orig[f])
            assert np.allclose(h["opacity_activated"][sel], orig["opacity_activated"][sel])
            assert np.allclose(h["opacity_activated"][~sel], orig["opacity_activated"][~sel] * 0.1, atol=1e-7)
            rgb = np.stack([h[f"f_dc_{c}"] for c in range(3)], 1) * hl.SH_C0 + 0.5
            assert np.allclose(rgb[~sel], 0.5, atol=1e-6)
            assert np.allclose(rgb[slot == tokens[0]], hl.hex_rgb(hl.PALETTE[0][1]), atol=1e-6)

            only = hl.read_vertices(tmp_path / SCENE / m / f"{m}_only{k}tokens.ply")
            only_slot = only["slot_id"].astype(int)
            only_rank = only["selected_token_rank"].astype(int)
            expected = np.isin(orig["slot_id"].astype(int), tokens[:k])
            assert only.shape[0] == k * CHILDREN
            assert set(only_slot) == set(tokens[:k])
            assert set(only_rank) == set(range(k))
            for f in ("x", "y", "z", "opacity", "opacity_activated", "scale_0", "rot_0", "slot_id", "child_id"):
                assert np.array_equal(only[f], orig[f][expected])
            only_rgb = np.stack([only[f"f_dc_{c}"] for c in range(3)], 1) * hl.SH_C0 + 0.5
            for i, t in enumerate(tokens[:k]):
                assert np.allclose(only_rgb[only_slot == t], hl.hex_rgb(hl.PALETTE[i][1]), atol=1e-6)
