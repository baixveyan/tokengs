"""CPU check of scripts/compare_same_scene_ply.py on synthetic Gaussians and sidecars.

Two methods on one scene: the second moves every Gaussian of token 0 by a known offset and leaves the rest
untouched, so displacement statistics, the difference PLY and the metrics table can be checked exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_same_scene_ply import compare_scene, find_sidecars, method_sort_key  # noqa: E402

TOKENS, CHILDREN = 4, 8
SCENE = "abcdef0123456789"


def _gaussians(seed: int) -> torch.Tensor:
    g = torch.zeros(1, TOKENS * CHILDREN, 14)
    gen = torch.Generator().manual_seed(seed)
    g[0, :, 0:3] = torch.randn(TOKENS * CHILDREN, 3, generator=gen)
    g[0, :, 3] = 0.5
    g[0, :, 4:7] = 0.1
    g[0, :, 7] = 1.0
    g[0, :, 11:14] = 0.5
    return g


def _write_method(demo_dir: Path, name: str, g: torch.Tensor, sha: str = "sha", psnr: float = 25.0) -> Path:
    method_dir = demo_dir / f"scene_{SCENE[:12]}" / name  # one folder per method, everything flat inside
    method_dir.mkdir(parents=True)
    torch.save(g, method_dir / "gaussians.pt")
    sidecar = {
        "method": name, "scene_id": SCENE, "files": {"ply": f"{name}.ply", "gaussians_pt": "gaussians.pt"},
        "decoder_inputs": None, "redecode_check": None, "tuning_target": None, "tuning_loop": None,
        "tuning_steps": None, "tuning_lr": None, "token_lr": None, "tuning_loop_geo_weights": {},
        "checkpoint_sha256": sha, "children_per_slot": CHILDREN,
        "metrics": {"all_psnr": psnr, "bucket_d0_psnr": psnr + 1},
    }
    path = method_dir / f"{name}.json"
    path.write_text(json.dumps(sidecar))
    return path


def test_displacement_of_one_moved_token(tmp_path):
    out_dir = tmp_path / f"scene_{SCENE[:12]}"
    ref = _gaussians(0)
    moved = ref.clone()
    moved[0, :CHILDREN, 0] += 2.0  # token 0: all children move 2 along x
    sidecars = [_write_method(tmp_path, "no_tuning", ref), _write_method(tmp_path, "tuned", moved, psnr=27.0)]

    result = compare_scene(sidecars, out_dir, "no_tuning")

    diff = result["methods"]["tuned"]["difference_to_reference"]
    assert "difference_to_reference" not in result["methods"]["no_tuning"]
    assert result["methods"]["tuned"]["ply_folder"] == "tuned"
    assert result["methods"]["tuned"]["difference_to_reference"]["ply"] == "tuned/tuned_minus_no_tuning.ply"
    assert find_sidecars(tmp_path) == {out_dir: sidecars}
    assert diff["compared_gaussians_alive_in_either"] == TOKENS * CHILDREN
    assert diff["centre_displacement_median"] == pytest.approx(0.0)
    assert diff["centre_displacement_p90"] == pytest.approx(2.0)
    assert diff["slot_centroid_displacement_median"] == pytest.approx(0.0)
    assert diff["opacity_change_mean_abs"] == pytest.approx(0.0)

    from plyfile import PlyData
    v = PlyData.read(str(out_dir / "tuned" / "tuned_minus_no_tuning.ply"))["vertex"]
    assert len(v) == TOKENS * CHILDREN
    assert list(v["displacement"][:CHILDREN]) == pytest.approx([2.0] * CHILDREN)
    assert (v["red"][:CHILDREN] == 255).all() and (v["red"][CHILDREN:] == 0).all()
    assert list(v["slot_id"][CHILDREN - 1:CHILDREN + 1]) == [0.0, 1.0]

    md = (out_dir / "comparison.md").read_text()
    assert "| all PSNR | 25.000 | 27.000 |" in md
    assert "- tuned: not saved" in md
    assert json.loads((out_dir / "comparison.json").read_text())["scene_id"] == SCENE


def test_sidecars_from_different_checkpoints_are_rejected(tmp_path):
    sidecars = [_write_method(tmp_path, "no_tuning", _gaussians(0), sha="a"),
                _write_method(tmp_path, "tuned", _gaussians(1), sha="b")]
    with pytest.raises(ValueError, match="disagree"):
        compare_scene(sidecars, tmp_path / f"scene_{SCENE[:12]}", "no_tuning")


def test_sidecar_must_name_its_folder(tmp_path):
    path = _write_method(tmp_path, "no_tuning", _gaussians(0))
    path.rename(path.with_name("other.json"))
    path.parent.rename(path.parent.with_name("other"))
    with pytest.raises(ValueError, match="not its folder name"):
        find_sidecars(tmp_path)


def test_methods_are_listed_scene_token_geo_order():
    expected = ["no_tuning", "scene_lr1e-2_tuning", "scene_lr1e-2_geo_tuning", "token_lr1e-4_tuning",
                "token_lr1e-2_tuning", "token_lr1e-4_geo_tuning", "scene_lr1e-2_token_lr1e-4_tuning",
                "scene_lr1e-2_token_lr1e-4_geo_tuning", "other_method"]
    assert sorted(reversed(expected), key=method_sort_key) == expected
