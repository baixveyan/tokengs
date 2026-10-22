"""CPU check of scripts/experiment_registry.py: names follow the naming rule, and the arguments of every 140-scene
run and every demo are accepted by evaluate_with_scene_latent_tuning.py with the settings the name promises."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_with_scene_latent_tuning as ev  # noqa: E402
from experiment_registry import EXPERIMENTS, METHOD_ORDER, NONE, method_of, method_sort_key  # noqa: E402

NAME = re.compile(
    r"^(no_tuning|"
    r"(?:scene_lr(?P<scene_lr>[0-9e.-]+)_)?(?:token_lr(?P<token_lr>[0-9e.-]+)_)?"
    r"(?:(?P<geo>geo)_|(?P<vis>vis_(?:xyz|opac)(?:_core_gate[0-9.]+)?)_|(?P<depth>(?:colmap(?:_clean)?|vggt_omega)_depth[0-9.]+)_)?tuning)$"
)
SCENE = "2f3e1c0f688c84cec67f9a1ea219c54c14ffabf31a046e620dacc690cac2f1bd"


def _parse(monkeypatch, argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["x", *argv, "--output", "unused"])
    return ev.parse_args()


def test_names_follow_the_rule_and_the_method_order():
    names = [e.name for e in EXPERIMENTS]
    assert len(set(names)) == len(names)
    assert sorted(reversed(names), key=method_sort_key) == names
    for e in EXPERIMENTS:
        m = NAME.match(e.name)
        assert m, e.name
        assert method_of(e.name) in METHOD_ORDER
        assert ("_geo_" in e.name) == (e.geo_weights == "0.1 / 0.001 / 0.01")
        if e.name != "no_tuning":
            assert (m["scene_lr"] or NONE) == e.scene_lr, e.name
            assert (m["token_lr"] or NONE) == e.token_lr, e.name


@pytest.mark.parametrize("e", EXPERIMENTS, ids=lambda e: e.name)
def test_run_and_demo_arguments_match_the_name(monkeypatch, e):
    run = _parse(monkeypatch, e.run_args())
    demo = _parse(monkeypatch, e.demo_args(SCENE))
    assert run.expected_scenes == 140 and not run.demo and run.scene_ids is None
    assert demo.demo and demo.scene_ids == [SCENE] and demo.ply_name == e.name and demo.expected_scenes is None
    assert run.mixed_precision == demo.mixed_precision == "no"
    target = {"scene_tuning": "scene_latents", "scene_geo_tuning": "scene_latents", "token_tuning": "gs_tokens",
              "token_geo_tuning": "gs_tokens", "no_tuning": None}.get(method_of(e.name), ev.JOINT)
    for a in (run, demo):
        assert a.tuning_target == target
        assert (a.geo_sw, a.geo_packing, a.geo_inter) == ((0.1, 0.001, 0.01) if "_geo_" in e.name else (0, 0, 0))
        if e.token_lr != NONE:
            assert (a.token_lr if a.tuning_target == ev.JOINT else a.tuning_lr) == float(e.token_lr)
        if e.scene_lr != NONE and a.tuning_target == ev.JOINT:
            assert a.tuning_lr == float(e.scene_lr)
        if e.name != "no_tuning" and a.tuning_steps is not None:
            assert a.tuning_steps == 50
    assert run.tuning_loop == {"官方": "official", "复刻": "reimplemented", NONE: "official"}[e.loop]
    assert demo.tuning_loop == ("official" if e.loop == NONE else "reimplemented")
