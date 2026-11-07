"""Every experiment of EXP_TokenGS, in one place: one entry = one 140-scene run in EXP_TokenGS/<name>/.

The launchers (EXP_TokenGS/run_140_scenes.sh, EXP_TokenGS/run_demo.sh), the reports and the total table all
read this list, so a name is the same everywhere: directory, report label, table row, demo folder
(EXP_TokenGS/demo/scene_<id[:12]>/<name>/) and PLY file names.

Name = no_tuning | [scene_lr<x>_][token_lr<y>_][geo_]tuning
  words           always in the order scene -> token -> geo -> tuning
  learning rates  attached directly to the tensor they tune: scene_lr1e-2_token_lr1e-4_geo_tuning
Same for every run, so not in the name: FP32, release latent model dl3dv_latent_6v_ssim, 50 tuning steps,
geometry weights sw / packing / inter = GEO_WEIGHTS (a run with other weights adds only the changed one after geo,
e.g. _geo_sw0.2_tuning).

    python scripts/experiment_registry.py names                  # all experiments, in table order
    python scripts/experiment_registry.py demo-names             # finished experiments that get a demo
    python scripts/experiment_registry.py args <name>            # arguments of the 140-scene run
    python scripts/experiment_registry.py demo-args <name> <scene_id>
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

EXP = "EXP_TokenGS"
DEMO_DIR = f"{EXP}/demo"
DEMO_SCENE_JSON = f"{DEMO_DIR}/demo_scene.json"
REPORT_DIR = f"{EXP}/reports"
GEO_WEIGHTS = "0.1 / 0.001 / 0.01"
GEO_ARGS = ("--geo-sw", "0.1", "--geo-packing", "0.001", "--geo-inter", "0.01")
MODEL_ARGS = ("--preset", "eval_dl3dv_latent_6view_ssim",
              "--resume", "checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors", "--mixed-precision", "no")
# fixed order of the methods (tables, legends); the words are always scene -> token -> geo / vis
METHOD_ORDER = (
    "no_tuning",
    "scene_tuning",
    "scene_geo_tuning",
    "token_tuning",
    "token_geo_tuning",
    "scene_token_tuning",
    "scene_token_geo_tuning",
    "scene_token_vis_xyz_tuning",
    "scene_token_vis_opac_tuning",
    "scene_token_vis_opac_core_gate2.5_tuning",
    "scene_token_colmap_depth0.01_tuning",
    "scene_token_colmap_clean_depth0.01_tuning",
    "scene_token_vggt_omega_depth0.01_tuning",
)
NONE = "—"


@dataclass(frozen=True)
class Experiment:
    name: str
    title: str                      # row title in the total table
    tuned: str                      # tensors that are tuned
    loop: str                       # tuning loop of the 140-scene run: 官方 / 复刻 / —
    scene_lr: str
    token_lr: str
    geo_weights: str                # sw / packing / inter, or —
    method_args: tuple[str, ...]

    @property
    def dir(self) -> str:
        return f"{EXP}/{self.name}"

    def run_args(self) -> list[str]:
        loop = ["--tuning-loop", "reimplemented"] if self.loop == "复刻" else []
        return [*MODEL_ARGS, "--expected-scenes", "140", *self.method_args, *loop]

    def demo_args(self, scene_id: str) -> list[str]:
        # the demo always uses the reimplemented loop (tuning curve + tuned tensors); it is bit-identical to the
        # official loop on CPU (tests/test_reimplemented_tuning_loops.py)
        loop = ["--tuning-loop", "reimplemented"] if self.loop != NONE else []
        return [*MODEL_ARGS, *self.method_args, *loop, "--demo", "--scene-ids", scene_id, "--ply-name", self.name]


def _scene() -> tuple[str, ...]:
    return ("--scene-latent-tuning",)  # lr 1e-2 = preset ttt_lr, 50 steps = preset ttt_n_steps


def _token(lr: str) -> tuple[str, ...]:
    return ("--token-tuning", "--tuning-lr", lr, "--tuning-steps", "50")


def _joint(token_lr: str) -> tuple[str, ...]:
    return ("--scene-latent-and-token-tuning", "--tuning-lr", "1e-2", "--token-lr", token_lr, "--tuning-steps", "50")


SCENE, TOKEN, BOTH = "scene latents", "Gaussian tokens", "scene latents + Gaussian tokens"
EXPERIMENTS = (
    Experiment("no_tuning", "不微调", "无", NONE, NONE, NONE, NONE, ("--no-tuning",)),
    Experiment("scene_lr1e-2_tuning", "只调 scene latents", SCENE, "复刻", "1e-2", NONE, NONE, _scene()),
    Experiment("scene_lr1e-2_geo_tuning", "只调 scene latents + 几何损失", SCENE, "复刻", "1e-2", NONE, GEO_WEIGHTS,
               (*_scene(), *GEO_ARGS)),
    Experiment("token_lr1e-4_tuning", "只调 token（lr 1e-4）", TOKEN, "官方", NONE, "1e-4", NONE, _token("1e-4")),
    Experiment("token_lr1e-4_geo_tuning", "只调 token（lr 1e-4）+ 几何损失", TOKEN, "复刻", NONE, "1e-4", GEO_WEIGHTS,
               (*_token("1e-4"), *GEO_ARGS)),
    Experiment("scene_lr1e-2_token_lr1e-4_tuning", "两者一起调（token lr 1e-4）", BOTH, "复刻", "1e-2", "1e-4", NONE,
               _joint("1e-4")),
    Experiment("scene_lr1e-2_token_lr1e-4_geo_tuning", "两者一起调（token lr 1e-4）+ 几何损失", BOTH, "复刻", "1e-2", "1e-4",
               GEO_WEIGHTS, (*_joint("1e-4"), *GEO_ARGS)),
    Experiment("scene_lr1e-2_token_lr1e-4_vis_xyz_tuning", "两者一起调 + 解析视锥拉 xyz", BOTH, "复刻", "1e-2", "1e-4",
               "vis_xyz 1.0", (*_joint("1e-4"), "--vis-mode", "vis_xyz")),
    Experiment("scene_lr1e-2_token_lr1e-4_vis_opac_tuning", "两者一起调 + 解析视锥压 opacity", BOTH, "复刻", "1e-2", "1e-4",
               "vis_opac 2.0", (*_joint("1e-4"), "--vis-mode", "vis_opac")),
    Experiment("scene_lr1e-2_token_lr1e-4_vis_opac_core_gate2.5_tuning",
               "两者一起调 + 视锥压 opacity + 离核(>2.5r)门控清理", BOTH, "复刻", "1e-2", "1e-4",
               "vis_opac 2.0 + core_gate 2.5",
               (*_joint("1e-4"), "--vis-mode", "vis_opac", "--core-gate-tau", "2.5")),
    Experiment("scene_lr1e-2_token_lr1e-4_colmap_depth0.01_tuning",
               "两者一起调 + COLMAP 原始稀疏深度（λ=0.01）", BOTH, "复刻", "1e-2", "1e-4",
               "colmap_raw_depth 0.01",
               (*_joint("1e-4"), "--depth-prior", "colmap_raw", "--depth-prior-weight", "0.01")),
    Experiment("scene_lr1e-2_token_lr1e-4_colmap_clean_depth0.01_tuning",
               "两者一起调 + COLMAP 去噪稀疏深度（λ=0.01，默认 COLMAP 基线）", BOTH, "复刻", "1e-2", "1e-4",
               "colmap_clean_depth 0.01",
               (*_joint("1e-4"), "--depth-prior", "colmap", "--depth-prior-weight", "0.01")),
    Experiment("scene_lr1e-2_token_lr1e-4_vggt_omega_depth0.01_tuning",
               "两者一起调 + VGGT-Ω Sim3 稠密深度（λ=0.01）", BOTH, "复刻", "1e-2", "1e-4",
               "vggt_omega_depth 0.01",
               (*_joint("1e-4"), "--depth-prior", "vggt_omega", "--depth-prior-weight", "0.01")),
)
BY_NAME = {e.name: e for e in EXPERIMENTS}


def method_of(name: str) -> str | None:
    """The canonical method of an experiment name (with inline _lr<...> stripped), None if unknown."""
    stripped = re.sub(r"_lr[0-9e.+-]+", "", name)
    return next((m for m in METHOD_ORDER if stripped == m or stripped.startswith(m + "_")), None)


def method_sort_key(name: str) -> tuple[int, int, str]:
    """Method order first, then the order of EXPERIMENTS, then the name; unknown names go last."""
    m = method_of(name)
    order = list(BY_NAME).index(name) if name in BY_NAME else len(BY_NAME)
    return (METHOD_ORDER.index(m) if m else len(METHOD_ORDER)), order, name


def run_status(name: str) -> str:
    st = Path(BY_NAME[name].dir) / "status.json"
    return json.loads(st.read_text()).get("status", "unknown") if st.is_file() else "missing"


def demo_names() -> list[str]:
    """Experiments that get a same-scene demo: the 140-scene run finished (a diverged run has no meaningful
    Gaussians)."""
    out = []
    for e in EXPERIMENTS:
        if run_status(e.name) == "complete":
            out.append(e.name)
        else:
            print(f"no demo for {e.name}: 140-scene run {run_status(e.name)}", file=sys.stderr)
    return out


def main(argv: list[str]) -> int:
    cmd, rest = (argv[0], argv[1:]) if argv else ("names", [])
    if cmd == "names":
        print("\n".join(BY_NAME))
    elif cmd == "demo-names":
        print("\n".join(demo_names()))
    elif cmd == "args" and len(rest) == 1 and rest[0] in BY_NAME:
        print(" ".join(BY_NAME[rest[0]].run_args()))
    elif cmd == "demo-args" and len(rest) == 2 and rest[0] in BY_NAME:
        print(" ".join(BY_NAME[rest[0]].demo_args(rest[1])))
    else:
        print(__doc__, file=sys.stderr)
        print(f"unknown command or experiment: {argv}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
