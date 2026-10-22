"""Highlight a few tokens in the same-scene PLY files: selected tokens in distinct colours, with both a
context version (other Gaussians grey and almost transparent) and a selected-only version (other Gaussians removed),
so the Gaussians of one token can be followed through every method.

Reads the <method>.json sidecars and true-colour <method>.ply files that evaluate_with_scene_latent_tuning.py
--demo --ply-name writes (<demo-dir>/scene_<id[:12]>/<method>/); no GPU. Per scene:

- token selection, done once on the reference method (default no_tuning) and reused for every method:
  active tokens = opacity mass (sum of the 64 children's opacities) >= the median over all tokens; draw
  max(--counts) of them at random (fixed --seed). The first k drawn are the k-token set, so the 10 tokens of
  the 10-token file are also the first 10 of the 20-token file and keep the same colour.
- scene_<id>/<method>/<method>_highlight<k>tokens.ply for every method and every k: selected
  Gaussians keep position / shape / opacity and get their token colour; all other Gaussians become grey with
  opacity x --grey-opacity-scale. Extra field selected_token_rank (0..k-1, -1 = not selected) for Points mode.
- scene_<id>/<method>/<method>_only<k>tokens.ply: the same coloured selected Gaussians, with every other
  Gaussian completely removed. Original slot_id / child_id are preserved.
- scene_<id>/highlighted_tokens.{md,json}: legend (token id -> colour), both output variants and per-method stats.

    python scripts/highlight_selected_tokens_ply.py --demo-dir EXP_TokenGS/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from compare_same_scene_ply import find_sidecars
from experiment_registry import method_sort_key

SH_C0 = 0.28209479177387814
ALIVE_OPACITY = 1.0 / 255.0
GREY = (0.5, 0.5, 0.5)
# Sasha Trubetskoy's 20 distinct colours (grey replaced by black, since the unselected Gaussians are grey);
# the first 10 are the most distinct and are the colours of the 10-token file
PALETTE = (("red", "#e6194b"), ("green", "#3cb44b"), ("yellow", "#ffe119"), ("blue", "#4363d8"),
           ("orange", "#f58231"), ("purple", "#911eb4"), ("cyan", "#42d4f4"), ("magenta", "#f032e6"),
           ("lime", "#bfef45"), ("pink", "#fabed4"), ("teal", "#469990"), ("lavender", "#dcbeff"),
           ("brown", "#9a6324"), ("beige", "#fffac8"), ("maroon", "#800000"), ("mint", "#aaffc3"),
           ("olive", "#808000"), ("apricot", "#ffd8b1"), ("navy", "#000075"), ("black", "#000000"))


def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0


def read_vertices(path: Path) -> np.ndarray:
    from plyfile import PlyData

    return PlyData.read(str(path))["vertex"].data


def token_mass(v: np.ndarray, children: int) -> np.ndarray:
    return v["opacity_activated"].reshape(-1, children).sum(axis=1)


def select_tokens(ref: np.ndarray, children: int, n: int, seed: int) -> tuple[np.ndarray, dict]:
    mass = token_mass(ref, children)
    threshold = float(np.median(mass))
    active = np.flatnonzero(mass >= threshold)
    chosen = np.random.default_rng(seed).choice(active, size=n, replace=False)
    rule = {"reference_opacity_mass_median": threshold, "active_tokens": int(active.size),
            "total_tokens": int(mass.size), "seed": seed}
    return chosen, rule


def coloured_selected(v: np.ndarray, children: int, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copy vertices, colour selected tokens, add their palette rank and return (vertices, selected mask)."""
    out = np.empty(v.shape[0], dtype=v.dtype.descr + [("selected_token_rank", "f4")])
    for name in v.dtype.names:
        out[name] = v[name]
    slot = np.arange(v.shape[0]) // children
    rank = np.full(v.shape[0], -1, dtype=np.int64)
    for i, t in enumerate(tokens):
        rank[slot == t] = i
    selected = rank >= 0
    rgb = np.stack([hex_rgb(PALETTE[i][1]) for i in range(len(tokens))])
    for c in range(3):
        out[f"f_dc_{c}"][selected] = (rgb[rank[selected], c] - 0.5) / SH_C0
    out["selected_token_rank"] = rank
    return out, selected


def write_highlight(path: Path, v: np.ndarray, children: int, tokens: np.ndarray, grey_scale: float) -> None:
    from plyfile import PlyData, PlyElement

    out, selected = coloured_selected(v, children, tokens)
    for c in range(3):
        out[f"f_dc_{c}"][~selected] = (GREY[c] - 0.5) / SH_C0
    opacity = v["opacity_activated"].astype(np.float64)
    opacity[~selected] *= grey_scale
    out["opacity_activated"] = opacity
    clipped = np.clip(opacity, 1e-6, 1 - 1e-6)
    out["opacity"] = np.log(clipped / (1 - clipped))
    PlyData([PlyElement.describe(out, "vertex")]).write(str(path))


def write_selected_only(path: Path, v: np.ndarray, children: int, tokens: np.ndarray) -> None:
    """Write only the selected tokens; preserve every original field plus selected_token_rank."""
    from plyfile import PlyData, PlyElement

    out, selected = coloured_selected(v, children, tokens)
    PlyData([PlyElement.describe(out[selected], "vertex")]).write(str(path))


def token_stats(v: np.ndarray, children: int, t: int) -> dict:
    g = slice(t * children, (t + 1) * children)
    xyz = np.stack([v["x"][g], v["y"][g], v["z"][g]], axis=1).astype(np.float64)
    w = v["opacity_activated"][g].astype(np.float64)
    centre = (xyz * w[:, None]).sum(0) / max(w.sum(), 1e-12)
    spread = float(np.sqrt((w * ((xyz - centre) ** 2).sum(1)).sum() / max(w.sum(), 1e-12)))
    return {"alive": int((w > ALIVE_OPACITY).sum()), "opacity_mass": float(w.sum()),
            "centre": centre.tolist(), "spread": spread}


def markdown(scene: str, ref_method: str, rule: dict, counts: list[int], tokens: np.ndarray, stats: dict,
             files: dict, grey_scale: float) -> str:
    methods = list(stats)
    lines = [f"# 高亮 token 图例：{scene}", "",
             f"- 选法：在 `{ref_method}` 中按 token 的不透明度总量（64 个子 Gaussian 的不透明度之和）取前一半"
             f"（≥ 中位数 {rule['reference_opacity_mass_median']:.3f}，{rule['active_tokens']}/{rule['total_tokens']} 个），"
             f"固定随机种子 {rule['seed']} 随机抽 {max(counts)} 个；前 k 个就是 k 个 token 的文件，所以颜色在各文件里一致。",
             "- 所有方法用同一批 token 编号；选中的 Gaussian 保留原位置、形状和不透明度，只换颜色。",
             f"- `_highlight{{k}}tokens.ply` 保留其他 Gaussian，但将其设为灰色、不透明度 ×{grey_scale}；"
             "`_only{k}tokens.ply` 完全删除其他 Gaussian，只显示选中的 token。",
             "- Points 模式下可以按字段 `selected_token_rank` 着色（0…k−1 为选中，−1 为其余；only 文件没有 −1）。",
             "- 存活 = 不透明度 > 1/255 的子 Gaussian 数（满分 64）；中心 = 按不透明度加权的平均位置；"
             "散开半径 = 按不透明度加权的到中心的均方根距离；中心位移 = 与参考方法同一 token 中心的距离。", "",
             "## 文件", ""]
    lines += [f"- `{m}`：" + "、".join(f"`{p}`" for p in files[m]) for m in methods]
    lines += ["", "## 图例与各方法的 token 统计", "",
              "| 序号 | token | 颜色 | 在 10 个里 | " + " | ".join(
                  f"{m} 存活 / 不透明度总量 / 散开半径 / 中心位移" for m in methods) + " |",
              "|" + "---|" * (4 + len(methods))]
    for i, t in enumerate(tokens):
        name, hx = PALETTE[i]
        cells = []
        for m in methods:
            s = stats[m][int(t)]
            cells.append(f"{s['alive']} / {s['opacity_mass']:.2f} / {s['spread']:.3f} / {s['centre_shift']:.3f}")
        lines.append(f"| {i} | {int(t)} | {name} `{hx}` | {'是' if i < min(counts) else '否'} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo-dir", default="EXP_TokenGS/demo", help="folder with scene_<id[:12]>/<method>/")
    p.add_argument("--reference", default="no_tuning", help="method on which the tokens are selected")
    p.add_argument("--counts", type=int, nargs="+", default=[10, 20])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grey-opacity-scale", type=float, default=0.1)
    args = p.parse_args()
    counts = sorted(set(args.counts))
    if max(counts) > len(PALETTE):
        print(f"[FATAL] at most {len(PALETTE)} tokens (palette size)", file=sys.stderr)
        return 2
    scenes = {scene_dir: {path.parent.name: path for path in paths}
              for scene_dir, paths in find_sidecars(Path(args.demo_dir)).items()}
    if not scenes:
        print(f"[FATAL] no scene_*/<method>/<method>.json sidecars in {args.demo_dir}", file=sys.stderr)
        return 2
    for scene_dir, sidecars in scenes.items():
        scene = scene_dir.name
        if args.reference not in sidecars:
            print(f"[FATAL] {scene}: reference method {args.reference} missing", file=sys.stderr)
            return 2
        metas = {m: json.load(open(s, encoding="utf-8")) for m, s in sidecars.items()}
        methods = [args.reference, *sorted((m for m in sidecars if m != args.reference), key=method_sort_key)]
        verts = {m: read_vertices(sidecars[m].parent / metas[m]["files"]["ply"]) for m in methods}
        children = int(metas[args.reference]["children_per_slot"])
        if any(int(metas[m]["children_per_slot"]) != children or verts[m].shape != verts[args.reference].shape
               for m in methods):
            print(f"[FATAL] {scene}: methods differ in Gaussian count / children per token", file=sys.stderr)
            return 2
        tokens, rule = select_tokens(verts[args.reference], children, max(counts), args.seed)
        files, stats = {}, {}
        for m in methods:
            files[m] = []
            for k in counts:
                out = sidecars[m].parent / f"{m}_highlight{k}tokens.ply"
                only = sidecars[m].parent / f"{m}_only{k}tokens.ply"
                write_highlight(out, verts[m], children, tokens[:k], args.grey_opacity_scale)
                write_selected_only(only, verts[m], children, tokens[:k])
                files[m].extend((str(out.relative_to(scene_dir)), str(only.relative_to(scene_dir))))
                print("wrote", out)
                print("wrote", only)
            stats[m] = {int(t): token_stats(verts[m], children, int(t)) for t in tokens}
        for m in methods:
            for t, s in stats[m].items():
                s["centre_shift"] = float(np.linalg.norm(np.array(s["centre"]) - np.array(stats[args.reference][t]["centre"])))
        legend = {"scene": scene, "reference_method": args.reference, "selection": rule, "counts": counts,
                  "grey_opacity_scale": args.grey_opacity_scale,
                  "tokens": [{"rank": i, "token": int(t), "colour": PALETTE[i][0], "hex": PALETTE[i][1]}
                             for i, t in enumerate(tokens)],
                  "files": files, "token_stats": {m: {str(t): s for t, s in st.items()} for m, st in stats.items()}}
        base = scene_dir / "highlighted_tokens"
        base.with_suffix(".json").write_text(json.dumps(legend, indent=2) + "\n")
        base.with_suffix(".md").write_text(markdown(scene, args.reference, rule, counts, tokens, stats, files,
                                                    args.grey_opacity_scale))
        print("wrote", base.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
