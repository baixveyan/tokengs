"""Compare the PLY results of several methods on the same scene(s).

Reads the <method>.json sidecars that evaluate_with_scene_latent_tuning.py --demo --ply-name writes into each
method's demo folder (<demo-dir>/scene_<id[:12]>/<method>/, see EXP_TokenGS/run_demo.sh), loads every
method's gaussians.pt from the same folder and, per scene, writes

  scene_<id[:12]>/comparison.md / comparison.json
        metrics table (image, distance buckets, sharpness, slot geometry, needles)
        + Gaussian-level difference of every method to the reference method
  scene_<id[:12]>/<method>/<method>_minus_<reference>.ply
        the method's Gaussian centres coloured by how far each Gaussian moved relative to the same Gaussian
        (same token, same child) of the reference

All methods decode the same tokens x children (release model: 4096 x 64), so Gaussian i is the same token /
child in every method and the difference is taken index by index.

    python scripts/compare_same_scene_ply.py --demo-dir EXP_TokenGS/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiment_registry import method_sort_key

ALIVE_OPACITY = 1.0 / 255.0
TABLE = (
    ("all PSNR", "all_psnr", "{:.3f}"), ("novel PSNR", "novel_psnr", "{:.3f}"),
    ("d0 PSNR", "bucket_d0_psnr", "{:.3f}"), ("d4+ PSNR", "bucket_d4+_psnr", "{:.3f}"),
    ("all SSIM", "all_ssim", "{:.4f}"), ("all LPIPS", "all_lpips", "{:.4f}"),
    ("HF ratio", "all_hf_ratio", "{:.3f}"), ("edge L1", "all_edge_l1", "{:.4f}"),
    ("SW", "geo_sw_loss", "{:.3f}"), ("packing ratio", "geo_packing_ratio", "{:.1f}"),
    ("inter d", "geo_inter_distance", "{:.3f}"), ("too close", "geo_inter_too_close", "{:.3f}"),
    ("aniso p50", "needle_aniso_p50_weighted", "{:.2f}"),
    ("needle>0.9", "needle_long_axis_along_ray_gt0.9_weighted", "{:.3f}"),
)


def _fmt(fmt: str, value) -> str:
    return "-" if value is None else fmt.format(value)


def gaussian_difference(ref: torch.Tensor, other: torch.Tensor, children: int) -> tuple[dict, torch.Tensor]:
    """Index-by-index difference of two [N,14] Gaussian sets; Gaussians alive (opacity > 1/255) in either."""
    alive = (ref[:, 3] > ALIVE_OPACITY) | (other[:, 3] > ALIVE_OPACITY)
    disp = (other[:, 0:3] - ref[:, 0:3]).norm(dim=-1)
    d_opacity = other[:, 3] - ref[:, 3]
    log_scale = (other[:, 4:7].clamp_min(1e-8).log() - ref[:, 4:7].clamp_min(1e-8).log()).abs().mean(dim=-1)
    d_rgb = (other[:, 11:14] - ref[:, 11:14]).abs().mean(dim=-1)

    def centroids(g):
        slots = g.reshape(-1, children, 14)
        w = slots[..., 3:4].clamp_min(1e-12)
        return (slots[..., 0:3] * w).sum(1) / w.sum(1)

    slot_disp = (centroids(other) - centroids(ref)).norm(dim=-1)
    # scene size: spread of the reference's alive centres, so displacements can be read as a fraction of it
    ref_alive = ref[ref[:, 3] > ALIVE_OPACITY, 0:3]
    extent = float((ref_alive - ref_alive.median(dim=0).values).norm(dim=-1).quantile(0.9))
    q = lambda x, p: float(x[alive].quantile(p)) if bool(alive.any()) else None  # noqa: E731
    stats = {
        "compared_gaussians_alive_in_either": int(alive.sum()),
        "reference_scene_extent_p90": extent,
        "centre_displacement_median": q(disp, 0.5),
        "centre_displacement_p90": q(disp, 0.9),
        "centre_displacement_median_over_extent": q(disp, 0.5) / extent if extent > 0 else None,
        "slot_centroid_displacement_median": float(slot_disp.median()),
        "opacity_change_mean_abs": float(d_opacity[alive].abs().mean()),
        "opacity_mean_reference": float(ref[:, 3].mean()),
        "opacity_mean_method": float(other[:, 3].mean()),
        "log_scale_change_median_abs": q(log_scale, 0.5),
        "rgb_change_mean_abs": float(d_rgb[alive].mean()),
    }
    return stats, disp


def export_difference_ply(path: Path, other: torch.Tensor, ref: torch.Tensor, disp: torch.Tensor, children: int):
    """Method's centres; colour blue (did not move) -> red (moved >= p95 displacement); scalar fields kept."""
    from plyfile import PlyData, PlyElement

    n = other.shape[0]
    t = (disp / disp.quantile(0.95).clamp_min(1e-12)).clamp(0, 1)
    rgb = torch.stack([t, 0.2 * torch.ones_like(t), 1 - t], dim=-1)
    columns = {
        "x": other[:, 0], "y": other[:, 1], "z": other[:, 2],
        "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2],
        "displacement": disp, "opacity_method": other[:, 3], "opacity_reference": ref[:, 3],
        "delta_opacity": other[:, 3] - ref[:, 3],
        "log_scale_ratio": (other[:, 4:7].clamp_min(1e-8).log() - ref[:, 4:7].clamp_min(1e-8).log()).mean(dim=-1),
        "slot_id": (torch.arange(n) // children).float(), "child_id": (torch.arange(n) % children).float(),
    }
    dtype = [(k, "u1") if k in ("red", "green", "blue") else (k, "f4") for k in columns]
    arr = np.empty(n, dtype=dtype)
    for k, v in columns.items():
        v = v.numpy()
        arr[k] = (v * 255).round().astype(np.uint8) if k in ("red", "green", "blue") else v.astype(np.float32)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def compare_scene(
    sidecar_paths: list[Path], scene_dir: Path, reference: str, allow_different_checkpoints: bool = False
) -> dict:
    """sidecar_paths: <scene_dir>/<method>/<method>.json, one per method of the same scene."""
    sidecars, folders = {}, {}
    for path in sorted(sidecar_paths, key=lambda p: method_sort_key(p.parent.name)):
        meta = json.load(open(path, encoding="utf-8"))
        if meta["method"] in sidecars:
            raise ValueError(f"{path}: method {meta['method']!r} appears twice")
        if path.parent.parent != scene_dir:
            raise ValueError(f"{path}: not in a method folder of {scene_dir}")
        meta["ply_folder"] = path.parent.name  # relative to scene_dir
        sidecars[meta["method"]] = meta
        folders[meta["method"]] = path.parent
    if not sidecars:
        raise ValueError(f"{scene_dir}: no method sidecars")
    scene_ids = {m["scene_id"] for m in sidecars.values()}
    shas = {m["checkpoint_sha256"] for m in sidecars.values()}
    children = {m["children_per_slot"] for m in sidecars.values()}
    if len(scene_ids) != 1 or (not allow_different_checkpoints and len(shas) != 1) or len(children) != 1:
        raise ValueError(f"{scene_dir}: sidecars disagree on scene {scene_ids} / checkpoint {shas} / "
                         f"children per token {children}")
    children = children.pop()
    gaussians = {name: torch.load(folders[name] / m["files"]["gaussians_pt"], map_location="cpu")[0].float()
                 for name, m in sidecars.items()}
    result = {
        "scene_id": scene_ids.pop(),
        "reference": reference,
        "multiple_checkpoints": len(shas) > 1,
        "methods": {},
    }
    for name, meta in sidecars.items():
        entry = {k: meta[k] for k in ("tuning_target", "tuning_loop", "tuning_steps", "tuning_lr", "token_lr",
                                      "tuning_loop_geo_weights", "ply_folder", "files", "decoder_inputs",
                                      "redecode_check")}
        entry["checkpoint_sha256"] = meta["checkpoint_sha256"]
        entry["metrics"] = meta["metrics"]
        if reference in gaussians and name != reference:
            stats, disp = gaussian_difference(gaussians[reference], gaussians[name], children)
            diff_ply = folders[name] / f"{name}_minus_{reference}.ply"
            export_difference_ply(diff_ply, gaussians[name], gaussians[reference], disp, children)
            entry["difference_to_reference"] = {**stats, "ply": f"{name}/{diff_ply.name}"}
        result["methods"][name] = entry
    with open(scene_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    (scene_dir / "comparison.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _markdown(result: dict) -> str:
    names = list(result["methods"])
    lines = [f"# Same-scene comparison {result['scene_id']}", "",
             "One folder per method next to this file, named after the method: `<method>.ply` (true colour), "
             "`<method>_slotcolor.ply` (one colour per token), `cameras.ply` (input red, evaluation green), "
             f"`<method>_minus_{result['reference']}.ply` (blue = Gaussian did not move, red = moved most).", ""]
    lines += [f"- {n}: `{result['methods'][n]['ply_folder']}/`" for n in names]
    lines += ["", "## Metrics (this scene)", "", "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for label, key, fmt in TABLE:
        lines.append(f"| {label} | " + " | ".join(_fmt(fmt, result["methods"][n]["metrics"].get(key)) for n in names)
                     + " |")
    diffs = [n for n in names if "difference_to_reference" in result["methods"][n]]
    if diffs:
        lines += ["", f"## Gaussian-level difference to {result['reference']} (same token, same child)", ""]
        if result.get("multiple_checkpoints"):
            lines += [
                "> Note: methods in this comparison use different checkpoints; index-by-index displacement "
                "is only meaningful when fine-tuned from shared initialization without token reordering.",
                "",
            ]
        lines += ["| | " + " | ".join(diffs) + " |", "|---" * (len(diffs) + 1) + "|"]
        rows = (("centre displacement median", "centre_displacement_median", "{:.4f}"),
                ("centre displacement p90", "centre_displacement_p90", "{:.4f}"),
                ("median displacement / scene extent", "centre_displacement_median_over_extent", "{:.4f}"),
                ("token centroid displacement median", "slot_centroid_displacement_median", "{:.4f}"),
                ("mean abs opacity change", "opacity_change_mean_abs", "{:.4f}"),
                ("median abs log-scale change", "log_scale_change_median_abs", "{:.3f}"),
                ("mean abs RGB change", "rgb_change_mean_abs", "{:.4f}"))
        for label, key, fmt in rows:
            lines.append(f"| {label} | " + " | ".join(
                _fmt(fmt, result["methods"][n]["difference_to_reference"][key]) for n in diffs) + " |")
    lines += ["", "## Re-decoding check (decoder_inputs.pt -> Gaussians, max abs diff)", ""]
    for n in names:
        check = result["methods"][n]["redecode_check"]
        lines.append(f"- {n}: " + ("not saved" if check is None else
                                   f"{check['max_abs_diff_vs_evaluated_gaussians']:.3e}"))
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo-dir", default="EXP_TokenGS/demo", help="folder with scene_<id[:12]>/<method>/")
    p.add_argument("--reference", default="no_tuning", help="method the others are compared to Gaussian by Gaussian")
    p.add_argument("--allow-different-checkpoints", action="store_true",
                   help="allow comparing methods evaluated from different model checkpoints on the same scene")
    args = p.parse_args()
    by_scene = find_sidecars(Path(args.demo_dir))
    if not by_scene:
        print(f"[FATAL] no scene_*/<method>/<method>.json sidecars in {args.demo_dir}", file=sys.stderr)
        return 2
    for scene_dir, paths in by_scene.items():
        result = compare_scene(paths, scene_dir, args.reference, args.allow_different_checkpoints)
        print(f"{scene_dir.name}: {', '.join(result['methods'])} -> {scene_dir / 'comparison.md'}")
    return 0


def find_sidecars(demo_dir: Path) -> dict[Path, list[Path]]:
    """scene folder -> sidecars <scene>/<method>/<method>.json; a sidecar must name its own folder."""
    by_scene: dict[Path, list[Path]] = {}
    for folder in sorted(p for p in demo_dir.glob("scene_*/*") if p.is_dir()):
        path = folder / f"{folder.name}.json"
        if not path.is_file():
            continue  # demo still running or failed: no sidecar yet
        method = json.load(open(path, encoding="utf-8"))["method"]
        if method != folder.name:
            raise ValueError(f"{path}: sidecar method {method!r} is not its folder name")
        by_scene.setdefault(folder.parent, []).append(path)
    return by_scene


if __name__ == "__main__":
    sys.exit(main())
