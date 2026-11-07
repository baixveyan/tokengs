"""50-scene ScanNet++ evaluation of SfM-anchored checkpoints with the unchanged TokenGS protocol
(scripts/evaluate_scannetpp.py: 6 inputs, 8 novel targets, FP32, G1-G4 geometry metrics, demo bundle).

The only additions: the SfM-aware provider / split_data / options are swapped in, and the demo bundle also gets
`sfm_points.ply` (all triangulated points), `sfm_anchors.ply` (the points used as fixed token positions) and
`anchored_vs_free_gaussians.ply` (Gaussian centres with opacity >= 0.05; anchored tokens red, free tokens grey).

    CUDA_VISIBLE_DEVICES=<idle gpu> python -m geometrygs.evaluate --checkpoint <model.safetensors> \
        --output EXP/<name> --demo-output EXP/demo/scene_7b6477cb95/<name>
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

import geometrygs  # noqa: F401
import evaluate_scannetpp as ev
from geometrygs.data import SfmProvider, split_data_with_sfm
from geometrygs.options import config_defaults as sfm_defaults
from geometrygs.plyio import write_points_ply, write_section_preview
from geometrygs.spatial_model import options_from_architecture

_models = []
_original_model_init = geometrygs.SfmTokenGS.__init__


def _tracking_init(self, *args, **kwargs):
    _original_model_init(self, *args, **kwargs)
    _models.append(self)


_original_export = ev.export_demo_bundle


def _export_with_sfm(demo_dir, *args, **kwargs):
    _original_export(demo_dir, *args, **kwargs)
    rec = _models[-1].last_sfm[0]
    pts = rec["points"].float().cpu().numpy()
    write_points_ply(Path(demo_dir) / "sfm_points.ply", pts, rec["color"].float().cpu().numpy())
    sel = rec["anchor_index"].cpu().numpy()
    write_points_ply(Path(demo_dir) / "sfm_anchors.ply", pts[sel], np.tile([[1.0, 0.1, 0.1]], (len(sel), 1)))
    # Gaussian centres (opacity >= 0.05) of anchored tokens in red, of free tokens in grey
    g = kwargs["gaussians"].detach().float().cpu()[0].numpy()
    anchored = np.zeros(len(g), dtype=bool)
    model = _models[-1]
    if hasattr(model, "last_spatial"):
        info = model.last_spatial
        slots = info["anchored"][0].cpu().numpy()
        anchored = np.repeat(slots, model.opt.dec_patch_size**2)
        write_points_ply(Path(demo_dir) / "token_centers.ply", info["centers"][0].cpu().numpy(),
                         np.where(slots[:, None], [[1., .1, .1]], [[.2, .5, 1.]]))
    else:
        anchored[: len(sel) * model.opt.dec_patch_size**2] = True
    keep = g[:, 3] >= 0.05
    colors = np.where(anchored[:, None], [[1.0, 0.15, 0.15]], [[0.6, 0.6, 0.6]])
    write_points_ply(Path(demo_dir) / "anchored_vs_free_gaussians.ply", g[keep, :3], colors[keep])
    write_section_preview(Path(demo_dir), g, anchored, pts[sel],
                          kwargs["cam_to_world_input"].detach().cpu().numpy()[0],
                          float(kwargs["scene_scale"]))


geometrygs.SfmTokenGS.__init__ = _tracking_init
ev.Provider = SfmProvider
ev.split_data = split_data_with_sfm
ev.config_defaults = {**ev.config_defaults, "eval_scannetpp_latent_6view_ssim": sfm_defaults["eval_sfm_anchor"]}
ev.export_demo_bundle = _export_with_sfm

def evaluation_options(path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if "_spatial_architecture" in f.keys():
            return options_from_architecture(f.get_tensor("_spatial_architecture"))
    return sfm_defaults["eval_sfm_anchor"]


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        ev.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--demo-only", action="store_true")
    args, _ = parser.parse_known_args()
    opt = evaluation_options(args.checkpoint)
    if args.model_type and args.model_type != opt.model_type:
        raise ValueError("--model-type disagrees with checkpoint architecture")
    if args.model_type is None:
        sys.argv += ["--model-type", opt.model_type]
    if opt.num_gs_tokens * opt.dec_patch_size**2 != 262144:
        raise ValueError("formal evaluation requires 262144 Gaussians")
    ev.config_defaults["eval_scannetpp_latent_6view_ssim"] = opt
    if opt.model_type == "geometrygs_spatial":
        def tracked(options):
            model = geometrygs.SpatialTokenGS(options)
            _models.append(model)
            return model
        ev.model_registry[opt.model_type] = tracked
    ev.main()
    if args.output and not args.demo_only:
        path = args.output / "summary.json"
        result = json.loads(path.read_text())
        result["model_type"] = opt.model_type
        result["sfm_from_input_views"] = getattr(opt, "sfm_enabled", True)
        path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
