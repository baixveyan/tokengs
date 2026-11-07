"""Training entry: the TokenGS training loop with SfM-anchored options, model and data provider.

    accelerate launch --multi_gpu -m geometrygs.train sfm_anchor_k1024 --workspace <dir> --resume <official ckpt>
"""

import sys
import os
import subprocess

import tyro

import geometrygs  # noqa: F401  (registers model, provider, split_data)
import tokengs.train as tokengs_train
from geometrygs.options import AllSfmConfigs
from geometrygs.spatial_model import SpatialTokenGS, architecture_config


def load_training_options(args=None):
    return tyro.cli(AllSfmConfigs, args=list(sys.argv[1:] if args is None else args))


tokengs_train.load_training_options = load_training_options

_load_model_checkpoint = tokengs_train.load_model_checkpoint


def load_model_checkpoint(opt, model, accelerator, epoch_start):
    if not isinstance(model, SpatialTokenGS):
        return _load_model_checkpoint(opt, model, accelerator, epoch_start)
    if opt.resume is None or opt.resume == "None":
        return
    from tokengs.train import load_checkpoint_model
    from geometrygs.spatial_model import options_from_architecture
    state = load_checkpoint_model(opt.resume)
    if "_spatial_architecture" not in state:
        if epoch_start > 0:
            raise ValueError("resuming training requires a complete GeometryGS checkpoint")
        count = model.initialize_scene_backbone(state)
        accelerator.print(f"[GeometryGS] loaded {count / 1e6:.2f}M image/scene parameters; "
                          "all spatial decoder/head/SfM parameters are newly initialized")
        return
    restored = options_from_architecture(state["_spatial_architecture"])
    if architecture_config(restored) != architecture_config(opt):
        raise ValueError("spatial checkpoint architecture/options differ from requested model")
    model.load_state_dict(state, strict=True)


def ensure_val_depth_cache(opt, accelerator):
    if not opt.val_depth_metric:
        return
    if accelerator.is_main_process and not os.path.isfile(opt.val_depth_cache_path):
        subprocess.run(
            [sys.executable, "-m", "geometrygs.cache_depth", "--config",
             os.path.join(opt.workspace, "config.yaml"), "--out", opt.val_depth_cache_path],
            check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
    accelerator.wait_for_everyone()
    if not os.path.isfile(opt.val_depth_cache_path):
        raise RuntimeError("validation depth cache was not created")


_resolve_cache = tokengs_train.resolve_val_depth_cache_path


def resolve_cache(opt):
    if opt.val_depth_metric and not opt.val_depth_cache_path:
        import hashlib
        import json
        identity = (opt.validation_data_mode, opt.img_size, opt.num_input_views, opt.num_views,
                    opt.camera_normalization_method, opt.camera_scale_method,
                    opt.validation_dataset_kwargs or opt.dataset_kwargs)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        opt.val_depth_cache_path = os.path.join(os.path.dirname(os.path.abspath(opt.workspace)),
                                               "val_depth_cache", f"geometrygs_{digest}.pt")
    _resolve_cache(opt)


tokengs_train.load_model_checkpoint = load_model_checkpoint
tokengs_train.ensure_val_depth_cache = ensure_val_depth_cache
tokengs_train.resolve_val_depth_cache_path = resolve_cache

if __name__ == "__main__":
    tokengs_train.main()
