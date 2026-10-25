# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tyro
import time
import os
import datetime
import sys
from dataclasses import asdict

import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs, gather_object

import imageio
import numpy as np

from tokengs.options import AllConfigs, Options
from tokengs.checkpoint import (
    atomic_write_json, capture_provenance, checkpoint_disk_preflight, commit_checkpoint,
    load_checkpoint_model, options_sha256, release_checkpoint_read_lock,
    require_finite, restore_checkpoint, sha256_file, synchronized_call,
)
from tokengs.data import get_multi_dataloader
from tokengs.models import model_registry

import warnings

from tokengs.utils.gaussians import Gaussians
warnings.filterwarnings("ignore")


def setup_workspace_and_status(opt, accelerator):
    """Setup workspace directory and check for existing completion."""
    status_dir = os.path.join(opt.workspace, "status")
    complete_file = os.path.join(status_dir, "COMPLETE")
    
    def setup():
        os.makedirs(status_dir, exist_ok=True)
        if os.path.exists(complete_file):
            raise RuntimeError(f"Found existing COMPLETE file at {complete_file}, remove it if you want to run the job again")

    synchronized_call(accelerator, setup, main_only=True)


def load_checkpoint_and_resume(opt, accelerator):
    """核验当前来源；从已提交 last 恢复进度并对账已提交 best。"""
    return restore_checkpoint(opt, accelerator)


def setup_wandb(opt, accelerator, epoch_start, wandb_run_id):
    """Setup wandb logging."""
    if not accelerator.is_main_process or not opt.use_wandb:
        return None, None
    
    import wandb
    run_name = datetime.datetime.now().strftime("%b %d, %I:%M%p")
    
    # Initialize wandb - resume if we have a run_id, otherwise create new run
    if wandb_run_id and epoch_start > 0:
        wandb.init(config=asdict(opt), project=opt.project_name, group=opt.experiment_name, 
                  id=wandb_run_id, resume="must")
        print(f"Resuming wandb run {wandb_run_id}")
    else:
        wandb.init(config=asdict(opt), project=opt.project_name, group=opt.experiment_name, 
                  name=f"{opt.experiment_name} {run_name}")
        
    # Store wandb run id for potential future resuming
    wandb_run_id = wandb.run.id
    print(f"wandb run id: {wandb.run.id}")
    
    tensorboard_root_dir = f'{opt.out_dir}/{opt.experiment_name}' if opt.experiment_name else None
    wandb.tensorboard.patch(root_logdir=tensorboard_root_dir, save=False)
    writer = SummaryWriter(log_dir=tensorboard_root_dir)
    print(f"tensorboard root dir: {tensorboard_root_dir}")
    
    return wandb_run_id, writer


def load_model_checkpoint(opt, model, accelerator, epoch_start):
    """Load model checkpoint with tolerance for shape mismatches."""
    if opt.resume is None or opt.resume == 'None':
        return
    
    ckpt = load_checkpoint_model(opt.resume)
    
    if epoch_start > 0:
        require_finite(ckpt, 'restored model')
        model.load_state_dict(ckpt, strict=True)
        return

    # tolerant load (only load matching shapes)
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        else:
            accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')

    if opt.init_tokens_from_existing and epoch_start == 0:
        _initialize_tokens_from_existing(ckpt, state_dict, accelerator)

    if opt.init_latents_from_existing and epoch_start == 0:
        _initialize_latents_from_existing(ckpt, state_dict, accelerator)
    
    if opt.init_dynamic_tokens_from_static and epoch_start == 0 and 'gs_tokens' in ckpt:
        _initialize_dynamic_tokens_from_static(ckpt, state_dict, accelerator)


def _initialize_tokens_from_existing(ckpt, state_dict, accelerator):
    """Initialize tokens from existing checkpoint."""
    with torch.no_grad():
        for token_type in ['gs_tokens', 'gs_tokens_dynamic']:
            if token_type not in ckpt or token_type not in state_dict:
                continue
            pretrained_tokens = ckpt[token_type]
            current_tokens = state_dict[token_type]    
            N_old = pretrained_tokens.shape[0]
            N_new = current_tokens.shape[0]
            
            if N_new != N_old:
                accelerator.print(f'[INFO] Initializing {token_type} from pretrained tokens, N_old: {N_old}, N_new: {N_new}')
            
            if N_new <= N_old:
                # Downsample / take subset if fewer tokens
                idx = torch.linspace(0, N_old - 1, N_new).long()
                current_tokens.copy_(pretrained_tokens[idx])
            else:
                # Copy existing tokens first
                current_tokens[:N_old].copy_(pretrained_tokens)

                # Initialize additional tokens by sampling from pretrained tokens + noise
                extra_tokens = current_tokens[N_old:]
                repeat_factor = (extra_tokens.shape[0] + N_old - 1) // N_old

                expanded = pretrained_tokens.repeat((repeat_factor, 1))[:extra_tokens.shape[0]]
                noise = 0.01 * torch.randn_like(expanded)   # small perturbation
                extra_tokens.copy_(expanded + noise)


def _initialize_latents_from_existing(ckpt, state_dict, accelerator):
    """Initialize latent bottleneck tokens from an existing latent checkpoint."""
    key = "enc_dec_backbone.latents"
    if key not in ckpt or key not in state_dict:
        return

    with torch.no_grad():
        pretrained_latents = ckpt[key]
        current_latents = state_dict[key]
        N_old = pretrained_latents.shape[0]
        N_new = current_latents.shape[0]

        if N_new != N_old:
            accelerator.print(
                f"[INFO] Initializing latent bottleneck from checkpoint, N_old: {N_old}, N_new: {N_new}"
            )

        if N_new <= N_old:
            idx = torch.linspace(0, N_old - 1, N_new).long()
            current_latents.copy_(pretrained_latents[idx])
        else:
            current_latents[:N_old].copy_(pretrained_latents)
            extra_latents = current_latents[N_old:]
            repeat_factor = (extra_latents.shape[0] + N_old - 1) // N_old
            expanded = pretrained_latents.repeat((repeat_factor, 1))[: extra_latents.shape[0]]
            noise = 0.01 * torch.randn_like(expanded)
            extra_latents.copy_(expanded + noise)


def _initialize_dynamic_tokens_from_static(ckpt, state_dict, accelerator):
    """Initialize dynamic tokens from static tokens."""
    accelerator.print(f'[INFO] Initializing dynamic tokens from static tokens')
    
    with torch.no_grad():
        static_tokens = ckpt["gs_tokens"]  # shape: [N_static, D]
        dynamic_tokens = state_dict['gs_tokens_dynamic']          # shape: [N_dynamic, D]
        N_dynamic = dynamic_tokens.shape[0]
        N_static = static_tokens.shape[0]

        if N_dynamic == N_static:
            # direct copy
            dynamic_tokens.copy_(static_tokens)
        elif N_dynamic > N_static:
            # replicate or pad
            repeat_factor = (N_dynamic + N_static - 1) // N_static
            dynamic_tokens.copy_(
                static_tokens.repeat((repeat_factor, 1))[:N_dynamic]
            )
        else:
            # random subset if fewer dynamic tokens
            idx = torch.randperm(N_static)[:N_dynamic]
            dynamic_tokens.copy_(static_tokens[idx])


def setup_optimizer(opt, model, accelerator, epoch_start):
    """Setup optimizer with optional per-parameter `_lr_scale` support. Call before accelerator.prepare()."""
    groups_map = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if (param.dim() == 1 or getattr(param, '_no_weight_decay', False)) else 0.05
        lr_scale = float(getattr(param, '_lr_scale', 1.0))
        key = (wd, lr_scale)
        groups_map.setdefault(key, []).append(param)

    optim_groups = [
        {'params': params, 'weight_decay': wd, 'lr': opt.lr * lr_scale, '_lr_scale': lr_scale}
        for (wd, lr_scale), params in groups_map.items()
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=opt.lr, betas=(0.9, 0.95), fused=True)

    if epoch_start > 0:
        state_dir = getattr(opt, '_resume_state_dir', opt.workspace)
        state = torch.load(os.path.join(state_dir, 'optimizer.pth'), map_location='cpu')
        require_finite(state, 'restored optimizer')
        optimizer.load_state_dict(state)

    return optimizer


def setup_scheduler(opt, optimizer, iters_per_epoch, accelerator, epoch_start):
    """Setup scheduler. Call after accelerator.prepare() with per-GPU iters_per_epoch."""
    steps_per_epoch = iters_per_epoch // opt.gradient_accumulation_steps
    total_steps = opt.num_epochs * steps_per_epoch
    pct_start = min(opt.pct_start_steps / total_steps, 0.99)
    max_lrs = [opt.lr * float(g.get('_lr_scale', 1.0)) for g in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs, total_steps=total_steps,
        pct_start=pct_start, final_div_factor=opt.final_div_factor,
    )

    if epoch_start > 0:
        state_dir = getattr(opt, '_resume_state_dir', opt.workspace)
        state = torch.load(os.path.join(state_dir, 'scheduler.pth'), map_location='cpu')
        require_finite(state, 'restored scheduler')
        scheduler.load_state_dict(state)
        # OneCycleLR 构造会重置 lr/betas；重新加载保证 optimizer 与 scheduler 同步恢复。
        optimizer.load_state_dict(torch.load(os.path.join(state_dir, 'optimizer.pth'), map_location='cpu'))
        release_checkpoint_read_lock(opt)

    return scheduler


def save_checkpoint(
    opt,
    accelerator,
    model,
    optimizer,
    scheduler,
    epoch,
    wandb_run_id,
    val_metrics,
    best_val_psnr,
    config_sha256,
    initialization_checkpoint_sha256,
    metric_split='val',
):
    """只从有限 validation 指标选择 best；每 epoch 原子更新 last。"""
    return commit_checkpoint(
        opt, accelerator, model, optimizer, scheduler, epoch, wandb_run_id,
        val_metrics, best_val_psnr, config_sha256, initialization_checkpoint_sha256,
        metric_split=metric_split,
    )


def log_training_images(opt, accelerator, data, out, epoch, i, writer, is_train=True):
    """Log training/evaluation images or videos."""
    if not accelerator.is_main_process:
        return
        
    prefix = "train" if is_train else "eval"
    
    # Ensure images directory exists
    os.makedirs(f'{opt.workspace}/images', exist_ok=True)
    
    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
    pred_images = np.clip(out['images_pred'].detach().cpu().numpy(), 0, 1) # [B, V, 3, output_size, output_size]
    
    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[4], 3) # [B*output_size, V*output_size, 3]
    imageio.imwrite(f'{opt.workspace}/images/{prefix}_gt_images_{epoch}_{i}.jpg', (np.clip(gt_images, 0, 1) * 255).astype(np.uint8))

    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[4], 3)
    imageio.imwrite(f'{opt.workspace}/images/{prefix}_pred_images_{epoch}_{i}.jpg', (np.clip(pred_images, 0, 1) * 255).astype(np.uint8))

    if opt.use_wandb:
        writer.add_image(f'image/{prefix}_gt', gt_images.clip(0,1.0), epoch, dataformats='HWC')
        writer.add_image(f'image/{prefix}_pred', pred_images.clip(0,1.0), epoch, dataformats='HWC')


def train_epoch(opt, accelerator, model, optimizer, scheduler, train_dataloader, 
                iters_per_epoch, epoch, writer, start_time, train_dataset):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_psnr = 0
    log_time = time.time()

    def train_step(data, should_log_images, iteration):
        """Execute a single training step and return metrics."""
        global_step_for_aux = epoch * iters_per_epoch + iteration
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "compute_lambda_dyn_aux_eff"):
            unwrapped_model.lambda_dyn_aux_eff = unwrapped_model.compute_lambda_dyn_aux_eff(
                global_step_for_aux,
                opt,
            )

        out = model(data)
        loss = out['loss']
        psnr = out['psnr']
        backward_loss = out.get('backward_loss', loss)
        accelerator.backward(backward_loss)

        if opt.use_geo_losses and epoch == 0 and iteration == 0 and accelerator.is_main_process:
            activation_weight = accelerator.unwrap_model(model).activation_head.deconv.weight
            grad = activation_weight.grad
            grad_norm = float("nan") if grad is None else grad.detach().float().norm().item()
            grad_finite = grad is not None and torch.isfinite(grad).all().item()
            print(
                f"[GEO_GRAD] activation_head.deconv.weight "
                f"norm: {grad_norm:.9f} finite: {grad_finite}"
            )

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

        optimizer.step()
        if accelerator.sync_gradients:
            scheduler.step()
        optimizer.zero_grad()

        loss_value = loss.detach()
        psnr_value = psnr.detach()
        loss_value_detailed = {
            'loss_rgb': out['loss_rgb'].detach(),
        }
        if 'loss_ssim' in out:
            loss_value_detailed['loss_ssim'] = out['loss_ssim'].detach()
        if 'loss_visibility' in out:
            loss_value_detailed['loss_visibility'] = out['loss_visibility'].detach()
        if 'loss_opacity' in out:
            loss_value_detailed['loss_opacity'] = out['loss_opacity'].detach()
        for key in ('loss_dyn_aux', 'psnr_dyn_aux', 'lambda_dyn_aux_eff'):
            if key in out:
                loss_value_detailed[key] = out[key].detach()
        for key in (
            'sw_loss',
            'packing_loss',
            'inter_loss',
            'total_geo_loss',
            'geo_packing_ratio',
            'geo_packing_target',
            'geo_effective_children',
            'geo_center_variance',
            'geo_local_scale_variance',
            'geo_inter_distance',
        ):
            if key in out:
                loss_value_detailed[key] = out[key].detach()
        
        if should_log_images:
            log_training_images(opt, accelerator, data, out, epoch, iteration, writer, is_train=True)
        
        return loss_value, loss_value_detailed, psnr_value

    model.zero_grad(set_to_none=True)

    train_dataset.set_rng_epoch(epoch)
    if accelerator.is_main_process:
        print(f"[INFO] Setting RNG epoch to {epoch}")

    for i, data in enumerate(iter(train_dataloader)):
        if i >= opt.max_iters_per_epoch:
            break

        # 正式 run 的 use_wandb=False 同时关闭本地训练媒体；正频率仅服务显式媒体日志。
        should_log_images = (accelerator.is_main_process and opt.use_wandb
                             and opt.log_image_freq > 0 and i % opt.log_image_freq == 0)
            
        with accelerator.accumulate(model):
            loss_value, loss_value_detailed, psnr_value = train_step(data, should_log_images, i)
            total_loss += loss_value
            total_psnr += psnr_value

            if opt.use_wandb and accelerator.is_main_process:
                global_step = epoch * iters_per_epoch + i
                writer.add_scalar(f"psnr/train_iteration", psnr_value.item(), global_step)
                writer.add_scalar(f"loss/train_iteration", loss_value.item(), global_step)
                writer.add_scalar(f"lr/train_iteration", scheduler.get_last_lr()[0], global_step)
                try:
                    writer.add_scalar(f"time/train_iteration", time.time() - start_time, global_step)
                    for key, value in loss_value_detailed.items():
                        writer.add_scalar(f"loss_detailed/{key}/train_iteration", value.item(), global_step)
                except: 
                    pass

        if accelerator.is_main_process:
            # logging
            if i % opt.print_freq == 0:
                mem_free, mem_total = torch.cuda.mem_get_info()    
                elapsed = time.time() - log_time
                speed = opt.print_freq / elapsed if elapsed > 0 else 0
                geometry_log = "".join(
                    f" {key}: {value.item():.6f}"
                    for key, value in loss_value_detailed.items()
                    if key in {
                        'sw_loss',
                        'packing_loss',
                        'inter_loss',
                        'total_geo_loss',
                        'geo_packing_ratio',
                        'geo_packing_target',
                        'geo_effective_children',
                        'geo_center_variance',
                        'geo_local_scale_variance',
                        'geo_inter_distance',
                    }
                )
                print(f"[INFO] {i}/{iters_per_epoch} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.10f} loss: {loss_value.item():.6f} speed: {speed:.2f} it/s{geometry_log}")
                log_time = time.time()

    total_loss = accelerator.gather_for_metrics(total_loss).mean()
    total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
    
    if accelerator.is_main_process:
        total_loss /= iters_per_epoch
        total_psnr /= iters_per_epoch
        accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")

        if opt.use_wandb:
            writer.add_scalar(f"psnr/train", total_psnr.item(), epoch)
            writer.add_scalar(f"loss/train", total_loss.item(), epoch)


def log_gaussian_histograms(opt, all_gaussians, epoch, writer):
    """Log histograms of Gaussian properties to wandb."""
    # Concatenate all gaussians: [B, N, 14] -> [B*N, 14]
    gaussians = Gaussians.from_raw(torch.cat(all_gaussians, dim=0).reshape(-1, 14))
    
    # Log position histograms (x, y, z separately)
    # Clamp to bin range so out-of-bounds values appear in extreme bins
    pos_bins = torch.linspace(-25, 25, 100)
    for i, label in enumerate("xyz"):
        writer.add_histogram(f'gaussian/pos_{label}', gaussians.xyz[:, i].clamp(-25, 25), global_step=epoch, bins=pos_bins)
    
    # Log opacity histogram [0, 1]
    opacity_bins = torch.linspace(0, 1, 100)
    writer.add_histogram('gaussian/opacity', gaussians.opacity.flatten().clamp(0, 1), global_step=epoch, bins=opacity_bins)
    
    scale_bins = torch.linspace(0, opt.gaussian_scale_cap, 100)
    for i, label in enumerate("xyz"):
        writer.add_histogram(f'gaussian/scale_{label}', gaussians.scaling[:, i].clamp(0, opt.gaussian_scale_cap), global_step=epoch, bins=scale_bins)
    
    # Log rotation histograms [-1, 1] (quaternion components)
    rotation_bins = torch.linspace(-1, 1, 100)
    for i, label in enumerate("wxyz"):
        writer.add_histogram(f'gaussian/rotation_{label}', gaussians.rotation[:, i].clamp(-1, 1), global_step=epoch, bins=rotation_bins)
    
    # Log RGB histograms [0, 1]
    rgb_bins = torch.linspace(0, 1, 100)
    for i, label in enumerate("rgb"):
        writer.add_histogram(f'gaussian/rgb_{label}', gaussians.rgb[:, i].clamp(0, 1), global_step=epoch, bins=rgb_bins)


def _batch_scene_ids(value):
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _batch_frame_ids(value):
    if isinstance(value, torch.Tensor):
        rows = value.detach().cpu()
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        return [[int(item) for item in row.tolist()] for row in rows]
    if value and isinstance(value[0], (int, np.integer)):
        return [[int(item) for item in value]]
    return [[int(item) for item in row] for row in value]


def validate_batch_identity(data):
    required = ('scene_id', 'frame_ids', 'requested_scene_id', 'requested_frame_ids')
    missing = [name for name in required if name not in data]
    if missing:
        raise RuntimeError(f"Evaluation batch is missing identity fields: {missing}")
    actual_scenes = _batch_scene_ids(data['scene_id'])
    requested_scenes = _batch_scene_ids(data['requested_scene_id'])
    actual_frames = _batch_frame_ids(data['frame_ids'])
    requested_frames = _batch_frame_ids(data['requested_frame_ids'])
    if not (len(actual_scenes) == len(requested_scenes) == len(actual_frames) == len(requested_frames)):
        raise RuntimeError("Evaluation identity batch dimensions do not match")
    identities = []
    for actual_scene, requested_scene, actual_frame_ids, requested_frame_ids in zip(
        actual_scenes, requested_scenes, actual_frames, requested_frames
    ):
        if actual_scene != requested_scene:
            raise RuntimeError(
                f"Evaluation scene identity mismatch: requested={requested_scene}, actual={actual_scene}"
            )
        if actual_frame_ids != requested_frame_ids:
            raise RuntimeError(
                f"Evaluation frame identity mismatch for {actual_scene}: "
                f"requested={requested_frame_ids}, actual={actual_frame_ids}"
            )
        identities.append({'scene_id': actual_scene, 'frame_ids': actual_frame_ids})
    return identities


def deduplicate_scene_records(records, expected_scene_ids):
    expected = [str(scene_id) for scene_id in expected_scene_ids]
    if not expected:
        raise RuntimeError('Validation scene set is empty')
    if len(expected) != len(set(expected)):
        raise RuntimeError("Expected validation scene IDs are not unique")
    unique = {}
    for record in records:
        normalized = {
            'scene_id': str(record['scene_id']),
            'frame_ids': [int(value) for value in record['frame_ids']],
            'psnr': float(record['psnr']),
        }
        require_finite(normalized['psnr'], 'gathered validation PSNR')
        scene_id = normalized['scene_id']
        if scene_id in unique and unique[scene_id] != normalized:
            raise RuntimeError(f"Inconsistent duplicate validation record for scene {scene_id}")
        unique[scene_id] = normalized
    expected_set = set(expected)
    actual_set = set(unique)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        raise RuntimeError(
            f"Validation scene set mismatch: missing={missing}, unexpected={unexpected}"
        )
    return [unique[scene_id] for scene_id in expected]


def expected_scene_ids(dataset):
    providers = getattr(dataset, 'datasets', [dataset])
    scene_ids = []
    for provider in providers:
        samples = getattr(getattr(provider, 'dataset', provider), 'sample_list', None)
        if samples is None:
            raise RuntimeError("Validation dataset does not expose sample_list")
        scene_ids.extend(os.path.splitext(os.path.basename(str(path)))[0] for path in samples)
    return scene_ids


def _per_scene_psnr(out, data):
    if 'psnr_per_scene' in out:
        return out['psnr_per_scene'].detach().float().reshape(-1)
    pred = out['images_pred'].detach().float()
    target = data['images_output'].detach().float()
    per_view_mse = (pred - target).square().mean(dim=(-1, -2, -3)).clamp_min(1e-12)
    return (-10.0 * torch.log10(per_view_mse)).mean(dim=1)


def evaluate_epoch(opt, accelerator, model, test_dataloader, epoch, writer, expected_scenes=None):
    """Evaluate validation by unique scene identity, ignoring DDP sampler padding."""
    use_input_supervision = opt.use_input_supervision
    opt.use_input_supervision = False
    local_records = []
    all_gaussians = []
    try:
        with torch.inference_mode():
            model.eval()
            for i, data in enumerate(iter(test_dataloader)):
                identities = validate_batch_identity(data)
                out = model(data)
                psnr_values = _per_scene_psnr(out, data)
                synchronized_call(
                    accelerator,
                    lambda: (require_finite(psnr_values, 'validation per-scene PSNR'),
                             require_finite(out.get('psnr'), 'validation batch PSNR')),
                )
                if len(psnr_values) != len(identities):
                    raise RuntimeError("Validation PSNR count does not match scene identities")
                local_records.extend(
                    identity | {'psnr': float(psnr)}
                    for identity, psnr in zip(identities, psnr_values.cpu().tolist())
                )
                if accelerator.is_main_process and opt.use_wandb:
                    all_gaussians.append(out['gaussians'].detach().cpu())
                    if opt.log_image_freq > 0 and i % opt.log_image_freq == 0:
                        log_training_images(opt, accelerator, data, out, epoch, i, writer, is_train=False)

            torch.cuda.empty_cache()
            gathered_records = gather_object(local_records)
            if expected_scenes is None:
                expected_scenes = expected_scene_ids(test_dataloader.dataset)
            unique_records = deduplicate_scene_records(gathered_records, expected_scenes)
            val_psnr = sum(record['psnr'] for record in unique_records) / len(unique_records)
            require_finite(val_psnr, 'aggregated validation PSNR')
            metrics = {'psnr': val_psnr, 'scene_count': len(unique_records)}
            if accelerator.is_main_process:
                accelerator.print(
                    f"[eval] epoch: {epoch} psnr: {val_psnr:.4f} unique_scenes: {len(unique_records)}"
                )
                if opt.use_wandb:
                    writer.add_scalar("psnr/eval", val_psnr, epoch)
                    if all_gaussians:
                        log_gaussian_histograms(opt, all_gaussians, epoch, writer)
            return metrics
    finally:
        opt.use_input_supervision = use_input_supervision


def complete_training(opt, accelerator, val_metrics, best_val_psnr):
    """成功标记只能在所有 rank 的有限 validation 后发布。"""
    synchronized_call(
        accelerator,
        lambda: (require_finite(val_metrics, 'completion validation'),
                 require_finite(best_val_psnr, 'completion best_val_psnr')),
    )

    def publish():
        status_dir = os.path.join(opt.workspace, 'status')
        os.makedirs(status_dir, exist_ok=True)
        atomic_write_json(os.path.join(status_dir, 'COMPLETE'), {
            'num_epochs': opt.num_epochs, 'val_metrics': val_metrics,
            'best_val_psnr': best_val_psnr,
        })
        print(f"[INFO] Training completed successfully after {opt.num_epochs} epochs")

    synchronized_call(accelerator, publish, main_only=True)
    accelerator.wait_for_everyone()


def load_training_options(args=None):
    """可选配置快照入口；未提供--config时沿用全部原preset。"""
    args = list(sys.argv[1:] if args is None else args)
    if '--config' not in args:
        return tyro.cli(AllConfigs, args=args)
    index = args.index('--config')
    if index + 1 >= len(args):
        raise ValueError('--config requires a YAML path')
    config_path = args[index + 1]
    del args[index:index + 2]
    with open(config_path, encoding='utf-8') as handle:
        default_opt = tyro.extras.from_yaml(Options, handle.read())
    return tyro.cli(Options, default=default_opt, args=args)


def main():    
    start_time = time.time()
    opt = load_training_options()

    torch.manual_seed(opt.seed)

    kwargs_handlers = []
    if "TOKENGS_DDP_BACKEND" in os.environ:
        kwargs_handlers.append(InitProcessGroupKwargs(backend=os.environ["TOKENGS_DDP_BACKEND"]))
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        kwargs_handlers=kwargs_handlers,
    )

    # Freeze provenance before resume mutates opt.resume to the last checkpoint.
    config_yaml = tyro.extras.to_yaml(opt)

    # Setup workspace and status
    setup_workspace_and_status(opt, accelerator)

    # Load checkpoint and resume
    resume_state = load_checkpoint_and_resume(opt, accelerator)
    epoch_start = resume_state['epoch_start']
    best_val_psnr = resume_state['best_val_psnr']
    wandb_run_id = resume_state['wandb_run_id']
    # 来源由恢复入口在修改 opt.resume 前重新捕获，不使用保存状态中的旧 SHA。
    provenance = opt._checkpoint_provenance
    config_sha256 = provenance['config_sha256']
    initialization_checkpoint_sha256 = provenance['initialization_checkpoint_sha256']

    # Setup wandb
    wandb_run_id, writer = setup_wandb(opt, accelerator, epoch_start, wandb_run_id)
    
    if accelerator.is_main_process:
        print(opt)

        config_save_path = os.path.join(opt.workspace, "config.yaml")
        if not os.path.exists(config_save_path):
            with open(config_save_path, "w", encoding="utf-8") as f:
                f.write(config_yaml)
            print(f"[INFO] Config saved to {config_save_path=}")

    # model
    model = model_registry[opt.model_type](opt)

    # Load model checkpoint
    load_model_checkpoint(opt, model, accelerator, epoch_start)
    if hasattr(model, "configure_sft_trainable_parameters"):
        model.configure_sft_trainable_parameters(accelerator)
    
    # Data
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_multi_dataloader(opt, accelerator)
    validation_scene_ids = expected_scene_ids(test_dataset)

    # Optimizer (before prepare)
    optimizer = setup_optimizer(opt, model, accelerator, epoch_start)

    # accelerate (shards dataloader across GPUs)
    ddp_cls = torch.nn.parallel.DistributedDataParallel
    original_ddp_init = ddp_cls.__init__
    if accelerator.num_processes > 1:
        def ddp_init_without_initial_sync(self, *args, **kwargs):
            kwargs["init_sync"] = False
            return original_ddp_init(self, *args, **kwargs)

        ddp_cls.__init__ = ddp_init_without_initial_sync
    try:
        model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, test_dataloader
        )
    finally:
        ddp_cls.__init__ = original_ddp_init

    # Compute per-GPU iterations from the prepared (sharded) dataloader
    iters_per_epoch = min(len(train_dataloader), opt.max_iters_per_epoch)

    # Scheduler (after prepare, so iters_per_epoch is correct)
    # NOTE: do NOT call accelerator.prepare(scheduler) here -- total_steps is already
    # computed from the per-GPU iters_per_epoch, and prepare() would divide it again
    # by num_processes, causing the LR to decay to zero far too early.
    scheduler = setup_scheduler(opt, optimizer, iters_per_epoch, accelerator, epoch_start)
    # 在首次 validation/训练前按正式双并发策略核查 checkpoint 峰值，所有 rank 同步拒绝。
    checkpoint_disk_preflight(opt, accelerator, model, optimizer, scheduler)

    # loop
    os.makedirs(opt.workspace, exist_ok=True)

    val_metrics = evaluate_epoch(
        opt, accelerator, model, test_dataloader, epoch_start - 1, writer,
        expected_scenes=validation_scene_ids,
    )
    
    epoch = epoch_start
    while epoch < opt.num_epochs:
        # train
        train_epoch(opt, accelerator, model, optimizer, scheduler, train_dataloader, 
                    iters_per_epoch, epoch, writer, start_time, train_dataset)

        # Only the formal, identity-checked, deduplicated validation metric selects best.
        val_metrics = evaluate_epoch(
            opt, accelerator, model, test_dataloader, epoch, writer,
            expected_scenes=validation_scene_ids,
        )
        best_val_psnr = save_checkpoint(
            opt,
            accelerator,
            model,
            optimizer,
            scheduler,
            epoch,
            wandb_run_id,
            val_metrics,
            best_val_psnr,
            config_sha256,
            initialization_checkpoint_sha256,
            metric_split='val',
        )

        epoch += 1
            
    # If we get here, we've completed all epochs
    # Signal successful completion
    complete_training(opt, accelerator, val_metrics, best_val_psnr)
    return


if __name__ == "__main__":
    main()
