#!/usr/bin/env bash
# [Exp-SFT-4 / 修复 3 大掉点病根 + 面积守恒去细针 + 纯旋转转刀片, 40 Epochs] 8-GPU Post-Training SFT on ScanNet++
# - Fix 1: `loss_blade_orient` only optimizes rotation quaternion `rot_all` (zero gradient on scale `sc_all`), with soft hinge at cos > 0.35 for side walls.
# - Fix 2: `loss_aniso` is strictly area-preserving (`s1_tgt = sqrt(2*s1*s2), s2_tgt = sqrt(0.5*s1*s2)`), preserving 2D disk area `s1 * s2 = const` and never touching `s3`.
# - Fix 3: `probe_head(raw_latents.detach())` isolates early random probe errors from shocking pretrained `latent_blocks.8..11`.
# - Trains for 40 Epochs (~50 minutes on 8 GPUs) and automatically runs full 50-scene evaluation (w/o TTT and w/ TTT).
#
# Usage:
#   bash scripts/run_sft4_lidar_area_preserving_40ep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENGS_CONCURRENT_RUNS=1

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS="${#GPU_ARR[@]}"

PRESET="sft4_scannetpp_lidar_area_preserving_40ep"
WORKSPACE="${WORKSPACE:-workspace_scannetpp/sft4_lidar_area_preserving_40ep}"
CKPT_INIT="checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors"
mkdir -p "$WORKSPACE"
LOG_FILE="$WORKSPACE/train_$(date '+%Y%m%d_%H%M%S').log"

echo "========================================================================"
echo " [Exp-SFT-4] Area-Preserving LiDAR SFT + Unfrozen deconv, 40 Epochs (${NUM_GPUS} GPUs: ${GPUS})"
echo " Init ckpt : ${CKPT_INIT}"
echo " Workspace : ${WORKSPACE}"
echo " Log file  : ${LOG_FILE}"
echo "========================================================================"

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
  --multi_gpu \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "${MAIN_PORT:-29524}" \
  -m tokengs.train "$PRESET" \
  --workspace "$WORKSPACE" \
  --resume "$CKPT_INIT" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

if [ "${RUN_FULL_EVAL:-1}" = "1" ] && [ -f "$WORKSPACE/checkpoints/best/model.safetensors" ]; then
  echo "========================================================================"
  echo " Evaluating [Exp-SFT-4] Best Checkpoint on 50 ScanNet++ Val Scenes (w/o TTT & w/ TTT)..."
  echo "========================================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -u scripts/evaluate_scannetpp_full_metrics.py \
    --checkpoint "$WORKSPACE/checkpoints/best/model.safetensors" \
    --model-type tokengs_lidar_sft \
    --lidar-sft-mode fused_full_decoder \
    --ttt-mode none \
    --output-dir "EXP_ScanNetPP/09_sft4_lidar_area_preserving_40ep_no_ttt" \
    2>&1 | tee -a "$LOG_FILE"

  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -u scripts/evaluate_scannetpp_full_metrics.py \
    --checkpoint "$WORKSPACE/checkpoints/best/model.safetensors" \
    --model-type tokengs_lidar_sft \
    --lidar-sft-mode fused_full_decoder \
    --ttt-mode scene_token \
    --output-dir "EXP_ScanNetPP/10_sft4_lidar_area_preserving_40ep_with_ttt" \
    2>&1 | tee -a "$LOG_FILE"
fi
