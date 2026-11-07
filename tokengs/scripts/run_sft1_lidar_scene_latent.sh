#!/usr/bin/env bash
# [Exp-SFT-1 / 方式一] 8-GPU Post-Training SFT on ScanNet++ with LiDAR 3D Surface Introspection & Zero-Init Residual Injection
# - Frozen   : [部位 1] 12-layer ViT + [部位 2 前8层] + [部位 4+5] gs_tokens + decoder_blocks + deconv
# - Trainable: [部位 2 后4层 + 部位 3] (`latents`, `latent_blocks.8..11`, `k_proj/v_proj`) + `LiDARSceneLatentProbeAdapter`
#
# Usage:
#   bash scripts/run_sft1_lidar_scene_latent.sh

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

PRESET="sft1_scannetpp_lidar_scene_latent"
WORKSPACE="${WORKSPACE:-workspace_scannetpp/sft1_lidar_scene_latent}"
CKPT_INIT="checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors"
mkdir -p "$WORKSPACE"
LOG_FILE="$WORKSPACE/train_$(date '+%Y%m%d_%H%M%S').log"

echo "========================================================================"
echo " [Exp-SFT-1 / 方式一] LiDAR Scene-Latent Probe SFT (${NUM_GPUS} GPUs: ${GPUS})"
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
  --main_process_port "${MAIN_PORT:-29521}" \
  -m tokengs.train "$PRESET" \
  --workspace "$WORKSPACE" \
  --resume "$CKPT_INIT" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

if [ "${RUN_FULL_EVAL:-1}" = "1" ] && [ -f "$WORKSPACE/checkpoints/best/model.safetensors" ]; then
  echo "========================================================================"
  echo " Evaluating [Exp-SFT-1] Best Checkpoint on 50 ScanNet++ Val Scenes..."
  echo "========================================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -u scripts/evaluate_scannetpp_full_metrics.py \
    --checkpoint "$WORKSPACE/checkpoints/best/model.safetensors" \
    --model-type tokengs_lidar_sft \
    --lidar-sft-mode scene_latent_probe \
    --ttt-mode none \
    --output-dir "EXP_ScanNetPP/03_sft1_lidar_scene_latent_probe_no_ttt" \
    2>&1 | tee -a "$LOG_FILE"
fi
