#!/usr/bin/env bash
# [Exp-SFT-2 / 方式二] 8-GPU Post-Training SFT on ScanNet++ with LiDAR 3D Surface Introspection + Waking up `gs_self_attn` (0.00036)
# - Frozen   : [部位 1] 12-layer ViT + [部位 2 前8层] + [部位 5] activation_head.deconv
# - Trainable: [部位 2 后4层 + 部位 3] + `LiDARSceneLatentProbeAdapter` + [部位 4] (`gs_tokens` + `decoder_blocks[0]`)
#
# Usage:
#   bash scripts/run_sft2_lidar_surface_sa.sh

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

PRESET="sft2_scannetpp_lidar_surface_sa"
WORKSPACE="${WORKSPACE:-workspace_scannetpp/sft2_lidar_surface_sa}"
CKPT_INIT="checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors"
mkdir -p "$WORKSPACE"
LOG_FILE="$WORKSPACE/train_$(date '+%Y%m%d_%H%M%S').log"

echo "========================================================================"
echo " [Exp-SFT-2 / 方式二] LiDAR Surface Self-Attn SFT (${NUM_GPUS} GPUs: ${GPUS})"
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
  --main_process_port "${MAIN_PORT:-29522}" \
  -m tokengs.train "$PRESET" \
  --workspace "$WORKSPACE" \
  --resume "$CKPT_INIT" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

if [ "${RUN_FULL_EVAL:-1}" = "1" ] && [ -f "$WORKSPACE/checkpoints/best/model.safetensors" ]; then
  echo "========================================================================"
  echo " Evaluating [Exp-SFT-2] Best Checkpoint on 50 ScanNet++ Val Scenes..."
  echo "========================================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -u scripts/evaluate_scannetpp_full_metrics.py \
    --checkpoint "$WORKSPACE/checkpoints/best/model.safetensors" \
    --model-type tokengs_lidar_sft \
    --lidar-sft-mode surface_self_attn \
    --ttt-mode none \
    --output-dir "EXP_ScanNetPP/04_sft2_lidar_surface_self_attn_no_ttt" \
    2>&1 | tee -a "$LOG_FILE"
fi
