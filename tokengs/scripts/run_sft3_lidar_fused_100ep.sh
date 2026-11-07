#!/usr/bin/env bash
# [Exp-SFT-3 / 融合方案 + 解冻 deconv, 100 Epochs] 8-GPU Post-Training SFT on ScanNet++
# - Freezes  : [部位 1] 12-layer ViT + [部位 2 前8层] + decoder_blocks[0].gs_cross_attn
# - Trainable:
#     * 1.0x LR (4e-5): New SFT modules (`lidar_adapter` + `token_3d_grounding`)
#     * 0.25x LR (1e-5): Pretrained `[部位 2 后4层 + 部位 3]` + `[部位 4] gs_tokens & gs_self_attn & mlp` + `[部位 5] activation_head.deconv`
# - Automatically evaluates full 50-scene metrics (both w/o TTT and w/ scene+token TTT) upon completion!
#
# Usage:
#   bash scripts/run_sft3_lidar_fused_100ep.sh

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

PRESET="sft3_scannetpp_lidar_fused_100ep"
WORKSPACE="${WORKSPACE:-workspace_scannetpp/sft3_lidar_fused_100ep}"
CKPT_INIT="checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors"
mkdir -p "$WORKSPACE"
LOG_FILE="$WORKSPACE/train_$(date '+%Y%m%d_%H%M%S').log"

echo "========================================================================"
echo " [Exp-SFT-3] Fused LiDAR SFT + Unfrozen deconv, 100 Epochs (${NUM_GPUS} GPUs: ${GPUS})"
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
  --main_process_port "${MAIN_PORT:-29523}" \
  -m tokengs.train "$PRESET" \
  --workspace "$WORKSPACE" \
  --resume "$CKPT_INIT" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

if [ "${RUN_FULL_EVAL:-1}" = "1" ] && [ -f "$WORKSPACE/checkpoints/best/model.safetensors" ]; then
  echo "========================================================================"
  echo " Evaluating [Exp-SFT-3] Best Checkpoint on 50 ScanNet++ Val Scenes (w/o TTT & w/ TTT)..."
  echo "========================================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -u scripts/evaluate_scannetpp_full_metrics.py \
    --checkpoint "$WORKSPACE/checkpoints/best/model.safetensors" \
    --model-type tokengs_lidar_sft \
    --lidar-sft-mode fused_full_decoder \
    --ttt-mode none \
    --output-dir "EXP_ScanNetPP/07_sft3_lidar_fused_100ep_no_ttt" \
    2>&1 | tee -a "$LOG_FILE"

  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -u scripts/evaluate_scannetpp_full_metrics.py \
    --checkpoint "$WORKSPACE/checkpoints/best/model.safetensors" \
    --model-type tokengs_lidar_sft \
    --lidar-sft-mode fused_full_decoder \
    --ttt-mode scene_token \
    --output-dir "EXP_ScanNetPP/08_sft3_lidar_fused_100ep_with_ttt" \
    2>&1 | tee -a "$LOG_FILE"
fi
