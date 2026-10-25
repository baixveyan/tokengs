#!/usr/bin/env bash
# 7-GPU joint training of the TokenGS 6-view latent-bottleneck baseline on ScanNet++ v2 DSLR
# Dataset: /home/jms/xveyan/data/4dgs/scannetpp (853 train scenes from nvs_sem_train.txt, 50 val scenes from nvs_sem_val.txt)
#
# Usage (run directly in your terminal to watch live progress):
#   bash scripts/run_train_scannetpp_baseline.sh
#
# Optional overrides:
#   GPUS="1,2,3,4,5,6,7" WORKSPACE="workspace_scannetpp/tokengs_baseline_6v" bash scripts/run_train_scannetpp_baseline.sh
#   MODE="scratch" bash scripts/run_train_scannetpp_baseline.sh   # train from scratch instead of DL3DV pretrained init

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOKENGS_CONCURRENT_RUNS=1

GPUS="${GPUS:-1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS="${#GPU_ARR[@]}"

MODE="${MODE:-scratch}"  # Default: "scratch" (full from-scratch training matching official TokenGS recipe) or "finetune"
if [ "$MODE" = "scratch" ]; then
  PRESET="train_scannetpp_latent_6view_ssim_scratch"
  DEFAULT_WS="workspace_scannetpp/tokengs_baseline_6v_scratch"
  RESUME_ARGS=()
else
  PRESET="finetune_scannetpp_latent_6view_ssim"
  DEFAULT_WS="workspace_scannetpp/tokengs_baseline_6v_finetune"
  RESUME_ARGS=(--resume checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors)
fi

WORKSPACE="${WORKSPACE:-$DEFAULT_WS}"
mkdir -p "$WORKSPACE"
LOG_FILE="$WORKSPACE/train_$(date '+%Y%m%d_%H%M%S').log"

echo "========================================================================"
echo " TokenGS ScanNet++ v2 Baseline Training (${NUM_GPUS} GPUs: ${GPUS})"
echo " Preset    : ${PRESET}"
echo " Workspace : ${WORKSPACE}"
echo " Log file  : ${LOG_FILE}"
echo "========================================================================"

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
  --multi_gpu \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "${MAIN_PORT:-29517}" \
  -m tokengs.train "$PRESET" \
  --workspace "$WORKSPACE" \
  "${RESUME_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$LOG_FILE"
