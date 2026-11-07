#!/usr/bin/env bash
# One-click: post-train an SfM-anchored preset on all strictly idle GPUs, then evaluate the 50 ScanNet++ val scenes
# (6 inputs, 8 novel targets, FP32, G1-G4 geometry) and export the demo of scene 7b6477cb95.
#
#   bash EXP/run_train.sh                    # full-width spatial_sfm, pretrained image/scene path
#   bash EXP/run_train.sh spatial_free       # no-SfM spatial control
#   bash EXP/run_train.sh spatial_sfm_chain  # optional same-surface links
#   bash EXP/run_train.sh <preset>           # any preset in geometrygs/options.py
#   GPUS="1 2 3" bash EXP/run_train.sh       # restrict the GPUs that may be used
#
# Strict GPU rule: a GPU is used only if it has no compute process and <= 100 MiB memory, checked right before each
# task. Re-running resumes: training continues from the workspace's last checkpoint (same GPU count required),
# finished stages are skipped. One complete, unfiltered log: EXP/<preset>/run.log.
set -uo pipefail

NAME=${1:-spatial_sfm}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
if [ -z "${TOKENGS_ROOT:-}" ]; then
  if [ -d "$ROOT/../TokenGS" ]; then
    TOKENGS_ROOT=$(cd "$ROOT/../TokenGS" && pwd)
  else
    TOKENGS_ROOT=$(cd "$ROOT/../tokengs" && pwd)
  fi
fi
source "$TOKENGS_ROOT/scripts/one_click_launch_common.sh"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENGS_ROOT
export IDLE_MIN_FREE_MIB=999999999  # disables the "0 % utilisation" sharing rule: never share a GPU

OUT="EXP/$NAME"
LOG="$OUT/run.log"
WS="/mnt/disk/xveyan/workspace_scannetpp/geometrygs_$NAME"
INIT="$TOKENGS_ROOT/checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors"
case "$NAME" in
  spatial_sfm|spatial_free|spatial_sfm_chain|spatial_small_sfm) INIT="" ;; # initialization comes from preset
  sfm_anchor_k1024) ;;
  *) echo "[FATAL] Unsupported training preset: $NAME"; exit 2 ;;
esac
DEMO="EXP/demo/scene_7b6477cb95/$NAME"
mkdir -p "$OUT"
LOCK="$OUT/.task.lock"
mkdir "$LOCK" 2>/dev/null || { echo "[FATAL] $LOCK exists (another launch of $NAME is running?)"; exit 4; }
ACTIVE_GPUS=""
cleanup() { [ -n "$ACTIVE_GPUS" ] && one_click_release_gpu "$ACTIVE_GPUS"; rmdir "$LOCK" 2>/dev/null || true; }
trap 'echo "[$(date +%T)] interrupted"; cleanup; exit 130' INT TERM
trap cleanup EXIT

PY="$TOKENGS_PYTHON"
"$PY" -m geometrygs.train "$NAME" --help > /dev/null || { echo "[FATAL] unknown preset $NAME"; exit 2; }

if [ ! -f "$OUT/model.safetensors" ]; then
  if [ ! -f "$WS/status/COMPLETE" ]; then
    echo "--- GPU status"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    ACTIVE_GPUS=$(one_click_acquire_all_idle_gpus "$$") || { echo "[FATAL] no idle GPU"; exit 3; }
    IFS=',' read -r -a gpu_arr <<< "$ACTIVE_GPUS"
    n=${#gpu_arr[@]}
    if [ "$n" -gt 1 ]; then
      cmd=(accelerate launch --multi_gpu --num_processes "$n" --num_machines 1 --mixed_precision no
           --dynamo_backend no --main_process_port $((29500 + ($$ % 500))) -m geometrygs.train "$NAME")
    else
      cmd=("$PY" -u -m geometrygs.train "$NAME")
    fi
    cmd+=(--workspace "$WS")
    [ -z "$INIT" ] || cmd+=(--resume "$INIT")
    echo "[$(date +%T)] train $NAME on GPUs [$ACTIVE_GPUS] -> $LOG"
    {
      echo "=== Train started: $(date -Iseconds); host=$(hostname); GPUs=$ACTIVE_GPUS ($n)"
      echo "GeometryGS: $ROOT; TokenGS git: $(git -C "$TOKENGS_ROOT" rev-parse HEAD)"
      echo "Command: CUDA_VISIBLE_DEVICES=$ACTIVE_GPUS ${cmd[*]}"
    } >> "$LOG"
    CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" "${cmd[@]}" >> "$LOG" 2>&1
    rc=$?
    echo "Train exit code: $rc; Finished: $(date -Iseconds)" >> "$LOG"
    one_click_release_gpu "$ACTIVE_GPUS"; ACTIVE_GPUS=""
    [ "$rc" -eq 0 ] || { echo "[FATAL] training failed (exit $rc), see $LOG"; exit "$rc"; }
  fi
  [ -f "$WS/checkpoints/best/model.safetensors" ] || { echo "[FATAL] no best checkpoint in $WS"; exit 5; }
  cp "$WS/checkpoints/best/model.safetensors" "$OUT/model.safetensors.tmp" && mv "$OUT/model.safetensors.tmp" "$OUT/model.safetensors"
  echo "copied best checkpoint -> $OUT/model.safetensors"
fi

if [ ! -f "$OUT/summary.json" ]; then
  ACTIVE_GPUS=$(one_click_acquire_idle_gpu "$$") || { echo "[FATAL] no idle GPU"; exit 3; }
  cmd=("$PY" -u -m geometrygs.evaluate --checkpoint "$OUT/model.safetensors" --output "$OUT" --demo-output "$DEMO")
  echo "[$(date +%T)] evaluate $NAME on GPU $ACTIVE_GPUS -> $LOG"
  { echo "=== Eval started: $(date -Iseconds); GPU=$ACTIVE_GPUS"; echo "Command: CUDA_VISIBLE_DEVICES=$ACTIVE_GPUS ${cmd[*]}"; } >> "$LOG"
  CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" "${cmd[@]}" >> "$LOG" 2>&1
  rc=$?
  echo "Eval exit code: $rc; Finished: $(date -Iseconds)" >> "$LOG"
  one_click_release_gpu "$ACTIVE_GPUS"; ACTIVE_GPUS=""
  [ "$rc" -eq 0 ] || { echo "[FATAL] evaluation failed (exit $rc), see $LOG"; exit "$rc"; }
fi
echo "done: $OUT/summary.json, demo $DEMO"
