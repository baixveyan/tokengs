#!/usr/bin/env bash
# One-click: run ScanNet++ 50-scene feed-forward evaluation for all (or specified) experiments in
# scripts/scannetpp_evaluation_protocol.py on idle GPUs, then rebuild EXP_ScanNetPP/all_experiments_table.{md,csv}.
#
#   bash EXP_ScanNetPP/run_50_scenes.sh                 # every experiment in scannetpp_evaluation_protocol.py
#   bash EXP_ScanNetPP/run_50_scenes.sh <name> ...      # only these experiments
#   GPUS="1 2" bash EXP_ScanNetPP/run_50_scenes.sh      # optional: force GPU indices
#   DRY_RUN=1 bash EXP_ScanNetPP/run_50_scenes.sh       # only show what would run (no GPU, no log)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/one_click_launch_common.sh
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

protocol() { "$TOKENGS_PYTHON" scripts/scannetpp_evaluation_protocol.py "$@"; }
mapfile -t ALL < <(protocol names)
[ "${#ALL[@]}" -gt 0 ] || { echo "[FATAL] experiment list is empty"; exit 2; }
if [ $# -ge 1 ]; then
  GROUPS_IN_ORDER=()
  declare -A SEEN_EXP=()
  for n in "$@"; do
    printf '%s\n' "${ALL[@]}" | grep -qx -- "$n" || {
      echo "[FATAL] unknown experiment $n (see: python scripts/scannetpp_evaluation_protocol.py names)"
      exit 2
    }
    if [ -z "${SEEN_EXP[$n]:-}" ]; then
      SEEN_EXP[$n]=1
      GROUPS_IN_ORDER+=("$n")
    fi
  done
else
  GROUPS_IN_ORDER=("${ALL[@]}")
fi
echo "experiments: ${GROUPS_IN_ORDER[*]}"

mapfile -t DONE_RGB < <(protocol demo-names)
mapfile -t DONE_GEO < <(protocol complete-with-geo-names)
RECONS_CACHE="/tmp/tokengs_scannetpp_layout_20261026_115429/pretrained_6view_rgb_fp32_50val/reconstructions"
TODO=()
declare -A ACTION_OF=()
for g in "${GROUPS_IN_ORDER[@]}"; do
  out="EXP_ScanNetPP/$g"
  if printf '%s\n' "${DONE_GEO[@]}" | grep -qx -- "$g"; then
    echo "skip $g: already complete with G1-G4 geometry metrics ($out)"
  elif printf '%s\n' "${DONE_RGB[@]}" | grep -qx -- "$g"; then
    if [ "$g" = "baseline_pretrained_6view" ] && [ -d "$RECONS_CACHE" ]; then
      TODO+=("$g")
      ACTION_OF["$g"]="backfill_geo"
    else
      echo "[FATAL] $out has RGB metrics but is missing G1-G4 geometry metrics and no reconstruction cache; move $out aside to re-evaluate"
      exit 4
    fi
  elif [ -e "$out" ] && [ -n "$(find "$out" -mindepth 1 -maxdepth 1 ! -name 'run.log' ! -name 'model.safetensors' -print -quit 2>/dev/null)" ]; then
    echo "[FATAL] $out already exists with partial evaluation files; inspect or move it aside first"
    exit 4
  else
    TODO+=("$g")
    ACTION_OF["$g"]="full"
  fi
done

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN=1: would run ${#TODO[@]} experiment(s) sequentially, each using all currently idle GPUs"
  for g in "${TODO[@]}"; do
    act="${ACTION_OF[$g]}"
    preset="$(protocol train-preset "$g")"
    if [ "$act" = "backfill_geo" ]; then
      echo "  $g [backfill G1-G4] -> EXP_ScanNetPP/$g/run.log"
      echo "    CUDA_VISIBLE_DEVICES=<idle GPU> $TOKENGS_PYTHON -u scripts/evaluate_scannetpp.py $(protocol args "$g") --backfill-geometry-from-reconstructions $RECONS_CACHE"
    elif [ -n "$preset" ]; then
      echo "  $g [train $preset on ALL idle GPUs + eval 50 scenes] -> EXP_ScanNetPP/$g/run.log"
      echo "    CUDA_VISIBLE_DEVICES=<all idle GPUs> accelerate launch --multi_gpu -m tokengs.train $preset --workspace workspace_scannetpp/$g --resume checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors"
      echo "    CUDA_VISIBLE_DEVICES=<idle GPU> $TOKENGS_PYTHON -u scripts/evaluate_scannetpp.py $(protocol args "$g")"
    else
      echo "  $g [eval 50 scenes] -> EXP_ScanNetPP/$g/run.log"
      echo "    CUDA_VISIBLE_DEVICES=<idle GPU> $TOKENGS_PYTHON -u scripts/evaluate_scannetpp.py $(protocol args "$g")"
    fi
  done
  exit 0
fi

FAILED=0
ACTIVE_GPUS=""
ACTIVE_LOCK_DIR=""
cleanup_current_task() {
  [ -n "$ACTIVE_GPUS" ] && one_click_release_gpu "$ACTIVE_GPUS"
  [ -n "$ACTIVE_LOCK_DIR" ] && rmdir "$ACTIVE_LOCK_DIR" 2>/dev/null || true
}
trap 'echo "[$(date +%T)] interrupted: stopping current task and releasing GPU locks"; cleanup_current_task; exit 130' INT TERM
trap 'cleanup_current_task' EXIT

if [ "${#TODO[@]}" -gt 0 ]; then
  for g in "${TODO[@]}"; do
    act="${ACTION_OF[$g]}"
    preset="$(protocol train-preset "$g")"
    ckpt="$(protocol checkpoint "$g")"
    out="EXP_ScanNetPP/$g"
    log="$out/run.log"
    rc=0

    echo "--- live GPU status before starting $g"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

    if [ "$act" = "backfill_geo" ]; then
      ACTIVE_GPUS=$(one_click_acquire_idle_gpu "$$") || {
        echo "[FATAL] no idle GPU available for $g"
        FAILED=1
        continue
      }
      ACTIVE_LOCK_DIR="$out/.backfill.lock"
      if ! mkdir "$ACTIVE_LOCK_DIR" 2>/dev/null; then
        echo "[FATAL] $ACTIVE_LOCK_DIR already exists (concurrent backfill refused)"
        one_click_release_gpu "$ACTIVE_GPUS"
        ACTIVE_GPUS=""
        ACTIVE_LOCK_DIR=""
        FAILED=1
        continue
      fi
      if protocol complete-with-geo-names | grep -qx -- "$g"; then
        echo "skip $g: geometry metrics already completed"
      else
        read -r -a args <<< "$(protocol args "$g")"
        cmd=("$TOKENGS_PYTHON" -u scripts/evaluate_scannetpp.py "${args[@]}" --backfill-geometry-from-reconstructions "$RECONS_CACHE")
        echo "[$(date +%T)] start $g (backfill G1-G4) on verified idle GPU $ACTIVE_GPUS -> $log"
        {
          echo "=== Backfill G1-G4 started: $(date -Iseconds); GPU=$ACTIVE_GPUS; git=$(git rev-parse HEAD)"
          echo "Command: CUDA_VISIBLE_DEVICES=$ACTIVE_GPUS ${cmd[*]}"
        } >> "$log"
        CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" "${cmd[@]}" >> "$log" 2>&1
        rc=$?
        echo "Exit code: $rc; Finished: $(date -Iseconds)" >> "$log"
      fi
      rmdir "$ACTIVE_LOCK_DIR" 2>/dev/null || true
      ACTIVE_LOCK_DIR=""
      one_click_release_gpu "$ACTIVE_GPUS"
      ACTIVE_GPUS=""
    else
      if [ -n "$preset" ] && [ ! -f "$ckpt" ]; then
        ACTIVE_GPUS=$(one_click_acquire_all_idle_gpus "$$") || {
          echo "[FATAL] no idle GPU available for training $g"
          FAILED=1
          continue
        }
        IFS=',' read -r -a gpu_arr <<< "$ACTIVE_GPUS"
        num_gpus="${#gpu_arr[@]}"
        mkdir -p "$out"
        ACTIVE_LOCK_DIR="$out/.task.lock"
        if ! mkdir "$ACTIVE_LOCK_DIR" 2>/dev/null; then
          echo "[FATAL] $ACTIVE_LOCK_DIR already exists (concurrent launch refused)"
          one_click_release_gpu "$ACTIVE_GPUS"
          ACTIVE_GPUS=""
          ACTIVE_LOCK_DIR=""
          FAILED=1
          continue
        fi
        echo "=== Started: $(date -Iseconds); git=$(git rev-parse HEAD)" >> "$log"
        ws="workspace_scannetpp/$g"
        port=$((29500 + ($$ % 500)))
        if [ "$num_gpus" -gt 1 ]; then
          train_cmd=(
            accelerate launch
            --multi_gpu
            --num_processes "$num_gpus"
            --num_machines 1
            --mixed_precision no
            --dynamo_backend no
            --main_process_port "$port"
            -m tokengs.train "$preset"
            --workspace "$ws"
            --resume checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors
          )
        else
          train_cmd=(
            "$TOKENGS_PYTHON" -u -m tokengs.train "$preset"
            --workspace "$ws"
            --resume checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors
          )
        fi
        echo "[$(date +%T)] start $g (train $preset) on $num_gpus verified idle GPU(s) [$ACTIVE_GPUS] -> $log"
        echo "Train GPUs=$ACTIVE_GPUS ($num_gpus GPUs); Command: CUDA_VISIBLE_DEVICES=$ACTIVE_GPUS ${train_cmd[*]}" >> "$log"
        CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" "${train_cmd[@]}" >> "$log" 2>&1
        rc=$?
        one_click_release_gpu "$ACTIVE_GPUS"
        ACTIVE_GPUS=""
        if [ "$rc" -eq 0 ] && [ -f "$ws/checkpoints/best/model.safetensors" ]; then
          cp -f "$ws/checkpoints/best/model.safetensors" "$ckpt"
        else
          rc=${rc:-1}
          [ "$rc" -ne 0 ] || rc=1
        fi
      else
        mkdir -p "$out"
        ACTIVE_LOCK_DIR="$out/.task.lock"
        if ! mkdir "$ACTIVE_LOCK_DIR" 2>/dev/null; then
          echo "[FATAL] $ACTIVE_LOCK_DIR already exists (concurrent launch refused)"
          ACTIVE_LOCK_DIR=""
          FAILED=1
          continue
        fi
        echo "=== Started: $(date -Iseconds); git=$(git rev-parse HEAD)" >> "$log"
      fi

      if [ "$rc" -eq 0 ]; then
        echo "--- live GPU status before evaluating $g"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        ACTIVE_GPUS=$(one_click_acquire_idle_gpu "$$") || {
          echo "[FATAL] no idle GPU available for evaluating $g"
          rmdir "$ACTIVE_LOCK_DIR" 2>/dev/null || true
          ACTIVE_LOCK_DIR=""
          FAILED=1
          continue
        }
        read -r -a args <<< "$(protocol args "$g")"
        eval_cmd=("$TOKENGS_PYTHON" -u scripts/evaluate_scannetpp.py "${args[@]}")
        echo "[$(date +%T)] start $g (eval 50 scenes) on verified idle GPU $ACTIVE_GPUS -> $log"
        echo "Eval GPU=$ACTIVE_GPUS; Command: CUDA_VISIBLE_DEVICES=$ACTIVE_GPUS ${eval_cmd[*]}" >> "$log"
        CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" "${eval_cmd[@]}" >> "$log" 2>&1
        rc=$?
        one_click_release_gpu "$ACTIVE_GPUS"
        ACTIVE_GPUS=""
      fi
      echo "Exit code: $rc; Finished: $(date -Iseconds)" >> "$log"
      rmdir "$ACTIVE_LOCK_DIR" 2>/dev/null || true
      ACTIVE_LOCK_DIR=""
    fi

    echo "[$(date +%T)] end $g (exit $rc)"
    [ "$rc" -eq 0 ] || FAILED=1
  done
fi

echo "--- rebuilding EXP_ScanNetPP/all_experiments_table.{md,csv}"
protocol build-table || FAILED=1
echo "=== run_50_scenes finished $(date '+%F %T'), failed: $FAILED"
exit "$FAILED"
