#!/usr/bin/env bash
# 10 cm grid occupancy probe: can frozen RGB features + cameras tell which cells should hold Gaussians?
#   bash EXP/run_grid_occupancy.sh            # 128 train / 32 calibration / 50 validation scenes
# Uses every strictly idle GPU (no compute process, <= 100 MiB), checked right before launch.
# Per-scene grid features are cached in /mnt/disk/xveyan/workspace_scannetpp/geometrygs_<name>/ and reused when the
# protocol matches; one complete, unfiltered log: EXP/<name>/run.log.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
if [[ -z "${TOKENGS_ROOT:-}" ]]; then
  [[ -d "$ROOT/../TokenGS" ]] && TOKENGS_ROOT="$ROOT/../TokenGS" || TOKENGS_ROOT="$ROOT/../tokengs"
fi
source "$TOKENGS_ROOT/scripts/one_click_launch_common.sh"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 IDLE_MIN_FREE_MIB=999999999
NAME=${NAME:-grid_occupancy_10cm}
OUT="EXP/$NAME"
DEMO="EXP/demo/scene_7b6477cb95/$NAME"
WS="/mnt/disk/xveyan/workspace_scannetpp/geometrygs_$NAME"
if [[ -f "$OUT/summary.json" ]]; then
  echo "Completed experiment already exists: $OUT/summary.json"
  exit 0
fi
mkdir -p "$OUT"
mkdir "$OUT/.task.lock" || { echo "Experiment already running ($OUT/.task.lock)"; exit 4; }
ACTIVE_GPUS=""
cleanup() {
  if [[ -n "$ACTIVE_GPUS" ]]; then one_click_release_gpu "$ACTIVE_GPUS"; fi
  rmdir "$OUT/.task.lock" 2>/dev/null || true
}
trap cleanup EXIT
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ACTIVE_GPUS=$(one_click_acquire_all_idle_gpus "$$")
IFS=',' read -r -a cards <<< "$ACTIVE_GPUS"
cmd=(torchrun --standalone --nproc_per_node="${#cards[@]}" scripts/check_grid_occupancy.py
     --output "$OUT" --demo "$DEMO" --workspace "$WS" "$@")
{
  echo "=== Started $(date -Iseconds); host=$(hostname); GPUs=$ACTIVE_GPUS"
  echo "Command: CUDA_VISIBLE_DEVICES=$ACTIVE_GPUS ${cmd[*]}"
} >> "$OUT/run.log"
echo "Running $NAME on GPUs $ACTIVE_GPUS; full log: $OUT/run.log"
set +e
CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" "${cmd[@]}" >> "$OUT/run.log" 2>&1
code=$?
set -e
echo "=== Finished $(date -Iseconds); exit=$code" >> "$OUT/run.log"
exit "$code"
