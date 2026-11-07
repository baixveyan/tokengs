#!/usr/bin/env bash
# Flying-pixel diagnosis of DA3 depth on the 20 DA3-held-out ScanNet++ scenes, plus the fb5a96b1a2 demo.
#   bash EXP/run_da3_flying_pixels.sh
# Needs the label grids of EXP/run_grid_occupancy.sh. Uses one strictly idle GPU, checked right before launch.
# One complete, unfiltered log: EXP/da3_flying_pixels/run.log.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
if [[ -z "${TOKENGS_ROOT:-}" ]]; then
  [[ -d "$ROOT/../TokenGS" ]] && TOKENGS_ROOT="$ROOT/../TokenGS" || TOKENGS_ROOT="$ROOT/../tokengs"
fi
source "$TOKENGS_ROOT/scripts/one_click_launch_common.sh"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 IDLE_MIN_FREE_MIB=999999999
OUT=EXP/da3_flying_pixels
if [[ -f "$OUT/summary.json" ]]; then echo "Completed experiment already exists: $OUT/summary.json"; exit 0; fi
mkdir -p "$OUT"
mkdir "$OUT/.task.lock" || { echo "Experiment already running ($OUT/.task.lock)"; exit 4; }
ACTIVE_GPUS=""
cleanup() {
  if [[ -n "$ACTIVE_GPUS" ]]; then one_click_release_gpu "$ACTIVE_GPUS"; fi
  rmdir "$OUT/.task.lock" 2>/dev/null || true
}
trap cleanup EXIT
ACTIVE_GPUS=$(one_click_acquire_idle_gpu "$$")
cmd=(python -u scripts/diag_da3_flying_pixels.py --output "$OUT" "$@")
{ echo "=== Started $(date -Iseconds); GPU=$ACTIVE_GPUS"; echo "Command: CUDA_VISIBLE_DEVICES=$ACTIVE_GPUS ${cmd[*]}"; } >> "$OUT/run.log"
echo "Running on GPU $ACTIVE_GPUS; full log: $OUT/run.log"
set +e
CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" "${cmd[@]}" >> "$OUT/run.log" 2>&1
code=$?
set -e
echo "=== Finished $(date -Iseconds); exit=$code" >> "$OUT/run.log"
exit "$code"
