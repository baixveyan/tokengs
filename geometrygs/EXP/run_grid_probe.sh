#!/usr/bin/env bash
# Small grid study: exact contribution statistics + shared frozen-feature activation classifiers.
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
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 IDLE_MIN_FREE_MIB=999999999 MAX_JOBS=4
OUT=${OUT:-EXP/grid_activation}
DEMO=${DEMO:-"EXP/demo/scene_7b6477cb95/$(basename "$OUT")"}
if [[ -f "$OUT/summary.json" ]]; then
  echo "Completed experiment already exists: $OUT/summary.json"
  exit 0
fi
mkdir -p "$OUT"
mkdir "$OUT/.task.lock" || { echo "Experiment already running"; exit 4; }
ACTIVE_GPUS=""
cleanup() {
  if [[ -n "$ACTIVE_GPUS" ]]; then one_click_release_gpu "$ACTIVE_GPUS"; fi
  rmdir "$OUT/.task.lock" 2>/dev/null || true
}
trap cleanup EXIT
ACTIVE_GPUS=$(one_click_acquire_all_idle_gpus "$$")
IFS=',' read -r -a cards <<< "$ACTIVE_GPUS"
cmd=(torchrun --standalone --nproc_per_node="${#cards[@]}" scripts/check_grid_queries.py --output "$OUT" --demo "$DEMO")
cmd+=("$@")
{
  echo "Started $(date -Iseconds); GPUs=$ACTIVE_GPUS"
  echo "Command: CUDA_VISIBLE_DEVICES=$ACTIVE_GPUS ${cmd[*]}"
} >> "$OUT/run.log"
echo "Running grid study on GPUs $ACTIVE_GPUS; full log: $OUT/run.log"
set +e
CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" "${cmd[@]}" >> "$OUT/run.log" 2>&1
code=$?
set -e
echo "Finished $(date -Iseconds); exit=$code" >> "$OUT/run.log"
exit "$code"
