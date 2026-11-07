#!/usr/bin/env bash
# Pretrained DA3-GIANT-1.1 depth -> 10 cm cells vs laser labels (labels from EXP/grid_occupancy_10cm), then the demo.
#   bash EXP/run_da3_occupancy.sh
# Uses one strictly idle GPU (no compute process, <= 100 MiB), checked right before launch.
# One complete, unfiltered log: EXP/da3_occupancy_10cm/run.log.
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
NAME=da3_occupancy_10cm
OUT="EXP/$NAME"
DEMO_SCENE=fb5a96b1a2
LABELS=/mnt/disk/xveyan/workspace_scannetpp/geometrygs_grid_occupancy_10cm
WS="/mnt/disk/xveyan/workspace_scannetpp/geometrygs_$NAME"
[[ -f "$LABELS/$DEMO_SCENE.pt" ]] || { echo "[FATAL] label grids missing: run bash EXP/run_grid_occupancy.sh first"; exit 2; }
mkdir -p "$OUT"
mkdir "$OUT/.task.lock" || { echo "Experiment already running ($OUT/.task.lock)"; exit 4; }
ACTIVE_GPUS=""
cleanup() {
  if [[ -n "$ACTIVE_GPUS" ]]; then one_click_release_gpu "$ACTIVE_GPUS"; fi
  rmdir "$OUT/.task.lock" 2>/dev/null || true
}
trap cleanup EXIT
run() {  # run <description> <command...>: full output appended to the log
  local desc=$1; shift
  { echo "=== $desc started $(date -Iseconds); GPU=$ACTIVE_GPUS"; echo "Command: CUDA_VISIBLE_DEVICES=$ACTIVE_GPUS $*"; } >> "$OUT/run.log"
  set +e
  CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" "$@" >> "$OUT/run.log" 2>&1
  local code=$?
  set -e
  echo "=== $desc finished $(date -Iseconds); exit=$code" >> "$OUT/run.log"
  [[ $code -eq 0 ]] || { echo "[FATAL] $desc failed (exit $code), see $OUT/run.log"; exit "$code"; }
}
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ACTIVE_GPUS=$(one_click_acquire_idle_gpu "$$")
echo "Running $NAME on GPU $ACTIVE_GPUS; full log: $OUT/run.log"
if [[ ! -f "$OUT/summary.json" ]]; then
  run "DA3 occupancy" python -u scripts/check_da3_occupancy.py --output "$OUT" --labels "$LABELS" --workspace "$WS" \
      --demo-scene "$DEMO_SCENE"
fi
run "demo" python -u scripts/demo_grid_occupancy.py --workspace "$LABELS" --scene "$DEMO_SCENE" \
    --mask-file "$WS/$DEMO_SCENE.npz" --mask-key camera_scale__all_pixels --demo-extra "$WS/${DEMO_SCENE}_demo.npz" \
    --output "EXP/demo/scene_$DEMO_SCENE/$NAME" --title "DA3-GIANT-1.1 depth (camera scale)"
echo "done: $OUT/summary.json, demo EXP/demo/scene_$DEMO_SCENE/$NAME"
