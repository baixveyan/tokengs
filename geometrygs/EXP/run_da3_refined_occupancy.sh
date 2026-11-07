#!/usr/bin/env bash
# DA3 depth -> 10 cm cells with deployable refinements (edge filter, multi-view consistency, SfM scale), then the demo.
#   bash EXP/run_da3_refined_occupancy.sh
# Needs the label grids of EXP/run_grid_occupancy.sh. Uses one strictly idle GPU, checked right before launch.
# One complete, unfiltered log: EXP/da3_refined_occupancy_10cm/run.log.
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
NAME=da3_refined_occupancy_10cm
OUT="EXP/$NAME"
DEMO_SCENE=fb5a96b1a2
DEMO_VARIANT=sfm_global_edge  # fixed by design (edge filter + SfM scale), not chosen from the results
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
run() {
  local desc=$1; shift
  { echo "=== $desc started $(date -Iseconds); GPU=$ACTIVE_GPUS"; echo "Command: CUDA_VISIBLE_DEVICES=$ACTIVE_GPUS $*"; } >> "$OUT/run.log"
  set +e
  CUDA_VISIBLE_DEVICES="$ACTIVE_GPUS" "$@" >> "$OUT/run.log" 2>&1
  local code=$?
  set -e
  echo "=== $desc finished $(date -Iseconds); exit=$code" >> "$OUT/run.log"
  [[ $code -eq 0 ]] || { echo "[FATAL] $desc failed (exit $code), see $OUT/run.log"; exit "$code"; }
}
ACTIVE_GPUS=$(one_click_acquire_idle_gpu "$$")
echo "Running $NAME on GPU $ACTIVE_GPUS; full log: $OUT/run.log"
if [[ ! -f "$OUT/summary.json" ]]; then
  run "DA3 refined occupancy" python -u scripts/check_da3_refined_occupancy.py --output "$OUT" --labels "$LABELS" \
      --workspace "$WS" --demo-scene "$DEMO_SCENE"
fi
run "demo" python -u scripts/demo_grid_occupancy.py --workspace "$LABELS" --scene "$DEMO_SCENE" \
    --mask-file "$WS/$DEMO_SCENE.npz" --mask-key "$DEMO_VARIANT" --demo-extra "$WS/${DEMO_SCENE}_demo.npz" \
    --output "EXP/demo/scene_$DEMO_SCENE/$NAME" --title "DA3 + edge filter + SfM scale"
echo "done: $OUT/summary.json, demo EXP/demo/scene_$DEMO_SCENE/$NAME"
