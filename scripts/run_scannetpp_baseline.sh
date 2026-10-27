#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/one_click_launch_common.sh
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

OUTPUT="${OUTPUT:-EXP_ScanNetPP/baseline_pretrained_6view}"
cmd=("$TOKENGS_PYTHON" -u scripts/evaluate_scannetpp.py --output "$OUTPUT" "$@")
if [ "${DRY_RUN:-0}" = 1 ]; then
  printf 'CUDA_VISIBLE_DEVICES=<idle GPU> %s\n' "${cmd[*]}"
  exit 0
fi
[ ! -e "$OUTPUT" ] || { echo "Results already exist: $OUTPUT; refusing to overwrite."; exit 2; }
WAIT_FOR_IDLE_GPU="${WAIT_FOR_IDLE_GPU:-0}"
gpu=$(one_click_acquire_idle_gpu "$$") || {
  echo "No idle GPU selected (memory <=100 MiB and no compute app required)."
  exit 3
}
trap 'one_click_release_gpu "$gpu"' EXIT INT TERM
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT" 2>/dev/null || {
  echo "Results already exist: $OUTPUT; refusing to overwrite."
  exit 2
}
{
  echo "Started: $(date -Iseconds); GPU=$gpu; git=$(git rev-parse HEAD)"
  echo "Command: CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
  if CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"; then rc=0; else rc=$?; fi
  if [ "$rc" -eq 0 ]; then
    "$TOKENGS_PYTHON" scripts/scannetpp_evaluation_protocol.py build-table || rc=$?
  fi
  echo "Exit code: $rc; Finished: $(date -Iseconds)"
  exit "$rc"
} 2>&1 | tee "$OUTPUT/run.log"
