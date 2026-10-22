#!/usr/bin/env bash
# One-click: the 140-scene runs of EXP_TokenGS, one folder per experiment (EXP_TokenGS/<experiment>/), in
# parallel on idle GPUs, then the reports and the total table.
#
#   bash EXP_TokenGS/run_140_scenes.sh                 # every experiment in scripts/experiment_registry.py
#   bash EXP_TokenGS/run_140_scenes.sh <name> ...      # only these experiments
#   GPUS="1 2 3" bash ...                              # optional: force GPU indices
#   DRY_RUN=1 bash ...                                 # only show what would run (no GPU, no log)
#
# Names, settings and order: scripts/experiment_registry.py (python scripts/experiment_registry.py names).
# A finished experiment is skipped: complete, or failed (e.g. the four lr 1e-2 token runs that diverged,
# kept as evidence). A folder that exists but is neither stops the launch.
# Logs: EXP_TokenGS/<experiment>/console.log (complete, unfiltered) and
#       EXP_TokenGS/launch_logs/run_140_scenes.<timestamp>.log for everything shown in the terminal.
# Reports: EXP_TokenGS/reports/{scene_geo_tuning,scene_token_x_geo}.{md,json}, demo scene pick
#          EXP_TokenGS/demo/demo_scene.json, total table EXP_TokenGS/all_experiments_table.{md,csv}.
set -uo pipefail
source "$(dirname "$0")/../scripts/one_click_launch_common.sh"
one_click_setup run_140_scenes

registry() { "$TOKENGS_PYTHON" scripts/experiment_registry.py "$@"; }
mapfile -t ALL < <(registry names)
[ "${#ALL[@]}" -gt 0 ] || { echo "[FATAL] experiment list is empty"; exit 2; }
if [ $# -ge 1 ]; then
  for n in "$@"; do
    printf '%s\n' "${ALL[@]}" | grep -qx -- "$n" || { echo "[FATAL] unknown experiment $n (see: python scripts/experiment_registry.py names)"; exit 2; }
  done
  GROUPS_IN_ORDER=("$@")
else
  GROUPS_IN_ORDER=("${ALL[@]}")
fi
echo "experiments: ${GROUPS_IN_ORDER[*]}"

group_output() { echo "EXP_TokenGS/$1"; }
group_args() { registry args "$1"; }

PENDING=0
for g in "${GROUPS_IN_ORDER[@]}"; do [ -f "$(group_output "$g")/status.json" ] || PENDING=$((PENDING + 1)); done
if [ "$PENDING" -gt 0 ]; then
  one_click_pick_gpus "$PENDING"
fi
one_click_run_groups "${GROUPS_IN_ORDER[@]}"

echo "--- reports"
"$TOKENGS_PYTHON" scripts/report_scene_geo_tuning.py > /dev/null || ONE_CLICK_FAILED=1
"$TOKENGS_PYTHON" scripts/report_scene_token_x_geo.py > /dev/null || ONE_CLICK_FAILED=1
"$TOKENGS_PYTHON" scripts/build_all_experiments_table.py || ONE_CLICK_FAILED=1
ls EXP_TokenGS/reports EXP_TokenGS/all_experiments_table.md
echo "=== run_140_scenes finished $(date '+%F %T'), failed: ${ONE_CLICK_FAILED:-0}"
exit "${ONE_CLICK_FAILED:-0}"
