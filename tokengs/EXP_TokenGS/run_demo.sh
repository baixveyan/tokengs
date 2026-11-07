#!/usr/bin/env bash
# One-click: the same scene(s) through every experiment whose 140-scene run finished, one folder per
# experiment, then the comparison and the highlighted-token PLY files. Diverged experiments get no demo.
#
#   bash EXP_TokenGS/run_demo.sh                 # scene in EXP_TokenGS/demo/demo_scene.json
#   bash EXP_TokenGS/run_demo.sh <scene_id> ...  # other scenes
#   GPUS="5 6 7" bash ...                        # optional: force GPU indices
#   DRY_RUN=1 bash ...                           # only show what would run (no GPU, no log)
#
# Settings = the experiment's 140-scene run (scripts/experiment_registry.py demo-args), always with the
# reimplemented loop (tuning curve + tuned tensors; bit-identical to the official loop on CPU).
# Everything of one experiment on one scene goes into one folder:
#   EXP_TokenGS/demo/scene_<id[:12]>/<experiment>/
#     <experiment>.ply, <experiment>_slotcolor.ply, cameras.ply, <experiment>.json (sidecar)
#     <experiment>_highlight{10,20}tokens.ply     same tokens in every experiment, others grey, opacity x0.1
#     <experiment>_only{10,20}tokens.ply          same colours, with every unselected Gaussian removed
#     <experiment>_minus_no_tuning.ply            how far each Gaussian moved relative to no_tuning
#     gaussians.pt, decoder_inputs.pt, tuning_curve.{jsonl,png}, render_strip.{png,json}, needle_stats.json
#     console.log (complete, unfiltered), run.log, status.json, summary.json, scene_metrics.*, ...
#   EXP_TokenGS/demo/scene_<id[:12]>/comparison.{md,json}            metrics + Gaussian-level differences
#   EXP_TokenGS/demo/scene_<id[:12]>/highlighted_tokens.{md,json}    legend: token id -> colour
# A finished experiment x scene is skipped, so adding an experiment only runs the new one.
# Master log: EXP_TokenGS/launch_logs/run_demo.<timestamp>.log.
# Default scene: pre-registered rule (scene latent tuning gain closest to the 140-scene median), written by
# scripts/report_scene_geo_tuning.py.
set -uo pipefail
PROGRESS_SECONDS=${PROGRESS_SECONDS:-10}
source "$(dirname "$0")/../scripts/one_click_launch_common.sh"
one_click_setup run_demo
DEMO=EXP_TokenGS/demo

if [ $# -ge 1 ]; then
  SCENES=("$@")
else
  DEMO_JSON=$DEMO/demo_scene.json
  [ -f "$DEMO_JSON" ] || { echo "[FATAL] $DEMO_JSON missing; pass scene ids explicitly"; exit 2; }
  SCENES=("$("$TOKENGS_PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1]))['scene_id'])" "$DEMO_JSON")")
fi
declare -A SCENE_OF
for s in "${SCENES[@]}"; do
  [ -z "${SCENE_OF[${s:0:12}]:-}" ] || { echo "[FATAL] two scene ids share the prefix ${s:0:12}"; exit 2; }
  SCENE_OF[${s:0:12}]=$s
done
echo "scenes: ${SCENES[*]}"

registry() { "$TOKENGS_PYTHON" scripts/experiment_registry.py "$@"; }
mapfile -t EXPERIMENTS < <(registry demo-names)
[ "${#EXPERIMENTS[@]}" -gt 0 ] || { echo "[FATAL] no finished 140-scene experiment"; exit 2; }
echo "experiments with a demo: ${EXPERIMENTS[*]}"

# group = <experiment>__<first 12 chars of scene id>
GROUPS_IN_ORDER=()
for s in "${SCENES[@]}"; do for e in "${EXPERIMENTS[@]}"; do GROUPS_IN_ORDER+=("${e}__${s:0:12}"); done; done
group_output() { echo "$DEMO/scene_${1##*__}/${1%__*}"; }
group_args() { registry demo-args "${1%__*}" "${SCENE_OF[${1##*__}]}"; }

PENDING=0
for g in "${GROUPS_IN_ORDER[@]}"; do [ -f "$(group_output "$g")/status.json" ] || PENDING=$((PENDING + 1)); done
if [ "$PENDING" -gt 0 ]; then
  one_click_pick_gpus "$PENDING"
fi
one_click_run_groups "${GROUPS_IN_ORDER[@]}"

echo "--- comparing experiments per scene"
"$TOKENGS_PYTHON" scripts/compare_same_scene_ply.py --demo-dir "$DEMO" || ONE_CLICK_FAILED=1
echo "--- highlighting the same 10 / 20 tokens (with context and selected-only) in every experiment"
"$TOKENGS_PYTHON" scripts/highlight_selected_tokens_ply.py --demo-dir "$DEMO" > /dev/null || ONE_CLICK_FAILED=1
echo "--- files per scene (open the .ply files in the PLY viewer)"
for s in "${SCENES[@]}"; do
  find "$DEMO/scene_${s:0:12}" \( -name "*.ply" -o -name "comparison.md" -o -name "highlighted_tokens.md" \
    -o -name "tuning_curve.png" -o -name "render_strip.png" \) 2>/dev/null | sort
done
echo "=== run_demo finished $(date '+%F %T'), failed: ${ONE_CLICK_FAILED:-0}"
exit "${ONE_CLICK_FAILED:-0}"
