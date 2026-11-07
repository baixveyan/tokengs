#!/usr/bin/env bash
# One-click: export the same-scene Demo bundle for every completed ScanNet++ experiment into
# EXP_ScanNetPP/demo/scene_<id[:12]>/<experiment>/, then generate comparison.{md,json} and
# highlighted_tokens.{md,json} (plus _only{10,20}tokens_prune{0.01,0.05}.ply).
#
#   bash EXP_ScanNetPP/run_demo.sh                 # default scene in EXP_ScanNetPP/demo/demo_scene.json (7b6477cb95)
#   bash EXP_ScanNetPP/run_demo.sh <scene_id> ...  # specific validation scene(s)
#   GPUS="1 2" bash EXP_ScanNetPP/run_demo.sh      # optional: force GPU indices
#   DRY_RUN=1 bash EXP_ScanNetPP/run_demo.sh       # only show what would run (no GPU, no log)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/one_click_launch_common.sh
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

DEMO="EXP_ScanNetPP/demo"
DEMO_JSON="$DEMO/demo_scene.json"
if [ $# -ge 1 ]; then
  SCENES=()
  declare -A SEEN_SCENE=()
  for s in "$@"; do
    if [ -z "${SEEN_SCENE[${s:0:12}]:-}" ]; then
      SEEN_SCENE[${s:0:12}]=$s
      SCENES+=("$s")
    elif [ "${SEEN_SCENE[${s:0:12}]}" != "$s" ]; then
      echo "[FATAL] two scene ids share prefix ${s:0:12}"
      exit 2
    fi
  done
elif [ -f "$DEMO_JSON" ]; then
  SCENES=("$("$TOKENGS_PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1]))['scene_id'])" "$DEMO_JSON")")
else
  SCENES=("7b6477cb95")
fi
echo "scenes: ${SCENES[*]}"

protocol() { "$TOKENGS_PYTHON" scripts/scannetpp_evaluation_protocol.py "$@"; }
mapfile -t EXPERIMENTS < <(protocol demo-names)
[ "${#EXPERIMENTS[@]}" -gt 0 ] || { echo "[FATAL] no completed 50-scene experiment in EXP_ScanNetPP"; exit 2; }
echo "experiments with a demo: ${EXPERIMENTS[*]}"

TODO=()
for s in "${SCENES[@]}"; do
  for e in "${EXPERIMENTS[@]}"; do
    out="$DEMO/scene_${s:0:12}/$e"
    if [ -f "$out/status.json" ] && grep -q '"status": "complete"' "$out/status.json" && [ -f "$out/$e.json" ]; then
      echo "skip ${e}__${s:0:12}: already complete ($out)"
    elif [ -e "$out" ]; then
      echo "[FATAL] $out already exists but is not complete; inspect or move it aside first"
      exit 4
    else
      TODO+=("${e}__${s}")
    fi
  done
done

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN=1: would run ${#TODO[@]} demo group(s)"
  for item in "${TODO[@]}"; do
    e="${item%__*}"
    s="${item##*__}"
    echo "  ${e}__${s:0:12} -> $DEMO/scene_${s:0:12}/$e/run.log"
    echo "    CUDA_VISIBLE_DEVICES=<idle GPU> $TOKENGS_PYTHON -u scripts/evaluate_scannetpp.py $(protocol demo-args "$e" "$s")"
  done
  exit 0
fi

FAILED=0
if [ "${#TODO[@]}" -gt 0 ]; then
  echo "--- live GPU status before launch"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  EXIT_CODES=$(mktemp)
  pids=()
  for item in "${TODO[@]}"; do
    (
      worker_pid=$BASHPID
      e="${item%__*}"
      s="${item##*__}"
      out="$DEMO/scene_${s:0:12}/$e"
      log="$out/run.log"
      gpu=""
      trap '[ -n "$gpu" ] && one_click_release_gpu "$gpu"' EXIT INT TERM
      gpu=$(one_click_acquire_idle_gpu "$worker_pid") || {
        echo "[FATAL] no idle GPU available for ${e}__${s:0:12}"
        echo "$item 3" >> "$EXIT_CODES"
        exit 3
      }
      mkdir -p "$(dirname "$out")"
      if ! mkdir "$out" 2>/dev/null; then
        echo "[FATAL] $out already exists (concurrent launch refused)"
        echo "$item 4" >> "$EXIT_CODES"
        exit 4
      fi
      read -r -a args <<< "$(protocol demo-args "$e" "$s")"
      cmd=("$TOKENGS_PYTHON" -u scripts/evaluate_scannetpp.py "${args[@]}")
      echo "[$(date +%T)] start ${e}__${s:0:12} on verified idle GPU $gpu -> $log"
      {
        echo "Started: $(date -Iseconds); GPU=$gpu; git=$(git rev-parse HEAD)"
        echo "Command: CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
      } > "$log"
      CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >> "$log" 2>&1
      rc=$?
      one_click_release_gpu "$gpu"
      gpu=""
      echo "Exit code: $rc; Finished: $(date -Iseconds)" >> "$log"
      echo "$item $rc" >> "$EXIT_CODES"
      echo "[$(date +%T)] end ${e}__${s:0:12} (exit $rc)"
    ) &
    pids+=($!)
  done
  trap 'echo "[$(date +%T)] interrupted: stopping workers"; for p in "${pids[@]}"; do pkill -TERM -P "$p"; kill "$p"; done 2>/dev/null' INT TERM
  wait "${pids[@]}"
  trap - INT TERM
  for item in "${TODO[@]}"; do
    rc=$(awk -v g="$item" '$1 == g {print $2}' "$EXIT_CODES")
    [ "${rc:-1}" = 0 ] || FAILED=1
  done
  rm -f "$EXIT_CODES"
fi

echo "--- comparing experiments per scene"
"$TOKENGS_PYTHON" scripts/compare_same_scene_ply.py --demo-dir "$DEMO" --reference baseline_pretrained_6view --allow-different-checkpoints || FAILED=1
echo "--- highlighting the same 10 / 20 tokens (with context, selected-only, and prune 0.01/0.05) in every experiment"
"$TOKENGS_PYTHON" scripts/highlight_selected_tokens_ply.py --demo-dir "$DEMO" --reference baseline_pretrained_6view --prune-thresholds 0.01 0.05 > /dev/null || FAILED=1
echo "=== run_demo finished $(date '+%F %T'), failed: $FAILED"
exit "$FAILED"
