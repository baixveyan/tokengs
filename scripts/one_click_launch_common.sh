#!/usr/bin/env bash
# Shared helpers for the one-click experiment launchers (sourced, not executed).
#
# Used by EXP_TokenGS/run_140_scenes.sh and EXP_TokenGS/run_demo.sh. A launcher declares its groups and calls:
#   one_click_setup <launcher_name>                    # env, master log, sanity checks
#   one_click_pick_gpus <n_wanted>                     # fills GPUS=() with idle GPUs (or uses $GPUS)
#   one_click_run_groups <group>...                    # needs group_args <group> and group_output <group>
#
# Every group writes one complete, unfiltered log <group_output>/console.log (stdout + stderr, header with
# command / GPU / git state, footer with exit code and duration). The launcher itself writes
# EXP_TokenGS/launch_logs/<launcher_name>.<timestamp>.log with everything printed to the terminal.
# DRY_RUN=1: only print which groups would run / be skipped and their commands, then stop (no GPU, no log).

TOKENGS_PYTHON=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin/python
IDLE_MEMORY_MIB=${IDLE_MEMORY_MIB:-100}
PROGRESS_SECONDS=${PROGRESS_SECONDS:-30}

one_click_setup() {
  local name=$1
  REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  cd "$REPO_ROOT"
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
  export PYTHONUNBUFFERED=1
  if [ "${DRY_RUN:-0}" = 1 ]; then
    MASTER_LOG="(none: DRY_RUN=1)"
  else
    mkdir -p EXP_TokenGS/launch_logs
    MASTER_LOG="EXP_TokenGS/launch_logs/$name.$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$MASTER_LOG") 2>&1
  fi
  echo "=== $name started $(date '+%F %T') on $(hostname)"
  echo "repo: $REPO_ROOT   git HEAD: $(git rev-parse --short HEAD)   python: $TOKENGS_PYTHON"
  echo "master log: $MASTER_LOG"
  [ -x "$TOKENGS_PYTHON" ] || { echo "[FATAL] tokengs python not found: $TOKENGS_PYTHON"; exit 2; }
  command -v ninja > /dev/null || { echo "[FATAL] ninja not on PATH (gsplat JIT needs it)"; exit 2; }
}

# Prints idle GPU indices: memory.used <= IDLE_MEMORY_MIB and no compute process.
one_click_idle_gpus() {
  "$TOKENGS_PYTHON" - "$IDLE_MEMORY_MIB" <<'EOF'
import subprocess, sys
limit = float(sys.argv[1])
q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()
busy = {line.split(",")[0].strip() for line in q("--query-compute-apps=gpu_uuid") if line.strip()}
for line in q("--query-gpu=index,uuid,memory.used"):
    idx, uuid, used = [x.strip() for x in line.split(",")]
    if uuid not in busy and float(used) <= limit:
        print(idx)
EOF
}

# Clean up any stale GPU reservation locks whose owner PID is no longer running.
_one_click_clean_stale_gpu_locks() {
  local lock pid
  for lock in /tmp/tokengs_gpu_active_*.lock; do
    [ -d "$lock" ] || continue
    if [ -f "$lock/pid" ]; then
      pid=$(cat "$lock/pid" 2>/dev/null || true)
      if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        rm -rf "$lock" 2>/dev/null || true
      fi
    else
      rm -rf "$lock" 2>/dev/null || true
    fi
  done
}

# Dynamically checks nvidia-smi right before a run and atomically reserves one idle GPU.
# Respects optional allowed list in $ALLOWED_GPUS (space- or comma-separated).
# If WAIT_FOR_IDLE_GPU=1 (default), waits until an idle GPU is available; if 0, exits 3 immediately.
one_click_acquire_idle_gpu() {
  local owner_pid="${1:-$$}"
  local wait_mode="${WAIT_FOR_IDLE_GPU:-1}"
  local allowed_raw="${ALLOWED_GPUS:-${GPUS:-${GPU:-}}}"
  allowed_raw="${allowed_raw//,/ }"
  while true; do
    _one_click_clean_stale_gpu_locks
    local -a live_idle=()
    mapfile -t live_idle < <(one_click_idle_gpus)
    local cand
    for cand in "${live_idle[@]}"; do
      [ -n "$cand" ] || continue
      if [ -n "$allowed_raw" ] && [[ " $allowed_raw " != *" $cand "* ]]; then
        continue
      fi
      local lock="/tmp/tokengs_gpu_active_${cand}.lock"
      if mkdir "$lock" 2>/dev/null; then
        echo "$owner_pid" > "$lock/pid"
        # Re-verify via nvidia-smi after acquiring lock to close any race window
        if one_click_idle_gpus | grep -qx -- "$cand"; then
          echo "$cand"
          return 0
        fi
        rm -rf "$lock" 2>/dev/null || true
      fi
    done
    if [ "$wait_mode" != "1" ]; then
      return 3
    fi
    sleep "${GPU_POLL_SECONDS:-10}"
  done
}

one_click_release_gpu() {
  local gpu
  for gpu in ${1//,/ }; do
    [ -n "$gpu" ] && rm -rf "/tmp/tokengs_gpu_active_${gpu}.lock" 2>/dev/null || true
  done
}

# Dynamically checks nvidia-smi right before a run and atomically reserves ALL currently idle GPUs.
# Prints comma-separated GPU indices (e.g. "2,3,4,5,6,7").
one_click_acquire_all_idle_gpus() {
  local owner_pid="${1:-$$}"
  local wait_mode="${WAIT_FOR_IDLE_GPU:-1}"
  local allowed_raw="${ALLOWED_GPUS:-${GPUS:-${GPU:-}}}"
  allowed_raw="${allowed_raw//,/ }"
  while true; do
    _one_click_clean_stale_gpu_locks
    local -a live_idle=()
    mapfile -t live_idle < <(one_click_idle_gpus)
    local -a acquired=()
    local cand
    for cand in "${live_idle[@]}"; do
      [ -n "$cand" ] || continue
      if [ -n "$allowed_raw" ] && [[ " $allowed_raw " != *" $cand "* ]]; then
        continue
      fi
      local lock="/tmp/tokengs_gpu_active_${cand}.lock"
      if mkdir "$lock" 2>/dev/null; then
        echo "$owner_pid" > "$lock/pid"
        if one_click_idle_gpus | grep -qx -- "$cand"; then
          acquired+=("$cand")
        else
          rm -rf "$lock" 2>/dev/null || true
        fi
      fi
    done
    if [ "${#acquired[@]}" -gt 0 ]; then
      local IFS=','
      echo "${acquired[*]}"
      return 0
    fi
    if [ "$wait_mode" != "1" ]; then
      return 3
    fi
    sleep "${GPU_POLL_SECONDS:-10}"
  done
}

one_click_pick_gpus() {
  local wanted=$1
  if [ "${DRY_RUN:-0}" = 1 ]; then
    GPUS=("<idle GPU>")
    return 0
  fi
  echo "--- GPU status"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  if [ -n "${GPUS:-}" ]; then
    read -r -a GPUS <<< "$GPUS"
    echo "using GPUs from \$GPUS: ${GPUS[*]}"
  else
    mapfile -t GPUS < <(one_click_idle_gpus)
    GPUS=("${GPUS[@]:0:$wanted}")
    echo "idle GPUs picked: ${GPUS[*]:-none}"
  fi
  [ "${#GPUS[@]}" -gt 0 ] || { echo "[FATAL] no idle GPU (memory <= ${IDLE_MEMORY_MIB} MiB, no process); retry later"; exit 3; }
  [ "${#GPUS[@]}" -ge "$wanted" ] || echo "only ${#GPUS[@]} idle GPU(s) for $wanted groups: groups sharing a GPU run one after another"
}

# Runs one group in the foreground of its worker; complete log at <output>/console.log
# (evaluate_with_scene_latent_tuning.py accepts an output folder that only holds console.log).
_one_click_run_one() {
  local gpu=$1 group=$2 out log t0 rc
  out=$(group_output "$group")
  log="$out/console.log"
  mkdir -p "$out"
  local -a args
  read -r -a args <<< "$(group_args "$group")"
  local -a cmd=("$TOKENGS_PYTHON" -u scripts/evaluate_with_scene_latent_tuning.py "${args[@]}" --output "$out")
  t0=$(date +%s)
  {
    echo "=== group: $group"
    echo "started: $(date '+%F %T')   host: $(hostname)   GPU: $gpu"
    echo "git HEAD: $(git rev-parse HEAD)"
    echo "CUDA_HOME=$CUDA_HOME"
    echo "command: CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    echo "=== output (stdout + stderr, unfiltered)"
  } > "$log"
  CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" >> "$log" 2>&1
  rc=$?
  {
    echo "=== exit code: $rc"
    echo "finished: $(date '+%F %T')   duration: $(( $(date +%s) - t0 )) s"
  } >> "$log"
  echo "$group $rc" >> "$EXIT_CODES"
  return $rc
}

_one_click_progress() {
  local g st
  for g in "$@"; do
    st=$(group_output "$g")/status.json
    if [ -f "$st" ]; then
      "$TOKENGS_PYTHON" -c "import json,sys;s=json.load(open(sys.argv[1]));print(f\"  {sys.argv[2]}: {s.get('status')} {s.get('completed_scenes', s.get('scene_count', ''))}/{s.get('expected_scene_count', '')}\")" "$st" "$g" 2>/dev/null \
        || echo "  $g: (status unreadable)"
    else
      echo "  $g: waiting"
    fi
  done
}

# Skips finished groups (complete, or failed = e.g. diverged, kept as evidence), refuses partial outputs,
# distributes the rest over GPUS round-robin.
one_click_run_groups() {
  local -a todo=()
  local g out
  for g in "$@"; do
    out=$(group_output "$g")
    if [ -f "$out/status.json" ] && grep -q '"status": "complete"' "$out/status.json"; then
      echo "skip $g: already complete ($out)"
    elif [ -f "$out/status.json" ] && grep -q '"status": "failed"' "$out/status.json"; then
      echo "skip $g: already ran and failed, kept as evidence ($out/status.json); move it aside to run it again"
    elif [ -e "$out" ]; then
      echo "[FATAL] $out exists but is not complete; move it aside first (e.g. into a cancelled_* folder)"
      exit 4
    else
      todo+=("$g")
    fi
  done
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "DRY_RUN=1: would run ${#todo[@]} group(s)"
    for g in "${todo[@]}"; do
      echo "  $g -> $(group_output "$g")/console.log"
      echo "    $TOKENGS_PYTHON -u scripts/evaluate_with_scene_latent_tuning.py $(group_args "$g") --output $(group_output "$g")"
    done
    exit 0
  fi
  [ "${#todo[@]}" -gt 0 ] || { echo "nothing to run"; return 0; }
  EXIT_CODES=$(mktemp)
  local i n=${#GPUS[@]} p alive
  local -a pids=()
  for ((i = 0; i < n && i < ${#todo[@]}; i++)); do
    (
      local j
      for ((j = i; j < ${#todo[@]}; j += n)); do
        echo "[$(date +%T)] start ${todo[j]} on GPU ${GPUS[i]} -> $(group_output "${todo[j]}")/console.log"
        _one_click_run_one "${GPUS[i]}" "${todo[j]}"
        echo "[$(date +%T)] end ${todo[j]} (exit $?)"
      done
    ) &
    pids+=($!)
  done
  # Ctrl-C reaches the python processes directly (same process group); TERM is forwarded here.
  trap 'echo "[$(date +%T)] interrupted: stopping groups"; for p in "${pids[@]}"; do pkill -TERM -P "$p"; kill "$p"; done 2>/dev/null' INT TERM
  while true; do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" = 1 ] || break
    sleep "$PROGRESS_SECONDS"
    echo "[$(date +%T)] progress"
    _one_click_progress "${todo[@]}"
  done
  # Only the workers: a bare `wait` also waits for the `tee` behind the master log (exec > >(tee ...)),
  # which never exits while this shell is alive, so the launcher would hang after the last group.
  wait "${pids[@]}"
  trap - INT TERM
  echo "--- exit codes"
  ONE_CLICK_FAILED=0
  for g in "${todo[@]}"; do
    local rc
    rc=$(awk -v g="$g" '$1 == g {print $2}' "$EXIT_CODES")
    echo "  $g: ${rc:-missing}   log: $(group_output "$g")/console.log"
    [ "${rc:-1}" = 0 ] || ONE_CLICK_FAILED=1
  done
  rm -f "$EXIT_CODES"
  echo "--- last lines of each group log"
  for g in "${todo[@]}"; do
    echo "## $g"
    tail -n 12 "$(group_output "$g")/console.log"
  done
}
