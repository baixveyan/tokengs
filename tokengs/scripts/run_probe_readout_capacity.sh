#!/usr/bin/env bash
# Readout-capacity probe (scripts/probe_readout_capacity.py), queued on a fixed GPU list.
# Every job waits until ITS GPU is idle (<=100 MiB and no process, or 0 % utilisation with >=32 GB free), retrying
# every 60 s, so other users' jobs are never disturbed.  Finished steps are skipped.
#   GPU_LIST=0,1,2,3 bash scripts/run_probe_readout_capacity.sh
#   ALLOW_SHARED_GPU=1 GPU_LIST=0,1,2,3 bash ...   # user-approved: also use busy GPUs that have >= 32 GB free
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
export OMP_NUM_THREADS=8 PYTHONPATH=scripts:. PYTHONUNBUFFERED=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CACHE=/mnt/disk/xveyan/workspace_scannetpp/probe_readout_capacity
IFS=',' read -r -a GPUS <<< "${GPU_LIST:-0,1,2,3}"
N=${#GPUS[@]}
mkdir -p "$CACHE"

gpu_idle() {
  local g=$1 uuid used free util
  IFS=', ' read -r uuid used free util < <(nvidia-smi -i "$g" --query-gpu=uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
  if [ "$used" -le 100 ] && ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then return 0; fi
  # ALLOW_SHARED_GPU=1: explicit user-approved exception, share a busy GPU if >= 32 GB are free
  [ "${ALLOW_SHARED_GPU:-0}" = 1 ] && [ "$free" -ge 32768 ] && return 0
  [ "$util" -eq 0 ] && [ "$free" -ge 32768 ]
}

run_on() {  # run_on <gpu> <log> <cmd...>: wait for the GPU, run; retry if the GPU was taken at start-up
  local g=$1 log=$2; shift 2
  for attempt in $(seq 1 "${IDLE_RETRIES:-600}"); do
    if gpu_idle "$g"; then
      echo "[$(date +%T)] GPU $g: $*"
      CUDA_VISIBLE_DEVICES=$g "$@" > "$log" 2>&1 && { echo "[$(date +%T)] done: $*"; return 0; }
      tail -n 3 "$log" | grep -q "Selected GPU is no longer idle" || { echo "[$(date +%T)] FAILED: $* (see $log)"; return 1; }
    fi
    sleep 60
  done
  echo "[$(date +%T)] gave up waiting for GPU $g: $*"; return 1
}

# phase 1: feature / region extraction, one shard per GPU
pids=()
for ((k = 0; k < N; k++)); do
  [ -f "$CACHE/shard_$k.pt" ] && { echo "skip extract shard $k"; continue; }
  run_on "${GPUS[$k]}" "$CACHE/extract_$k.log" python -u scripts/probe_readout_capacity.py extract --shard $k --num-shards $N &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for ((k = 0; k < N; k++)); do [ -f "$CACHE/shard_$k.pt" ] || { echo "missing shard $k, stop"; exit 1; }; done

# phase 2: readouts (oracle / official deconv first: fast references), LR sweep for the linear readout
JOBS=("oracle|" "deconv0|" "linear|--lr-linear 1e-5 --tag _lr1e-5" "linear|--lr-linear 3e-5 --tag _lr3e-5"
      "linear|--lr-linear 1e-4 --tag _lr1e-4" "mlp_s|" "mlp_l|" "moe|" "set|")
queue() {
  local g=$1; shift
  for job in "$@"; do
    IFS='|' read -r head extra <<< "$job"
    tag=$(sed -n 's/.*--tag \([^ ]*\).*/\1/p' <<< "$extra")
    [ -f "$CACHE/fit_${head}${tag}.json" ] && { echo "skip fit ${head}${tag}"; continue; }
    # shellcheck disable=SC2086
    run_on "$g" "$CACHE/fit_${head}${tag}.log" python -u scripts/probe_readout_capacity.py fit --head "$head" $extra
  done
}
for ((i = 0; i < N; i++)); do
  q=()
  for ((j = i; j < ${#JOBS[@]}; j += N)); do q+=("${JOBS[$j]}"); done
  queue "${GPUS[$i]}" "${q[@]}" &
done
wait
python scripts/probe_readout_capacity.py report
