#!/usr/bin/env bash
# Deep LocusGS diagnosis (no training): 50 ScanNet++ validation scenes (laser G1-G4) + demo scene 7b6477cb95.
# Usage: GPU_LIST=1,2,3,4,5,6,7 bash EXP_LocusGS_Diagnosis/run_diag.sh
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IFS=, read -r -a GPUS <<< "${GPU_LIST:-1,2,3,4,5,6,7}"
N=${#GPUS[@]}
LOG=EXP_LocusGS_Diagnosis/run.log
echo "[$(date +%T)] stats: ${N} shards" | tee -a "$LOG"
pids=()
for k in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$k]} OMP_NUM_THREADS=8 PYTHONPATH=scripts:. \
    python -u scripts/diag_locusgs_deep.py stats --shard "$k" --num-shards "$N" >> "$LOG" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "[$(date +%T)] stats failed" | tee -a "$LOG"; exit 1; }
PYTHONPATH=scripts:. python -u scripts/diag_locusgs_deep.py summarize >> "$LOG" 2>&1 || exit 1
CUDA_VISIBLE_DEVICES=${GPUS[0]} OMP_NUM_THREADS=8 PYTHONPATH=scripts:. python -u scripts/diag_locusgs_deep.py demo >> "$LOG" 2>&1 || exit 1
echo "[$(date +%T)] done" | tee -a "$LOG"
