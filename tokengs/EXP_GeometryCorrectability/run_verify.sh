#!/usr/bin/env bash
# Pre-registered verification (EXP_GeometryCorrectability/README.md). No model training; only small probe scorers.
# Every job reserves one GPU that is idle right before it starts (scripts/one_click_launch_common.sh rules + lock) and
# waits otherwise, so other users' jobs are never shared.   Usage: bash EXP_GeometryCorrectability/run_verify.sh
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/one_click_launch_common.sh
LOG=EXP_GeometryCorrectability/run.log
PY="python -u scripts/verify_geometry_correctability.py"

job() {
  local g rc me=$BASHPID   # PID of this background job (not of the $(...) subshell), keeps the GPU lock alive
  g=$(one_click_acquire_idle_gpu "$me") || return 1
  echo "[$(date +%T)] start GPU $g: $*" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 PYTHONPATH=scripts:. $PY "$@" >> "$LOG" 2>&1
  rc=$?
  one_click_release_gpu "$g"
  echo "[$(date +%T)] exit $rc GPU $g: $*" | tee -a "$LOG"
  return $rc
}
waitall() { local fail=0; for p in "$@"; do wait "$p" || fail=1; done; return $fail; }

echo "[$(date +%T)] phase A: extract / conflict / recover" | tee -a "$LOG"
ext=()
for fam in tokengs locus; do
  for k in 0 1 2 3; do job extract --family $fam --split train --shard $k --num-shards 4 & ext+=($!); sleep 2; done
  for k in 0 1; do job extract --family $fam --split val --shard $k --num-shards 2 & ext+=($!); sleep 2; done
done
other=()
for k in 0 1 2 3; do job conflict --shard $k --num-shards 4 & other+=($!); sleep 2; done
for k in 0 1; do job recover --shard $k --num-shards 2 & other+=($!); sleep 2; done
waitall "${ext[@]}" || { echo "[$(date +%T)] extract failed" | tee -a "$LOG"; exit 1; }

echo "[$(date +%T)] phase B: probes" | tee -a "$LOG"
job probe --family tokengs --source release & b1=$!; sleep 2
job probe --family tokengs --source sft12 & b2=$!; sleep 2
job probe --family locus --source locus & b3=$!
waitall "${other[@]}" $b1 $b2 $b3 || { echo "[$(date +%T)] phase A/B failed" | tee -a "$LOG"; exit 1; }

echo "[$(date +%T)] phase C: summarize / demo" | tee -a "$LOG"
PYTHONPATH=scripts:. $PY summarize >> "$LOG" 2>&1 || exit 1
job demo || exit 1
echo "[$(date +%T)] done" | tee -a "$LOG"
