#!/usr/bin/env bash
# Diagnoses of frozen query-based models on DL3DV-10K-Benchmark (140 scenes; no training except light probes on frozen
# features).  Usage: GPU_LIST=1,2,3,4,5,6,7 MODELS="locusgs_finetune_4v tokengs_paper_4v" bash EXP_DL3DV/run_diag_dl3dv.sh
# Models: tokengs_release_latent_6v, tokengs_paper_4v, locusgs_finetune_4v (see scripts/diag_dl3dv.py).
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IFS=, read -r -a GPUS <<< "${GPU_LIST:-1,2,3,4,5,6,7}"
N=${#GPUS[@]}
LOG=EXP_DL3DV/run.log
for MODEL in ${MODELS:-tokengs_release_latent_6v tokengs_paper_4v locusgs_finetune_4v}; do
  echo "[$(date +%T)] start $MODEL: ${N} shards on GPUs ${GPU_LIST:-1,2,3,4,5,6,7}" | tee -a "$LOG"
  pids=()
  for k in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$k]} OMP_NUM_THREADS=8 PYTHONPATH=scripts:. \
      python -u scripts/diag_dl3dv.py extract --model "$MODEL" --shard "$k" --num-shards "$N" >> "$LOG" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" = 0 ] || { echo "[$(date +%T)] extract failed: $MODEL" | tee -a "$LOG"; exit 1; }
  PYTHONPATH=scripts:. python -u scripts/diag_dl3dv.py summarize --model "$MODEL" >> "$LOG" 2>&1 || exit 1
  CUDA_VISIBLE_DEVICES=${GPUS[0]} PYTHONPATH=scripts:. python -u scripts/diag_dl3dv.py probe --model "$MODEL" >> "$LOG" 2>&1 || exit 1
  echo "[$(date +%T)] done $MODEL" | tee -a "$LOG"
done
