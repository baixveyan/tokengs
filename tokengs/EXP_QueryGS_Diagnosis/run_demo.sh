#!/usr/bin/env bash
# Demo-style visual diagnosis (no training): ScanNet++ demo scene (laser mesh) + DL3DV demo scene (COLMAP points).
# Usage: GPU_A=1 GPU_B=2 bash EXP_QueryGS_Diagnosis/run_demo.sh
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=EXP_QueryGS_Diagnosis/run.log
CUDA_VISIBLE_DEVICES=${GPU_A:-1} OMP_NUM_THREADS=8 PYTHONPATH=scripts:. python -u scripts/demo_query_diagnosis.py --source scannetpp >> "$LOG" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=${GPU_B:-2} OMP_NUM_THREADS=8 PYTHONPATH=scripts:. python -u scripts/demo_query_diagnosis.py --source dl3dv >> "$LOG" 2>&1 &
p2=$!
wait $p1 && wait $p2 && echo "[$(date +%T)] done" | tee -a "$LOG"
