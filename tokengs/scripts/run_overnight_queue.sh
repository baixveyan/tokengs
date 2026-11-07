#!/usr/bin/env bash
# Overnight queue (2026-11-03): wait for the running sft_readout_rgb_depth_surface3d job, then train + evaluate the
# remaining ladder steps one after another, then extrapolation evaluation and thickness split for the new models.
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while kill -0 3843349 2>/dev/null; do sleep 60; done
echo "[$(date +%T)] previous job finished"
for exp in sft_readout_rgb_depth_surface3d_posmlp sft_latenttail_readout_rgb_depth_surface3d_posmlp sft_readout_rgb_depth_surface3d_posmlp_ep20; do
  for try in $(seq 1 30); do   # retry while no GPU is idle (another user), up to 30 x 10 min
    bash EXP_ScanNetPP/run_50_scenes.sh "$exp" && break
    grep -q "no idle GPU" "/tmp/tokengs_overnight_$exp.last" 2>/dev/null || true
    [ -f "EXP_ScanNetPP/$exp/summary.json" ] && break
    echo "[$(date +%T)] $exp not complete, retry $try in 10 min"; sleep 600
  done
done
NEW="sft_readout_rgb_depth_surface3d sft_readout_rgb_depth_surface3d_posmlp sft_latenttail_readout_rgb_depth_surface3d_posmlp sft_readout_rgb_depth_surface3d_posmlp_ep20"
GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_scannetpp_extrapolation.sh $NEW
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTHONPATH=scripts:. python -u scripts/diag_thickness_split.py $NEW
echo "[$(date +%T)] overnight queue done"
