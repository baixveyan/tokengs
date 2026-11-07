#!/usr/bin/env bash
# 2026-11-04: thickness split for the new models, then the 20-epoch latent-tail model (train + eval + extrapolation).
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEW="sft_readout_rgb_depth_surface3d sft_readout_rgb_depth_surface3d_posmlp sft_latenttail_readout_rgb_depth_surface3d_posmlp sft_readout_rgb_depth_surface3d_posmlp_ep20"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTHONPATH=scripts:. python -u scripts/diag_thickness_split.py $NEW
E=sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20
bash EXP_ScanNetPP/run_50_scenes.sh $E
GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_scannetpp_extrapolation.sh $E
echo "[$(date +%T)] next queue done"
