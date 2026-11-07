#!/usr/bin/env bash
# 2026-11-04: A1 = opacity-only zero-init residual MLP on top of the 10-epoch latent-tail model (compare with sft_latenttail_readout_rgb_depth_surface3d_posmlp).
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash EXP_ScanNetPP/run_50_scenes.sh sft_latenttail_readout_rgb_depth_surface3d_posmlp_opamlp
GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_scannetpp_extrapolation.sh sft_latenttail_readout_rgb_depth_surface3d_posmlp_opamlp
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTHONPATH=scripts:. python -u scripts/diag_junk_slots.py sft_latenttail_readout_rgb_depth_surface3d_posmlp sft_latenttail_readout_rgb_depth_surface3d_posmlp_opamlp
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTHONPATH=scripts:. python -u scripts/diag_thickness_split.py official sft_latenttail_readout_rgb_depth_surface3d_posmlp sft_latenttail_readout_rgb_depth_surface3d_posmlp_opamlp
echo "[$(date +%T)] opamlp queue done"
