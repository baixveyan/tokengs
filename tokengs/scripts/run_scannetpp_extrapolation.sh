#!/usr/bin/env bash
# Extrapolation evaluation: same 6 inputs as the official protocol, 8 targets OUTSIDE the 18-frame window.
#   ext_near: 3/6/9/12 frames before the window start and after the window end
#   ext_far : 18/24/30/36 frames
# Results: EXP_ScanNetPP/eval_extrapolation/<mode>/<experiment>/{summary.json,scene_metrics.csv,frame_metrics.csv,run.log}
# Jobs are spread over the GPUs idle at launch time (at most $MAX_GPUS; one evaluation per GPU at a time).
# Existing results are skipped.
#   MAX_GPUS=3 bash scripts/run_scannetpp_extrapolation.sh [experiment ...]
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/mnt/disk/jinsoo/anaconda3/envs/tokengs/bin:/usr/local/cuda-12.8/bin:$PATH
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXPS=("$@")
[ ${#EXPS[@]} -eq 0 ] && EXPS=(baseline_pretrained_6view sft_latenttail_rgb sft_readout_rgb sft_readout_rgb_depth \
  sft_readout_rgb_depth_thick sft_readout_rgb_depth_thick_sh1 sft_readout_rgb_depth_thick_posmlp sft_readout_rgb_depth_thick_posmlp_surfel_binar \
  sft_readout_rgb_depth_surface3d sft_readout_rgb_depth_surface3d_posmlp sft_latenttail_readout_rgb_depth_surface3d_posmlp sft_readout_rgb_depth_surface3d_posmlp_ep20 sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20 sft_latenttail_readout_rgb_depth_surface3d_posmlp_attrmlp_ep20 sft_latenttail_readout_rgb_depth_surface3d_posmlp_opamlp)
MODES=(interp ext_near ext_far)

outdir() {  # interp = official protocol, results live in the experiment directory itself
  if [ "$1" = interp ]; then echo "EXP_ScanNetPP/$2"; else echo "EXP_ScanNetPP/eval_extrapolation/$1/$2"; fi
}

JOBS=()
for mode in "${MODES[@]}"; do
  for exp in "${EXPS[@]}"; do
    out=$(outdir "$mode" "$exp")
    [ -f "$out/summary.json" ] && { echo "skip $mode/$exp (done)"; continue; }
    if [ "$exp" = baseline_pretrained_6view ]; then ckpt=checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors
    else ckpt="EXP_ScanNetPP/$exp/model.safetensors"; fi
    [ -f "$ckpt" ] || { echo "missing checkpoint for $exp: $ckpt"; continue; }
    JOBS+=("$mode|$exp|$ckpt")
  done
done
[ ${#JOBS[@]} -eq 0 ] && { echo "nothing to do"; exit 0; }

if [ -n "${GPU_LIST:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"   # fixed GPUs, e.g. GPU_LIST=5,6,7 (each job still checks idleness)
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$2>=32768 && $3==0 {print $1}')
fi
[ ${#GPUS[@]} -eq 0 ] && { echo "no idle GPU"; exit 1; }
MAX_GPUS=${MAX_GPUS:-${#GPUS[@]}}
GPUS=("${GPUS[@]:0:$MAX_GPUS}")
echo "GPUs: ${GPUS[*]}  jobs: ${#JOBS[@]}"

run_queue() {
  local gpu=$1; shift
  for job in "$@"; do
    IFS='|' read -r mode exp ckpt <<< "$job"
    out=$(outdir "$mode" "$exp")
    if [ "$mode" = interp ]; then
      # keep model.safetensors and the training run.log; drop only a partial frame CSV from a failed evaluation
      rm -f "$out/frame_metrics.csv"; log="$out/run.log"; echo "=== interp evaluation $(date -Is)" >> "$log"
    else
      rm -rf "$out"; mkdir -p "$out"; log="$out/run.log"; : > "$log"
    fi
    echo "[$(date +%T)] GPU $gpu: $mode/$exp"
    for attempt in $(seq 1 "${IDLE_RETRIES:-120}"); do
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=scripts:. python -u scripts/evaluate_scannetpp.py \
        --output "$out" --checkpoint "$ckpt" --target-mode "$mode" --no-demo >> "$log" 2>&1 \
        && { echo "[$(date +%T)] done $mode/$exp"; break; }
      if tail -n 3 "$log" | grep -q "Selected GPU is no longer idle"; then
        # GPU temporarily taken by someone else: wait and retry the same job on the same GPU
        [ "$mode" != interp ] && { rm -rf "$out"; mkdir -p "$out"; : > "$log"; }
        echo "[$(date +%T)] GPU $gpu busy, retry $attempt for $mode/$exp in 60 s"; sleep 60
      else
        echo "[$(date +%T)] FAILED $mode/$exp (see $log)"; break
      fi
    done
  done
}

n=${#GPUS[@]}
for ((g = 0; g < n; g++)); do
  q=()
  for ((j = g; j < ${#JOBS[@]}; j += n)); do q+=("${JOBS[$j]}"); done
  [ ${#q[@]} -gt 0 ] && run_queue "${GPUS[$g]}" "${q[@]}" &
done
wait
PYTHONPATH=scripts:. python scripts/build_extrapolation_table.py
