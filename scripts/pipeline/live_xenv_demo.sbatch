#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --job-name=live_xenv
#SBATCH -t 6:00:00
#SBATCH -o logs/live_xenv_demo_%j.out
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CF:$CF/scripts:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) $(date)"
${WM_PY:-python} -u $SC/live_xenv_demo.py "${1:-2}" "${2:-10}" "${3:-8}" 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating"
rc=${PIPESTATUS[0]}
echo "DONE $(date)"
exit ${rc:-0}
