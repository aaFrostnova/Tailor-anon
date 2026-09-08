#!/bin/bash
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH --job-name=clean_min
#SBATCH -t 0-4:00
#SBATCH --exclude=gpu058
#SBATCH -o logs/clean_min_%j.out
set -u -o pipefail
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
cd $REPO
export PYTHONPATH="$REPO:$REPO/scripts/defense:$SC" CUDA_VISIBLE_DEVICES=0 HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) $(date)"
${WM_PY:-python} -u $SC/measure_clean_minimum.py "${1:-100}" 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|it/s"
rc=${PIPESTATUS[0]}
echo "DONE $(date)"
exit ${rc:-0}
