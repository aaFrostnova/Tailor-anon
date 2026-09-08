#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 6
#SBATCH --job-name=cert_full
#SBATCH -t 8:00:00
#SBATCH -o logs/certify_full_%A_%a.out
# $1 = class index, $2 = shards, $3 = N images; array 0..shards-1
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CF:$CF/scripts:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
echo "node=$(hostname) class=$1 shard=$SLURM_ARRAY_TASK_ID/$2 N=${3:-100} $(date)"
${WM_PY:-python} -u $SC/certify_full.py measure "$1" "$SLURM_ARRAY_TASK_ID" "$2" "${3:-100}" 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating|exists, skipping|Model loaded"
rc=${PIPESTATUS[0]}
echo "DONE $(date)"
exit ${rc:-0}
