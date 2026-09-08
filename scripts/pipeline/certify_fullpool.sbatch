#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 6
#SBATCH --job-name=fullpool
#SBATCH -t 0-08:00
#SBATCH -o logs/fullpool_%A_%a.out
# $1 = space-separated class indices (default "0 1 2 4"), $2 = shards per class (16), $3 = N, $4 = family
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
CLASSES=(${1:-0 1 2 4}); NSH=${2:-16}; N=${3:-10000}; FAM=${4:-inprocess}
CLS=${CLASSES[$((SLURM_ARRAY_TASK_ID / NSH))]}; SHARD=$((SLURM_ARRAY_TASK_ID % NSH))
cd $CF
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CF:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) class=$CLS shard=$SHARD/$NSH N=$N family=$FAM $(date)"
${WM_PY:-python} -u $SC/certify_fullpool.py $CLS $SHARD $NSH $N $FAM 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating"
rc=${PIPESTATUS[0]}
echo "DONE $(date)"
exit ${rc:-0}
