#!/bin/bash
#SBATCH --partition=gpu-preempt,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH --job-name=variance
#SBATCH -t 0-3:00
#SBATCH -o logs/variance_%j.out
set -u -o pipefail
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/scripts/defense:$SC" OMP_NUM_THREADS=8
export HF_HOME="$SC/../hf_cache" HUGGINGFACE_HUB_CACHE="$SC/../hf_cache/hub"
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
echo "node=$(hostname) $(date)"
${WM_PY:-python} -u $SC/variance_profile.py 50 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating|Downloading"
rc=${PIPESTATUS[0]}   # the pipeline's python status, not grep's and not the trailing echo's
echo "DONE $(date)"
exit ${rc:-0}   # without this a crashed python still leaves the job COMPLETED
