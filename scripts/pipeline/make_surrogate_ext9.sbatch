#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH --job-name=mk_surr9
#SBATCH -t 0-12:00
#SBATCH -o logs/make_surrogate_ext9_%j.out
set -u -o pipefail
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/scripts/defense:$REPO/external/WatermarkAttacker:$SC" OMP_NUM_THREADS=8
export HF_HOME="$SC/../hf_cache" HUGGINGFACE_HUB_CACHE="$SC/../hf_cache/hub"
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export TS=$(date -u +%FT%TZ)
N="${1:-20}"
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) TS=$TS N=$N $(date)"
${WM_PY:-python} -u $SC/make_surrogate_ext9.py "$N" 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating|Downloading|it/s"
rc=${PIPESTATUS[0]}   # the pipeline's python status, not grep's and not the trailing echo's
echo "DONE $(date)"
exit ${rc:-0}   # without this a crashed python still leaves the job COMPLETED
