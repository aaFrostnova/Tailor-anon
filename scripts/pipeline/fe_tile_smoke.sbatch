#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH --job-name=fe_tile_sm
#SBATCH -t 0-01:00
#SBATCH -o logs/fe_tile_smoke_%j.out
set -u -o pipefail
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/scripts/defense:$REPO/external/WatermarkAttacker:$SC" OMP_NUM_THREADS=8
export HF_HOME="$SC/../hf_cache" HUGGINGFACE_HUB_CACHE="$SC/../hf_cache/hub"
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export TS=$(date -u +%FT%TZ)
echo "node=$(hostname) tile smoke: cell 27 = tile x border20, N=2"
${WM_PY:-python} $SC/make_frontend_components.py 2 27 28 2>&1 \
  | grep -vE "Warning|warn|weights_only|Loading|Migrating|Downloading|it/s"
echo "DONE $(date)"
