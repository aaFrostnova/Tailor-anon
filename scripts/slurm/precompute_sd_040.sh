#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-4
#SBATCH --job-name=sd_resid_040
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

export PYTHONUNBUFFERED=1

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/precompute_sd_residuals.py \
    --image_dir /project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017 \
    --output_dir /project/pi_shiqingma_umass_edu/mingzheli/cache/sd_residuals \
    --n_images 500 \
    --strengths 0.40 \
    --resolution 256

echo "Done: strength_0.40 residuals"
