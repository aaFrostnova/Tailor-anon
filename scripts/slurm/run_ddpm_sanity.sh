#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-8
#SBATCH --job-name=fp_ddpm_sanity
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

EPOCHS="${EPOCHS:-50}"
N_SAMPLES="${N_SAMPLES:-500}"
TAG="${TAG:-ddpm_cifar10}"

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/test_memorization.py \
    --output_dir ${ROOT_DIR}/results/${TAG} \
    --strategy pixel \
    --epochs ${EPOCHS} \
    --fp_ratios 1.0 0.0 \
    --n_samples ${N_SAMPLES} \
    --epsilon 0.0627 \
    --batch_size 128 \
    --lr 2e-4

echo "Done. Results in ${ROOT_DIR}/results/${TAG}"
