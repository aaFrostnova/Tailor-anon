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
#SBATCH --job-name=bench_v5
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Benchmark v5 encoder (matched-filter) on 100 test images.
# Tests classical + regen attacks. Uses held-out COCO images.

set -euo pipefail

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
# Use COCO val images (offset from training set start)
COCO=/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017
CKPT=${1:-${ROOT_DIR}/results/key_encoder_v1/enc_mf_regen60.pt}
OUT=${ROOT_DIR}/results/benchmark_v5/$(basename ${CKPT%.pt})

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

export PYTHONUNBUFFERED=1

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/benchmark_v5.py \
    --ckpt ${CKPT} \
    --src_dir ${COCO} \
    --n_images 100 \
    --resolution 256 \
    --out_dir ${OUT} \
    --attack_set all

echo "Done: ${OUT}"
