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
#SBATCH --job-name=bench_vine
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Benchmark VINE-R on same 100 test images with same attacks.
# Must run in vine conda env.

set -euo pipefail

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
COCO=/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017
OUT=${ROOT_DIR}/results/benchmark_v5/vine_baseline

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

export PYTHONUNBUFFERED=1

set +u
eval "$(conda shell.bash hook)"
conda activate vine
set -u

python scripts/benchmark_vine_comprehensive.py \
    --src_dir ${COCO} \
    --n_images 100 \
    --resolution 256 \
    --out_dir ${OUT} \
    --attack_set all

echo "Done: ${OUT}"
