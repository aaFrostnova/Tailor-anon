#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 0-4
#SBATCH --job-name=bench_sdt
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
COCO=/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017
CKPT=${ROOT_DIR}/results/key_encoder_v1/enc_sd_turbo.pt
OUT=${ROOT_DIR}/results/benchmark_v5/enc_sd_turbo

export PYTHONUNBUFFERED=1

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/benchmark_sd_turbo.py \
    --ckpt ${CKPT} \
    --src_dir ${COCO} \
    --n_images 100 \
    --resolution 256 \
    --out_dir ${OUT} \
    --attack_set all

echo "Done: ${OUT}"
