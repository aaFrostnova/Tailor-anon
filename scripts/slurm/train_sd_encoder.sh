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
#SBATCH --job-name=sd_enc
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# SD-pipeline-based watermark encoder (VINE-inspired conservative route).
# Uses SD-v1.5 UNet (with LoRA) + ConditionAdaptor + matched filter loss.
# Needs more memory than pixel encoder (UNet + VAE in VRAM).

set -euo pipefail

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
COCO=/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017
CACHE=/project/pi_shiqingma_umass_edu/mingzheli/cache/sd_residuals
OUT=${ROOT_DIR}/results/key_encoder_v1/enc_sd_pipeline.pt

export PYTHONUNBUFFERED=1

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/train_sd_encoder.py \
    --image_dir ${COCO} \
    --output_ckpt ${OUT} \
    --sd_residual_cache ${CACHE} \
    --attack_strengths 0.05 0.10 0.15 0.20 0.30 \
    --steps 5000 \
    --batch_size 2 \
    --max_images 2000 \
    --lr 1e-4 \
    --lambda_dist 1.0 \
    --lambda_lpips 0.5 \
    --p_regen 0.30 \
    --att_weight 0.8 \
    --warmup_frac 0.15 \
    --save_every 500 \
    --log_every 10

echo "Done: ${OUT}"
