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
#SBATCH --job-name=v5_regen_v2
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# v2 matched-filter training with curriculum regen augmentation.
# Warm-starts from enc_mf_regen60, adds strength_0.30, higher p_regen.

set -euo pipefail

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
COCO=/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017
CACHE=/project/pi_shiqingma_umass_edu/mingzheli/cache/sd_residuals
RESUME=${ROOT_DIR}/results/key_encoder_v1/enc_mf_regen60.pt
OUT=${ROOT_DIR}/results/key_encoder_v1/enc_regen_v2.pt

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

export PYTHONUNBUFFERED=1

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/train_key_encoder_v2.py \
    --image_dir ${COCO} \
    --output_ckpt ${OUT} \
    --resume_ckpt ${RESUME} \
    --sd_residual_cache ${CACHE} \
    --attack_strengths 0.05 0.10 0.15 0.20 0.30 \
    --steps 15000 \
    --batch_size 4 \
    --max_images 2000 \
    --lr 3e-4 \
    --lambda_dist 0.5 \
    --p_regen 0.65 \
    --att_weight 0.7 \
    --att_weight_max 0.85 \
    --warmup_frac 0.08 \
    --save_every 2500 \
    --log_every 50

echo "Done: ${OUT}"
