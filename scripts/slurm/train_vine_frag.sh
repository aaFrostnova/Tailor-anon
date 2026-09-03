#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH -t 0-4
#SBATCH --job-name=vine_frag
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Multi-fragment VINE-style: SD-Turbo + ConvNeXt decoder + 4-fragment crypto payload.

set -euo pipefail

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
COCO=/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017
CACHE=/project/pi_shiqingma_umass_edu/mingzheli/cache/sd_residuals
OUT=${ROOT_DIR}/results/key_encoder_v1/enc_vine_frag.pt

export PYTHONUNBUFFERED=1

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/train_vine_style_frag.py \
    --image_dir ${COCO} \
    --output_ckpt ${OUT} \
    --sd_residual_cache ${CACHE} \
    --attack_strengths 0.05 0.10 0.15 0.20 0.30 \
    --steps 5000 \
    --batch_size 2 \
    --max_images 2000 \
    --lr 1e-4 \
    --secret_loss_scale 1.5 \
    --l2_loss_scale 2.0 \
    --l2_loss_ramp 10000 \
    --lpips_loss_scale 1.5 \
    --lpips_loss_ramp 10000 \
    --no_im_loss_steps 1000 \
    --p_regen 0.30 \
    --num_fragments 4 \
    --frag_t 3 \
    --save_every 1000 \
    --log_every 10

echo "Done: ${OUT}"
