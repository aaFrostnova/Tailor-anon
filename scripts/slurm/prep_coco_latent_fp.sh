#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-2
#SBATCH --job-name=fp_prep_latent
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
OUT=/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_latent_fp_2k

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/prepare_coco_latent_fp.py \
    --output_dir ${OUT} \
    --max_images 2000 \
    --epsilon_latent 0.3 \
    --num_fragments 8 \
    --resolution 512 \
    --skip_download

echo "Done: ${OUT}"
