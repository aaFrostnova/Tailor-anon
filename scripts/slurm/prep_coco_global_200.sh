#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0-1
#SBATCH --job-name=fp_prep_global
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
OUT=/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fp_global_200

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/prepare_coco_fingerprinted.py \
    --output_dir ${OUT} \
    --max_images 200 \
    --epsilon 0.0627 \
    --num_fragments 8 \
    --resolution 512 \
    --global_fingerprint \
    --skip_download

echo "Done: ${OUT}"
