#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0-2
#SBATCH --job-name=fp_prep_global
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Parameterized global-fingerprint dataset prep.
#   sbatch --export=ALL,N_IMAGES=500 scripts/slurm/prep_coco_global.sh
#   sbatch --export=ALL,N_IMAGES=2000 scripts/slurm/prep_coco_global.sh

set -euo pipefail

N_IMAGES="${N_IMAGES:-500}"
EPS="${EPS:-0.0627}"  # 16/255
ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
OUT=/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fp_global_${N_IMAGES}

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

PYTHON=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
echo "Python: ${PYTHON}"
echo "Output: ${OUT}"
echo "N_IMAGES=${N_IMAGES}, epsilon=${EPS}"

${PYTHON} scripts/prepare_coco_fingerprinted.py \
    --output_dir ${OUT} \
    --max_images ${N_IMAGES} \
    --epsilon ${EPS} \
    --num_fragments 8 \
    --resolution 512 \
    --global_fingerprint \
    --skip_download

echo "Done: ${OUT}"
