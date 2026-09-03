#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-3
#SBATCH --job-name=fp_redetect
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Re-run detection on an already-trained checkpoint using the updated
# verify_multi_domain protocol (same as test_robustness.py).
#
# Usage:
#   sbatch --export=ALL,OUTPUT_DIR=results/memorization_hf_aligned_overfit50,\
#          DATASET_DIR=/project/.../coco_latent_fp_50,\
#          N_SAMPLES=50,FULL_UNET=1 scripts/slurm/rerun_detect.sh

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
DATASET_DIR="${DATASET_DIR:?DATASET_DIR required}"
N_SAMPLES="${N_SAMPLES:-100}"
FULL_UNET="${FULL_UNET:-1}"
MODEL_NAME="${MODEL_NAME:-/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5}"

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

PYTHON=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
echo "Python: ${PYTHON}"
echo "Output: ${OUTPUT_DIR}"
echo "Dataset: ${DATASET_DIR}"
echo "N_SAMPLES: ${N_SAMPLES}"

EXTRA_ARGS=""
[[ "${FULL_UNET}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --full_unet"

${PYTHON} scripts/run_memorization_sd.py \
    --model_name ${MODEL_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --max_train_steps 15000 \
    --n_samples ${N_SAMPLES} \
    --fp_ratios 1.0 \
    --skip_train \
    --mixed_precision fp16 \
    ${EXTRA_ARGS}

echo "Done: ${OUTPUT_DIR}"
