#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 0-12
#SBATCH --job-name=fp_robust
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Robustness test: fingerprint images → apply transforms → detect.
#
# Optional env vars:
#   NUM_FRAGMENTS  — fragment count (default: 11)
#   EPSILONS       — perturbation budgets (default: "0.0157 0.0314")
#   MAX_IMAGES     — number of images (default: 100)
#   WORKERS        — parallel workers (default: 16)
#   TAG            — output dir tag (default: "k11")
#
# Examples:
#   sbatch scripts/slurm/run_robustness.sh
#   sbatch --export=NUM_FRAGMENTS=8,TAG=k8 scripts/slurm/run_robustness.sh
#   sbatch --export=NUM_FRAGMENTS="8 11",TAG=k8_vs_k11 scripts/slurm/run_robustness.sh

set -euo pipefail

NUM_FRAGMENTS="${NUM_FRAGMENTS:-11}"
EPSILONS="${EPSILONS:-0.0157 0.0314}"
MAX_IMAGES="${MAX_IMAGES:-100}"
WORKERS="${WORKERS:-8}"
TAG="${TAG:-k8}"
TRANSFORMS="${TRANSFORMS:-jpeg_q80 jpeg_q40 resize_75% center_crop_80% random_crop_60% center_crop_resize_80% center_crop_resize_60% noise_0.05 blur_1.0 blur_2.0 bright_+20% contrast_-20% jpeg_q60+center_crop_80% jpeg_q40+random_crop_resize_60%}"

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
INPUT_DIR=/home/mingzhel_umass_edu/inverse/LatentTracer/data/lexica
OUTPUT_DIR=${ROOT_DIR}/results/robustness_${TAG}

echo "=================================================="
echo "Fingerprint Robustness Test"
echo "=================================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "Input:        ${INPUT_DIR}"
echo "Output:       ${OUTPUT_DIR}"
echo "Fragments:    ${NUM_FRAGMENTS}"
echo "Epsilons:     ${EPSILONS}"
echo "Max images:   ${MAX_IMAGES}"
echo "Workers:      ${WORKERS}"
echo "=================================================="
echo ""

mkdir -p ${ROOT_DIR}/slurm_logs
cd ${ROOT_DIR}

set +u
eval "$(conda shell.bash hook)"
conda activate fingerprint
set -u

python scripts/test_robustness.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_fragments ${NUM_FRAGMENTS} \
    --epsilons ${EPSILONS} \
    --max_images ${MAX_IMAGES} \
    --workers ${WORKERS} \
    --transforms ${TRANSFORMS}

echo ""
echo "Done! Results in ${OUTPUT_DIR}"
