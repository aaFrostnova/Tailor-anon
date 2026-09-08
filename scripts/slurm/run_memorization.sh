#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1-0
#SBATCH --job-name=fp_memorize
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# End-to-end memorization test: fine-tune SD on fingerprinted COCO, detect fingerprint.
#
# Optional env vars (set via sbatch --export):
#   MAX_TRAIN_STEPS   — training steps (default: 5000)
#   N_SAMPLES         — detection samples (default: 500)
#   FP_RATIOS         — fingerprint ratios (default: "1.0 0.0")
#   BATCH_SIZE        — training batch size (default: 4)
#   GRAD_ACCUM        — gradient accumulation (default: 4)
#   TAG               — experiment tag for output dir (default: "default")
#
# Examples:
#   # Quick test
#   sbatch --export=MAX_TRAIN_STEPS=1000,N_SAMPLES=200,TAG=quick \
#     scripts/slurm/run_memorization.sh
#
#   # Full experiment with multiple fp ratios
#   sbatch --export=MAX_TRAIN_STEPS=5000,N_SAMPLES=500,FP_RATIOS="1.0 0.5 0.1 0.0",TAG=full \
#     scripts/slurm/run_memorization.sh

set -euo pipefail

# ---- Config — defaults mirror HF text2image Naruto example verbatim ----
# https://huggingface.co/docs/diffusers/training/text2image#finetuning
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-15000}"   # HF: --max_train_steps=15000
N_SAMPLES="${N_SAMPLES:-500}"
FP_RATIOS="${FP_RATIOS:-1.0 0.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"                 # HF: --train_batch_size=1
GRAD_ACCUM="${GRAD_ACCUM:-4}"                 # HF: --gradient_accumulation_steps=4
LR="${LR:-1e-5}"                              # HF: --learning_rate=1e-05
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"         # HF: --max_grad_norm=1
LR_SCHEDULER="${LR_SCHEDULER:-constant}"      # HF: --lr_scheduler="constant"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"       # HF: --lr_warmup_steps=0
USE_EMA="${USE_EMA:-1}"                       # HF example passes --use_ema
CENTER_CROP="${CENTER_CROP:-1}"               # HF example passes --center_crop
RANDOM_FLIP="${RANDOM_FLIP:-1}"               # HF passes --random_flip; WARNING: destroys spatial fingerprint
GRAD_CKPT="${GRAD_CKPT:-1}"                   # HF example passes --gradient_checkpointing
SNR_GAMMA="${SNR_GAMMA:-}"                    # empty = off; HF docs recommend 5.0 if set
FULL_UNET="${FULL_UNET:-1}"                   # HF text2image finetune is full UNet (no LoRA)
TAG="${TAG:-default}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
LORA_RANK="${LORA_RANK:-16}"                  # only used when FULL_UNET=0

ROOT_DIR=/home/mingzhel_umass_edu/cryptographic_fingerprint
DATASET_DIR="${DATASET_DIR:-/project/pi_shiqingma_umass_edu/mingzheli/datasets/coco_fingerprinted}"
MODEL_NAME="${MODEL_NAME:-/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5}"
OUTPUT_DIR=${ROOT_DIR}/results/memorization_sd_${TAG}

# ---- Setup ----
echo "=================================================="
echo "Fingerprint Memorization Test"
echo "=================================================="
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Node:           $(hostname)"
echo "GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Dataset:        ${DATASET_DIR}"
echo "Model:          ${MODEL_NAME}"
echo "Output:         ${OUTPUT_DIR}"
echo "Train steps:    ${MAX_TRAIN_STEPS}"
echo "Detect samples: ${N_SAMPLES}"
echo "FP ratios:      ${FP_RATIOS}"
echo "Batch:          ${BATCH_SIZE} × ${GRAD_ACCUM} grad accum"
echo "=================================================="
echo ""

mkdir -p ${ROOT_DIR}/slurm_logs

cd ${ROOT_DIR}

# Use fingerprint env's python directly — avoids fragile conda activation under sbatch
PYTHON=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
echo "Python:         ${PYTHON}"
${PYTHON} -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

# ---- Run ----
EXTRA_ARGS=""
[[ "${SKIP_TRAIN}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --skip_train"
[[ "${FULL_UNET}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --full_unet"
[[ "${USE_EMA}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --use_ema"
[[ "${CENTER_CROP}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --center_crop"
[[ "${RANDOM_FLIP}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --random_flip"
[[ "${GRAD_CKPT}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --gradient_checkpointing"
[[ -n "${SNR_GAMMA}" ]] && EXTRA_ARGS="${EXTRA_ARGS} --snr_gamma ${SNR_GAMMA}"

echo "HF-aligned flags: ${EXTRA_ARGS}"

${PYTHON} scripts/run_memorization_sd.py \
    --model_name ${MODEL_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --max_train_steps ${MAX_TRAIN_STEPS} \
    --n_samples ${N_SAMPLES} \
    --fp_ratios ${FP_RATIOS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation ${GRAD_ACCUM} \
    --lora_rank ${LORA_RANK} \
    --lr ${LR} \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_warmup_steps ${LR_WARMUP_STEPS} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --mixed_precision fp16 \
    ${EXTRA_ARGS}

echo ""
echo "Done! Results in ${OUTPUT_DIR}"
