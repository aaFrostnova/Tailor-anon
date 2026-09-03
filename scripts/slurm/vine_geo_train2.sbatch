#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH -t 0-6
#SBATCH --job-name=vine_geo_train2
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e
cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin
echo "node=$(hostname)"; nvidia-smi -L 2>/dev/null
$PY/accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  scripts/finetune_vine_geo.py \
    --dataset_folder /project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017 \
    --output_dir /scratch/workspace/mingzhel_umass_edu-ablator/vine_geo_ft2 \
    --train_batch_size 8 --learning_rate 2e-5 --mixed_precision bf16 \
    --max_train_steps 20000 --checkpointing_steps 2000 --no_im_loss_steps 200 \
    --max_rot_deg 30 --geometric_step 0 --geometric_ramp 3000 --geometric_prob 0.5 \
    --min_keep 0.75 --regen_prob 0.3 --dataloader_num_workers 6
echo "SBATCH_DONE"
