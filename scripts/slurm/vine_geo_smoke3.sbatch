#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=short
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=80G -t 0-1
#SBATCH --job-name=vine_geo_smoke3
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python scripts/finetune_vine_geo.py --smoke --train_batch_size 8 --mixed_precision bf16
echo SBATCH_DONE
