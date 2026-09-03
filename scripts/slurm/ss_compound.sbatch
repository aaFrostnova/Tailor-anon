#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 0-1
#SBATCH --job-name=ss_compound
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e
cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
echo "node=$(hostname)"; nvidia-smi -L 2>/dev/null | head -1
$PY scripts/defense/syncseal_vine_compound_test.py
echo "SBATCH_DONE"
