#!/bin/bash
#SBATCH --job-name=classes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/classes_gpu_%A_%a.log
# Margin run (2026-09-08 v4): 40 shards per class (C5 requests take 5-6 min each; 50 per shard keeps every task
# under 6 h), reverse order so that C5 starts first; a shard the local worker finished is skipped by
# class_scenarios.py's guard. Array 0-199 with %40.
set -u -o pipefail
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
export PYTHONPATH=$REPO:${PYTHONPATH:-}
cd $SC
T=$((199 - SLURM_ARRAY_TASK_ID)); CLS=$((T / 40)); SHARD=$((T % 40))
${WM_PY:-python} -u $SC/class_scenarios.py $CLS $SHARD 40 2000
rc=$?
echo "DONE $(date)"
exit ${rc:-0}
