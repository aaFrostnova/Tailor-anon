#!/bin/bash
#SBATCH --job-name=orch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/orchestrate_%j.log
# The post-class chain (v3, 200 shards) as a SLURM job of its own, so it outlives the interactive node.
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
exec bash $SC/chain_after_classes_v3.sh >> $SC/logs/chain_after_classes.out 2>&1
