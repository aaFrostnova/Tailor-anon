#!/bin/bash
#SBATCH --job-name=orch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/orchestrate_%j.log
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
exec bash $SC/chain_resume.sh >> $SC/logs/chain_after_classes.out 2>&1
