#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 6
#SBATCH --job-name=pi_stage
#SBATCH -t 4:00:00
#SBATCH -o logs/perimage_stage_%A_%a.out
# per-image values for the front-end cells: (component, fragment, knot) per task, original 9 knots plus
# the range-extension knots; scale is VINE-only and tile TrustMark-only
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CF:$CF/scripts:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC"
cd $CF          # src.dft_kred_modules does `from scripts.prototype_dft_carrier import ...`, a
                # namespace-package import that only resolves with the repo root as the working directory
V="0.14 0.17 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"; T="0.4 0.55 0.7 0.85 1.0 1.15 1.3 1.45 1.6 1.8 2.0"; S="0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.375 1.5 1.75 2.0 2.25 2.5 2.8 3.2"
CELLS=()
for c in resync angle; do for s in $V; do CELLS+=("$c VINE $s"); done; for s in $T; do CELLS+=("$c TrustMark $s"); done; for s in $S; do CELLS+=("$c VideoSeal $s"); done; done
for s in $V; do CELLS+=("scale VINE $s"); done
for s in $T; do CELLS+=("tile TrustMark $s"); done
ARGS=${CELLS[$SLURM_ARRAY_TASK_ID]}
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID/${#CELLS[@]} cell=$ARGS $(date)"
${WM_PY:-python} -u $SC/perimage_campaign.py stage $ARGS 2>&1 | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating|exists, skipping|Model loaded"
rc=${PIPESTATUS[0]}
echo "DONE $(date)"
exit ${rc:-0}
