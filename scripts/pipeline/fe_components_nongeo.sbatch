#!/bin/bash
# Front-end replacement curves on the NON-geometric columns for the three embed-changing stages
# (resync / scale / tile), one (stage, attack) cell per array task. Same protocol and images as the
# 28 geometric cells (make_frontend_components.py, cross-source N=100, offset 0).
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH --job-name=fe_nongeo
#SBATCH -t 0-8:00
#SBATCH --exclude=gpu058
#SBATCH --array=0-29
#SBATCH -o logs/fe_nongeo_%A_%a.out
set -u -o pipefail
CF=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}; SCR=$(dirname "$SC")
cd $CF
export PYTHONPATH="$CF:$CF/scripts/defense:$CF/external/WatermarkAttacker:$SC" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
export HF_HOME=$SCR/hf_cache HUGGINGFACE_HUB_CACHE=$SCR/hf_cache/hub
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export TS=$(date -u +%FT%TZ)
export FE_COMPONENTS=resync,scale,tile
export FE_ATTACKS=jpeg25,blur,noise,bright,contrast,vaeB,vaeC,regen,rinse2x,rinse4x
export FE_OUT_TAG=${FE_OUT_TAG:-nongeo} FE_PENALTY=0
N="${1:-100}"
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) N=$N cell=$SLURM_ARRAY_TASK_ID/30 $(date)"
${WM_PY:-python} -u $SC/make_frontend_components.py "$N" "$SLURM_ARRAY_TASK_ID" 30 2>&1 \
  | grep --line-buffered -vE "Warning|warn|weights_only|Loading|Migrating|Downloading|it/s"
rc=${PIPESTATUS[0]}
[ -f "$SC/surrogate_fe_components_${FE_OUT_TAG}_${SLURM_ARRAY_TASK_ID}of30.json" ] || { echo "FE_NONGEO_ABORT: no output"; rc=1; }
echo "DONE $(date)"
exit ${rc:-0}
