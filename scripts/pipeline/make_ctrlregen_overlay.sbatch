#!/bin/bash
# CtrlRegen+ strength-family overlay: same 21-cell embed set as the regen overlay, attacked with
# ctrlregen_batch.py at a DIFFERENT diffusion step. The solver's DIFFUSION family needs a continuous
# strength curve per ctrlregen step (s03/s05/s07); s05 == the canonical "regen" run already on disk,
# so this script covers the missing s03 and s07. Rinse is off here (rinse is defined on s05 only).
# Usage: sbatch make_ctrlregen_overlay.sbatch <N> <STEP> <LABEL>
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH --job-name=cr_ovl
#SBATCH -t 0-16:00
#SBATCH --exclude=gpu058
#SBATCH -o logs/cr_ovl_%j.out
# --exclude=gpu058: sole Blackwell (sm_120) node; the installed torch supports up to sm_90.
set -u
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/scripts/defense:$SC" OMP_NUM_THREADS=8
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
FPPY=${WM_PY:-python}
CRPY=${WM_PY_CTRLREGEN:-python}
N="${1:-50}"; STEP="${2:-0.7}"; LABEL="${3:-ctrlregen_s07}"
export REGEN_STEP="$STEP" ATT_LABEL="$LABEL" DO_RINSE=0
export WORK_SUFFIX="_${LABEL}" OUT_NAME="surrogate_${LABEL}_overlay.json"
WORK=$SC/regen_overlay_ext_n${N}${WORK_SUFFIX}
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) N=$N STEP=$STEP LABEL=$LABEL $(date)"

N_CELLS=21; N_EXP=$((N_CELLS * N))
export HF_HOME="$SC/../hf_cache" HUGGINGFACE_HUB_CACHE="$SC/../hf_cache/hub"
if [ -f "$WORK/embed_stage_results.json" ]; then
  echo ">>> [A] embed already done, skipping (resume)"
else
  echo ">>> [A] embed (fingerprint env): $N_CELLS cells x N=$N -> $WORK/embed"
  $FPPY $SC/make_regen_overlay_ext.py embed $N 2>&1 | grep -vE "Warning|warn|weights_only|Loading|Migrating|Downloading|it/s"
fi
[ -f "$WORK/embed_stage_results.json" ] || { echo "CR_OVL_ABORT: embed stage produced nothing"; exit 1; }

N_HAVE=$(ls "$WORK/attacked"/*.png 2>/dev/null | wc -l)
if [ "$N_HAVE" -ge "$N_EXP" ]; then
  echo ">>> [B] ctrlregen step=$STEP already has $N_HAVE/$N_EXP, skipping (resume)"
else
  echo ">>> [B] ctrlregen attack (ctrlregen env), step=$STEP, seed=1, single model load"
  export HF_HOME=${HF_HOME:-/work/pi_shiqingma_umass_edu/mingzheli/hf_cache} HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir $WORK/embed --out_dir $WORK/attacked --step $STEP --seed 1 \
    2>&1 | grep -vE "warn|Warning|weights_only|Loading pipeline|^\s*$|Migrating|cache for"
fi
N_ATT=$(ls "$WORK/attacked"/*.png 2>/dev/null | wc -l)
echo ">>> attacked dir has $N_ATT/$N_EXP images"
[ "$N_ATT" -ge 1 ] || { echo "CR_OVL_ABORT: ctrlregen produced 0 images"; exit 1; }

echo ">>> [C] decode + assemble (fingerprint env), attack column=$LABEL"
export HF_HOME="$SC/../hf_cache" HUGGINGFACE_HUB_CACHE="$SC/../hf_cache/hub"
$FPPY $SC/make_regen_overlay_ext.py decode $N 2>&1 | grep -vE "Warning|warn|weights_only|Loading|Migrating|Downloading|it/s"
echo "CR_OVL_DONE $(date)"
