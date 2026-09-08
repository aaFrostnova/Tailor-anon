#!/bin/bash
# CtrlRegen+ column at nine strength knots, cross-source N images, three stages across two envs:
#   [A] fingerprint env  make_regen_overlay_ext.py embed  -> $WORK/embed  (33 cells/image)
#   [B] ctrlregen env    ctrlregen_batch.py --step $STEP     -> $WORK/attacked
#   [B2] (rinse only)    ctrlregen_batch.py on the rinse-eligible 29 cells of [B]'s output -> $WORK/rinsed
#   [C] fingerprint env  make_regen_overlay_ext.py decode -> $SC/$OUT_NAME
# Usage: sbatch make_ctrlregen9.sbatch <N> <STEP> <LABEL> [RINSE_LABEL|none] [OUT_NAME]
#   e.g. sbatch make_ctrlregen9.sbatch 100 0.5 ctrlregen_s05 ctrlregen_s05_x2 surrogate_regen_overlay_ext.json
#        sbatch make_ctrlregen9.sbatch 100 0.3 ctrlregen_s03 none surrogate_ctrlregen_s03_overlay.json
# Every stage is skipped when its output is already complete, so a resubmit resumes.
#SBATCH --job-name=cr9
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 0-8:00
#SBATCH -o logs/cr9_%j.out
set -u -o pipefail
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/scripts/defense:$SC" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
FPPY=${WM_PY:-python}
CRPY=${WM_PY_CTRLREGEN:-python}
FILT="Warning|warn|weights_only|Loading|Migrating|Downloading|it/s|cache for|^\s*$"
N="${1:-100}"; STEP="${2:-0.5}"; LABEL="${3:-ctrlregen_s05}"; RINSE="${4:-none}"
export REGEN_STEP="$STEP" ATT_LABEL="$LABEL" WORK_SUFFIX="_${LABEL}"
export OUT_NAME="${5:-surrogate_${LABEL}_overlay.json}"
if [ "$RINSE" != "none" ]; then export DO_RINSE=1 RINSE_LABEL="$RINSE"; else export DO_RINSE=0; fi
WORK=$SC/regen_overlay_ext_n${N}${WORK_SUFFIX}
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) N=$N STEP=$STEP LABEL=$LABEL RINSE=$RINSE OUT=$OUT_NAME $(date)"

N_CELLS=33; N_EXP=$((N_CELLS * N))          # 3 x 9 solo + 6 ordered pairs
export HF_HOME="$SC/../hf_cache" HUGGINGFACE_HUB_CACHE="$SC/../hf_cache/hub"
if [ -f "$WORK/embed_stage_results.json" ] && [ "$(ls "$WORK/embed"/*.png 2>/dev/null | wc -l)" -ge "$N_EXP" ]; then
  echo ">>> [A] embed already complete, skipping (resume)"
else
  echo ">>> [A] embed (fingerprint env): $N_CELLS cells x N=$N -> $WORK/embed"
  $FPPY -u $SC/make_regen_overlay_ext.py embed "$N" 2>&1 | grep --line-buffered -vE "$FILT"
  rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || { echo "CR9_ABORT: embed stage rc=$rc"; exit $rc; }
fi
N_EMB=$(ls "$WORK/embed"/*.png 2>/dev/null | wc -l)
[ "$N_EMB" -ge "$N_EXP" ] || { echo "CR9_ABORT: embed dir has $N_EMB/$N_EXP images"; exit 1; }

export HF_HOME=${HF_HOME:-/work/pi_shiqingma_umass_edu/mingzheli/hf_cache} HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
N_HAVE=$(ls "$WORK/attacked"/*.png 2>/dev/null | wc -l)
if [ "$N_HAVE" -ge "$N_EXP" ]; then
  echo ">>> [B] ctrlregen step=$STEP already has $N_HAVE/$N_EXP, skipping (resume)"
else
  echo ">>> [B] ctrlregen attack (ctrlregen env), step=$STEP, seed=1, pass 1 ($N_HAVE/$N_EXP present)"
  $CRPY -u scripts/attack/ctrlregen_batch.py --in_dir $WORK/embed --out_dir $WORK/attacked --step "$STEP" --seed 1 \
    2>&1 | grep --line-buffered -vE "$FILT"
  rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || { echo "CR9_ABORT: ctrlregen pass 1 rc=$rc"; exit $rc; }
fi
N_ATT=$(ls "$WORK/attacked"/*.png 2>/dev/null | wc -l)
[ "$N_ATT" -ge "$N_EXP" ] || { echo "CR9_ABORT: attacked dir has $N_ATT/$N_EXP images"; exit 1; }

if [ "$DO_RINSE" = "1" ]; then
  # rinse = the same operator applied a second time to pass 1's own output, on the rinse-eligible
  # cells: all three solo grids (27) plus the VINE<->VideoSeal pair in both orders (2)
  N_RIN_CELLS=29; N_RIN_EXP=$((N_RIN_CELLS * N))
  mkdir -p "$WORK/attacked_for_rinse"
  for pfx in solo_V_ solo_T_ solo_S_ pair_VS_ pair_SV_; do
    for f in "$WORK/attacked/${pfx}"*.png; do [ -e "$f" ] && ln -sf "$f" "$WORK/attacked_for_rinse/$(basename "$f")"; done
  done
  N_LINKED=$(ls "$WORK/attacked_for_rinse"/*.png 2>/dev/null | wc -l)
  [ "$N_LINKED" -ge "$N_RIN_EXP" ] || { echo "CR9_ABORT: only $N_LINKED/$N_RIN_EXP rinse-eligible images"; exit 1; }
  N_HAVE2=$(ls "$WORK/rinsed"/*.png 2>/dev/null | wc -l)
  if [ "$N_HAVE2" -ge "$N_RIN_EXP" ]; then
    echo ">>> [B2] rinse already has $N_HAVE2/$N_RIN_EXP, skipping (resume)"
  else
    echo ">>> [B2] ctrlregen pass 2 on pass-1 output (rinse), step=$STEP, seed=2 ($N_HAVE2/$N_RIN_EXP present)"
    $CRPY -u scripts/attack/ctrlregen_batch.py --in_dir $WORK/attacked_for_rinse --out_dir $WORK/rinsed --step "$STEP" --seed 2 \
      2>&1 | grep --line-buffered -vE "$FILT"
    rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || { echo "CR9_ABORT: ctrlregen pass 2 rc=$rc"; exit $rc; }
  fi
  N_RIN=$(ls "$WORK/rinsed"/*.png 2>/dev/null | wc -l)
  [ "$N_RIN" -ge "$N_RIN_EXP" ] || { echo "CR9_ABORT: rinsed dir has $N_RIN/$N_RIN_EXP images"; exit 1; }
fi

echo ">>> [C] decode + assemble (fingerprint env) -> $SC/$OUT_NAME"
export HF_HOME="$SC/../hf_cache" HUGGINGFACE_HUB_CACHE="$SC/../hf_cache/hub"
$FPPY -u $SC/make_regen_overlay_ext.py decode "$N" 2>&1 | grep --line-buffered -vE "$FILT"
rc=${PIPESTATUS[0]}
[ -f "$SC/$OUT_NAME" ] || { echo "CR9_ABORT: decode wrote no $OUT_NAME"; rc=1; }
echo "CR9_DONE rc=$rc $(date)"
exit ${rc:-0}
