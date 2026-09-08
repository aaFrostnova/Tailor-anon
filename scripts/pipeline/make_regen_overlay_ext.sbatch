#!/bin/bash
# NOTE: partition is gpu-preempt ONLY (not ",gpu"). The smoke run (job 63696165, N=3, for the ORIGINAL
# non-ext overlay) landed on gypsum-gpu001 via the combined "gpu-preempt,gpu" spec and drew a Tesla M40
# (2015 Maxwell, no real fp16 throughput) -- 58min for just 27 ctrlregen images. gpu-preempt's node
# roster (checked via sinfo) is uniformly modern (a100/l40s/a16/a4000/l4/2080ti) and structurally
# EXCLUDES the ancient gypsum-gpu M40/titan_x/1080_ti nodes that only the plain "gpu" partition can
# hand out, so this avoids the slow-node tail for the expensive N=50 run. Trade-off: gpu-preempt jobs
# are preemptible (QoS=gpu-preempt) with no --requeue set here (matches make_regen_overlay.sbatch's
# documented choice), hence the skip-if-already-done guards below so a resubmit after a preemption
# doesn't redo a completed stage.
#
# EXT vs base make_regen_overlay.sbatch: 21 cells/image (3 fragments x 5 strengths + all 6 ordered
# pairs) instead of 9, PLUS a [B2] second ctrlregen pass = rinse (regen applied twice in sequence,
# this repo's existing rinse definition -- see make_regen_overlay_ext.py's header) over the
# rinse-eligible subset (17/21 cells: all 3 solo grids + the VINE<->VideoSeal pair, NOT the 4
# TrustMark cross pairs -- out of task scope for rinse). Walltime bumped 8h->16h accordingly
# (~1050 + 850 = 1900 ctrlregen forward passes at N=50, vs the original overlay's 450).
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH --job-name=regen_ovl_ext
#SBATCH -t 0-16:00
#SBATCH --exclude=gpu058
#SBATCH -o logs/regen_ovl_ext_%j.out
# --exclude=gpu058: that node is the cluster's one Blackwell-gen card (rtx_pro_6000_blackwell,
# MIG 2g.48gb, sm_120). Smoke job 63732164 landed there and the embed stage hard-crashed --
# "CUDA error: no kernel image is available for execution on the device" / "sm_120 is not
# compatible with the current PyTorch installation (supports up to sm_90)". Every other node in
# gpu-preempt's roster (2080ti/a100/a16/a40/l40s/h100/h200) is sm_75-sm_90, all torch-compatible.
set -u
REPO=${WM_REPO:-/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint}
SC=${WM_SCRATCH:-/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k}
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/scripts/defense:$SC" OMP_NUM_THREADS=8
export HF_HOME="$SC/../hf_cache" HUGGINGFACE_HUB_CACHE="$SC/../hf_cache/hub"
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN} HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
FPPY=${WM_PY:-python}
CRPY=${WM_PY_CTRLREGEN:-python}
N="${1:-50}"
WORK=$SC/regen_overlay_ext_n${N}
echo "node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) N=$N $(date)"

N_CELLS=21
N_EXP=$((N_CELLS * N))
if [ -f "$WORK/embed_stage_results.json" ]; then
  echo ">>> [A] embed: $WORK/embed_stage_results.json already exists, skipping (resume)"
else
  echo ">>> [A] embed (fingerprint env): $N_CELLS cells x N=$N -> $WORK/embed"
  $FPPY $SC/make_regen_overlay_ext.py embed $N 2>&1 | grep -vE "Warning|warn|weights_only|Loading|Migrating|Downloading|it/s"
fi
if [ ! -f "$WORK/embed_stage_results.json" ]; then
  echo "REGEN_OVL_EXT_ABORT: embed stage did not produce $WORK/embed_stage_results.json"; exit 1
fi

N_HAVE=$(ls "$WORK/attacked"/*.png 2>/dev/null | wc -l)
if [ "$N_HAVE" -ge "$N_EXP" ]; then
  echo ">>> [B] ctrlregen (regen, pass 1): $WORK/attacked already has $N_HAVE/$N_EXP images, skipping (resume)"
else
  echo ">>> [B] ctrlregen attack (ctrlregen env), step=0.5, seed=1, PASS 1 (regen=1x), single model load"
  export HF_HOME=${HF_HOME:-/work/pi_shiqingma_umass_edu/mingzheli/hf_cache} HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir $WORK/embed --out_dir $WORK/attacked --step 0.5 --seed 1 \
    2>&1 | grep -vE "warn|Warning|weights_only|Loading pipeline|^\s*$|Migrating|cache for"
fi
N_ATT=$(ls "$WORK/attacked"/*.png 2>/dev/null | wc -l)
echo ">>> ctrlregen pass-1 dir now has $N_ATT/$N_EXP images"
if [ "$N_ATT" -lt 1 ]; then
  echo "REGEN_OVL_EXT_ABORT: ctrlregen pass 1 produced 0 output images in $WORK/attacked"; exit 1
fi

echo ">>> [B1.5] filtering rinse-eligible subset of attacked/ -> attacked_for_rinse/ (symlinks; all 3"
echo "    solo grids + the VINE<->VideoSeal pair only, NOT the 4 TrustMark cross pairs -- see .py header)"
RIN_PREFIXES=(solo_V_ solo_T_ solo_S_ pair_VS_ pair_SV_)
N_RIN_CELLS=17
N_RIN_EXP=$((N_RIN_CELLS * N))
mkdir -p "$WORK/attacked_for_rinse"
N_LINKED=$(ls "$WORK/attacked_for_rinse"/*.png 2>/dev/null | wc -l)
if [ "$N_LINKED" -lt "$N_RIN_EXP" ]; then
  N_LINKED=0
  for pfx in "${RIN_PREFIXES[@]}"; do
    for f in "$WORK/attacked/${pfx}"*.png; do
      [ -e "$f" ] || continue
      ln -sf "$f" "$WORK/attacked_for_rinse/$(basename "$f")"
      N_LINKED=$((N_LINKED + 1))
    done
  done
fi
echo ">>> attacked_for_rinse/ has $N_LINKED/$N_RIN_EXP linked images"

N_HAVE2=$(ls "$WORK/rinsed"/*.png 2>/dev/null | wc -l)
if [ "$N_HAVE2" -ge "$N_RIN_EXP" ]; then
  echo ">>> [B2] ctrlregen (rinse, pass 2): $WORK/rinsed already has $N_HAVE2/$N_RIN_EXP images, skipping (resume)"
elif [ "$N_LINKED" -lt 1 ]; then
  echo ">>> [B2] SKIPPED: 0 images available in attacked_for_rinse -- rinse will be omitted from the overlay (decode stage records this explicitly, does not fabricate numbers)"
else
  echo ">>> [B2] ctrlregen attack (ctrlregen env), step=0.5, seed=2, PASS 2 on pass-1 output = rinse (2x regen)"
  export HF_HOME=${HF_HOME:-/work/pi_shiqingma_umass_edu/mingzheli/hf_cache} HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
  $CRPY scripts/attack/ctrlregen_batch.py --in_dir $WORK/attacked_for_rinse --out_dir $WORK/rinsed --step 0.5 --seed 2 \
    2>&1 | grep -vE "warn|Warning|weights_only|Loading pipeline|^\s*$|Migrating|cache for"
fi
N_RIN=$(ls "$WORK/rinsed"/*.png 2>/dev/null | wc -l)
echo ">>> ctrlregen pass-2 (rinse) dir now has $N_RIN/$N_RIN_EXP images"

echo ">>> [C] decode + assemble extended overlay (fingerprint env)"
export HF_HOME="$SC/../hf_cache" HUGGINGFACE_HUB_CACHE="$SC/../hf_cache/hub"
$FPPY $SC/make_regen_overlay_ext.py decode $N 2>&1 | grep -vE "Warning|warn|weights_only|Loading|Migrating|Downloading|it/s"
echo "REGEN_OVL_EXT_DONE $(date)"
