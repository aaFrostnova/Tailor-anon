#!/bin/bash
# UnMarker parameter sweep on the sweep/comp images (after ctrlregen, single GPU0).
# PHASED (stage1-all -> free -> stage2-all) + lite_filter (drop 3 largest stage2 kernels)
# so it fits the 14.6GB GPU. Spectral arm: stage1 iters (quality high->low). Crop arm:
# center-crop ratio (geometric). UnMarker is iterative & slow -> N small, iters modest.
set -e
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO
UMPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/unmarker/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EMB=results/defense/sweep/comp
N=${1:-8}

# --- spectral arm (pure FFT/perceptual, no preprocess crop): low & high strength ---
for SPEC in "300 75" "1200 150"; do
  set -- $SPEC; I=$1; S2=$2; ODIR=results/defense/sweep/um_spec_i${I}
  echo "=== UnMarker spectral stage1=$I stage2=$S2 -> $ODIR ==="
  CUDA_VISIBLE_DEVICES=0 $UMPY scripts/attack/unmarker_batch.py --in_dir $EMB --out_dir $ODIR \
    --n $N --no_preprocess --lite_filter --max_iter_stage1 $I --max_iter_stage2 $S2 2>&1 \
    | grep -aE "\[unmarker\]|stage2 \[$N|UNMARKER_BATCH_DONE|OutOfMemory|Traceback \(most" | tail -3
done

# --- crop arm (geometric; minimal spectral iters since crop alone kills VINE) ---
for C in 0.95 0.90 0.85; do
  TAG=$(echo $C | tr -d '.'); ODIR=results/defense/sweep/um_crop_c${TAG}
  echo "=== UnMarker crop=$C stage1=200 stage2=50 -> $ODIR ==="
  CUDA_VISIBLE_DEVICES=0 $UMPY scripts/attack/unmarker_batch.py --in_dir $EMB --out_dir $ODIR \
    --n $N --crop_ratio $C --lite_filter --max_iter_stage1 200 --max_iter_stage2 50 2>&1 \
    | grep -aE "\[unmarker\]|stage2 \[$N|UNMARKER_BATCH_DONE|OutOfMemory|Traceback \(most" | tail -3
done
echo "UNMARKER_SWEEP_ALL_DONE"
