#!/bin/bash
# N=200 UnMarker (pure-spectral) for the 3-frag composite, slightly HIGHER strength
# (max_iter 300/150 -> 400/200). embed reused from ext_vtv200 (run ctrlregen driver first).
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# do NOT export PYTHONPATH=$REPO (shadows ai-watermark's own src package)
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
UMPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/unmarker/bin/python
GPU=${1:-0}; N=${2:-200}; S1=${3:-400}; S2=${4:-200}
EMB=results/defense/ext_vtv200
ADIR=${EMB}_unmarker_s${S1}_${S2}
echo "## UnMarker pure-spectral N=$N strength stage1=$S1 stage2=$S2  in=$EMB out=$ADIR"
CUDA_VISIBLE_DEVICES=$GPU $UMPY scripts/attack/unmarker_batch.py --in_dir $EMB --out_dir $ADIR \
  --n $N --no_preprocess --max_iter_stage1 $S1 --max_iter_stage2 $S2 2>&1 \
  | grep -aE "\[unmarker\]|stage2|UNMARKER_BATCH_DONE|OutOfMemory|Traceback" | tail -6
nout=$(ls $ADIR/*.png 2>/dev/null | wc -l); echo "   attacked images: $nout"
CUDA_VISIBLE_DEVICES=$GPU $FPPY scripts/defense/decode_breakdown.py \
  --embed_dir $EMB --attacked_dir $ADIR --attack_name unmarker200_s${S1}_${S2} \
  --output results/defense/unmarker200_s${S1}_${S2}.json 2>&1 | grep -aE "\]|BREAKDOWN_DONE|Error|Traceback" | tail -2
echo "UM200_DONE"
