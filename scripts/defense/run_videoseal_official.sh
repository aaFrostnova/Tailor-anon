#!/bin/bash
# Official RAVEN-suite comparison: VINE+TrustMark (baseline) vs VINE+VideoSeal (TM replacement).
# Full 18-attack in-env suite at n=64. GPU index passed as $1 (probe first; mapping is unstable).
set -e
REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
cd $REPO
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export PYTHONPATH=$REPO
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
GPU=${1:-0}
ATTACKS="clean jpeg blur noise bright contrast bm3d regen rinse2x rinse4x vae_b vae_c rs256 hflip crop75 crop50 rot9 crop_jpeg"
GREP='^attack|^clean|^jpeg|^blur|^noise|^bright|^contrast|^bm3d|^regen|^rinse|^vae|^rs256|^hflip|^crop|^rot|coexist|\[done\]|Error|Traceback'

run_cfg () {  # $1=fragments  $2=tag
  echo "########## $2  (fragments: $1) ##########"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/defense/benchmark_composite_defense.py \
    --fragments $1 --tm_variant B --n_images 64 --attacks $ATTACKS \
    --output results/defense/official_$2.json 2>&1 | grep -aE "$GREP"
}

run_cfg "vine trustmark"  "vine_tm"        # baseline (current pipeline)
run_cfg "vine videoseal"  "vine_videoseal" # TM replaced by VideoSeal
echo "OFFICIAL_DONE"
