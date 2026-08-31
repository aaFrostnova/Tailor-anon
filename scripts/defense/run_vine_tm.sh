#!/bin/bash
cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
echo "##### DETECTION (VINE+TrustMark-B, n=100) #####"
$PY scripts/defense/benchmark_composite_defense.py --fragments vine trustmark --tm_variant B --n_images 100 \
  --attacks clean jpeg blur noise bright contrast bm3d regen rinse2x rinse4x vae_b vae_c rs256 hflip crop75 crop50 rot9 crop_jpeg \
  --output results/defense/composite_vine_tm.json
echo "##### QUALITY #####"
$PY scripts/defense/composite_quality.py --fragments vine trustmark --n_images 20 \
  --output results/defense/composite_quality_vine_tm.json
echo "##### FPR #####"
$PY scripts/defense/composite_fpr.py --fragments vine trustmark --n_clean 100 --n_regen 30 --n_wrong 30 \
  --output results/defense/composite_fpr_vine_tm.json
echo "VINE_TM_ALL_DONE"
