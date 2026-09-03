#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=short
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=6 --mem=48G -t 0-2
#SBATCH --job-name=geo_cascade_bench2
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python; SS=/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_integ; FR="vine trustmark videoseal"
echo "node=$(hostname)"
for A in rot30 crop75 crop50; do
  echo "########## geo_cascade+gate  attack=$A ##########"
  $PY scripts/defense/composite_external_eval.py --mode decode --fragments $FR --geo_cascade \
    --embed_dir "$SS/embed" --attacked_dir "$SS/bench_$A" --attack_name "${A}_casc_gate" \
    --output "$SS/bench_out_${A}_casc_gate.json" 2>&1 | grep -aE "COMPOSITE_or|SPEED"
done
echo "SBATCH_DONE"
