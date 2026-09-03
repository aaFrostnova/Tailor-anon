#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G -t 0-6
#SBATCH --job-name=crsweep
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
CRPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/ctrlregen/bin/python
SC=/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks
SL="0.3,0.5,0.7,0.9"
for M in ours vine; do
  case "$M" in ours) FR="vine trustmark videoseal"; EX="--geo_cascade";; *) FR="$M"; EX="";; esac
  echo "=== CtrlRegen+ sweep $M steps=$SL ==="
  (cd external/CtrlRegen && $CRPY $REPO/scripts/attack/ctrlregen_batch.py --in_dir "$SC/embed_$M" --out_dir "$SC/crsw_$M" --step_list "$SL" --steps 50 2>&1 | grep -aE "ctrlregen|Error|Traceback")
  for TAG in 03 05 07 09; do
    $FPPY scripts/defense/composite_external_eval.py --mode decode --fragments $FR $EX \
      --embed_dir "$SC/embed_$M" --attacked_dir "$SC/crsw_${M}_s${TAG}" --attack_name "crsw_${M}_s${TAG}" \
      --output "$SC/out_crsw_${M}_s${TAG}.json" 2>&1 | grep -aE "COMPOSITE_or"
  done
done
echo "CR_SWEEP_DONE"
