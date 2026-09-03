#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G -t 0-8
#SBATCH --job-name=advatk
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/adv_%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/adv_%x_%j.err
set -e; REPO=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint; cd $REPO
export HF_HOME=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache HF_HUB_OFFLINE=0 HUGGINGFACE_HUB_CACHE=/scratch/workspace/mingzhel_umass_edu-ablator/hf_cache/hub TOKENIZERS_PARALLELISM=false
FPPY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
CRPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/ctrlregen/bin/python
UMPY=/project/pi_shiqingma_umass_edu/mingzheli/.conda/envs/unmarker/bin/python
SC=/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks; M="$1"; case "$M" in ours) FR="vine trustmark videoseal"; EX="--geo_cascade";; *) FR="$M"; EX="";; esac
mkdir -p $SC; echo "node=$(hostname) method=$M frags=$FR"
[ -f "$SC/embed_$M/meta.json" ] || $FPPY scripts/defense/composite_external_eval.py --mode embed --fragments $FR $EX \
  --n_images 50 --start_idx 6000 --image_dir "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017" --image_glob "*.jpg" --embed_dir "$SC/embed_$M" --strength 0.70 2>&1 | grep -aE "EMBED_DONE"
echo "=== CtrlRegen+ $M ==="
(cd external/CtrlRegen && $CRPY $REPO/scripts/attack/ctrlregen_batch.py --in_dir "$SC/embed_$M" --out_dir "$SC/cr_$M" --step 0.5 --steps 50 2>&1 | tail -3)
$FPPY scripts/defense/composite_external_eval.py --mode decode --fragments $FR $EX --embed_dir "$SC/embed_$M" --attacked_dir "$SC/cr_$M" --attack_name "cr_$M" --output "$SC/out_cr_$M.json" 2>&1 | grep -aE "COMPOSITE_or"
echo "=== UnMarker $M ==="
(cd external/ai-watermark && $UMPY $REPO/scripts/attack/unmarker_batch.py --in_dir "$SC/embed_$M" --out_dir "$SC/um_$M" --n 50 2>&1 | tail -3)
$FPPY scripts/defense/composite_external_eval.py --mode decode --fragments $FR $EX --embed_dir "$SC/embed_$M" --attacked_dir "$SC/um_$M" --attack_name "um_$M" --output "$SC/out_um_$M.json" 2>&1 | grep -aE "COMPOSITE_or"
echo "ADV_DONE $M"
