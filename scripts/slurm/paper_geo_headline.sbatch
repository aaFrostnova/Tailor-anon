#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G -t 0-8
#SBATCH --job-name=paper_geo
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
WB=/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image; OUT=/scratch/workspace/mingzhel_umass_edu-ablator/paper_geo; N=500
echo "node=$(hostname)"; nvidia-smi -L 2>/dev/null | head -1; mkdir -p $OUT
CE="$PY scripts/defense/composite_external_eval.py"
# ---- embed each method on N W-Bench images (ours adds sync mark via --geo_cascade) ----
for M in ours vine trustmark videoseal; do
  case $M in
    ours) FR="vine trustmark videoseal"; EX="--geo_cascade";;
    *)    FR="$M"; EX="";;
  esac
  [ -f "$OUT/embed_$M/meta.json" ] && { echo "embed_$M exists, skip"; continue; }
  echo "=== EMBED $M ==="
  $CE --mode embed --fragments $FR $EX --n_images $N --start_idx 0 \
    --image_dir "$WB" --image_glob "*.png" --embed_dir "$OUT/embed_$M" --strength 0.70 2>&1 | grep -aE "EMBED_DONE"
done
# ---- build geo-attacked dirs from each embed ----
$PY - <<'PYA'
import os,json,numpy as np
from PIL import Image
OUT="/scratch/workspace/mingzhel_umass_edu-ablator/paper_geo"
def rotb(im,d): return im.rotate(d,resample=Image.BICUBIC)
def cz(im,r): s=int(512*r); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
ATT={"rot9":lambda x:rotb(x,9),"rot30":lambda x:rotb(x,30),"crop75":lambda x:cz(x,.75),"crop50":lambda x:cz(x,.5)}
for M in ("ours","vine","trustmark","videoseal"):
    ed=os.path.join(OUT,"embed_"+M); meta=json.load(open(os.path.join(ed,"meta.json")))
    for name,fn in ATT.items():
        d=os.path.join(OUT,f"atk_{M}_{name}"); os.makedirs(d,exist_ok=True); json.dump(meta,open(os.path.join(d,"meta.json"),"w"))
        for it in meta["items"]:
            im=Image.open(os.path.join(ed,f"img_{it['i']:05d}.png")).convert("RGB").resize((512,512))
            fn(im).save(os.path.join(d,f"img_{it['i']:05d}.png"))
print("attacks_built",flush=True)
PYA
# ---- decode each method on each attack + clean ----
for M in ours vine trustmark videoseal; do
  case $M in ours) FR="vine trustmark videoseal"; EX="--geo_cascade";; *) FR="$M"; EX="";; esac
  for A in clean rot9 rot30 crop75 crop50; do
    AD=$([ $A = clean ] && echo "$OUT/embed_$M" || echo "$OUT/atk_${M}_$A")
    echo "########## method=$M attack=$A ##########"
    $CE --mode decode --fragments $FR $EX --embed_dir "$OUT/embed_$M" --attacked_dir "$AD" \
      --attack_name "${M}_$A" --output "$OUT/out_${M}_$A.json" 2>&1 | grep -aE "COMPOSITE_or|SPEED"
  done
done
echo "SBATCH_DONE"
