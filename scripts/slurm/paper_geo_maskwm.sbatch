#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=6 --mem=48G -t 0-3
#SBATCH --job-name=paper_geo_mwm
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python; WB=/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image; OUT=/scratch/workspace/mingzhel_umass_edu-ablator/paper_geo; N=500
CE="$PY scripts/defense/composite_external_eval.py"
[ -f "$OUT/embed_maskwm/meta.json" ] || $CE --mode embed --fragments maskwm --n_images $N --start_idx 0 \
  --image_dir "$WB" --image_glob "*.png" --embed_dir "$OUT/embed_maskwm" --strength 0.70 2>&1 | grep -aE "EMBED_DONE"
$PY - <<'PYA'
import os,json
from PIL import Image
OUT="/scratch/workspace/mingzhel_umass_edu-ablator/paper_geo"; ed=os.path.join(OUT,"embed_maskwm"); meta=json.load(open(os.path.join(ed,"meta.json")))
def rotb(im,d): return im.rotate(d,resample=Image.BICUBIC)
def cz(im,r): s=int(512*r); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
for name,fn in {"rot9":lambda x:rotb(x,9),"rot30":lambda x:rotb(x,30),"crop75":lambda x:cz(x,.75),"crop50":lambda x:cz(x,.5)}.items():
    d=os.path.join(OUT,f"atk_maskwm_{name}"); os.makedirs(d,exist_ok=True); json.dump(meta,open(os.path.join(d,"meta.json"),"w"))
    for it in meta["items"]:
        im=Image.open(os.path.join(ed,f"img_{it['i']:05d}.png")).convert("RGB").resize((512,512)); fn(im).save(os.path.join(d,f"img_{it['i']:05d}.png"))
print("attacks_built")
PYA
for A in clean rot9 rot30 crop75 crop50; do
  AD=$([ $A = clean ] && echo "$OUT/embed_maskwm" || echo "$OUT/atk_maskwm_$A")
  echo "########## method=maskwm attack=$A ##########"
  $CE --mode decode --fragments maskwm --embed_dir "$OUT/embed_maskwm" --attacked_dir "$AD" \
    --attack_name "maskwm_$A" --output "$OUT/out_maskwm_$A.json" 2>&1 | grep -aE "COMPOSITE_or"
done
echo "SBATCH_DONE"
