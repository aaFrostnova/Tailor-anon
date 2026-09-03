#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 0-3
#SBATCH --job-name=syncseal_integ
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e
cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
SS=/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_integ; COCO=/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017; FRAGS="vine trustmark videoseal"
echo "node=$(hostname)"; nvidia-smi -L 2>/dev/null | head -1
mkdir -p $SS
# --- Step A: embed composite + SyncSeal sync mark on 48 COCO images ---
echo "=== EMBED (composite + syncseal) ==="
$PY scripts/defense/composite_external_eval.py --mode embed --fragments $FRAGS --syncseal \
  --n_images 48 --start_idx 4000 --image_dir "$COCO" --image_glob "*.jpg" \
  --embed_dir "$SS/embed" --strength 0.70 2>&1 | grep -aE "EMBED_DONE|embedded \[4"
# --- Step B: geometry attacks + FPR negatives (unmarked COCO 8000+, rot30'd) ---
$PY - <<'PYA'
import os,glob,json,numpy as np
from PIL import Image
SS=os.environ.get("SS","/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_integ"); COCO="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
def rotb(im,d): return im.rotate(d,resample=Image.BICUBIC)   # black-fill rotation (realistic attack)
def cz(im,r): s=int(512*r); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
meta=json.load(open(os.path.join(SS,"embed","meta.json")))
for name,fn in [("rot30",lambda x:rotb(x,30)),("crop75",lambda x:cz(x,.75)),("crop50",lambda x:cz(x,.5))]:
    d=os.path.join(SS,name); os.makedirs(d,exist_ok=True); json.dump(meta,open(os.path.join(d,"meta.json"),"w"))
    for it in meta["items"]:
        im=Image.open(os.path.join(SS,"embed",f"img_{it['i']:05d}.png")).convert("RGB").resize((512,512))
        fn(im).save(os.path.join(d,f"img_{it['i']:05d}.png"))
# FPR negatives: UNMARKED coco (idx 8000+), rot30'd, reuse embed image_ids
d=os.path.join(SS,"fpr"); os.makedirs(d,exist_ok=True); json.dump(meta,open(os.path.join(d,"meta.json"),"w"))
files=sorted(glob.glob(os.path.join(COCO,"*.jpg")))[8000:8000+len(meta["items"])]
for it,fp in zip(meta["items"],files):
    rotb(Image.open(fp).convert("RGB").resize((512,512)),30).save(os.path.join(d,f"img_{it['i']:05d}.png"))
print("attacks_and_fpr_ready",flush=True)
PYA
# --- Step C: decode --syncseal on clean + each attack + FPR ---
for A in embed rot30 crop75 crop50 fpr; do
  echo "=== DECODE --syncseal : $A ==="
  $PY scripts/defense/composite_external_eval.py --mode decode --fragments $FRAGS --syncseal \
    --embed_dir "$SS/embed" --attacked_dir "$SS/$A" --attack_name "ss_$A" \
    --output "$SS/out_$A.json" 2>&1 | grep -aE "COMPOSITE_or|composite vs"
done
echo "SBATCH_DONE"
