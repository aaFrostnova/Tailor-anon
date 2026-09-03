#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_shiqingma_umass_edu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH -t 0-2
#SBATCH --job-name=resync_fpr
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e
cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
SCRATCH=/scratch/workspace/mingzhel_umass_edu-ablator/resync_fpr
echo "node=$(hostname)"; nvidia-smi -L 2>/dev/null | head -1
# Build FPR negatives: 100 CLEAN (unmarked) COCO imgs (idx offset 8000), rot30'd, ext_vtv100 image_ids
$PY - <<'PYA'
import os,glob,json,numpy as np
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
COCO="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
DST="/scratch/workspace/mingzhel_umass_edu-ablator/resync_fpr"; os.makedirs(DST,exist_ok=True)
def rot(img,deg):
    a=np.asarray(img); p=256
    big=Image.fromarray(np.pad(a,((p,p),(p,p),(0,0)),"reflect")).rotate(deg,resample=Image.BICUBIC)
    bw,bh=big.size; l=(bw-512)//2; return big.crop((l,l,l+512,l+512))
src=json.load(open(os.path.join(REPO,"results/defense/ext_vtv100/meta.json")))
meta={k:src[k] for k in src if k!="items"}; meta["items"]=src["items"][:100]
json.dump(meta,open(os.path.join(DST,"meta.json"),"w"))
files=sorted(glob.glob(os.path.join(COCO,"*.jpg")))[8000:8000+100]  # unmarked negatives
for it,fp in zip(meta["items"],files):
    im=Image.open(fp).convert("RGB").resize((512,512))
    rot(im,30.0).save(os.path.join(DST,f"img_{it['i']:05d}.png"))
print("fpr_negatives_ready",len(meta["items"]),flush=True)
PYA
echo "=== FPR check: --resync on CLEAN unmarked images (expect composite_or=0, resync=0) ==="
$PY scripts/defense/composite_external_eval.py --mode decode --fragments vine trustmark videoseal \
  --embed_dir "$SCRATCH" --attacked_dir "$SCRATCH" --attack_name resync_fpr --resync \
  --output "$SCRATCH/out_fpr.json" 2>&1 | grep -aE "COMPOSITE_or|composite vs"
echo "SBATCH_DONE"
