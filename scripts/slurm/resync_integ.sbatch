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
#SBATCH --job-name=resync_integ
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e
cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
SCRATCH=/scratch/workspace/mingzhel_umass_edu-ablator/resync_validation
echo "node=$(hostname)"; nvidia-smi -L 2>/dev/null | head -1
# Step A: build rot30-attacked dir (first 48 imgs) in scratch, reuse ext_vtv100 meta.json
$PY - <<'PYA'
import os,json,shutil,numpy as np
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
SRC=os.path.join(REPO,"results/defense/ext_vtv100")
DST="/scratch/workspace/mingzhel_umass_edu-ablator/resync_validation"; os.makedirs(DST,exist_ok=True)
def rot(img,deg):
    a=np.asarray(img); p=256
    big=Image.fromarray(np.pad(a,((p,p),(p,p),(0,0)),"reflect")).rotate(deg,resample=Image.BICUBIC)
    bw,bh=big.size; l=(bw-512)//2; return big.crop((l,l,l+512,l+512))
meta=json.load(open(os.path.join(SRC,"meta.json"))); meta["items"]=meta["items"][:48]
json.dump(meta,open(os.path.join(DST,"meta.json"),"w"))
for it in meta["items"]:
    im=Image.open(os.path.join(SRC,f"img_{it['i']:05d}.png")).convert("RGB").resize((512,512))
    rot(im,30.0).save(os.path.join(DST,f"img_{it['i']:05d}.png"))
print("attacked_dir_ready",len(meta["items"]),flush=True)
PYA
# Step B: decode WITHOUT resync (default) -> expect ~0 on rot30
echo "=== NO RESYNC ==="
$PY scripts/defense/composite_external_eval.py --mode decode --fragments vine trustmark videoseal \
  --embed_dir "$SCRATCH" --attacked_dir "$SCRATCH" --attack_name rot30_noresync \
  --output "$SCRATCH/out_noresync.json" 2>&1 | grep -aE "COMPOSITE_or|composite vs"
# Step C: decode WITH resync -> expect ~1.0
echo "=== WITH RESYNC ==="
$PY scripts/defense/composite_external_eval.py --mode decode --fragments vine trustmark videoseal \
  --embed_dir "$SCRATCH" --attacked_dir "$SCRATCH" --attack_name rot30_resync --resync \
  --output "$SCRATCH/out_resync.json" 2>&1 | grep -aE "COMPOSITE_or|composite vs"
echo "SBATCH_DONE"
