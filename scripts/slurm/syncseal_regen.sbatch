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
#SBATCH --job-name=syncseal_regen
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e
cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
SS2=/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_regen; COCO=/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017; JIT=/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_ckpt/syncmodel.jit.pt; FRAGS="vine trustmark videoseal"
echo "node=$(hostname)"; nvidia-smi -L 2>/dev/null | head -1
mkdir -p $SS2
# --- Step A: embed composite + sync on 32 COCO ---
echo "=== EMBED (composite + syncseal) ==="
$PY scripts/defense/composite_external_eval.py --mode embed --fragments $FRAGS --syncseal \
  --n_images 32 --start_idx 5000 --image_dir "$COCO" --image_glob "*.jpg" \
  --embed_dir "$SS2/embed" --strength 0.70 2>&1 | grep -aE "EMBED_DONE"
# --- Step B: regen + compound (rot30->regen, crop75->regen) + sync-survival diagnostic ---
$PY - <<'PYA'
import os,sys,json,numpy as np,torch
from PIL import Image
sys.path.insert(0,"scripts/defense")
from _regen_util import build_regen_pipe, stable_regen
from torchvision.transforms.functional import to_tensor
SS2="/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_regen"; JIT="/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_ckpt/syncmodel.jit.pt"; dev="cuda"
def rotb(im,d): return im.rotate(d,resample=Image.BICUBIC)
def cz(im,r): s=int(512*r); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
meta=json.load(open(os.path.join(SS2,"embed","meta.json")))
pipe=build_regen_pipe()
sync=torch.jit.load(JIT).to(dev).eval()
def sscore(pil):
    with torch.no_grad(): return float(sync.detect(to_tensor(pil).unsqueeze(0).to(dev))["preds"][0,0])
clean_s=[]; regen_s=[]
ATT={"regen":lambda x,s:stable_regen(pipe,x,s),
     "rot30_regen":lambda x,s:stable_regen(pipe,rotb(x,30),s),
     "crop75_regen":lambda x,s:stable_regen(pipe,cz(x,.75),s)}
for name in ATT: d=os.path.join(SS2,name); os.makedirs(d,exist_ok=True); json.dump(meta,open(os.path.join(d,"meta.json"),"w"))
for k,it in enumerate(meta["items"]):
    im=Image.open(os.path.join(SS2,"embed",f"img_{it['i']:05d}.png")).convert("RGB").resize((512,512))
    clean_s.append(sscore(im))
    for name,fn in ATT.items():
        att=fn(im,1234+k); att.save(os.path.join(SS2,name,f"img_{it['i']:05d}.png"))
        if name=="regen": regen_s.append(sscore(att))
    if (k+1)%8==0: print(f"  regen [{k+1}/{len(meta['items'])}]",flush=True)
print(f"SYNC-SURVIVAL: presence score clean {np.mean(clean_s):+.3f} -> regen {np.mean(regen_s):+.3f}",flush=True)
print("regen_attacks_ready",flush=True)
PYA
# --- Step C: decode --syncseal on each compound attack ---
for A in regen rot30_regen crop75_regen; do
  echo "=== DECODE --syncseal : $A ==="
  $PY scripts/defense/composite_external_eval.py --mode decode --fragments $FRAGS --syncseal \
    --embed_dir "$SS2/embed" --attacked_dir "$SS2/$A" --attack_name "ss_$A" \
    --output "$SS2/out_$A.json" 2>&1 | grep -aE "COMPOSITE_or|composite vs"
done
echo "SBATCH_DONE"
