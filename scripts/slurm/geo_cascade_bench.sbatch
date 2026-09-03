#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=short
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=6 --mem=48G -t 0-3
#SBATCH --job-name=geo_cascade_bench
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
SS=/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_integ; FR="vine trustmark videoseal"
echo "node=$(hostname)"; nvidia-smi -L 2>/dev/null | head -1
# --- build attacked dirs from the composite+sync embed set ---
$PY - <<'PYA'
import os,json,numpy as np
from PIL import Image
SS="/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_integ"
def rotb(im,d): return im.rotate(d,resample=Image.BICUBIC)
def cz(im,r): s=int(512*r); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
meta=json.load(open(os.path.join(SS,"embed","meta.json")))
ATT={"rot9":lambda x:rotb(x,9),"rot30":lambda x:rotb(x,30),"crop75":lambda x:cz(x,.75),"crop50":lambda x:cz(x,.5)}
for name,fn in ATT.items():
    d=os.path.join(SS,"bench_"+name); os.makedirs(d,exist_ok=True); json.dump(meta,open(os.path.join(d,"meta.json"),"w"))
    for it in meta["items"]:
        im=Image.open(os.path.join(SS,"embed",f"img_{it['i']:05d}.png")).convert("RGB").resize((512,512))
        fn(im).save(os.path.join(d,f"img_{it['i']:05d}.png"))
print("attacks_built",flush=True)
PYA
# --- run 3 configs x 4 attacks ---
for A in rot9 rot30 crop75 crop50; do
  for CFG in resync syncseal geo_cascade; do
    echo "########## attack=$A  config=--$CFG ##########"
    $PY scripts/defense/composite_external_eval.py --mode decode --fragments $FR --$CFG \
      --embed_dir "$SS/embed" --attacked_dir "$SS/bench_$A" --attack_name "${A}_${CFG}" \
      --output "$SS/bench_out_${A}_${CFG}.json" 2>&1 | grep -aE "COMPOSITE_or|SPEED|composite vs"
  done
done
echo "SBATCH_DONE"
