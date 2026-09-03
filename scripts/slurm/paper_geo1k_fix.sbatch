#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=long
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G -t 0-4
#SBATCH --job-name=geo1k_fix
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python
OUTD="/scratch/workspace/mingzhel_umass_edu-ablator/paper_geo1k"
# generate geometry attacks into the CORRECT dir from existing embeds
$PY - "$OUTD" <<'PYA'
import os,sys,json
from PIL import Image
OUT=sys.argv[1]
def rotb(im,d): return im.rotate(d,resample=Image.BICUBIC)
def cz(im,r): s=int(512*r); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
ATT={"rot9":lambda x:rotb(x,9),"rot30":lambda x:rotb(x,30),"crop75":lambda x:cz(x,.75),"crop50":lambda x:cz(x,.5)}
for M in ("ours","vine","trustmark","videoseal","maskwm"):
    ed=os.path.join(OUT,"embed_"+M)
    if not os.path.exists(os.path.join(ed,"meta.json")): print("skip",M); continue
    meta=json.load(open(os.path.join(ed,"meta.json")))
    for name,fn in ATT.items():
        d=os.path.join(OUT,f"atk_{M}_{name}"); os.makedirs(d,exist_ok=True); json.dump(meta,open(os.path.join(d,"meta.json"),"w"))
        for it in meta["items"]:
            im=Image.open(os.path.join(ed,f"img_{it['i']:05d}.png")).convert("RGB").resize((512,512)); fn(im).save(os.path.join(d,f"img_{it['i']:05d}.png"))
    print("built",M,flush=True)
print("ATTACKS_DONE")
PYA
CE="$PY scripts/defense/composite_external_eval.py"
for M in ours vine trustmark videoseal maskwm; do
  case $M in ours) FR="vine trustmark videoseal"; EX="--geo_cascade";; *) FR="$M"; EX="";; esac
  for A in rot9 rot30 crop75 crop50; do
    echo "########## $M $A ##########"
    $CE --mode decode --fragments $FR $EX --embed_dir "$OUTD/embed_$M" --attacked_dir "$OUTD/atk_${M}_$A" \
      --attack_name "${M}_$A" --output "$OUTD/out_${M}_$A.json" 2>&1 | grep -aE "COMPOSITE_or"
  done
done
echo "SBATCH_DONE"
