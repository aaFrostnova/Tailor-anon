#!/bin/bash
#SBATCH --partition=superpod-a100 --account=pi_shiqingma_umass_edu --qos=short
#SBATCH --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=6 --mem=48G -t 0-1
#SBATCH --job-name=synthid_def
#SBATCH --output=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.out
#SBATCH --error=/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/slurm_logs/%x_%j.err
set -e; cd /work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint
export HF_HOME=/work/pi_shiqingma_umass_edu/mingzheli/hf_cache HF_HUB_OFFLINE=0
PY=/home/mingzhel_umass_edu/.conda/envs/fingerprint/bin/python; SS=/scratch/workspace/mingzhel_umass_edu-ablator/synthid_defense
CE="$PY scripts/defense/composite_external_eval.py"
# 1) embed OUR composite (+sync) on the 5 SynthID images
echo "=== EMBED our composite on SynthID images ==="
$CE --mode embed --fragments vine trustmark videoseal --geo_cascade \
  --n_images 5 --start_idx 0 --image_dir results/synthid/wm_test5 --image_glob "sid_*.png" \
  --embed_dir "$SS/embed" --strength 0.70 2>&1 | grep -aE "EMBED_DONE"
# 2) imperceptibility of our layer on SynthID pixels + build attacks (rot30/crop75/regen)
$PY - <<'PYA'
import os,sys,json,numpy as np
from PIL import Image
sys.path.insert(0,"scripts/defense")
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from _regen_util import build_regen_pipe, stable_regen
SS="/scratch/workspace/mingzhel_umass_edu-ablator/synthid_defense"; ed=os.path.join(SS,"embed"); meta=json.load(open(os.path.join(ed,"meta.json")))
def rotb(im,d): return im.rotate(d,resample=Image.BICUBIC)
def cz(im,r): s=int(512*r); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
# PSNR our-watermark vs the SynthID input
ps=[]
for it in meta["items"]:
    orig=Image.open(f"results/synthid/wm_test5/sid_{it['i']:05d}.png").convert("RGB").resize((512,512))
    wm=Image.open(os.path.join(ed,f"img_{it['i']:05d}.png")).convert("RGB").resize((512,512))
    ps.append(psnr(np.asarray(orig),np.asarray(wm),data_range=255))
print(f"OUR-WATERMARK imperceptibility on SynthID pixels: PSNR {np.mean(ps):.2f} dB",flush=True)
# geometry attacks
for name,fn in {"rot30":lambda x:rotb(x,30),"crop75":lambda x:cz(x,.75)}.items():
    d=os.path.join(SS,name); os.makedirs(d,exist_ok=True); json.dump(meta,open(os.path.join(d,"meta.json"),"w"))
    for it in meta["items"]:
        im=Image.open(os.path.join(ed,f"img_{it['i']:05d}.png")).convert("RGB").resize((512,512)); fn(im).save(os.path.join(d,f"img_{it['i']:05d}.png"))
# regen attack
pipe=build_regen_pipe(); d=os.path.join(SS,"regen"); os.makedirs(d,exist_ok=True); json.dump(meta,open(os.path.join(d,"meta.json"),"w"))
for it in meta["items"]:
    im=Image.open(os.path.join(ed,f"img_{it['i']:05d}.png")).convert("RGB").resize((512,512))
    stable_regen(pipe,im,1234+it['i']).resize((512,512)).save(os.path.join(d,f"img_{it['i']:05d}.png"))
print("attacks_built",flush=True)
PYA
# 3) detect OUR watermark on SynthID pixels, clean + under attack
for A in clean rot30 crop75 regen; do
  AD=$([ $A = clean ] && echo "$SS/embed" || echo "$SS/$A")
  echo "########## OUR detection on SynthID+ourWM : $A ##########"
  $CE --mode decode --fragments vine trustmark videoseal --geo_cascade \
    --embed_dir "$SS/embed" --attacked_dir "$AD" --attack_name "sid_$A" --output "$SS/out_$A.json" 2>&1 | grep -aE "COMPOSITE_or"
done
echo "SBATCH_DONE"
