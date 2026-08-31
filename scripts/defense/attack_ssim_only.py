"""Compute SSIM(attack(C), C) per attack on clean images (fast, watermark-independent), with per-attack timing.
Merge with composite_vine_tm.json detection to see detection under SSIM>=0.8 fidelity constraint."""
import glob, io, os, sys, time, json, tempfile, numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/scripts/defense"); sys.path.insert(0,os.path.join(REPO,"external","WatermarkAttacker"))
from _regen_util import build_regen_pipe, stable_regen
RES=512; dev="cuda"; N=5
def arr(p): return np.asarray(p.convert("RGB").resize((RES,RES)),np.float32)
pipe=build_regen_pipe()
from wmattacker import (VAEWMAttacker,GaussianBlurAttacker,GaussianNoiseAttacker,JPEGAttacker,BrightnessAttacker,ContrastAttacker,BM3DAttacker)
tmp=tempfile.mkdtemp()
sig={"jpeg":JPEGAttacker(quality=25),"blur":GaussianBlurAttacker(5,1),"noise":GaussianNoiseAttacker(std=0.05),
     "bright":BrightnessAttacker(0.2),"contrast":ContrastAttacker(0.2),"bm3d":BM3DAttacker()}
vae={}
for k,mn in [("vae_b","bmshj2018-hyperprior"),("vae_c","cheng2020-anchor")]:
    try: vae[k]=VAEWMAttacker(mn,quality=3,metric="mse",device=dev)
    except Exception as e: print("vae skip",k,e,flush=True)
def sapply(a,pil):
    ip=os.path.join(tmp,"i.png"); op=os.path.join(tmp,"o.png"); pil.save(ip); a.attack([ip],[op]); return Image.open(op).convert("RGB").resize((RES,RES))
def geom(name,pil):
    if name=="hflip": return pil.transpose(Image.FLIP_LEFT_RIGHT)
    if name=="rs256": return pil.resize((256,256),Image.BICUBIC).resize((RES,RES),Image.BICUBIC)
    if name in("crop75","crop50"):
        r=0.75 if name=="crop75" else 0.5; cw=int(RES*r); l=(RES-cw)//2; return pil.crop((l,l,l+cw,l+cw)).resize((RES,RES),Image.BICUBIC)
    if name=="rot9":
        a=np.array(pil); pad=RES//2; big=Image.fromarray(np.pad(a,((pad,pad),(pad,pad),(0,0)),'reflect')).rotate(9,resample=Image.BICUBIC); bw,bh=big.size; l=(bw-RES)//2; return big.crop((l,l,l+RES,l+RES))
    if name=="crop_jpeg":
        c=geom("crop75",pil); b=io.BytesIO(); c.save(b,"JPEG",quality=25); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def attack(name,W,seed):
    if name in sig: return sapply(sig[name],W)
    if name in vae: return sapply(vae[name],W)
    if name=="regen": return stable_regen(pipe,W,seed,denoise_steps=8)
    if name=="rinse2x":
        x=stable_regen(pipe,W,seed,denoise_steps=8); return stable_regen(pipe,x,seed+1,denoise_steps=8)
    return geom(name,W)
ATKS=["jpeg","blur","noise","bright","contrast","bm3d","regen","rinse2x","vae_b","vae_c","rs256","hflip","crop75","crop50","rot9","crop_jpeg"]
files=[Image.open(f).convert("RGB").resize((RES,RES)) for f in sorted(glob.glob('/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg'))[4000:4000+N]]
out={}
for a in ATKS:
    t0=time.time(); ss=[]
    for i,C in enumerate(files):
        At=attack(a,C,1234+i); ss.append(ssim(arr(At),arr(C),channel_axis=2,data_range=255))
    out[a]=float(np.mean(ss)); print(f"  {a:<11} SSIM={out[a]:.3f}   ({time.time()-t0:.0f}s for {N} imgs)",flush=True)
det=json.load(open(REPO+"/results/defense/composite_vine_tm.json"))["attacks"]
print(f"\n{'attack':<11}{'SSIM':>7}{'legal>=.8':>10}{'comp_or(n=100)':>16}")
legal=[]
for a in ATKS:
    o=det.get(a,{}).get("composite_or",float('nan')); leg=out[a]>=0.8
    if leg: legal.append(o)
    print(f"{a:<11}{out[a]:>7.3f}{('YES' if leg else 'no'):>10}{o:>16.3f}")
print(f"\nLEGAL (SSIM>=0.8): {sum(1 for a in ATKS if out[a]>=0.8)} attacks, mean composite detection={np.mean(legal):.3f}")
print("SSIM_ONLY_DONE")
