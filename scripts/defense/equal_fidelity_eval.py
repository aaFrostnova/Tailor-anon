"""Equal-fidelity eval anchored on REGEN. For each attack, measure PSNR/SSIM/LPIPS (vs clean).
An attack is 'legal' only if its fidelity is >= regen on ALL three (PSNR>=, SSIM>=, LPIPS<=).
Attacks weaker-fidelity-than-regen are excluded (too destructive). Merge with composite detection."""
import glob, io, os, sys, time, json, tempfile, numpy as np
from PIL import Image
import torch, lpips
from skimage.metrics import structural_similarity as ssim
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/scripts/defense"); sys.path.insert(0,os.path.join(REPO,"external","WatermarkAttacker"))
from _regen_util import build_regen_pipe, stable_regen
RES=512; dev="cuda"; N=8
def arr(p): return np.asarray(p.convert("RGB").resize((RES,RES)),np.float32)
def psnr(a,b):
    m=np.mean((a-b)**2); return 99.0 if m<1e-9 else 10*np.log10(255.0**2/m)
lp=lpips.LPIPS(net='alex').to(dev).eval()
def lpd(a,b):
    ta=torch.from_numpy(a/127.5-1).permute(2,0,1)[None].float().to(dev); tb=torch.from_numpy(b/127.5-1).permute(2,0,1)[None].float().to(dev)
    with torch.no_grad(): return float(lp(ta,tb).item())
pipe=build_regen_pipe()
from wmattacker import (VAEWMAttacker,GaussianBlurAttacker,GaussianNoiseAttacker,JPEGAttacker,BrightnessAttacker,ContrastAttacker,BM3DAttacker)
tmp=tempfile.mkdtemp()
sig={"jpeg":JPEGAttacker(quality=25),"blur":GaussianBlurAttacker(5,1),"noise":GaussianNoiseAttacker(std=0.05),
     "bright":BrightnessAttacker(0.2),"contrast":ContrastAttacker(0.2),"bm3d":BM3DAttacker()}
vae={}
for k,mn in [("vae_b","bmshj2018-hyperprior"),("vae_c","cheng2020-anchor")]:
    try: vae[k]=VAEWMAttacker(mn,quality=3,metric="mse",device=dev)
    except Exception as e: print("vae skip",k,flush=True)
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
ATKS=["regen","jpeg","blur","noise","bright","contrast","bm3d","rinse2x","vae_b","vae_c","rs256","hflip","crop75","crop50","rot9","crop_jpeg"]
files=[Image.open(f).convert("RGB").resize((RES,RES)) for f in sorted(glob.glob('/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg'))[4000:4000+N]]
M={}
for a in ATKS:
    t0=time.time(); P,S,L=[],[],[]
    for i,C in enumerate(files):
        Ca=arr(C); At=arr(attack(a,C,1234+i)); P.append(psnr(At,Ca)); S.append(ssim(At,Ca,channel_axis=2,data_range=255)); L.append(lpd(At,Ca))
    M[a]=(np.mean(P),np.mean(S),np.mean(L)); print(f"  {a:<11} PSNR={M[a][0]:5.1f} SSIM={M[a][1]:.3f} LPIPS={M[a][2]:.3f}  ({time.time()-t0:.0f}s)",flush=True)
rp,rs,rl=M["regen"]; det=json.load(open(REPO+"/results/defense/composite_vine_tm.json"))["attacks"]
print(f"\n=== equal-fidelity (anchor=regen: PSNR>={rp:.1f} SSIM>={rs:.3f} LPIPS<={rl:.3f}) ===")
print(f"{'attack':<11}{'PSNR':>7}{'SSIM':>7}{'LPIPS':>7}{'>=regen?':>10}{'comp_or':>9}")
legal=[]
for a in ATKS:
    p,s,l=M[a]; ok=(p>=rp-0.5 and s>=rs-0.02 and l<=rl+0.02); o=det.get(a,{}).get("composite_or",float('nan'))
    if a!="regen" and ok: legal.append(o)
    print(f"{a:<11}{p:>7.1f}{s:>7.3f}{l:>7.3f}{('LEGAL' if ok else 'excl'):>10}{o:>9.3f}")
print(f"\nregen itself: comp_or={det.get('regen',{}).get('composite_or',float('nan')):.3f}")
print(f"attacks with fidelity >= regen (excl regen): {len(legal)}, mean composite detection = {np.mean(legal) if legal else float('nan'):.3f}")
print("EQUAL_FIDELITY_DONE")
