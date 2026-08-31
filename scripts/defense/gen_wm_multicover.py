"""Multi-cover comparison WITH original: rows = 6 covers (gray + 5 natural), columns =
[original | VINE-B resid | VINE-R resid | C0(latest) resid | C5(latest) resid]. Residual cells
per-cell normalized + border ratio. Original column = the cover image itself (for comparison)."""
import sys, os, glob, numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
dev="cuda"; IMG=256
RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"
def latest(cfg):
    ds=sorted(glob.glob(os.path.join(RT,cfg,"checkpoint-*")),key=lambda d:int(d.split("-")[-1])); return ds[-1], int(ds[-1].split("-")[-1])
c0d,c0s=latest("vine_official_C0"); c5d,c5s=latest("vine_official_C5")
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))
covers=[("gray",np.full((IMG,IMG,3),0.5,np.float32))]+[(f"nat{i}",np.asarray(Image.open(NAT[i]).convert("RGB").resize((IMG,IMG)),np.float32)/255) for i in [3,7,12,20,30]]
models=[("original",None),("VINE-B\n(released)","pretrained:Shilin-LU/VINE-B-Enc"),("VINE-R\n(released)","pretrained:Shilin-LU/VINE-R-Enc"),
        (f"C0 full\nstep {c0s//1000}k",c0d),(f"C5 crop\nstep {c5s//1000}k",c5d)]
yy,xx=np.mgrid[0:IMG,0:IMG]; D=np.minimum(np.minimum(yy,IMG-1-yy),np.minimum(xx,IMG-1-xx)); BORD=D<(IMG//10)
def to_t(x): return (torch.from_numpy(x).permute(2,0,1)[None].float()*2-1).to(dev)
torch.manual_seed(0); sec=torch.randint(0,2,(1,100),device=dev).float()
def load(s): return (VINE_Turbo.from_pretrained(s.split(":",1)[1]) if str(s).startswith("pretrained:") else VINE_Turbo(ckpt_path=s,device=dev)).to(dev).eval()
# precompute residuals per (model,cover)
enc_cache={}
nr,ncl=len(covers),len(models)
fig,ax=plt.subplots(nr,ncl,figsize=(2.1*ncl,2.15*nr))
for j,(mlabel,spec) in enumerate(models):
    enc=None if spec is None else load(spec)
    T=999
    if enc is not None:
        try: T=int(enc.sched.config.num_train_timesteps)-1
        except: pass
    for i,(cname,cov) in enumerate(covers):
        b=ax[i,j]; b.set_xticks([]); b.set_yticks([])
        if enc is None:
            b.imshow(np.clip(cov,0,1))   # original
            if i==0: b.set_title("original",fontsize=9)
        else:
            with torch.no_grad(): wm=enc(to_t(cov),secret=sec,timesteps=torch.tensor([T],device=dev).long())
            w01=(wm[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()
            r=np.abs(w01-cov).mean(2); ratio=r[BORD].mean()/max(r[~BORD].mean(),1e-9)
            b.imshow(r,cmap="viridis")
            if i==0: b.set_title(f"{mlabel}\nratio={ratio:.2f}",fontsize=8.5)
            else: b.set_title(f"ratio={ratio:.2f}",fontsize=7)
        if j==0: b.set_ylabel(cname,fontsize=9)
    if enc is not None: del enc; torch.cuda.empty_cache()
    print(f"  {mlabel} done",flush=True)
plt.suptitle("Watermark residual across covers: original vs VINE-B vs VINE-R vs official C0/C5 (per-cell normalized)",fontsize=11)
plt.tight_layout(); out="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/fig_wm_multicover.png"
plt.savefig(out,dpi=125,bbox_inches="tight"); print(f"SAVED {out}\nMULTI_DONE",flush=True)
