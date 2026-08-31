"""Gallery: 8 covers (gray + 7 natural). Columns = original | VINE-B watermarked | VINE-B residual |
VINE-R residual | C0(latest) residual | C5(latest) residual. Original vs watermarked shows imperceptibility;
residuals (per-cell norm) show WHERE the mark sits + border ratio."""
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
idx=[3,7,12,20,30,45,60,80]
covers=[("gray",np.full((IMG,IMG,3),0.5,np.float32))]+[(f"img{k}",np.asarray(Image.open(NAT[k]).convert("RGB").resize((IMG,IMG)),np.float32)/255) for k in idx[:7]]
yy,xx=np.mgrid[0:IMG,0:IMG]; D=np.minimum(np.minimum(yy,IMG-1-yy),np.minimum(xx,IMG-1-xx)); BORD=D<(IMG//10)
def to_t(x): return (torch.from_numpy(x).permute(2,0,1)[None].float()*2-1).to(dev)
torch.manual_seed(0); sec=torch.randint(0,2,(1,100),device=dev).float()
def load(s): return (VINE_Turbo.from_pretrained(s.split(":",1)[1]) if str(s).startswith("pretrained:") else VINE_Turbo(ckpt_path=s,device=dev)).to(dev).eval()
specs={"VINE-B":"pretrained:Shilin-LU/VINE-B-Enc","VINE-R":"pretrained:Shilin-LU/VINE-R-Enc","C0":c0d,"C5":c5d}
# precompute
wm_img={}; res={}
for mn,sp in specs.items():
    enc=load(sp)
    try: T=int(enc.sched.config.num_train_timesteps)-1
    except: T=999
    wm_img[mn]=[]; res[mn]=[]
    for _,cov in covers:
        with torch.no_grad(): w=enc(to_t(cov),secret=sec,timesteps=torch.tensor([T],device=dev).long())
        w01=(w[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()
        r=np.abs(w01-cov).mean(2); wm_img[mn].append(w01); res[mn].append((r,r[BORD].mean()/max(r[~BORD].mean(),1e-9)))
    del enc; torch.cuda.empty_cache(); print(f"  {mn} done",flush=True)
# columns: (header, kind, model)
cols=[("original","orig",None),("VINE-B\nwatermarked","wm","VINE-B"),(f"VINE-B\nresidual","res","VINE-B"),
      ("VINE-R\nresidual","res","VINE-R"),(f"C0 (full) {c0s//1000}k\nresidual","res","C0"),(f"C5 (crop) {c5s//1000}k\nresidual","res","C5")]
nr,ncl=len(covers),len(cols)
fig,ax=plt.subplots(nr,ncl,figsize=(2.05*ncl,2.05*nr))
for j,(hdr,kind,mn) in enumerate(cols):
    for i,(cname,cov) in enumerate(covers):
        b=ax[i,j]; b.set_xticks([]); b.set_yticks([])
        if kind=="orig": b.imshow(np.clip(cov,0,1))
        elif kind=="wm": b.imshow(np.clip(wm_img[mn][i],0,1))
        else:
            r,ratio=res[mn][i]; b.imshow(r,cmap="viridis")
        if i==0: b.set_title(hdr+("" if kind!="res" else f"\nratio={res[mn][i][1]:.2f}"),fontsize=8)
        elif kind=="res": b.set_title(f"ratio={res[mn][i][1]:.2f}",fontsize=7)
        if j==0: b.set_ylabel(cname,fontsize=9)
plt.suptitle("Original vs watermarked vs residual across 8 covers (VINE-B / VINE-R / official C0 / C5)",fontsize=11)
plt.tight_layout(); out="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/fig_wm_gallery.png"
plt.savefig(out,dpi=120,bbox_inches="tight"); print(f"SAVED {out}\nGALLERY_DONE",flush=True)
