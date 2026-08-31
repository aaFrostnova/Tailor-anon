"""Current watermark residual of official C0 vs C5 (latest checkpoint) vs released VINE-B.
Rows = gray (intrinsic pattern) + natural. Each cell = |residual| heatmap (per-cell norm) + border ratio + PSNR."""
import sys, os, glob, numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
dev="cuda"; IMG=256
RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"
def latest(cfg):
    ds=sorted(glob.glob(os.path.join(RT,cfg,"checkpoint-*")),key=lambda d:int(d.split("-")[-1]))
    return ds[-1], int(ds[-1].split("-")[-1])
c0d,c0s=latest("vine_official_C0"); c5d,c5s=latest("vine_official_C5")
cols=[(f"C0 (full)\nstep {c0s//1000}k", c0d), (f"C5 (crop)\nstep {c5s//1000}k", c5d), ("VINE-B\n(released)","pretrained:Shilin-LU/VINE-B-Enc")]
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))
covers=[("gray",np.full((IMG,IMG,3),0.5,np.float32)),("natural",np.asarray(Image.open(NAT[3]).convert("RGB").resize((IMG,IMG)),np.float32)/255)]
yy,xx=np.mgrid[0:IMG,0:IMG]; D=np.minimum(np.minimum(yy,IMG-1-yy),np.minimum(xx,IMG-1-xx)); BORD=D<(IMG//10)
def to_t(x): return (torch.from_numpy(x).permute(2,0,1)[None].float()*2-1).to(dev)
torch.manual_seed(0); sec=torch.randint(0,2,(1,100),device=dev).float()
def load(s): return (VINE_Turbo.from_pretrained(s.split(":",1)[1]) if str(s).startswith("pretrained:") else VINE_Turbo(ckpt_path=s,device=dev)).to(dev).eval()
fig,ax=plt.subplots(2,3,figsize=(8.5,6))
for j,(label,spec) in enumerate(cols):
    enc=load(spec)
    try: T=int(enc.sched.config.num_train_timesteps)-1
    except: T=999
    for i,(cname,cov) in enumerate(covers):
        with torch.no_grad(): wm=enc(to_t(cov),secret=sec,timesteps=torch.tensor([T],device=dev).long())
        w01=(wm[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()
        r=np.abs(w01-cov).mean(2); ratio=r[BORD].mean()/max(r[~BORD].mean(),1e-9)
        rms=np.sqrt(np.mean((w01-cov)**2))*255
        ax[i,j].imshow(r,cmap="viridis"); ax[i,j].set_xticks([]); ax[i,j].set_yticks([])
        ax[i,j].set_title(f"{label}\nratio={ratio:.2f} RMS={rms:.1f}" if i==0 else f"ratio={ratio:.2f}",fontsize=9)
        if j==0: ax[i,j].set_ylabel(cname,fontsize=11)
    del enc; torch.cuda.empty_cache(); print(f"  {label} done",flush=True)
plt.suptitle("Current watermark residual: official C0 vs C5 vs released VINE-B (gray row = intrinsic pattern)",fontsize=11)
plt.tight_layout(); out="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/fig_wm_current_C0_C5.png"
plt.savefig(out,dpi=130,bbox_inches="tight"); print(f"SAVED {out}\nCURRENT_DONE",flush=True)
