"""Watermark residual SPATIAL distribution over TRAINING vs released VINE-B.
Rows = gray cover (intrinsic border pattern) + a natural cover. Columns = checkpoints (given --steps)
then released VINE-B-Enc (border-concentrated target). Each cell = |residual| heatmap (per-cell normalized)
annotated with border/center ratio. Args: --dir <ckpt dir> --steps 5000,20000,... --out <png> --tag <label>."""
import sys, os, glob, argparse, numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
dev="cuda"; IMG=256
ap=argparse.ArgumentParser()
ap.add_argument("--dir", default="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts/vine_official_C0")
ap.add_argument("--steps", default="")   # comma-sep; empty = all available
ap.add_argument("--out", required=True); ap.add_argument("--tag", default="C0")
a=ap.parse_args()
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))
covers=[("gray", np.full((IMG,IMG,3),0.5,np.float32)),
        ("natural", np.asarray(Image.open(NAT[3]).convert("RGB").resize((IMG,IMG)),np.float32)/255)]
avail=sorted(int(d.split("-")[-1]) for d in glob.glob(os.path.join(a.dir,"checkpoint-*")))
steps=[int(s) for s in a.steps.split(",") if s] if a.steps else avail
steps=[s for s in steps if s in avail]
cols=[(f"{a.tag} {s//1000}k", os.path.join(a.dir,f"checkpoint-{s}")) for s in steps]
cols.append(("VINE-B\n(released)", "pretrained:Shilin-LU/VINE-B-Enc"))
cols.append(("VINE-R\n(released)", "pretrained:Shilin-LU/VINE-R-Enc"))
yy,xx=np.mgrid[0:IMG,0:IMG]; D=np.minimum(np.minimum(yy,IMG-1-yy),np.minimum(xx,IMG-1-xx)); BORD=D<(IMG//10)
def to_t(x): return (torch.from_numpy(x).permute(2,0,1)[None].float()*2-1).to(dev)
torch.manual_seed(0); sec=torch.randint(0,2,(1,100),device=dev).float()
def load(spec):
    if str(spec).startswith("pretrained:"): return VINE_Turbo.from_pretrained(spec.split(":",1)[1]).to(dev).eval()
    return VINE_Turbo(ckpt_path=spec,device=dev).to(dev).eval()
nr,ncl=len(covers),len(cols)
fig,ax=plt.subplots(nr,ncl,figsize=(2.15*ncl,2.5*nr))
for j,(label,spec) in enumerate(cols):
    try:
        enc=load(spec)
        try: T=int(enc.sched.config.num_train_timesteps)-1
        except: T=999
        for i,(cname,cov) in enumerate(covers):
            with torch.no_grad(): wm=enc(to_t(cov),secret=sec,timesteps=torch.tensor([T],device=dev).long())
            w01=(wm[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()
            r=np.abs(w01-cov).mean(2); ratio=r[BORD].mean()/max(r[~BORD].mean(),1e-9)
            b=ax[i,j]; b.imshow(r,cmap="viridis"); b.set_xticks([]); b.set_yticks([])
            b.set_title(f"{label}\nratio={ratio:.2f}" if i==0 else f"ratio={ratio:.2f}",fontsize=8)
            if j==0: b.set_ylabel(cname,fontsize=10)
        del enc; torch.cuda.empty_cache(); print(f"  col {label} done",flush=True)
    except Exception as e: print(f"  col {label} ERR {repr(e)[:150]}",flush=True)
plt.suptitle(f"{a.tag}: watermark residual over training (per-cell normalized) vs released VINE-B  |  gray row = intrinsic border pattern",fontsize=11)
plt.tight_layout(); plt.savefig(a.out,dpi=125,bbox_inches="tight"); print(f"SAVED {a.out}\nEVOLUTION_DONE",flush=True)
