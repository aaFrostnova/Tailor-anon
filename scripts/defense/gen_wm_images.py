"""Visual comparison: watermarked image + residual PATTERN (position) + residual on SHARED scale (strength),
for from-scratch C0/C1/C5 (latest ckpt) vs released VINE-B/R. One row per model, on a fixed cover."""
import sys, os, glob, numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
dev="cuda"; IMG=256
ROOT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts/vine_abl_scratch"
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))
cover=np.asarray(Image.open(NAT[3]).convert("RGB").resize((IMG,IMG)),np.float32)/255
def to_t(a): return (torch.from_numpy(a).permute(2,0,1)[None].float()*2-1).to(dev)
def latest(cfg):
    ds=sorted(glob.glob(os.path.join(ROOT,cfg,"checkpoint-*")),key=lambda d:int(d.split("-")[-1])); return ds[-1] if ds else None
specs=[("C0_full (from-scratch)",latest("C0_full")),("C1_secretonly (no imperceptibility)",latest("C1_secretonly")),
       ("C5_crop (from-scratch)",latest("C5_crop")),("VINE-B-Enc (released)","pretrained:Shilin-LU/VINE-B-Enc"),
       ("VINE-R-Enc (released)","pretrained:Shilin-LU/VINE-R-Enc")]
torch.manual_seed(0); sec=torch.randint(0,2,(1,100),device=dev).float()
rows=[]
for name,spec in specs:
    if spec is None: continue
    if str(spec).startswith("pretrained:"): enc=VINE_Turbo.from_pretrained(spec.split(":",1)[1]); step="rel"
    else: enc=VINE_Turbo(ckpt_path=spec,device=dev); step=os.path.basename(spec).split("-")[-1]
    enc.to(dev).eval()
    try: T=int(enc.sched.config.num_train_timesteps)-1
    except: T=999
    with torch.no_grad(): wm=enc(to_t(cover),secret=sec,timesteps=torch.tensor([T],device=dev).long())
    w01=(wm[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()
    res=w01-cover; rms=np.sqrt(np.mean(res**2))*255; ares=np.abs(res).mean(2)
    rows.append((f"{name}\nstep {step} | RMS {rms:.1f} | PSNR {10*np.log10(1/max(np.mean(res**2),1e-12)):.1f}dB", w01, ares))
    del enc; torch.cuda.empty_cache()
# shared scale for strength column = 99th pct across all abs-residuals
vmax_shared=np.percentile(np.concatenate([r[2].ravel() for r in rows]),99.5)
n=len(rows); fig,ax=plt.subplots(n,3,figsize=(9.5,3.1*n))
for i,(title,w01,ares) in enumerate(rows):
    ax[i,0].imshow(np.clip(w01,0,1)); ax[i,0].set_ylabel(title,fontsize=8,rotation=0,ha="right",va="center",labelpad=8)
    ax[i,1].imshow(ares,cmap="viridis"); # per-model normalized (position pattern)
    im=ax[i,2].imshow(ares,cmap="inferno",vmin=0,vmax=vmax_shared)  # shared scale (strength)
    for j in range(3): ax[i,j].set_xticks([]); ax[i,j].set_yticks([])
ax[0,0].set_title("watermarked image",fontsize=10); ax[0,1].set_title("residual PATTERN\n(per-model norm → WHERE)",fontsize=10)
ax[0,2].set_title("residual STRENGTH\n(shared scale → HOW BIG)",fontsize=10)
plt.tight_layout(); out="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/vine_abl_scratch/fig_wm_visual_compare.png"
plt.savefig(out,dpi=125,bbox_inches="tight"); print(f"SAVED {out}\nWM_IMG_DONE",flush=True)
