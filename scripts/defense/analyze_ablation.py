"""Ablation analysis: C0(full) vs C2(no-GAN) vs C3(no-LPIPS) at MATCHED step 45k.
Outputs: (1) residual-map comparison figure, (2) radial energy profile, (3) border-ratio/PSNR/RMS table.
Reveals whether removing GAN vs LPIPS are different failure modes for border emergence."""
import sys, os, glob, numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
dev="cuda"; IMG=256
RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"
OUT="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/VINE_border_report/figs"
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))
covers=[("gray",np.full((IMG,IMG,3),0.5,np.float32))]+[(f"img{k}",np.asarray(Image.open(NAT[k]).convert("RGB").resize((IMG,IMG)),np.float32)/255) for k in [3,7,20]]
yy,xx=np.mgrid[0:IMG,0:IMG]; D=np.minimum(np.minimum(yy,IMG-1-yy),np.minimum(xx,IMG-1-xx)); BORD=D<(IMG//10)
def to_t(x): return (torch.from_numpy(x).permute(2,0,1)[None].float()*2-1).to(dev)
def load(spec): return (VINE_Turbo.from_pretrained(spec.split(":",1)[1]) if str(spec).startswith("pretrained:") else VINE_Turbo(ckpt_path=spec,device=dev)).to(dev).eval()
torch.manual_seed(0); sec=torch.randint(0,2,(1,100),device=dev).float()
# columns: (label, spec)
cols=[("C0 full\n105k",f"{RT}/vine_official_C0/checkpoint-105000"),
      ("C2 no-GAN\n110k",f"{RT}/vine_official_C2/checkpoint-110000"),
      ("C3 no-LPIPS\n110k",f"{RT}/vine_official_C3/checkpoint-110000"),
      ("VINE-B\nreleased","pretrained:Shilin-LU/VINE-B-Enc")]
# radial bins (normalized distance from center, 0=center 1=corner)
cy=cx=(IMG-1)/2; R=np.sqrt((yy-cy)**2+(xx-cx)**2); R=R/R.max(); nb=12; bins=np.linspace(0,1,nb+1)
resid={}; radial={}; stats={}
for lab,spec in cols:
    enc=load(spec)
    try: T=int(enc.sched.config.num_train_timesteps)-1
    except: T=999
    rs=[]; rad=np.zeros(nb); psnrs=[]; rmss=[]; brs=[]
    for cname,cov in covers:
        with torch.no_grad(): w=enc(to_t(cov),secret=sec,timesteps=torch.tensor([T],device=dev).long())
        w01=(w[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()
        r=np.abs(w01-cov).mean(2)
        mse=np.mean((w01-cov)**2); psnrs.append(10*np.log10(1/max(mse,1e-12))); rmss.append(np.sqrt(mse)*255)
        brs.append(r[BORD].mean()/max(r[~BORD].mean(),1e-9))
        rs.append((cname,r))
        if cname=="gray":  # radial profile on gray (intrinsic pattern)
            for b in range(nb):
                m=(R>=bins[b])&(R<bins[b+1]); rad[b]=r[m].mean() if m.any() else 0
    resid[lab]=rs; radial[lab]=rad; stats[lab]=(np.mean(brs),np.mean(psnrs),np.mean(rmss))
    del enc; torch.cuda.empty_cache(); print(f"  {lab.split(chr(10))[0]:12s} done",flush=True)
# ---- table ----
print("\n=== 消融定量 (4 covers avg) ===")
print(f"{'model':16s} {'border_ratio':>12s} {'PSNR(dB)':>9s} {'RMS':>6s}")
for lab,_ in cols:
    br,ps,rm=stats[lab]; print(f"{lab.replace(chr(10),' '):16s} {br:12.2f} {ps:9.2f} {rm:6.2f}")
# ---- figure 1: residual maps ----
nr,ncl=len(covers),len(cols)
fig,ax=plt.subplots(nr,ncl,figsize=(2.05*ncl,2.05*nr))
for j,(lab,_) in enumerate(cols):
    for i,(cname,r) in enumerate(resid[lab]):
        b=ax[i,j]; b.imshow(r,cmap="viridis"); b.set_xticks([]); b.set_yticks([])
        if i==0: b.set_title(f"{lab}\nratio={stats[lab][0]:.2f}",fontsize=8)
        if j==0: b.set_ylabel(cname,fontsize=9)
plt.suptitle("Ablation residual maps: full vs no-GAN vs no-LPIPS (fully-trained 110k) + C0-105k / VINE-B",fontsize=11)
plt.tight_layout(); f1=f"{OUT}/fig_ablation_residual.png"; plt.savefig(f1,dpi=120,bbox_inches="tight"); plt.close()
# ---- figure 2: radial profile (gray) ----
plt.figure(figsize=(7,4.5)); ctr=(bins[:-1]+bins[1:])/2
for lab,_ in cols:
    rr=radial[lab]; rr=rr/max(rr.max(),1e-9)  # normalize each to its own peak
    plt.plot(ctr,rr,marker="o",ms=3,label=lab.replace(chr(10)," "))
plt.xlabel("normalized distance from center  (0=center, 1=corner)"); plt.ylabel("residual energy (self-normalized)")
plt.title("Radial energy profile on gray cover: where does the watermark sit?"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); f2=f"{OUT}/fig_ablation_radial.png"; plt.savefig(f2,dpi=120,bbox_inches="tight"); plt.close()
print(f"\nSAVED {f1}\nSAVED {f2}\nABL_ANALYSIS_DONE",flush=True)
