"""VINE-B vs VINE-R watermark strength AT THE BORDER (absolute residual energy), not just the ratio.
Report mean |residual| in the outer-10% border band vs the center, for both, on gray + natural covers,
and save a radial profile (energy vs distance-to-edge)."""
import sys, glob, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
dev="cuda"; IMG=256
encB = VINE_Turbo.from_pretrained("Shilin-LU/VINE-B-Enc").to(dev).eval()
encR = VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc").to(dev).eval()
T = int(encB.sched.config.num_train_timesteps)-1
yy,xx=np.mgrid[0:IMG,0:IMG]; Dedge=np.minimum(np.minimum(yy,IMG-1-yy),np.minimum(xx,IMG-1-xx)); BORD=Dedge<IMG//10
def to_t(a01): return (torch.from_numpy(a01).permute(2,0,1)[None].float()*2-1).to(dev)
def resid(enc,cov,sec):
    with torch.no_grad(): wm=enc(to_t(cov),secret=sec,timesteps=torch.tensor([T],device=dev).long())
    return np.abs((wm[0]*0.5+0.5).permute(1,2,0).cpu().numpy()-cov).mean(2)*255   # |residual| in /255 units
def radial(r,nb=48):
    bins=np.linspace(0,IMG//2,nb+1); idx=np.clip(np.digitize(Dedge.ravel(),bins)-1,0,nb-1)
    prof=np.array([r.ravel()[idx==b].mean() if np.any(idx==b) else 0 for b in range(nb)])
    return (0.5*(bins[:-1]+bins[1:])).tolist(), prof.tolist()
covers={"gray":np.full((IMG,IMG,3),0.5,np.float64)}
for s in sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:5]:
    covers[f"nat{len(covers)}"]=np.asarray(Image.open(s).convert("RGB").resize((IMG,IMG)),np.float64)/255
prof_store={}
print(f"=== |residual| (in /255) : BORDER band vs CENTER ===")
print(f"{'cover':8} | {'VINE-B border':>13} {'center':>7} | {'VINE-R border':>13} {'center':>7}")
aggB=[]; aggR=[]
for name,cov in covers.items():
    sec=torch.randint(0,2,(1,100),device=dev).float()
    rB=resid(encB,cov,sec); rR=resid(encR,cov,sec)
    bB,cB=rB[BORD].mean(),rB[~BORD].mean(); bR,cR=rR[BORD].mean(),rR[~BORD].mean()
    print(f"{name:8} | {bB:13.3f} {cB:7.3f} | {bR:13.3f} {cR:7.3f}")
    if name.startswith("nat"): aggB.append((bB,cB)); aggR.append((bR,cR))
    if name in ("gray","nat1"):
        x,yB=radial(rB); _,yR=radial(rR); prof_store[name]=(x,yB,yR)
aggB=np.array(aggB); aggR=np.array(aggR)
print(f"--- natural avg: VINE-B border={aggB[:,0].mean():.3f} center={aggB[:,1].mean():.3f} | VINE-R border={aggR[:,0].mean():.3f} center={aggR[:,1].mean():.3f}")
print(f"--- BORDER strength ratio R/B = {aggR[:,0].mean()/aggB[:,0].mean():.2f}x")
# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(11,4)); 
for ax,(nm,(x,yB,yR)) in zip(axs,prof_store.items()):
    ax.plot(x,yB,color="#2a78d6",lw=2,label="VINE-B"); ax.plot(x,yR,color="#e34948",lw=2,label="VINE-R")
    ax.set_xlabel("distance to nearest edge (px)"); ax.set_ylabel("|residual| (/255)"); ax.set_title(f"cover: {nm}",fontsize=10,loc="left")
    ax.axvspan(0,IMG//10,color="#eee",zorder=0); ax.legend(frameon=False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.suptitle("VINE-B vs VINE-R watermark strength vs distance-to-edge (border = shaded)",y=1.02)
fig.savefig("results/defense/final_figs/fig_bvr_border_strength.png",dpi=150,bbox_inches="tight")
print("BVR_BORDER_DONE")
