"""Apples-to-apples border ratio of the RELEASED VINE-B-Enc vs VINE-R-Enc, same averaged method
(16 secrets x 5 covers = 80) as the from-scratch trajectory. Answers: what is the true reference '5'?"""
import sys, glob, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
dev="cuda"; IMG=256; NSEC=16
yy,xx=np.mgrid[0:IMG,0:IMG]; Dist=np.minimum(np.minimum(yy,IMG-1-yy),np.minimum(xx,IMG-1-xx)); BORD=Dist<(IMG//10)
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))
covers=[np.full((IMG,IMG,3),0.5,np.float32)]+[np.asarray(Image.open(p).convert("RGB").resize((IMG,IMG)),np.float32)/255 for p in NAT[:4]]
def to_t(a01): return (torch.from_numpy(a01).permute(2,0,1)[None].float()*2-1).to(dev)
for name,repo in [("VINE-B-Enc","Shilin-LU/VINE-B-Enc"),("VINE-R-Enc","Shilin-LU/VINE-R-Enc")]:
    try:
        enc=VINE_Turbo.from_pretrained(repo).to(dev).eval()
        try: T=int(enc.sched.config.num_train_timesteps)-1
        except: T=999
        ratios=[]; psnrs=[]; bmeans=[]
        for cov in covers:
            for _ in range(NSEC):
                s=torch.randint(0,2,(1,100),device=dev).float()
                with torch.no_grad(): wm=enc(to_t(cov),secret=s,timesteps=torch.tensor([T],device=dev).long())
                w01=(wm[0]*0.5+0.5).permute(1,2,0).cpu().numpy(); res=w01-cov; r=np.abs(res).mean(2)
                ratios.append(r[BORD].mean()/max(r[~BORD].mean(),1e-9)); bmeans.append(float(r[BORD].mean()))
                mse=np.mean((w01-cov)**2); psnrs.append(10*np.log10(1/max(mse,1e-12)))
        print(f"{name}: border/center ratio = {np.mean(ratios):.3f} ± {np.std(ratios):.3f}  | border|res| {np.mean(bmeans):.4f} | PSNR {np.mean(psnrs):.2f}  (n={len(ratios)})",flush=True)
    except Exception as e: print(f"{name}: ERR {repr(e)[:200]}",flush=True)
print("RELEASED_DONE",flush=True)
