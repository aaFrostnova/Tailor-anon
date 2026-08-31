"""Noise-reduced position measurement: average border/center ratio over N secrets x K covers per checkpoint.
Tests whether C0 (imperceptibility) has a robustly higher border ratio than C1 (secret-only) even at low PSNR."""
import sys, os, glob, argparse, csv, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
dev="cuda"; IMG=256; NSEC=16
ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--configs",nargs="+",default=["C0_full","C1_secretonly","C5_crop"])
ap.add_argument("--minstep",type=int,default=0)   # skip checkpoints below this step (faster: only re-measure new ones)
a=ap.parse_args()
yy,xx=np.mgrid[0:IMG,0:IMG]; Dist=np.minimum(np.minimum(yy,IMG-1-yy),np.minimum(xx,IMG-1-xx)); BORD=Dist<(IMG//10)
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))
covers=[np.full((IMG,IMG,3),0.5,np.float32)]+[np.asarray(Image.open(p).convert("RGB").resize((IMG,IMG)),np.float32)/255 for p in NAT[:4]]
def to_t(a01): return (torch.from_numpy(a01).permute(2,0,1)[None].float()*2-1).to(dev)
def step_of(d):
    try: return int(os.path.basename(d).split("-")[1])
    except: return -1
rows=[]
for cfg in a.configs:
    for cd in sorted(glob.glob(os.path.join(a.root,cfg,"checkpoint-*")),key=step_of):
        st=step_of(cd)
        if st < a.minstep: continue
        if not all(os.path.exists(os.path.join(cd,f)) for f in ["UNet2DConditionModel.pth","ConditionAdaptor.pth","vae.pth","CustomConvNeXt.pth"]): continue
        try:
            enc=VINE_Turbo(ckpt_path=cd,device=dev).to(dev).eval()
            try: T=int(enc.sched.config.num_train_timesteps)-1
            except: T=999
            ratios=[]
            for cov in covers:
                for _ in range(NSEC):
                    s=torch.randint(0,2,(1,100),device=dev).float()
                    with torch.no_grad(): wm=enc(to_t(cov),secret=s,timesteps=torch.tensor([T],device=dev).long())
                    res=(wm[0]*0.5+0.5).permute(1,2,0).cpu().numpy()-cov; r=np.abs(res).mean(2)
                    ratios.append(r[BORD].mean()/max(r[~BORD].mean(),1e-9))
            m,sd=float(np.mean(ratios)),float(np.std(ratios))
            rows.append((cfg,st,m,sd)); print(f"  {cfg} step {st:5d}: ratio={m:.3f}±{sd:.3f}  (n={len(ratios)})",flush=True)
            del enc; torch.cuda.empty_cache()
        except Exception as e: print(f"  {cfg} step {st}: ERR {repr(e)[:150]}",flush=True)
with open(a.out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["config","step","ratio_mean","ratio_std"]); [w.writerow(r) for r in rows]
print(f"WROTE {len(rows)} -> {a.out}",flush=True)
