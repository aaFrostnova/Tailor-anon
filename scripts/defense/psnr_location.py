"""Direct test of 'border = perceptually cheapest': if the crop-ft training pushes the payload INTO the interior,
does clean PSNR DROP (interior payload more visible = more perceptual cost)? Compare baseline VINE-B vs crop-ft
checkpoints: clean PSNR + payload border/center ratio. If PSNR falls as the border ratio falls -> border was cheaper."""
import sys, glob, numpy as np, torch
from PIL import Image
sys.path.insert(0,"/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
dev="cuda"; IMG=256
yy,xx=np.mgrid[0:IMG,0:IMG]; Dedge=np.minimum(np.minimum(yy,IMG-1-yy),np.minimum(xx,IMG-1-xx)); BORD=Dedge<IMG//10
def to_t(a): return (torch.from_numpy(a).permute(2,0,1)[None].float()*2-1).to(dev)
def load(spec): return (VINE_Turbo.from_pretrained(spec.split(":",1)[1]) if spec.startswith("pretrained:") else VINE_Turbo(ckpt_path=spec,device=dev)).to(dev).eval()
srcs=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:6]
covs=[np.asarray(Image.open(s).convert("RGB").resize((IMG,IMG)),np.float64)/255 for s in srcs]
import sys as _s; models=_s.argv[1:]
for spec in models:
    enc=load(spec); T=int(enc.sched.config.num_train_timesteps)-1; ps=[]; rat=[]
    for cov in covs:
        sec=torch.randint(0,2,(1,100),device=dev).float()
        with torch.no_grad(): wm=enc(to_t(cov),secret=sec,timesteps=torch.tensor([T],device=dev).long())
        res=(wm[0]*0.5+0.5).permute(1,2,0).cpu().numpy()-cov
        ps.append(10*np.log10(1.0/max((res**2).mean(),1e-12)))
        r=np.abs(res).mean(2); rat.append(r[BORD].mean()/max(r[~BORD].mean(),1e-9))
    name=(spec.split(":",1)[1].split("/")[-1] if spec.startswith("pretrained:") else "/".join(spec.rstrip("/").split("/")[-2:]))
    print(f"  {name:22}: clean PSNR={np.mean(ps):.2f}dB  border/center={np.mean(rat):.2f}",flush=True)
print("PSNR_LOC_DONE")
