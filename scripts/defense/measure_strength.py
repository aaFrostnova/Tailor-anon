"""Watermark STRENGTH = residual RMS (in [0,255]) + PSNR, for from-scratch C0/C1/C5 (latest ckpt)
vs released VINE-B-Enc / VINE-R-Enc. Strength = how big the embedded mark is (magnitude, not location)."""
import sys, os, glob, numpy as np, torch
from PIL import Image
sys.path.insert(0,"/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
dev="cuda"; IMG=256; K=4
ROOT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts/vine_abl_scratch"
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:8]
covers=[np.asarray(Image.open(p).convert("RGB").resize((IMG,IMG)),np.float32)/255 for p in NAT]
def to_t(a): return (torch.from_numpy(a).permute(2,0,1)[None].float()*2-1).to(dev)
def latest(cfg):
    ds=sorted(glob.glob(os.path.join(ROOT,cfg,"checkpoint-*")),key=lambda d:int(d.split("-")[-1]))
    return ds[-1] if ds else None
specs=[("C0_full",latest("C0_full")),("C1_secretonly",latest("C1_secretonly")),("C5_crop",latest("C5_crop")),
       ("VINE-B-Enc(release)","pretrained:Shilin-LU/VINE-B-Enc"),("VINE-R-Enc(release)","pretrained:Shilin-LU/VINE-R-Enc")]
print(f"{'model':24s} {'step':>7s}  {'RMS(0-255)':>11s}  {'PSNR(dB)':>9s}",flush=True)
for name,spec in specs:
    if spec is None: print(f"{name:24s}  (no ckpt)"); continue
    try:
        if str(spec).startswith("pretrained:"): enc=VINE_Turbo.from_pretrained(spec.split(":",1)[1]); step="-"
        else: enc=VINE_Turbo(ckpt_path=spec,device=dev); step=os.path.basename(spec).split("-")[-1]
        enc.to(dev).eval()
        try: T=int(enc.sched.config.num_train_timesteps)-1
        except: T=999
        rmss=[]; psnrs=[]
        for cov in covers:
            for _ in range(K):
                s=torch.randint(0,2,(1,100),device=dev).float()
                with torch.no_grad(): wm=enc(to_t(cov),secret=s,timesteps=torch.tensor([T],device=dev).long())
                w01=(wm[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()
                res=w01-cov; mse=np.mean(res**2)
                rmss.append(np.sqrt(mse)*255); psnrs.append(10*np.log10(1/max(mse,1e-12)))
        print(f"{name:24s} {str(step):>7s}  {np.mean(rmss):7.2f}±{np.std(rmss):4.2f}  {np.mean(psnrs):9.2f}",flush=True)
        del enc; torch.cuda.empty_cache()
    except Exception as e: print(f"{name:24s}  ERR {repr(e)[:120]}",flush=True)
print("STRENGTH_DONE",flush=True)
