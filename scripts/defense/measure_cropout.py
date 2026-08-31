"""FAIR crop test: apply the EXACT training-time Cropout (replace <=30% area with cover, aligned)
vs my geometric crop-zoom. Shows what C5's crop augmentation actually bought.
Cols: clean | cropout0.1 | cropout0.2 | cropout0.3 | geo_crop75(zoom)."""
import sys, os, glob, numpy as np, torch
from PIL import Image
sys.path.insert(0,"/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
from vine.src.training_src.other_noises import Cropout
os.chdir("/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")  # ./config_mask/*.yaml for irregular mask
dev="cuda"; IMG=256; NCOV=8; NSEC=2; NMASK=4
RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"
def latest(cfg):
    ds=sorted(glob.glob(os.path.join(RT,cfg,"checkpoint-*")),key=lambda d:int(d.split("-")[-1])); return ds[-1],int(ds[-1].split("-")[-1])
c0d,c0s=latest("vine_official_C0"); c5d,c5s=latest("vine_official_C5")
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:NCOV]
covers=[np.asarray(Image.open(p).convert("RGB").resize((IMG,IMG)),np.float32)/255 for p in NAT]
def to_t(x): return (torch.from_numpy(x).permute(2,0,1)[None].float()*2-1).to(dev)
def load_enc(s): return (VINE_Turbo.from_pretrained(s.split(":",1)[1]) if str(s).startswith("pretrained:") else VINE_Turbo(ckpt_path=s,device=dev)).to(dev).eval()
def load_dec(spec):
    if str(spec).startswith("pretrained:"): return CustomConvNeXt.from_pretrained("Shilin-LU/VINE-B-Dec").to(dev).eval()
    d=CustomConvNeXt(secret_size=100); d.load_state_dict(torch.load(os.path.join(spec,"CustomConvNeXt.pth"),map_location="cpu")); return d.to(dev).eval()
def decT(dec,t):
    with torch.no_grad(): return dec(t)
cropout=Cropout().to(dev)
def geo_crop(x01,r=0.75):
    im=Image.fromarray((x01*255).astype(np.uint8)); s=int(IMG*r); o=(IMG-s)//2
    return np.asarray(im.crop((o,o,o+s,o+s)).resize((IMG,IMG)),np.float32)/255
LEVELS=[0.1,0.2,0.3]
models=[("C0",c0d),("C5",c5d),("VINE-B","pretrained:Shilin-LU/VINE-B-Enc")]
torch.manual_seed(0); np.random.seed(0)
hdr=["clean"]+[f"cropout{l}" for l in LEVELS]+["geoCrop75"]
print(f"{'model':8s} "+" ".join(f"{h:>10s}" for h in hdr),flush=True)
for mname,spec in models:
    enc=load_enc(spec); dec=load_dec(spec)
    try: T=int(enc.sched.config.num_train_timesteps)-1
    except: T=999
    acc={h:[] for h in hdr}
    for cov in covers:
        covt=torch.from_numpy(cov).permute(2,0,1)[None].float().to(dev)  # [0,1]
        for _ in range(NSEC):
            sec=torch.randint(0,2,(1,100),device=dev).float()
            with torch.no_grad(): wm=enc(to_t(cov),secret=sec,timesteps=torch.tensor([T],device=dev).long())
            w01=(wm[0]*0.5+0.5).clamp(0,1)  # [3,H,W]
            acc["clean"].append((( decT(dec,w01[None]) >0.5).float()==sec).float().mean().item())
            for l in LEVELS:
                for _ in range(NMASK):
                    out=cropout([w01[None].clone(),covt.clone()],max_tamper_area=l,
                                height_ratio_range=(max(1-l,0.8),max(1-l,0.9)),width_ratio_range=(max(1-l,0.8),max(1-l,0.9)))[0]
                    acc[f"cropout{l}"].append(((decT(dec,out)>0.5).float()==sec).float().mean().item())
            g=torch.from_numpy(geo_crop(w01.permute(1,2,0).cpu().numpy())).permute(2,0,1)[None].float().to(dev)
            acc["geoCrop75"].append(((decT(dec,g)>0.5).float()==sec).float().mean().item())
    print(f"{mname:8s} "+" ".join(f"{np.mean(acc[h]):10.3f}" for h in hdr),flush=True)
    del enc,dec; torch.cuda.empty_cache()
print("CROPOUT_DONE",flush=True)
