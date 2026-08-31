"""Simple robustness: bit-acc of official C0/C5 (latest ckpt) + released VINE-B under common attacks.
Attacks: clean, jpeg50/25, gaussblur, gaussnoise, resize0.5, centercrop0.75, rotate10."""
import sys, os, glob, io, numpy as np, torch
from PIL import Image, ImageFilter
sys.path.insert(0,"/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
dev="cuda"; IMG=256; NCOV=8; NSEC=2
RT="/scratch/workspace/mingzhel_umass_edu-ablator/vine_ckpts"
def latest(cfg):
    ds=sorted(glob.glob(os.path.join(RT,cfg,"checkpoint-*")),key=lambda d:int(d.split("-")[-1])); return ds[-1],int(ds[-1].split("-")[-1])
c0d,c0s=latest("vine_official_C0"); c5d,c5s=latest("vine_official_C5")
NAT=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:NCOV]
covers=[np.asarray(Image.open(p).convert("RGB").resize((IMG,IMG)),np.float32)/255 for p in NAT]
def to_t(x): return (torch.from_numpy(x).permute(2,0,1)[None].float()*2-1).to(dev)
def np01(t): return t
# attacks on [0,1] HxWx3
def a_clean(x): return x
def a_jpeg(x,q): 
    b=io.BytesIO(); Image.fromarray((x*255).astype(np.uint8)).save(b,"JPEG",quality=q); b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"),np.float32)/255
def a_blur(x,s=2): return np.asarray(Image.fromarray((x*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(s)),np.float32)/255
def a_noise(x,s=0.05): return np.clip(x+np.random.randn(*x.shape).astype(np.float32)*s,0,1)
def a_resize(x,r=0.5):
    im=Image.fromarray((x*255).astype(np.uint8)); s=int(IMG*r)
    return np.asarray(im.resize((s,s)).resize((IMG,IMG)),np.float32)/255
def a_crop(x,r=0.75):
    im=Image.fromarray((x*255).astype(np.uint8)); s=int(IMG*r); o=(IMG-s)//2
    return np.asarray(im.crop((o,o,o+s,o+s)).resize((IMG,IMG)),np.float32)/255
def a_rot(x,deg=10): return np.asarray(Image.fromarray((x*255).astype(np.uint8)).rotate(deg,resample=Image.BILINEAR),np.float32)/255
ATT=[("clean",a_clean),("jpeg50",lambda x:a_jpeg(x,50)),("jpeg25",lambda x:a_jpeg(x,25)),
     ("blur2",a_blur),("noise.05",a_noise),("resize.5",a_resize),("crop75",a_crop),("rot10",a_rot)]
def load_enc(s): return (VINE_Turbo.from_pretrained(s.split(":",1)[1]) if str(s).startswith("pretrained:") else VINE_Turbo(ckpt_path=s,device=dev)).to(dev).eval()
def load_dec(spec):
    if str(spec).startswith("pretrained:"): return CustomConvNeXt.from_pretrained("Shilin-LU/VINE-B-Dec").to(dev).eval()
    d=CustomConvNeXt(secret_size=100); d.load_state_dict(torch.load(os.path.join(spec,"CustomConvNeXt.pth"),map_location="cpu")); return d.to(dev).eval()
models=[(f"C0 {c0s//1000}k",c0d),(f"C5 {c5s//1000}k",c5d),("VINE-B(rel)","pretrained:Shilin-LU/VINE-B-Enc")]
torch.manual_seed(0); np.random.seed(0)
print(f"{'model':14s} "+" ".join(f"{a[0]:>9s}" for a in ATT),flush=True)
for mname,spec in models:
    enc=load_enc(spec); dec=load_dec(spec)
    try: T=int(enc.sched.config.num_train_timesteps)-1
    except: T=999
    accs={a[0]:[] for a in ATT}
    for cov in covers:
        for _ in range(NSEC):
            sec=torch.randint(0,2,(1,100),device=dev).float()
            with torch.no_grad(): wm=enc(to_t(cov),secret=sec,timesteps=torch.tensor([T],device=dev).long())
            w01=(wm[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()
            for aname,af in ATT:
                att=np.ascontiguousarray(af(w01))
                t=torch.from_numpy(att).permute(2,0,1)[None].float().to(dev)
                with torch.no_grad(): bits=dec(t)
                accs[aname].append(((bits>0.5).float()==sec).float().mean().item())
    print(f"{mname:14s} "+" ".join(f"{np.mean(accs[a[0]]):9.3f}" for a in ATT),flush=True)
    del enc,dec; torch.cuda.empty_cache()
print("ROBUST_DONE",flush=True)
