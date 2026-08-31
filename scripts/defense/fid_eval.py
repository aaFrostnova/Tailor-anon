"""FID at W-Bench scale (n=1000) for the deployed 3-frag+head composite.
Reports: (1) finite-sample FLOOR = FID(clean split A, clean split B) — the FID you get from
n/2 samples of the SAME distribution (the small-n bias baseline); (2) imperceptibility
FID(watermarked, clean); (3) optional attack FID(regen_s, clean) as the WAVES FID component.
Interpretation: watermark is distributionally imperceptible iff FID_wm ~= FID_floor."""
import os, sys, glob, argparse
import numpy as np, torch
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts")]: sys.path.insert(0,p)
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from torchmetrics.image.fid import FrechetInceptionDistance
KEY=b"v5_key_encoder_master"; dev="cuda"; ALPHA=0.70
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=1000); ap.add_argument("--regen",action="store_true")
ap.add_argument("--regen_strengths",default="0.3,0.5"); a=ap.parse_args()
sb=ShortenedBCH(); n=sb.n
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=n,device=dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def scale(c,w):
    C=np.asarray(to512(c),np.float64); W=np.asarray(to512(w),np.float64)
    return Image.fromarray(np.clip(C+ALPHA*(W-C),0,255).astype(np.uint8))
def embed(orig,iid):
    tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
    x=scale(orig,to512(vine.embed_with_target(orig,apply_crypto(tx,pv,Mv))))
    x=scale(x,   to512(tm.embed_with_target(x,apply_crypto(tx,pt,Mt))))
    x=scale(x,   to512(vsf.embed_with_target(x,apply_crypto(tx,ps,Ms))))
    return x
def t8(pil): return torch.tensor(np.asarray(pil.convert("RGB"),np.uint8)).permute(2,0,1)[None]  # 1x3xHxW uint8

imgs=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:a.n]
print(f"FID eval n={len(imgs)}",flush=True)
clean=[to512(Image.open(f).convert("RGB")) for f in imgs]
wm=[]
for j,c in enumerate(clean):
    wm.append(embed(c,f"fid_{j:05d}"))
    if (j+1)%200==0: print(f"  embedded {j+1}/{len(imgs)}",flush=True)

def fid_of(realset,fakeset):
    m=FrechetInceptionDistance(feature=2048,normalize=False).to(dev); m.set_dtype(torch.float64)
    for i in range(0,len(realset),32): m.update(torch.cat([t8(x) for x in realset[i:i+32]]).to(dev),real=True)
    for i in range(0,len(fakeset),32): m.update(torch.cat([t8(x) for x in fakeset[i:i+32]]).to(dev),real=False)
    return float(m.compute())

h=len(clean)//2
floor=fid_of(clean[:h],clean[h:])              # finite-sample floor (same distribution)
fid_wm=fid_of(clean,wm)                          # imperceptibility
print(f"\n=== FID (n={len(imgs)}, Inception-2048) ===")
print(f"  FINITE-SAMPLE FLOOR  FID(clean_A, clean_B) [n/2 each] = {floor:.3f}")
print(f"  IMPERCEPTIBILITY     FID(watermarked, clean)          = {fid_wm:.3f}")
print(f"  -> excess over floor = {fid_wm-floor:.3f}  (watermark distribution shift)")

if a.regen:
    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
    SD="/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
    pipe=StableDiffusionImg2ImgPipeline.from_pretrained(SD,torch_dtype=torch.float16,safety_checker=None,local_files_only=True).to(dev)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True)
    for s in [float(x) for x in a.regen_strengths.split(",")]:
        att=[]
        for j,w in enumerate(wm):
            g=torch.Generator(dev).manual_seed(7000+j)
            att.append(to512(pipe(prompt="",image=w,strength=s,num_inference_steps=50,guidance_scale=1.0,generator=g).images[0]))
        fr=fid_of(clean,att)
        print(f"  ATTACK regen s={s}   FID(attacked, clean) = {fr:.3f}  (excess {fr-floor:.3f})")
print("FID_DONE")
