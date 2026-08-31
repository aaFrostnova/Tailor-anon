"""Fidelity cost of the 3-fragment composite. Embed VINE -> TrustMark -> VideoSeal
(sequential alpha=0.70 residual blends, the composite order) and report PSNR/SSIM of each
cumulative config vs the cover: VINE+TM (current 2-frag), VINE+VideoSeal, VINE+TM+VideoSeal.
"""
import os, sys, glob
import numpy as np
from PIL import Image
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts"), os.path.join(REPO,"scripts/defense")]:
    sys.path.insert(0, p)
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
try:
    from skimage.metrics import structural_similarity as _ssim
    def ssim(a,b): return float(_ssim(a,b,channel_axis=2))
except Exception:
    def ssim(a,b): return float("nan")

KEY=b"v5_key_encoder_master"; dev="cuda"; ALPHA=0.70
N=int(sys.argv[1]) if len(sys.argv)>1 else 16
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def sr(c,w,a): C=np.asarray(to512(c),np.float64); W=np.asarray(to512(w),np.float64); return Image.fromarray(np.clip(C+a*(W-C),0,255).astype(np.uint8))
def psnr(a,b): e=np.mean((np.asarray(a,np.float64)-np.asarray(b,np.float64))**2); return 99 if e<1e-9 else 10*np.log10(255*255/e)
imgs=(sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png")))+sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[180:180+N]
P={"vine_tm":[], "vine_vs":[], "vine_tm_vs":[]}; S={"vine_tm":[], "vine_tm_vs":[]}
for j,fp in enumerate(imgs):
    iid=f"sw_{180+j:05d}"; orig=to512(Image.open(fp).convert("RGB"))
    cw=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
    v=sr(orig, to512(vine.embed_with_target(orig,apply_crypto(cw,pv,Mv))), ALPHA)
    vt=sr(v, to512(tm.embed_with_target(v,apply_crypto(cw,pt,Mt))), ALPHA)            # VINE+TM
    vvs=sr(v, to512(vsf.embed_with_target(v,apply_crypto(cw,ps,Ms))), ALPHA)          # VINE+VideoSeal
    vtv=sr(vt, to512(vsf.embed_with_target(vt,apply_crypto(cw,ps,Ms))), ALPHA)        # VINE+TM+VideoSeal
    P["vine_tm"].append(psnr(orig,vt)); P["vine_vs"].append(psnr(orig,vvs)); P["vine_tm_vs"].append(psnr(orig,vtv))
    S["vine_tm"].append(ssim(np.asarray(orig),np.asarray(vt))); S["vine_tm_vs"].append(ssim(np.asarray(orig),np.asarray(vtv)))
    if (j+1)%8==0: print(f"  [{j+1}/{len(imgs)}]",flush=True)
print(f"\nn={N}  fidelity (alpha={ALPHA})")
print(f"  VINE+TM           PSNR {np.mean(P['vine_tm']):.1f}dB  SSIM {np.mean(S['vine_tm']):.4f}")
print(f"  VINE+VideoSeal    PSNR {np.mean(P['vine_vs']):.1f}dB")
print(f"  VINE+TM+VideoSeal PSNR {np.mean(P['vine_tm_vs']):.1f}dB  SSIM {np.mean(S['vine_tm_vs']):.4f}  (3-frag cost {np.mean(P['vine_tm'])-np.mean(P['vine_tm_vs']):.1f}dB vs 2-frag)")
print("PSNR3_DONE")
