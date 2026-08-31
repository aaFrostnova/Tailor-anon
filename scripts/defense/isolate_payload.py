"""Separate the TWO components of VINE's residual:
  total    = |watermarked(msg) - original|      (contains VAE round-trip error + payload)
  vae_only = |watermarked(null) - original|      (VAE round-trip error alone, zero secret)
  payload  = |watermarked(msg) - watermarked(null)|  (message-caused change; VAE error subtracted out)
Question: is the PAYLOAD (not the VAE artifact) also on the border? And how big is payload vs VAE error?"""
import sys, glob, numpy as np
from PIL import Image
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0,"scripts/defense")
import torch
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
KEY=b"v5_key_encoder_master"; sb=ShortenedBCH(); n=sb.n
V=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device="cuda")
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
yy,xx=np.mgrid[0:512,0:512]; D=np.minimum(np.minimum(yy,511-yy),np.minimum(xx,511-xx)).astype(int)
def bfrac(r): return float(r[D<51].sum()/(r.sum()+1e-9))
def wm(cov_arr,tgt): return np.asarray(to512(V.embed_with_target(Image.fromarray(cov_arr.astype(np.uint8)),tgt)),np.float64)
def tgt(iid): p,M=V.get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)),p,M)
for name,cov in [("natural",np.asarray(to512(Image.open(sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[0]).convert("RGB")),np.float64)),
                 ("gray128",np.full((512,512,3),128.0))]:
    wm_msg=wm(cov,tgt("whyA")); wm_null=wm(cov,np.zeros(n,dtype=np.uint8)); wm_null2=wm(cov,np.zeros(n,dtype=np.uint8))
    total=np.abs(wm_msg-cov).mean(2); vae=np.abs(wm_null-cov).mean(2); payload=np.abs(wm_msg-wm_null).mean(2)
    detn=np.abs(wm_null-wm_null2).mean(2)  # determinism check
    print(f"[{name}]")
    print(f"  total   |wm(msg)-orig|  : mean={total.mean():.3f}  border-frac={bfrac(total):.3f}")
    print(f"  vae     |wm(null)-orig| : mean={vae.mean():.3f}  border-frac={bfrac(vae):.3f}   (VAE round-trip error alone)")
    print(f"  PAYLOAD |wm(msg)-wm(null)|: mean={payload.mean():.3f}  border-frac={bfrac(payload):.3f}   <-- the actual watermark")
    print(f"  payload/vae energy ratio = {payload.sum()/vae.sum():.3f}   determinism |null-null2| mean={detn.mean():.4f}")
print("ISOLATE_DONE")
