"""DECISIVE test for H1: does a NULL-secret (all-zero 100-bit) VAE round-trip already show the border ring?
If yes on natural AND flat covers -> the border is the zero-padded autoencoder boundary effect, message-independent."""
import sys, glob, json, numpy as np
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
def resid(cov_arr,tgt):
    wm=to512(V.embed_with_target(Image.fromarray(cov_arr.astype(np.uint8)),tgt))
    return np.abs(np.asarray(wm,np.float64)-cov_arr).mean(axis=2)
def bfrac(r): return float(r[D<51].sum()/r.sum())   # fraction of energy in outer-10% frame band
def bcorr(x,y):
    m=D<51; xv=x[m]-x[m].mean(); yv=y[m]-y[m].mean(); return float((xv*yv).sum()/(np.sqrt((xv**2).sum()*(yv**2).sum())+1e-9))
nat=np.asarray(to512(Image.open(sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[0]).convert("RGB")),np.float64)
gray=np.full((512,512,3),128.0)
zero=np.zeros(n,dtype=np.uint8)                          # TRUE null secret (no crypto)
def tgt(iid): p,M=V.get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)),p,M)
rnat0=resid(nat,zero); rnatA=resid(nat,tgt("whyA")); rgray0=resid(gray,zero)
print(f"  natural  null-secret : border-band frac={bfrac(rnat0):.3f}   (message-A frac={bfrac(rnatA):.3f})")
print(f"  gray128  null-secret : border-band frac={bfrac(rgray0):.3f}")
print(f"  corr(null-secret , message-A) on border = {bcorr(rnat0,rnatA):.3f}")
print("NULL_DONE")
