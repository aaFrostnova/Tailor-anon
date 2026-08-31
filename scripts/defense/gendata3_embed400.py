"""Embed VINE+TM+VideoSeal composite @alpha=0.7 on N UltraEdit imgs, SAVE the watermarked
512 PNGs (image_id = fh3_{i:05d}, same convention as gendata3) so external-env attackers
(CtrlRegen+, UnMarker) can process them. -> /scratch/.../fh3_embed400/{i:05d}.png"""
import os,sys,glob
import numpy as np
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"external/videoseal"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.payload import image_id_to_payload
KEY=b"v5_key_encoder_master"; ALPHA=0.70; dev="cuda"
N=int(sys.argv[1]) if len(sys.argv)>1 else 80
OUT="/scratch/workspace/mingzhel_umass_edu-ablator/fh3_embed400"; os.makedirs(OUT,exist_ok=True)
imgs=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vs=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def scale_resid(c,w,a):
    c=np.asarray(to512(c),np.float64); w=np.asarray(to512(w),np.float64)
    return Image.fromarray(np.clip(c+a*(w-c),0,255).astype(np.uint8))
for i,fp in enumerate(imgs):
    iid=f"fh3_{i:05d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    orig=to512(Image.open(fp).convert("RGB"))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vs.get_perm_M(iid)
    a1=scale_resid(orig,to512(vine.embed_with_target(orig,apply_crypto(tx,pv,Mv))),ALPHA)
    a2=scale_resid(a1,to512(tm.embed_with_target(a1,apply_crypto(tx,pt,Mt))),ALPHA)
    comp=scale_resid(a2,to512(vs.embed_with_target(a2,apply_crypto(tx,ps,Ms))),ALPHA)
    comp.save(os.path.join(OUT,f"{i:05d}.png"))
    if (i+1)%10==0: print(f"  embedded {i+1}/{len(imgs)}",flush=True)
print(f"[saved] {len(imgs)} composite imgs -> {OUT}\nEMBED_DONE")
