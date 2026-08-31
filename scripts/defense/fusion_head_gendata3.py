"""3-fragment (VINE+TrustMark+VideoSeal) fusion-head data gen, WITH attacked image saved
(64x64) for the image-conditioned head. Composite embed @alpha=0.7, diverse attacks, aligned
codeword LLRs per fragment. -> results/defense/fusion_head_data3.npz (AV,AT,AS,TX,ATK,IMG,IMGD)."""
import os, sys, glob, io
import numpy as np, torch
from PIL import Image, ImageEnhance, ImageFilter
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"scripts")); sys.path.insert(0,os.path.join(REPO,"scripts/defense"))
sys.path.insert(0,os.path.join(REPO,"external/videoseal"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
KEY=b"v5_key_encoder_master"; ALPHA=0.70; dev="cuda"
N_IMG=int(sys.argv[1]) if len(sys.argv)>1 else 150
imgs=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N_IMG]
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vs=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
regen_pipe=build_regen_pipe()
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def scale_resid(c,w,a):
    c=np.asarray(to512(c),np.float64); w=np.asarray(to512(w),np.float64)
    return Image.fromarray(np.clip(c+a*(w-c),0,255).astype(np.uint8))
def att_jpeg(im,r): q=int(round(90-80*r)); b=io.BytesIO(); im.save(b,"JPEG",quality=q); b.seek(0); return Image.open(b).convert("RGB")
def att_blur(im,r): return im.filter(ImageFilter.GaussianBlur(radius=0.5+7.5*r))
def att_noise(im,r):
    a=np.asarray(im,np.float64)+np.random.RandomState(int(r*1e6)+1).normal(0,(0.02+0.08*r)*255,(512,512,3)); return Image.fromarray(np.clip(a,0,255).astype(np.uint8))
def att_bright(im,r): return ImageEnhance.Brightness(im).enhance(0.6+1.2*r)
def att_contrast(im,r): return ImageEnhance.Contrast(im).enhance(0.6+1.2*r)
def att_crop(im,r):  # RAVEN-style: keep 0.9..0.75 linear + resize
    ratio=0.90-0.15*r; s=int(round(512*ratio)); off=(512-s)//2; return im.crop((off,off,off+s,off+s)).resize((512,512))
def att_rot(im,r): return im.rotate(3+27*r, resample=Image.BILINEAR)  # rotation -> VideoSeal territory
def att_regen(im,r,sd): return stable_regen(regen_pipe,im,seed=sd)
def att_rinse(im,r,sd): return stable_regen(regen_pipe,stable_regen(regen_pipe,im,seed=sd),seed=sd+1)
ATTACKS=["clean","jpeg","blur","noise","bright","contrast","crop","rot","regen","rinse"]
def apply_attack(name,im,r,sd):
    return {"clean":lambda:im,"jpeg":lambda:att_jpeg(im,r),"blur":lambda:att_blur(im,r),"noise":lambda:att_noise(im,r),
            "bright":lambda:att_bright(im,r),"contrast":lambda:att_contrast(im,r),"crop":lambda:att_crop(im,r),
            "rot":lambda:att_rot(im,r),"regen":lambda:att_regen(im,r,sd),"rinse":lambda:att_rinse(im,r,sd)}[name]()
AV,AT,AS,TX,ATK,IMG,IMGD=[],[],[],[],[],[],[]
for i,fp in enumerate(imgs):
    iid=f"fh3_{i:05d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    orig=to512(Image.open(fp).convert("RGB"))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vs.get_perm_M(iid)
    tv=apply_crypto(tx,pv,Mv); tt=apply_crypto(tx,pt,Mt); ts=apply_crypto(tx,ps,Ms)
    a1=scale_resid(orig,to512(vine.embed_with_target(orig,tv)),ALPHA)
    a2=scale_resid(a1,to512(tm.embed_with_target(a1,tt)),ALPHA)
    comp=scale_resid(a2,to512(vs.embed_with_target(a2,ts)),ALPHA)
    rng=np.random.RandomState(1000+i)
    for k in ATTACKS:
        r=float(rng.uniform(0.3,1.0)); att=to512(apply_attack(k,comp,r,2000+i))
        av=method_soft_to_codeword_llr(vine.raw_probs(att),pv,Mv,kind="prob",n_codeword=sb.n)
        at=method_soft_to_codeword_llr(tm.raw_logits(att),pt,Mt,kind="logit",n_codeword=sb.n)
        as_=method_soft_to_codeword_llr(vs.raw_logits(att),ps,Ms,kind="logit",n_codeword=sb.n)
        AV.append(av.astype(np.float32)); AT.append(at.astype(np.float32)); AS.append(as_.astype(np.float32))
        TX.append(tx.astype(np.uint8)); ATK.append(k); IMG.append(i)
        IMGD.append(np.asarray(att.resize((64,64)),np.uint8))  # image for conditioning
    if (i+1)%10==0: print(f"  [{i+1}/{len(imgs)}] samples={len(AV)}",flush=True)
out=os.path.join(REPO,"results/defense/fusion_head_data3.npz")
np.savez_compressed(out,AV=np.array(AV),AT=np.array(AT),AS=np.array(AS),TX=np.array(TX),
                    ATK=np.array(ATK),IMG=np.array(IMG),IMGD=np.array(IMGD,dtype=np.uint8),
                    n=sb.n,alpha=ALPHA,attacks=np.array(ATTACKS))
print(f"[saved] {len(AV)} samples -> {out}\nGENDATA3_DONE")
