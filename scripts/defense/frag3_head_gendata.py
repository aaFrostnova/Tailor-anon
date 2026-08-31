"""Generate 3-way fusion-head training data: VINE + TrustMark + VideoSeal aligned codeword
LLRs under a diverse attack suite that covers ALL the death modes the gate must learn:
  regen/rinse -> TM & VideoSeal die, VINE survives   (the CtrlRegen+ dilution case)
  crop        -> VINE dies, TM/VS partial
  rotate      -> VINE dies, TM dies (>5deg), VideoSeal survives
  clean/jpeg/blur/noise/bright/contrast -> all alive
Cache (AV, AT, AS, TX, ATK, IMG) -> results/defense/frag3_head_data.npz. Encoders frozen.
"""
import os, sys, glob, io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts"), os.path.join(REPO,"scripts/defense")]:
    sys.path.insert(0, p)
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from _regen_util import build_regen_pipe, stable_regen

KEY=b"v5_key_encoder_master"; dev="cuda"; ALPHA=0.70; CLAMP=15.0
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
pipe=build_regen_pipe()
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def sr(c,w,a): C=np.asarray(to512(c),np.float64); W=np.asarray(to512(w),np.float64); return Image.fromarray(np.clip(C+a*(W-C),0,255).astype(np.uint8))

def a_jpeg(im,r): q=int(round(90-80*r)); b=io.BytesIO(); im.save(b,"JPEG",quality=q); b.seek(0); return Image.open(b).convert("RGB")
def a_blur(im,r): return im.filter(ImageFilter.GaussianBlur(0.5+7.5*r))
def a_noise(im,r): x=np.asarray(im,np.float64)+np.random.RandomState(int(r*1e6)+1).normal(0,(0.02+0.08*r)*255,(512,512,3)); return Image.fromarray(np.clip(x,0,255).astype(np.uint8))
def a_bright(im,r): return ImageEnhance.Brightness(im).enhance(0.6+1.2*r)
def a_contrast(im,r): return ImageEnhance.Contrast(im).enhance(0.6+1.2*r)
def a_crop(im,r): area=0.95-0.40*r; s=int(round(512*np.sqrt(area))); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
def a_rot(im,r): return im.rotate(round(5+85*r), resample=Image.BILINEAR)   # 5..90 deg
def a_regen(im,r,sd): return to512(stable_regen(pipe,im,seed=sd))
def a_rinse(im,r,sd): return to512(stable_regen(pipe,to512(stable_regen(pipe,im,seed=sd)),seed=sd+1))
ATT={"clean":lambda im,r,sd:im,"jpeg":lambda im,r,sd:a_jpeg(im,r),"blur":lambda im,r,sd:a_blur(im,r),
     "noise":lambda im,r,sd:a_noise(im,r),"bright":lambda im,r,sd:a_bright(im,r),"contrast":lambda im,r,sd:a_contrast(im,r),
     "crop":lambda im,r,sd:a_crop(im,r),"rotate":lambda im,r,sd:a_rot(im,r),"regen":a_regen,"rinse":a_rinse}
ATTACKS=list(ATT.keys())

imgs=(sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png")))+
      sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[:N]
AV,AT,AS,TX,ATK,IMG=[],[],[],[],[],[]
for i,fp in enumerate(imgs):
    iid=f"f3_{i:05d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)); orig=to512(Image.open(fp).convert("RGB"))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
    v=sr(orig,to512(vine.embed_with_target(orig,apply_crypto(tx,pv,Mv))),ALPHA)
    vt=sr(v,to512(tm.embed_with_target(v,apply_crypto(tx,pt,Mt))),ALPHA)
    comp=sr(vt,to512(vsf.embed_with_target(vt,apply_crypto(tx,ps,Ms))),ALPHA)
    rng=np.random.RandomState(1000+i)
    for k in ATTACKS:
        r=float(rng.uniform(0.3,1.0)); att=to512(ATT[k](comp,r,2000+i))
        AV.append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(att),pv,Mv,kind="prob",n_codeword=sb.n),-CLAMP,CLAMP).astype(np.float32))
        AT.append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(att),pt,Mt,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP).astype(np.float32))
        AS.append(np.clip(method_soft_to_codeword_llr(vsf.raw_logits(att),ps,Ms,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP).astype(np.float32))
        TX.append(tx.astype(np.uint8)); ATK.append(k); IMG.append(i)
    if (i+1)%10==0: print(f"  [{i+1}/{len(imgs)}] samples={len(AV)}",flush=True)
out=os.path.join(REPO,"results/defense/frag3_head_data.npz")
np.savez_compressed(out,AV=np.array(AV),AT=np.array(AT),AS=np.array(AS),TX=np.array(TX),ATK=np.array(ATK),IMG=np.array(IMG),n=sb.n,attacks=np.array(ATTACKS))
print(f"[saved] {len(AV)} samples ({len(imgs)}x{len(ATTACKS)}) -> {out}\nFRAG3GEN_DONE")
