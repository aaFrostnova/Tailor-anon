"""END-TO-END (attack-in-the-loop) training of the 3-way fusion head, VINE-style.

Encoders/decoders stay FROZEN (pretrained external models). The 'end-to-end' part is the
NOISE/ATTACK LAYER in the loop: each step we sample watermarked images, apply a RANDOM
attack online (fast geo/signal-proc on a clean-wm OR a regen-wm base), decode the 3
fragments online, fuse with the head, BCE on the codeword, backprop into the head. So the
head sees fresh random attack instances every step (cf. VINE's online noise layer), not a
fixed offline LLR cache. Compared head-to-head against the offline head (frag3_head.pt) on
CtrlRegen+ / FPR / held-out.
"""
import os, sys, glob, io, json, random
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import binom
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts"), os.path.join(REPO,"scripts/defense")]:
    sys.path.insert(0,p)
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.fusion_head3 import Head3, load_head3
from _regen_util import build_regen_pipe, stable_regen
torch.manual_seed(0); np.random.seed(0); random.seed(0)
KEY=b"v5_key_encoder_master"; dev="cuda"; ALPHA=0.70; CLAMP=15.0
NTRAIN=int(sys.argv[1]) if len(sys.argv)>1 else 100
STEPS=int(sys.argv[2]) if len(sys.argv)>2 else 1200
BATCH=8
sb=ShortenedBCH(); tau=float(binom.ppf(0.99,sb.n,0.5)+1)/sb.n
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
pipe=build_regen_pipe()
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def sr(c,w,a): C=np.asarray(to512(c),np.float64); W=np.asarray(to512(w),np.float64); return Image.fromarray(np.clip(C+a*(W-C),0,255).astype(np.uint8))
def llrs(att,pv,Mv,pt,Mt,ps,Ms):
    av=np.clip(method_soft_to_codeword_llr(vine.raw_probs(att),pv,Mv,kind="prob",n_codeword=sb.n),-CLAMP,CLAMP)
    at=np.clip(method_soft_to_codeword_llr(tm.raw_logits(att),pt,Mt,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP)
    as_=np.clip(method_soft_to_codeword_llr(vsf.raw_logits(att),ps,Ms,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP)
    return av.astype(np.float32),at.astype(np.float32),as_.astype(np.float32)

# ---- online fast attacks (the noise layer) ----
def a_jpeg(im,r): q=int(round(90-80*r)); b=io.BytesIO(); im.save(b,"JPEG",quality=q); b.seek(0); return Image.open(b).convert("RGB")
def a_blur(im,r): return im.filter(ImageFilter.GaussianBlur(0.5+6.0*r))
def a_noise(im,r): x=np.asarray(im,np.float64)+np.random.normal(0,(0.02+0.07*r)*255,(512,512,3)); return Image.fromarray(np.clip(x,0,255).astype(np.uint8))
def a_bright(im,r): return ImageEnhance.Brightness(im).enhance(0.6+1.0*r)
def a_contrast(im,r): return ImageEnhance.Contrast(im).enhance(0.6+1.0*r)
def a_crop(im,r): area=0.95-0.45*r; s=int(round(512*np.sqrt(area))); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
def a_rot(im,r): return im.rotate(round(5+85*r),resample=Image.BILINEAR)
FAST=[lambda im:im,a_jpeg,a_blur,a_noise,a_bright,a_contrast,a_crop,a_rot]
def rand_attack(im):
    f=random.choice(FAST); return to512(f(im) if f is (lambda x:x) else (f(im, random.uniform(0.3,1.0)) if f.__code__.co_argcount==2 else f(im)))

# ---- pre-embed train pool (clean-wm + one regen-wm) ----
imgs=(sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png")))+sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[:NTRAIN]
POOL=[]
print(f"pre-embedding {len(imgs)} composite imgs (+regen base)...",flush=True)
for i,fp in enumerate(imgs):
    iid=f"on_{i:05d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)); orig=to512(Image.open(fp).convert("RGB"))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
    v=sr(orig,to512(vine.embed_with_target(orig,apply_crypto(tx,pv,Mv))),ALPHA)
    vt=sr(v,to512(tm.embed_with_target(v,apply_crypto(tx,pt,Mt))),ALPHA)
    comp=sr(vt,to512(vsf.embed_with_target(vt,apply_crypto(tx,ps,Ms))),ALPHA)
    regen=to512(stable_regen(pipe,comp,seed=7000+i))
    POOL.append(dict(clean=comp,regen=regen,pm=(pv,Mv,pt,Mt,ps,Ms),tx=tx.astype(np.uint8)))
    if (i+1)%20==0: print(f"  [{i+1}/{len(imgs)}]",flush=True)

# ---- online training ----
head=Head3().to(dev); opt=torch.optim.Adam(head.parameters(),lr=2e-3,weight_decay=1e-5)
print(f"online training {STEPS} steps batch {BATCH} (attack-in-the-loop)...",flush=True)
for step in range(STEPS):
    AV,AT,AS,TX=[],[],[],[]
    for _ in range(BATCH):
        s=random.choice(POOL); base=s["regen"] if random.random()<0.4 else s["clean"]
        att=rand_attack(base); av,at,as_=llrs(att,*s["pm"]); AV.append(av);AT.append(at);AS.append(as_);TX.append(s["tx"])
    av=torch.tensor(np.array(AV),device=dev);at=torch.tensor(np.array(AT),device=dev);as_=torch.tensor(np.array(AS),device=dev)
    y=torch.tensor(np.array(TX),dtype=torch.float32,device=dev)
    opt.zero_grad(); f=head(av,at,as_); loss=F.binary_cross_entropy_with_logits(f,y); loss.backward(); opt.step()
    if (step+1)%200==0: print(f"  step{step+1} loss={loss.item():.4f}",flush=True)
torch.save(head.state_dict(),os.path.join(REPO,"results/defense/frag3_head_online.pt")); head.eval()

# ---- eval: online vs offline head on real CtrlRegen+ + FPR ----
off=load_head3(os.path.join(REPO,"results/defense/frag3_head.pt"),dev)
def hl(h,a,b,c):
    with torch.no_grad(): return h(torch.tensor(a,device=dev),torch.tensor(b,device=dev),torch.tensor(c,device=dev)).cpu().numpy()
def detvec(Fz,tx,ids):
    d=np.zeros(len(Fz))
    for i in range(len(Fz)):
        ver=bool(decode_and_verify(Fz[i],ids[i],codec=sb)["detected"]); d[i]=1.0 if (ver or ((Fz[i]>0).astype(np.uint8)==tx[i]).mean()>=tau) else 0.0
    return d
meta=json.load(open(os.path.join(REPO,"results/defense/ext_vtv/meta.json")))
def dec(dirn):
    A,B,C,T,I=[],[],[],[],[]
    for it in meta["items"]:
        fp=os.path.join(REPO,dirn,f"img_{it['i']:05d}.png")
        if not os.path.exists(fp): continue
        iid=it["image_id"]; att=to512(Image.open(fp).convert("RGB")); tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
        pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
        av,at,as_=llrs(att,pv,Mv,pt,Mt,ps,Ms); A.append(av);B.append(at);C.append(as_);T.append(tx.astype(np.uint8));I.append(iid)
    return np.array(A),np.array(B),np.array(C),np.array(T),I
print(f"\n{'CtrlRegen+':10s}|{'VINE':>7s}|{'eq3':>7s}|{'OFFLINE':>8s}|{'ONLINE':>8s}")
for step_,tag in [("0.3","03"),("0.5","05"),("0.7","07")]:
    a,b,cc,tx2,ids2=dec(f"results/defense/ext_vtv_ctrlregen_{tag}")
    if len(a)==0: continue
    dvV=detvec(a,tx2,ids2); dvE=detvec(a+b+cc,tx2,ids2); dvO=detvec(hl(off,a,b,cc),tx2,ids2); dvN=detvec(hl(head,a,b,cc),tx2,ids2)
    print(f"s={step_:7s}|{dvV.mean():7.2f}|{dvE.mean():7.2f}|{dvO.mean():8.2f}|{dvN.mean():8.2f}")
# FPR
neg=(sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png")))+sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[260:300]
NV,NB,NC,NT,NID=[],[],[],[],[]
for j,fp in enumerate(neg):
    iid=f"neg_{j:05d}"; im=to512(Image.open(fp).convert("RGB"))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
    av,at,as_=llrs(im,pv,Mv,pt,Mt,ps,Ms); NV.append(av);NB.append(at);NC.append(as_);NT.append(sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8));NID.append(iid)
NV,NB,NC,NT=np.array(NV),np.array(NB),np.array(NC),np.array(NT)
print(f"[FPR unwm n={len(NV)}] eq3={detvec(NV+NB+NC,NT,NID).mean():.3f}  OFFLINE={detvec(hl(off,NV,NB,NC),NT,NID).mean():.3f}  ONLINE={detvec(hl(head,NV,NB,NC),NT,NID).mean():.3f}")
print("FRAG3ONLINE_DONE")
