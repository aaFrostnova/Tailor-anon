"""Is rot45/60 failure just the +-30 blind-search range? Re-run large angles with widened range."""
import os,sys,glob,json,numpy as np
from PIL import Image
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0,"scripts/defense")
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from src.syncseal_frontend import load_sync, sync_embed, sync_rectify
KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15.0; ALPHA=0.70
sb=ShortenedBCH(); n=sb.n; tau=float(binom.ppf(0.99,n,0.5)+1)/n
SPEC={"vine":("prob","raw_probs"),"trustmark":("logit","raw_logits"),"videoseal":("logit","raw_logits")}
F={"vine":VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device=dev),
   "trustmark":TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=n,model_type="B",device=dev),
   "videoseal":VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=n,device=dev)}
sync=load_sync(dev=dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def scl(c,w):
    C=np.asarray(to512(c),np.float64); W=np.asarray(to512(w),np.float64)
    return Image.fromarray(np.clip(C+ALPHA*(W-C),0,255).astype(np.uint8))
def txc(iid,nm):
    p,M=F[nm].get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)),p,M)
def vine_nested(x,iid):
    x=to512(x)
    for K in (1.0,0.75,0.5):
        if K>=0.999: x=to512(F["vine"].embed_with_target(x,txc(iid,"vine")))
        else:
            s=int(512*K); o=(512-s)//2
            m=F["vine"].embed_with_target(x.crop((o,o,o+s,o+s)),txc(iid,"vine"))
            out=x.copy(); out.paste(m.resize((s,s)),(o,o)); x=out
    return x
def embed(orig,iid,frags):
    x=to512(orig)
    for nm in frags: x=vine_nested(x,iid) if nm=="vine" else scl(x,to512(F[nm].embed_with_target(x,txc(iid,nm))))
    return to512(sync_embed(sync,to512(x),dev))
def llr(nm,pil,iid):
    kind,g=SPEC[nm]; p,M=F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm],g)(pil),p,M,kind=kind,n_codeword=n),-CLAMP,CLAMP).astype(np.float32)
def cv(rl,iid): return bool(decode_and_verify(rl,iid,codec=sb)["detected"])
def rot(img,deg):
    a=np.asarray(img); pad=256
    big=Image.fromarray(np.pad(a,((pad,pad),(pad,pad),(0,0)),"reflect")).rotate(deg,resample=Image.BICUBIC)
    l=(big.size[0]-512)//2; return big.crop((l,l,l+512,l+512))
def detect(att,iid,tx,frags,RANGE):
    al={nm:llr(nm,att,iid) for nm in frags}
    fused=fuse_llrs(al,weights=None,n_codeword=n) if len(al)>1 else next(iter(al.values()))
    if cv(fused,iid) or ((fused>0).astype(np.uint8)==tx).mean()>=tau: return 1.0
    if any(cv(al[nm],iid) for nm in frags): return 1.0
    rect,_=sync_rectify(sync,att,dev)
    if any(cv(llr(nm,rect,iid),iid) for nm in frags): return 1.0
    best_d,best_ba=0.0,-1.0
    for d in np.arange(-RANGE,RANGE+0.01,10.0):
        rl=llr("trustmark",rot(att,float(d)),iid)
        if cv(rl,iid): return 1.0
        ba=float(((rl>0).astype(np.uint8)==tx).mean())
        if ba>best_ba: best_ba,best_d=ba,d
    if best_ba<0.60: return 0.0
    for d in np.arange(best_d-10,best_d+10.01,3.0):
        r=rot(att,float(d))
        for nm in ("trustmark","videoseal"):
            if nm in frags and cv(llr(nm,r,iid),iid): return 1.0
    return 0.0
srcs=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:20]
FR=["vine","trustmark","videoseal"]; ANG=[45,60,75,120,150,180]
res={}
wm_cache=[]
for j,fp in enumerate(srcs):
    iid=f"wr_{j:04d}"; orig=to512(Image.open(fp).convert("RGB"))
    tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
    wm_cache.append((iid,embed(orig,iid,FR),tx))
for RANGE in (30,90,180):
    row={}
    for ang in ANG:
        d=[detect(rot(wm,ang),iid,tx,FR,RANGE) for iid,wm,tx in wm_cache]
        row[ang]=round(float(np.mean(d)),3)
    res[f"range{RANGE}"]=row
    print(f"blind range +-{RANGE:3d}deg : "+"  ".join(f"rot{a}={row[a]:.2f}" for a in ANG),flush=True)
json.dump(res,open("results/defense/wide_rot.json","w"),indent=2)
print("WIDE_ROT_DONE")
