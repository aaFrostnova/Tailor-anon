"""RE-RUN the attack suite with the FIXED fine scale search (the old 0.03 grid missed the layers).
n=50 nested-VINE composite. Crop staircase + off-center + re-decode of existing UnMarker/CtrlRegen+ sets."""
import os,sys,glob,json,argparse,itertools,numpy as np
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
SC="/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks"
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=50)
ap.add_argument("--vine_scale_step",type=float,default=0.005)   # FIXED grid (was 0.03 -> missed layers)
a=ap.parse_args()
SPEC={"vine":("prob","raw_probs"),"trustmark":("logit","raw_logits"),"videoseal":("logit","raw_logits")}
F={"vine":VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device=dev),
   "trustmark":TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=n,model_type="B",device=dev),
   "videoseal":VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=n,device=dev)}
FR=["vine","trustmark","videoseal"]; sync=load_sync(dev=dev)
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
def embed(orig,iid):
    x=to512(orig)
    for nm in FR: x=vine_nested(x,iid) if nm=="vine" else scl(x,to512(F[nm].embed_with_target(x,txc(iid,nm))))
    return to512(sync_embed(sync,to512(x),dev))
def llr(nm,pil,iid):
    kind,g=SPEC[nm]; p,M=F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm],g)(pil),p,M,kind=kind,n_codeword=n),-CLAMP,CLAMP).astype(np.float32)
def cv(rl,iid): return bool(decode_and_verify(rl,iid,codec=sb)["detected"])
def rot(img,deg):
    ar=np.asarray(img); pad=256
    big=Image.fromarray(np.pad(ar,((pad,pad),(pad,pad),(0,0)),"reflect")).rotate(deg,resample=Image.BICUBIC)
    l=(big.size[0]-512)//2; return big.crop((l,l,l+512,l+512))
VS=np.arange(0.34,1.0001,a.vine_scale_step)
def vine_scale_search(pil,iid,tx):
    P=to512(pil)
    for f in VS:
        if f>=0.999: view=P
        else:
            s=int(round(512*float(f))); o=(512-s)//2; view=P.crop((o,o,o+s,o+s))
        rl=llr("vine",view,iid)
        if cv(rl,iid) or ((rl>0).astype(np.uint8)==tx).mean()>=tau: return True
    return False
def detect(att,iid,tx):
    al={nm:llr(nm,att,iid) for nm in FR}
    fused=fuse_llrs(al,weights=None,n_codeword=n)
    if cv(fused,iid) or ((fused>0).astype(np.uint8)==tx).mean()>=tau: return 1.0
    if any(cv(al[nm],iid) for nm in FR): return 1.0
    if vine_scale_search(att,iid,tx): return 1.0
    rect,_=sync_rectify(sync,att,dev)
    if any(cv(llr(nm,rect,iid),iid) for nm in FR): return 1.0
    best_d,best_ba=0.0,-1.0
    for d in np.arange(-180,180.01,10.0):                      # full-circle (the other fixed default)
        rl=llr("trustmark",rot(att,float(d)),iid)
        if cv(rl,iid): return 1.0
        ba=float(((rl>0).astype(np.uint8)==tx).mean())
        if ba>best_ba: best_ba,best_d=ba,d
    if best_ba<0.60: return 0.0
    for d in np.arange(best_d-10,best_d+10.01,3.0):
        r=rot(att,float(d))
        for nm in ("trustmark","videoseal"):
            if cv(llr(nm,r,iid),iid): return 1.0
    return 0.0
srcs=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:a.n]
wms=[]
for j,fp in enumerate(srcs):
    iid=f"rf_{j:04d}"; orig=to512(Image.open(fp).convert("RGB"))
    tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
    wms.append((iid,embed(orig,iid),tx))
    if (j+1)%10==0: print(f"embed {j+1}/{len(srcs)}",flush=True)
def a_crop(x,c,frac=0.5):
    s=int(512*c); mx=512-s; o=int(round(mx*frac))
    return to512(x).crop((o,o,o+s,o+s)).resize((512,512))
res={}
CROPS=[0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4]
res["crop_staircase"]={c:round(float(np.mean([detect(a_crop(wm,c),iid,tx) for iid,wm,tx in wms])),3) for c in CROPS}
print("crop staircase:",res["crop_staircase"],flush=True)
res["offcenter"]={}
for c in (0.9,0.75,0.6):
    for fr in (0.0,0.5,1.0):
        k=f"c{int(c*100)}_off{fr}"
        res["offcenter"][k]=round(float(np.mean([detect(a_crop(wm,c,fr),iid,tx) for iid,wm,tx in wms])),3)
print("off-center:",res["offcenter"],flush=True)
# re-decode existing advanced-attack sets with the fixed search
def redecode(embed_dir,att_dir):
    meta=json.load(open(f"{embed_dir}/meta.json"))["items"]; d=[]
    for m in meta:
        p=f"{att_dir}/{m['fname']}"
        if not os.path.exists(p): continue
        iid=m["iid"]; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
        d.append(detect(to512(Image.open(p).convert("RGB")),iid,tx))
    return round(float(np.mean(d)),3) if d else None, len(d)
res["advanced_refixed"]={}
for name,ed,ad in [("UnMarker_default",f"{SC}/p3embed_Dum",f"{SC}/p3um_D"),
                   ("UnMarker_strong",f"{SC}/p3embed_Dum",f"{SC}/p3um_strong"),
                   ("CtrlRegen_s05",f"{SC}/p3embed_D",f"{SC}/p3cr_D_s05"),
                   ("CtrlRegen_s09",f"{SC}/p3embed_D",f"{SC}/p3cr_D_s09")]:
    if os.path.isdir(ad):
        v,cnt=redecode(ed,ad); res["advanced_refixed"][name]={"detect":v,"n":cnt}
        print(f"  {name}: {v} (n={cnt})",flush=True)
json.dump({"vine_scale_step":a.vine_scale_step,"n":len(wms),**res},open("results/defense/rerun_fixed_search.json","w"),indent=2)
print("RERUN_FIXED_DONE")
