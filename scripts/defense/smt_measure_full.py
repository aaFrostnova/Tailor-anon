"""TRUE end-to-end fused-detector measurement + n>=100 for the SMT solver inputs.
Embeds the deployed 3-frag composite, and per attack computes (N images):
  - per-fragment composite bit-acc  -> best_single = max over fragments (what the solver currently models)
  - FUSED bit-acc (equal-MRC of the 3 aligned codeword LLRs)  -> the real deployed decode
  - detection rate = fraction with (fused zero-bit ba>=tau=0.63) OR crypto-verify (hard-BCH id match)
So we can (a) confirm fused >= best_single (tightness), (b) stabilize near-threshold points at n>=100.
Output -> results/defense/smt_inputs_full.json"""
import os,sys,io,glob,json,numpy as np
from PIL import Image, ImageEnhance, ImageFilter
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,CF); sys.path.insert(0,CF+"/scripts/defense"); sys.path.insert(0,CF+"/external/videoseal")
import torch
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
KEY=b"v5_key_encoder_master"; dev="cuda"; RES=512; TAU=0.63
def to(im): return im.resize((RES,RES)) if im.size!=(RES,RES) else im
def sr(c,w,a):
    c=np.asarray(to(c),np.float64); w=np.asarray(to(w),np.float64); return Image.fromarray(np.clip(c+a*(w-c),0,255).astype(np.uint8))
def p2l(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
def align1d(raw,perm,M): return np.asarray(M)[perm]*np.asarray(raw)[perm]
from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor
VAEB=bmshj2018_hyperprior(quality=3,pretrained=True).eval().to(dev); VAEC=cheng2020_anchor(quality=3,pretrained=True).eval().to(dev)
def vae(im,m):
    t=torch.from_numpy(np.asarray(to(im),np.float32)/255).permute(2,0,1)[None].to(dev)
    with torch.no_grad(): o=m(t)["x_hat"].clamp(0,1)
    return Image.fromarray((o[0].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
def crop(im,r): s=int(RES*r); o=(RES-s)//2; return im.crop((o,o,o+s,o+s)).resize((RES,RES))
def att(name,im,k):
    if name=="clean": return im
    if name=="jpeg50": b=io.BytesIO(); im.save(b,"JPEG",quality=50); b.seek(0); return Image.open(b).convert("RGB")
    if name=="jpeg25": b=io.BytesIO(); im.save(b,"JPEG",quality=25); b.seek(0); return Image.open(b).convert("RGB")
    if name=="blur": return im.filter(ImageFilter.GaussianBlur(2))
    if name=="noise": x=np.asarray(im,np.float32)/255+np.random.RandomState(k).randn(RES,RES,3).astype(np.float32)*0.05; return Image.fromarray((np.clip(x,0,1)*255).astype('uint8'))
    if name=="bright": return ImageEnhance.Brightness(im).enhance(0.5)
    if name=="contrast": return ImageEnhance.Contrast(im).enhance(0.5)
    if name=="vaeB": return vae(im,VAEB)
    if name=="vaeC": return vae(im,VAEC)
    if name=="crop90": return crop(im,0.9)
    if name=="crop75": return crop(im,0.75)
    if name=="rot9": return im.rotate(9,resample=Image.BICUBIC)
    if name=="rot30": return im.rotate(30,resample=Image.BICUBIC)
ATTS=["clean","jpeg50","jpeg25","blur","noise","bright","contrast","vaeB","vaeC","crop90","crop75","rot9","rot30"]
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vs=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
def crypto_ok(hard, payload):
    try: return int(np.array_equal(sb.decode(hard.astype(np.uint8))[:sb.data_bits], payload[:sb.data_bits]))
    except Exception: return 0
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
OFF=int(sys.argv[2]) if len(sys.argv)>2 else 1000     # disjoint from fit(0-24)/heldout(500-515)
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[OFF:OFF+N]
R={a:{"vine":[],"tm":[],"vs":[],"best":[],"fused":[],"det":[]} for a in ATTS}
for i,fp in enumerate(files):
    iid=f"f_{i:05d}"; pay=image_id_to_payload(iid,n_bits=sb.data_bits); tx=sb.encode(pay)
    orig=to(Image.open(fp).convert("RGB"))
    pv_p,Mv=vine.get_perm_M(iid); pt_p,Mt=tm.get_perm_M(iid); ps_p,Ms=vs.get_perm_M(iid)
    a1=sr(orig, to(vs.embed_with_target(orig, apply_crypto(tx,ps_p,Ms))), 0.70)
    a2=sr(a1,   to(tm.embed_with_target(a1,  apply_crypto(tx,pt_p,Mt))), 0.70)
    comp=sr(a2, to(vine.embed_with_target(a2, apply_crypto(tx,pv_p,Mv))), 1.00)
    for a in ATTS:
        x=att(a,comp,9000+i)
        LV=align1d(np.clip(p2l(vine.raw_probs(x)),-15,15),pv_p,Mv)
        LT=align1d(np.clip(tm.raw_logits(x),-15,15),pt_p,Mt)
        LS=method_soft_to_codeword_llr(vs.raw_logits(x),ps_p,Ms,kind="logit",n_codeword=sb.n)
        bv=((LV>0).astype(int)==tx).mean(); bt=((LT>0).astype(int)==tx).mean(); bs=((LS>0).astype(int)==tx).mean()
        LF=np.clip(LV,-15,15)+np.clip(LT,-15,15)+np.clip(LS,-15,15)            # equal-MRC fusion
        bf=((LF>0).astype(int)==tx).mean()
        # detection: zero-bit(fused ba>=tau) OR crypto-verify on best-path (fused, or any single fragment)
        det=int(bf>=TAU or crypto_ok((LF>0).astype(int),pay) or crypto_ok((LV>0).astype(int),pay))
        R[a]["vine"].append(bv);R[a]["tm"].append(bt);R[a]["vs"].append(bs)
        R[a]["best"].append(max(bv,bt,bs));R[a]["fused"].append(bf);R[a]["det"].append(det)
    if (i+1)%20==0: print(f"  [{i+1}/{N}]",flush=True)
def m(x): return round(float(np.mean(x)),3)
OUT={"n":N,"tau":TAU,"per_attack":{a:{k:m(R[a][k]) for k in R[a]} for a in ATTS}}
json.dump(OUT,open(CF+"/results/defense/smt_inputs_full.json","w"),indent=1)
print(f"\n{'attack':9s} {'VINE':>6s} {'TM':>6s} {'VS':>6s} | {'best':>6s} {'fused':>6s} {'Δf-b':>6s} | {'detect':>7s}")
for a in ATTS:
    b=m(R[a]["best"]);f=m(R[a]["fused"])
    print(f"{a:9s} {m(R[a]['vine']):6.3f} {m(R[a]['tm']):6.3f} {m(R[a]['vs']):6.3f} | {b:6.3f} {f:6.3f} {f-b:+6.3f} | {m(R[a]['det']):7.3f}")
print("SAVED results/defense/smt_inputs_full.json\nFULL_DONE")
