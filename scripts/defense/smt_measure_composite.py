"""Measure per-fragment bit-acc INSIDE the deployed 3-frag composite (VideoSeal@.7 -> TM@.7 -> VINE@1.0),
so the SMT solver uses REALISTIC (composite, post-interference) numbers instead of optimistic solo ones.
Also measures composite PSNR for 1-/2-/3-fragment stacks (replaces the calibrated stacking model).
Reliable attacks measured fresh; regen/rinse taken from hidden_big composite (VINE 0.85) to avoid the flaky regen proxy.
Output -> results/defense/smt_inputs_composite.json"""
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
KEY=b"v5_key_encoder_master"; dev="cuda"; RES=512
def to(im): return im.resize((RES,RES)) if im.size!=(RES,RES) else im
def sr(c,w,a):
    c=np.asarray(to(c),np.float64); w=np.asarray(to(w),np.float64); return Image.fromarray(np.clip(c+a*(w-c),0,255).astype(np.uint8))
def p2l(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
def align1d(raw,perm,M): return np.asarray(M)[perm]*np.asarray(raw)[perm]   # de-whiten to codeword space
def psnr(a,b):
    mse=np.mean((np.asarray(a,np.float64)-np.asarray(b,np.float64))**2); return 99.0 if mse<1e-9 else 10*np.log10(255**2/mse)
# attacks (reliable; on 512 composite)
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
N=int(sys.argv[1]) if len(sys.argv)>1 else 24
OFF=int(sys.argv[2]) if len(sys.argv)>2 else 0     # held-out offset (disjoint image set for validation)
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[OFF:OFF+N]
accV={a:[] for a in ATTS}; accT={a:[] for a in ATTS}; accS={a:[] for a in ATTS}
ps1=[]; ps2=[]; ps3=[]
for i,fp in enumerate(files):
    iid=f"c_{i:05d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    orig=to(Image.open(fp).convert("RGB"))
    pv_p,Mv=vine.get_perm_M(iid); pt_p,Mt=tm.get_perm_M(iid); ps_p,Ms=vs.get_perm_M(iid)
    a1=sr(orig, to(vs.embed_with_target(orig, apply_crypto(tx,ps_p,Ms))), 0.70)
    a2=sr(a1,   to(tm.embed_with_target(a1,  apply_crypto(tx,pt_p,Mt))), 0.70)
    comp=sr(a2, to(vine.embed_with_target(a2, apply_crypto(tx,pv_p,Mv))), 1.00)
    # PSNR anchors: VINE-only, VINE+TM, full-3
    vonly=sr(orig, to(vine.embed_with_target(orig, apply_crypto(tx,pv_p,Mv))), 1.00)
    vtm=sr(sr(orig, to(tm.embed_with_target(orig, apply_crypto(tx,pt_p,Mt))),0.70), to(vine.embed_with_target(sr(orig,to(tm.embed_with_target(orig,apply_crypto(tx,pt_p,Mt))),0.70), apply_crypto(tx,pv_p,Mv))),1.00)
    ps1.append(psnr(orig,vonly)); ps2.append(psnr(orig,vtm)); ps3.append(psnr(orig,comp))
    for a in ATTS:
        x=att(a,comp,9000+i)
        FV=align1d(p2l(vine.raw_probs(x)),pv_p,Mv); accV[a].append(float(((FV>0).astype(int)==tx).mean()))
        FT=align1d(np.clip(tm.raw_logits(x),-15,15),pt_p,Mt); accT[a].append(float(((FT>0).astype(int)==tx).mean()))
        AS=method_soft_to_codeword_llr(vs.raw_logits(x),ps_p,Ms,kind="logit",n_codeword=sb.n); accS[a].append(float(((AS>0).astype(int)==tx).mean()))
    if (i+1)%8==0: print(f"  [{i+1}/{N}]",flush=True)
def avg(d): return {a:round(float(np.mean(d[a])),3) for a in ATTS}
# regen/rinse from hidden_big composite (VINE 0.85 measured; TM/VS dead); rinse VINE ~0.73
REGEN_COMP={"VINE":{"regen":0.853,"rinse":0.73},"TrustMark":{"regen":0.50,"rinse":0.50},"VideoSeal":{"regen":0.50,"rinse":0.50}}
OUT={"n":N,"note":"per-fragment bit-acc INSIDE 3-frag composite (VS@.7->TM@.7->VINE@1.0); regen/rinse from hidden_big",
     "fragments":{
       "VINE":{"bit_acc":{**avg(accV),**REGEN_COMP["VINE"]}},
       "TrustMark":{"bit_acc":{**avg(accT),**REGEN_COMP["TrustMark"]}},
       "VideoSeal":{"bit_acc":{**avg(accS),**REGEN_COMP["VideoSeal"]}}},
     "psnr_stack":{"1frag":round(float(np.mean(ps1)),2),"2frag":round(float(np.mean(ps2)),2),"3frag":round(float(np.mean(ps3)),2)}}
OUTF=CF+"/results/defense/smt_inputs_composite.json" if OFF==0 else CF+f"/results/defense/smt_inputs_composite_off{OFF}.json"
json.dump(OUT,open(OUTF,"w"),indent=1)
print("\ncomposite bit-acc (VINE / TM / VideoSeal):")
for a in ATTS: print(f"  {a:9s}  {avg(accV)[a]:.3f}  {avg(accT)[a]:.3f}  {avg(accS)[a]:.3f}")
print(f"PSNR stack: 1frag {OUT['psnr_stack']['1frag']}  2frag {OUT['psnr_stack']['2frag']}  3frag {OUT['psnr_stack']['3frag']}")
print("SAVED results/defense/smt_inputs_composite.json\nCOMP_DONE")
