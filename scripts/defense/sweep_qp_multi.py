"""Multi-attack WAVES Performance-Quality (Q@P) sweep for the deployed 3-frag+head pipeline.
Beyond CtrlRegen+: jpeg / blur / noise / brightness / contrast / crop-zoom / rotate, each over
a strength grid, plus the existing CtrlRegen+ attacked dirs. P = TPR@0.1%FPR (fused-ba>=0.66).
Q = WAVES quantile-normalized degradation over {PSNR,SSIM,LPIPS}, pooled over ALL attacked
images across ALL attacks+strengths -> Q is comparable across attacks. Reports Q@0.95P /
Q@0.70P / AvgP per attack (inf if the attack never pushes P below the threshold)."""
import os, sys, glob, json, io
import numpy as np, torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import binom
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts")]: sys.path.insert(0,p)
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.fusion_head3 import load_head3
from skimage.metrics import structural_similarity as _ssim, peak_signal_noise_ratio as _psnr
import lpips as lpips_mod
KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15.0
sb=ShortenedBCH(); n=sb.n; TAU01=0.66
EMB="results/defense/ext_vtv100"
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=n,device=dev)
head=load_head3(os.path.join(REPO,"results/defense/frag3_head.pt"),dev)
lp=lpips_mod.LPIPS(net='alex').to(dev).eval()
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def lpips_d(a,b):
    ta=torch.tensor(np.asarray(a,np.float32)/127.5-1).permute(2,0,1)[None].to(dev)
    tb=torch.tensor(np.asarray(b,np.float32)/127.5-1).permute(2,0,1)[None].to(dev)
    with torch.no_grad(): return float(lp(ta,tb).item())
meta=json.load(open(os.path.join(REPO,EMB,"meta.json")))
items=meta["items"]
def fused_ba(att,iid,tx):
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
    av=np.clip(method_soft_to_codeword_llr(vine.raw_probs(att),pv,Mv,kind="prob", n_codeword=n),-CLAMP,CLAMP).astype(np.float32)
    at=np.clip(method_soft_to_codeword_llr(tm.raw_logits(att),pt,Mt,kind="logit",n_codeword=n),-CLAMP,CLAMP).astype(np.float32)
    as_=np.clip(method_soft_to_codeword_llr(vsf.raw_logits(att),ps,Ms,kind="logit",n_codeword=n),-CLAMP,CLAMP).astype(np.float32)
    with torch.no_grad():
        H=head(torch.tensor(av[None],device=dev),torch.tensor(at[None],device=dev),torch.tensor(as_[None],device=dev)).cpu().numpy()[0]
    return float(((H>0).astype(np.uint8)==tx).mean())

# ---- attack functions (strength increasing) ----
def a_jpeg(im,q): b=io.BytesIO(); im.save(b,"JPEG",quality=int(q)); b.seek(0); return Image.open(b).convert("RGB")
def a_blur(im,s): return im.filter(ImageFilter.GaussianBlur(float(s)))
def a_noise(im,s): x=np.asarray(im,np.float64)+np.random.RandomState(0).normal(0,float(s)*255,(512,512,3)); return Image.fromarray(np.clip(x,0,255).astype(np.uint8))
def a_bright(im,f): return ImageEnhance.Brightness(im).enhance(float(f))
def a_contrast(im,f): return ImageEnhance.Contrast(im).enhance(float(f))
def a_cropzoom(im,area): s=int(round(512*np.sqrt(area))); o=(512-s)//2; return im.crop((o,o,o+s,o+s)).resize((512,512))
def a_rotate(im,a): return im.rotate(float(a),resample=Image.BILINEAR)
ATT={
 "jpeg":(a_jpeg,[90,70,50,30,15,8]),
 "blur":(a_blur,[0.5,1.5,3,5,7]),
 "noise":(a_noise,[0.02,0.06,0.1,0.15,0.2]),
 "bright":(a_bright,[1.2,1.5,1.9,2.3,2.8]),
 "contrast":(a_contrast,[1.3,1.6,2.0,2.4,2.8]),
 "crop_zoom":(a_cropzoom,[0.9,0.75,0.6,0.45,0.3]),
 "rotate":(a_rotate,[3,9,20,35,60,90]),
}
# severity x in [0,1] per attack (for ordering/x-axis): normalize index
def sev(name,i,k): return (i+1)/k

records=[]  # each: {attack, sev, fused_ba list, quality list}
def ref_of(it): return to512(Image.open(os.path.join(REPO,EMB,f"img_{it['i']:05d}.png")).convert("RGB"))

for name,(fn,grid) in ATT.items():
    for gi,g in enumerate(grid):
        bas,qs=[],[]
        for it in items:
            iid=it["image_id"]; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
            ref=ref_of(it); att=to512(fn(ref,g))
            bas.append(fused_ba(att,iid,tx))
            a,b=np.asarray(att),np.asarray(ref); qs.append((float(_ssim(a,b,channel_axis=2)),float(_psnr(b,a,data_range=255)),lpips_d(att,ref)))
        records.append({"attack":name,"param":g,"sev":sev(name,gi,len(grid)),"ba":bas,"q":qs})
    print(f"  done {name}",flush=True)

# ---- CtrlRegen+ from existing attacked dirs (n=100) ----
for s in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    adir=f"results/defense/ext_vtv100_cr_s{str(s).replace('.','')}"
    if not os.path.isdir(os.path.join(REPO,adir)): continue
    bas,qs=[],[]
    for it in items:
        fp=os.path.join(REPO,adir,f"img_{it['i']:05d}.png")
        if not os.path.exists(fp): continue
        iid=it["image_id"]; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
        ref=ref_of(it); att=to512(Image.open(fp).convert("RGB"))
        bas.append(fused_ba(att,iid,tx)); a,b=np.asarray(att),np.asarray(ref)
        qs.append((float(_ssim(a,b,channel_axis=2)),float(_psnr(b,a,data_range=255)),lpips_d(att,ref)))
    records.append({"attack":"ctrlregen","param":s,"sev":s,"ba":bas,"q":qs})
print("  done ctrlregen",flush=True)

# ---- WAVES quantile-norm Q over ALL attacked images pooled ----
deg={"ssim":[],"psnr":[],"lpips":[]}
for r in records:
    for (ss,ps,lp_) in r["q"]: deg["ssim"].append(1-ss); deg["psnr"].append(-ps); deg["lpips"].append(lp_)
qn={m:(float(np.quantile(deg[m],0.10)),float(np.quantile(deg[m],0.90))) for m in deg}
def Qnorm(qtuple):
    ss,ps,lp_=qtuple; d={"ssim":1-ss,"psnr":-ps,"lpips":lp_}
    return float(np.mean([np.clip(0.1+0.8*(d[m]-qn[m][0])/(qn[m][1]-qn[m][0]+1e-9),0,1) for m in d]))
for r in records:
    r["P"]=float(np.mean(np.array(r["ba"])>=TAU01))
    r["Q"]=float(np.mean([Qnorm(q) for q in r["q"]]))
    r["ssim"]=float(np.mean([q[0] for q in r["q"]])); r["psnr"]=float(np.mean([min(q[1],60) for q in r["q"]])); r["lpips"]=float(np.mean([q[2] for q in r["q"]]))

# ---- per-attack curve + Q@P ----
def q_at_P(curve,T):  # curve: list of (Q,P) sorted by severity; P decreasing
    Ps=[p for _,p in curve]
    if all(p>T for p in Ps): return "inf"
    if all(p<T for p in Ps): return "-inf"
    # anchor clean at (0,1.0)
    pts=[(0.0,1.0)]+curve
    for i in range(len(pts)-1):
        (q0,p0),(q1,p1)=pts[i],pts[i+1]
        if p0>=T>=p1 and p0!=p1:
            f=(p0-T)/(p0-p1); return float(q0+f*(q1-q0))
    return "-inf"
attacks=["jpeg","blur","noise","bright","contrast","rotate","crop_zoom","ctrlregen"]
out={"tau01":TAU01,"qn":qn,"per_attack":{}}
print("\n%-10s %8s %8s %8s %8s"%("attack","Q@0.95P","Q@0.70P","AvgP","AvgQ"))
for a in attacks:
    rs=sorted([r for r in records if r["attack"]==a],key=lambda r:r["sev"])
    curve=[(r["Q"],r["P"]) for r in rs]
    q95=q_at_P(curve,0.95); q70=q_at_P(curve,0.70)
    avgP=float(np.mean([r["P"] for r in rs])); avgQ=float(np.mean([r["Q"] for r in rs]))
    out["per_attack"][a]={"Q@0.95P":q95,"Q@0.70P":q70,"AvgP":avgP,"AvgQ":avgQ,
                          "curve":[{"param":r["param"],"Q":r["Q"],"P":r["P"],"ssim":r["ssim"],"psnr":r["psnr"],"lpips":r["lpips"]} for r in rs]}
    f=lambda x:(x if isinstance(x,str) else "%.3f"%x)
    print("%-10s %8s %8s %8.3f %8.3f"%(a,f(q95),f(q70),avgP,avgQ))
json.dump(out,open(os.path.join(REPO,"results/defense/sweep_qp_multi.json"),"w"),indent=2)
print("\nSWEEP_QP_MULTI_DONE")
