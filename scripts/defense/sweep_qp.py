"""WAVES-style Q@0.95P / Q@0.7P for the deployed 3-frag+head pipeline under a CtrlRegen+
strength sweep. P = TPR@0.1%FPR (fused-ba score, 0.1%FPR threshold tau01=0.66, validated by
fpr_test.py: head null ba ~ Binom(100,.5)/100, P(ba>=.66)=8.95e-4). Q = WAVES quantile-
normalized degradation (10%q->0.1, 90%q->0.9) over {PSNR, SSIM, LPIPS} (8->3 fidelity core;
FID/CLIP-FID/aesthetics/artifacts omitted -> absolute Q not cross-comparable to WAVES tables,
but the Q@P operating points are faithful for THIS pipeline). Curve includes clean (s=0,
P~1, Q=0). Reports HEAD(deployed)/VINE/eq3/best-path; raw SSIM/PSNR/LPIPS at crossings too.
"""
import os, sys, glob, json
import numpy as np, torch
from PIL import Image
from scipy.stats import binom
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts")]: sys.path.insert(0,p)
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.fusion_head3 import load_head3
from skimage.metrics import structural_similarity as _ssim, peak_signal_noise_ratio as _psnr
import lpips as lpips_mod
KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15.0
sb=ShortenedBCH(); n=sb.n
TAU01=0.66                         # 0.1%FPR operating point (validated by fpr_test.py)
EMB="results/defense/ext_vtv100"
STR=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
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
def decode_point(adir, want_quality):
    """Return per-image fused/vine/eq3 ba, crypto-verify flags, and (ssim,psnr,lpips) vs clean-embed."""
    A,B,C,T,I,Q=[],[],[],[],[],[]
    for it in meta["items"]:
        ref=os.path.join(REPO,EMB,f"img_{it['i']:05d}.png")
        fp=os.path.join(REPO,adir,f"img_{it['i']:05d}.png") if adir!=EMB else ref
        if not os.path.exists(fp): continue
        iid=it["image_id"]; att=to512(Image.open(fp).convert("RGB")); tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
        pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
        A.append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(att),pv,Mv,kind="prob", n_codeword=n),-CLAMP,CLAMP).astype(np.float32))
        B.append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(att),pt,Mt,kind="logit",n_codeword=n),-CLAMP,CLAMP).astype(np.float32))
        C.append(np.clip(method_soft_to_codeword_llr(vsf.raw_logits(att),ps,Ms,kind="logit",n_codeword=n),-CLAMP,CLAMP).astype(np.float32))
        T.append(tx.astype(np.uint8)); I.append(iid)
        if want_quality:
            ref_im=to512(Image.open(ref).convert("RGB")); a=np.asarray(att); b=np.asarray(ref_im)
            Q.append((float(_ssim(a,b,channel_axis=2)), float(_psnr(b,a,data_range=255)), lpips_d(att,ref_im)))
    A,B,C,T=np.array(A),np.array(B),np.array(C),np.array(T)
    with torch.no_grad():
        H=head(torch.tensor(A,device=dev),torch.tensor(B,device=dev),torch.tensor(C,device=dev)).cpu().numpy()
    def ba(F): return ((F>0).astype(np.uint8)==T).mean(1)
    eq=A+B+C
    out={"vine_ba":ba(A),"eq3_ba":ba(eq),"head_ba":ba(H),
         "vine_crypto":np.array([decode_and_verify(A[i],I[i],codec=sb)["detected"] for i in range(len(A))],float),
         "head_crypto":np.array([decode_and_verify(H[i],I[i],codec=sb)["detected"] for i in range(len(H))],float)}
    if want_quality: out["quality"]=np.array(Q)  # [N,3] ssim,psnr,lpips
    return out

# ---- collect curve: clean (s=0) + 9 strengths ----
pts=[("0.0",EMB)]+[(f"{s:.1f}",f"results/defense/ext_vtv100_cr_s{str(s).replace('.','')}") for s in STR]
data={}
for tag,adir in pts:
    if adir!=EMB and not os.path.isdir(os.path.join(REPO,adir)): print(f"  skip {tag} (no dir)"); continue
    data[tag]=decode_point(adir, want_quality=True)
    print(f"  decoded s={tag} n={len(data[tag]['head_ba'])}",flush=True)

# ---- WAVES quantile-normalized Q over pooled ATTACKED images (exclude clean) ----
deg_pool={"ssim":[],"psnr":[],"lpips":[]}
for tag in data:
    if tag=="0.0": continue
    q=data[tag]["quality"]; deg_pool["ssim"]+=list(1-q[:,0]); deg_pool["psnr"]+=list(-q[:,1]); deg_pool["lpips"]+=list(q[:,2])
qn={}
for m,v in deg_pool.items():
    v=np.array(v); q10,q90=np.quantile(v,0.10),np.quantile(v,0.90); qn[m]=(q10,q90)
def waves_q(quality):  # quality [N,3] ssim,psnr,lpips -> mean normalized degradation per image
    deg={"ssim":1-quality[:,0],"psnr":-quality[:,1],"lpips":quality[:,2]}
    norms=[]
    for m in ["ssim","psnr","lpips"]:
        q10,q90=qn[m]; norms.append(np.clip(0.1+0.8*(deg[m]-q10)/(q90-q10+1e-9),0,1))
    return np.mean(norms,0)  # [N]

# ---- per-strength P (TPR@0.1%FPR) and Q for each detector ----
def P_of(tag,score):  # TPR@0.1%FPR using ba score
    return float(np.mean(data[tag][score]>=TAU01))
rows=[]
for tag in [p[0] for p in pts if p[0] in data]:
    if tag=="0.0": Q=0.0; rawq=(1.0,99.0,0.0)
    else:
        qv=data[tag]["quality"]; Q=float(np.mean(waves_q(qv))); rawq=(float(qv[:,0].mean()),float(qv[:,1].mean()),float(qv[:,2].mean()))
    rows.append({"s":float(tag),"Q":Q,"ssim":rawq[0],"psnr":rawq[1],"lpips":rawq[2],
                 "P_head":P_of(tag,"head_ba"),"P_vine":P_of(tag,"vine_ba"),"P_eq3":P_of(tag,"eq3_ba"),
                 "P_bestpath":float(np.mean(np.maximum(data[tag]["head_ba"]>=TAU01,data[tag]["vine_ba"]>=TAU01)))})
rows=sorted(rows,key=lambda r:r["s"])

def q_at_P(rows,pkey,target):
    """Q (and raw quality) at strength where P crosses target (P decreasing in s). inf/-inf edges."""
    Ps=[r[pkey] for r in rows]
    if all(p>target for p in Ps): return ("inf",None)
    if all(p<target for p in Ps): return ("-inf",None)
    for i in range(len(rows)-1):
        p0,p1=rows[i][pkey],rows[i+1][pkey]
        if p0>=target>=p1 and p0!=p1:
            f=(p0-target)/(p0-p1)
            interp=lambda k: rows[i][k]+f*(rows[i+1][k]-rows[i][k])
            return (interp("Q"),{"s":interp("s"),"ssim":interp("ssim"),"psnr":interp("psnr"),"lpips":interp("lpips")})
    return ("-inf",None)

print("\n=== CtrlRegen+ sweep curve (n=%d) ==="%len(data[list(data)[0]]["head_ba"]))
print(f"{'s':>4} {'Q':>5} {'SSIM':>5} {'PSNR':>5} {'LPIPS':>5} | {'P_head':>6} {'P_vine':>6} {'P_eq3':>6} {'P_best':>6}")
for r in rows:
    print(f"{r['s']:>4} {r['Q']:>5.2f} {r['ssim']:>5.2f} {r['psnr']:>5.1f} {r['lpips']:>5.2f} | {r['P_head']:>6.2f} {r['P_vine']:>6.2f} {r['P_eq3']:>6.2f} {r['P_bestpath']:>6.2f}")
print("\n=== WAVES Q@P (deployed detector = HEAD; also VINE/eq3/best-path) ===")
res={}
for pkey,nm in [("P_head","HEAD"),("P_vine","VINE"),("P_eq3","eq3"),("P_bestpath","best-path")]:
    avgP=float(np.mean([r[pkey] for r in rows])); avgQ=float(np.mean([r["Q"] for r in rows]))
    q95,c95=q_at_P(rows,pkey,0.95); q70,c70=q_at_P(rows,pkey,0.70)
    f=lambda x:(x if isinstance(x,str) else f"{x:.3f}")
    cs=lambda c:(f"(s={c['s']:.2f} SSIM={c['ssim']:.2f} PSNR={c['psnr']:.1f} LPIPS={c['lpips']:.2f})" if c else "")
    print(f"  {nm:9s}: Q@0.95P={f(q95):>6} {cs(c95)}")
    print(f"  {'':9s}  Q@0.70P={f(q70):>6} {cs(c70)}   AvgP={avgP:.3f} AvgQ={avgQ:.3f}")
    res[nm]={"Q@0.95P":q95,"Q@0.70P":q70,"AvgP":avgP,"AvgQ":avgQ,"cross95":c95,"cross70":c70}
json.dump({"rows":rows,"qn_quantiles":{k:list(map(float,v)) for k,v in qn.items()},"tau01":TAU01,"results":res},
          open(os.path.join(REPO,"results/defense/sweep_qp.json"),"w"),indent=2)
print("\nSWEEP_QP_DONE")
