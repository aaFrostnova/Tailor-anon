"""Train a 3-WAY context-gated calibrated MRC fusion head over VINE+TM+VideoSeal LLRs.
Goal: learn to down-weight whichever fragments are dead (TM+VideoSeal under regen -> trust
VINE, no dilution; VINE under crop/rotate -> trust TM/VideoSeal) while preserving fragment
LLR scale so the fused codeword still Chase-BCH crypto-verifies. Output is a codeword LLR ->
crypto-ID 2^-37 FPR preserved by construction.

Eval: held-out mild attacks AND the real CtrlRegen+ attacked images (ext_vtv_ctrlregen_0X),
comparing equal-MRC-3 / learned-head-3 / VINE-only / best-of-paths oracle. FPR on unwm.
"""
import os, sys, glob, json
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.stats import binom
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts"), os.path.join(REPO,"scripts/defense")]:
    sys.path.insert(0,p)
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
torch.manual_seed(0); np.random.seed(0)
dev="cuda"; CLAMP=15.0
d=np.load(os.path.join(REPO,"results/defense/frag3_head_data.npz"),allow_pickle=True)
AV=np.clip(d["AV"],-CLAMP,CLAMP); AT=np.clip(d["AT"],-CLAMP,CLAMP); AS=np.clip(d["AS"],-CLAMP,CLAMP)
TX,ATK,IMG=d["TX"],d["ATK"],d["IMG"]; sb=ShortenedBCH(); n=int(d["n"]); tau=float(binom.ppf(0.99,n,0.5)+1)/n
attacks=list(d["attacks"]); cut=int(np.quantile(np.unique(IMG),0.7)); tr=IMG<cut; te=IMG>=cut
print(f"n={n} tau={tau:.3f} train={tr.sum()} test={te.sum()}",flush=True)

class Head3(nn.Module):
    def __init__(s,hid=24):
        super().__init__()
        s.cal=nn.ModuleList([nn.Sequential(nn.Linear(1,hid),nn.ReLU(),nn.Linear(hid,1)) for _ in range(3)])
        s.gate=nn.Sequential(nn.Linear(15,hid),nn.ReLU(),nn.Linear(hid,3))
    @staticmethod
    def feats(a):
        aa=a.abs(); return torch.stack([aa.mean(1),aa.std(1),(aa<0.5).float().mean(1),torch.tanh(aa).mean(1)],1)
    def ctx(s,av,at,as_):
        agr=lambda x,y:(torch.sign(x)==torch.sign(y)).float().mean(1,keepdim=True)
        return torch.cat([s.feats(av),s.feats(at),s.feats(as_),agr(av,at),agr(av,as_),agr(at,as_)],1)
    def forward(s,av,at,as_):
        B,m=av.shape; w=F.softplus(s.gate(s.ctx(av,at,as_)))
        cs=[s.cal[i](x.reshape(-1,1)).reshape(B,m) for i,x in enumerate([av,at,as_])]
        return w[:,0:1]*cs[0]+w[:,1:2]*cs[1]+w[:,2:3]*cs[2], w

head=Head3().to(dev)
xv=torch.tensor(AV[tr],device=dev);xt=torch.tensor(AT[tr],device=dev);xs=torch.tensor(AS[tr],device=dev)
y=torch.tensor(TX[tr].astype(np.float32),device=dev)
opt=torch.optim.Adam(head.parameters(),lr=2e-3,weight_decay=1e-5)
for ep in range(1500):
    opt.zero_grad(); f,_=head(xv,xt,xs); loss=F.binary_cross_entropy_with_logits(f,y); loss.backward(); opt.step()
    if (ep+1)%300==0: print(f"  ep{ep+1} loss={loss.item():.4f}",flush=True)
torch.save(head.state_dict(),os.path.join(REPO,"results/defense/frag3_head.pt")); head.eval()

e=1e-6
def norm(a): return a/(a.std(1,keepdims=True)+e)
def oracle(tx,*Fr):
    f=0.0
    for fr in Fr: b=((fr>0).astype(np.uint8)==tx).mean(1)[:,None]; f=f+np.clip(2*b-1,0,None)*norm(fr)
    return f
def hl(av,at,as_):
    with torch.no_grad():
        f,w=head(torch.tensor(av,device=dev),torch.tensor(at,device=dev),torch.tensor(as_,device=dev)); return f.cpu().numpy(),w.cpu().numpy()
def detvec(Fz,tx,ids):
    dv=np.zeros(len(Fz))
    for i in range(len(Fz)):
        ver=bool(decode_and_verify(Fz[i],ids[i],codec=sb)["detected"]); dv[i]=1.0 if (ver or ((Fz[i]>0).astype(np.uint8)==tx[i]).mean()>=tau) else 0.0
    return dv
def rate(Fz,tx,ids): dv=detvec(Fz,tx,ids); return float(dv.mean()),float(((Fz>0).astype(np.uint8)==tx).mean())

# ---- held-out mild split ----
av,at,as_,tx,atk,img=AV[te],AT[te],AS[te],TX[te],ATK[te],IMG[te]
ids=[f"f3_{int(i):05d}" for i in img]
Fhd,_=hl(av,at,as_)
print(f"\n[held-out mild] {'attack':9s} | {'eq3':>9s} | {'HEAD3':>9s} | {'oracle':>9s}")
for k in attacks:
    s=atk==k
    if s.sum()==0: continue
    def c(Fz): r=rate(Fz[s],tx[s],[ids[j] for j in np.where(s)[0]]); return f"{r[0]:.2f}/{r[1]:.2f}"
    print(f"  {k:9s} | {c(av+at+as_):>9s} | {c(Fhd):>9s} | {c(oracle(tx,av,at,as_)):>9s}")

# ---- real CtrlRegen+ attacked images (decode 3 frags, apply head) ----
print("\n[CtrlRegen+ real attacked imgs] decoding...",flush=True)
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
KEY=b"v5_key_encoder_master"
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tmf=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
meta=json.load(open(os.path.join(REPO,"results/defense/ext_vtv/meta.json")))
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def dec(dirn):
    A,B,C,T,I=[],[],[],[],[]
    for it in meta["items"]:
        fp=os.path.join(REPO,dirn,f"img_{it['i']:05d}.png")
        if not os.path.exists(fp): continue
        iid=it["image_id"]; att=to512(Image.open(fp).convert("RGB")); tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
        pv,Mv=vine.get_perm_M(iid); pt,Mt=tmf.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
        A.append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(att),pv,Mv,kind="prob",n_codeword=sb.n),-CLAMP,CLAMP).astype(np.float32))
        B.append(np.clip(method_soft_to_codeword_llr(tmf.raw_logits(att),pt,Mt,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP).astype(np.float32))
        C.append(np.clip(method_soft_to_codeword_llr(vsf.raw_logits(att),ps,Ms,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP).astype(np.float32))
        T.append(tx.astype(np.uint8)); I.append(iid)
    return np.array(A),np.array(B),np.array(C),np.array(T),I
print(f"{'CtrlRegen+':10s}|{'VINE':>9s}|{'eq3':>9s}|{'HEAD3':>9s}|{'best-path':>9s}")
for step,tag in [("0.3","03"),("0.5","05"),("0.7","07")]:
    a,b,cc,tx2,ids2=dec(f"results/defense/ext_vtv_ctrlregen_{tag}")
    if len(a)==0: continue
    fhd,_=hl(a,b,cc)
    dvV=detvec(a,tx2,ids2); dvH=detvec(fhd,tx2,ids2); dvE=detvec(a+b+cc,tx2,ids2); best=np.maximum(dvV,dvE)
    print(f"s={step:7s}|{dvV.mean():9.2f}|{dvE.mean():9.2f}|{dvH.mean():9.2f}|{best.mean():9.2f}")

# ---- FPR on unwatermarked ----
neg=(sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen/*.png")))+sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen2/*.png"))))[260:320]  # 60 unwm, disjoint from N=250 train (imgs 0..249)
NV,NB,NC,NT,NID=[],[],[],[],[]
for j,fp in enumerate(neg):
    iid=f"neg_{j:05d}"; im=to512(Image.open(fp).convert("RGB"))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tmf.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
    NV.append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(im),pv,Mv,kind="prob",n_codeword=sb.n),-CLAMP,CLAMP))
    NB.append(np.clip(method_soft_to_codeword_llr(tmf.raw_logits(im),pt,Mt,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP))
    NC.append(np.clip(method_soft_to_codeword_llr(vsf.raw_logits(im),ps,Ms,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP))
    NT.append(sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)); NID.append(iid)
NV,NB,NC,NT=np.array(NV,np.float32),np.array(NB,np.float32),np.array(NC,np.float32),np.array(NT,np.uint8)
fn,_=hl(NV,NB,NC)
print(f"\n[FPR unwm n={len(NV)}] eq3={detvec(NV+NB+NC,NT,NID).mean():.3f}  HEAD3={detvec(fn,NT,NID).mean():.3f}")
print("FRAG3HEAD_DONE")
