"""Architecture search for the fusion head on 18K VINE-scale data.
3 fragments: VINE(hidden HV[1000]+rawLLR PV) + TM(hidden HT[2048]+rawLLR PT) + VideoSeal(cw LLR AS only).
+ optional IMAGE input (clean cover reloaded by IMG index, 64x64).
All heads output 100-dim CODEWORD LLR -> crypto-ID FPR (2^-37) preserved by construction.
Compare vs eq-3frag baseline & per-sample oracle ceiling, overall + per-attack. By-image split (no leak)."""
import os,sys,glob,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"; sys.path.insert(0,REPO)
dev="cuda"; torch.manual_seed(0); np.random.seed(0); CLAMP=15.0
DATA=sys.argv[1] if len(sys.argv)>1 else "/scratch/workspace/mingzhel_umass_edu-ablator/hidden_big.npz"
USE_IMG = ("--noimg" not in sys.argv)
d=np.load(DATA,allow_pickle=True)
HV=d["HV"].astype(np.float32); HT=d["HT"].astype(np.float32)
SV=d["SV"].astype(np.int64); MV=d["MV"].astype(np.float32); ST=d["ST"].astype(np.int64); MT=d["MT"].astype(np.float32)
PV=d["PV"].astype(np.float32); PT=d["PT"].astype(np.float32); AS=d["AS"].astype(np.float32)
TX=d["TX"].astype(np.float32); ATK=d["ATK"].astype(str); IMG=d["IMG"].astype(int); m=int(d["n"])
def prob2llr(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
RAWV=np.clip(prob2llr(PV),-CLAMP,CLAMP).astype(np.float32)   # VINE message-space LLR
RAWT=np.clip(PT,-CLAMP,CLAMP).astype(np.float32)             # TM message-space LLR
def npalign(raw,sig,M): return np.take_along_axis(M,sig,1)*np.take_along_axis(raw,sig,1)
FV=npalign(RAWV,SV,MV).astype(np.float32)    # VINE frozen aligned codeword LLR
FT=npalign(RAWT,ST,MT).astype(np.float32)    # TM frozen aligned codeword LLR
FS=np.clip(AS,-CLAMP,CLAMP).astype(np.float32)  # VideoSeal aligned codeword LLR
# ---- optional image branch: reload clean cover by IMG index, 64x64 ----
IMGpix=None
if USE_IMG:
    srcs=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))
    uniq=np.unique(IMG); cache={}
    for i in uniq:
        im=Image.open(srcs[i]).convert("RGB").resize((64,64))
        cache[i]=(np.asarray(im,np.float32)/255).transpose(2,0,1)
    IMGpix=np.stack([cache[i] for i in IMG]).astype(np.float32)  # [N,3,64,64]
    print(f"[img] loaded {len(uniq)} covers -> {IMGpix.shape}",flush=True)
# ---- split by image ----
SPLIT_SEED=int(os.environ.get("SPLIT_SEED","0"))
uimg=np.unique(IMG); rng=np.random.RandomState(SPLIT_SEED); rng.shuffle(uimg)
te=set(uimg[:max(1,len(uimg)//5)].tolist()); trm=np.array([i not in te for i in IMG]); tem=~trm
def ba(L,mask=None): P=(L>0).astype(np.float32); acc=(P==TX); return float(acc[(tem if mask is None else (tem&mask))].mean())
T=lambda a: torch.tensor(a,device=dev)
def talign(raw,sig,M): return torch.gather(M,1,sig)*torch.gather(raw,1,sig)
# ================= architectures (all -> [N,100] codeword LLR) =================
class GatedWSum(nn.Module):   # L1: per-frag scalar weight from reliability stats
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,3))
    def stats(self,F): a=F.abs(); return torch.stack([a.mean(1),(a>1).float().mean(1)],1)  # [N,2]
    def forward(self,b):
        s=torch.cat([self.stats(b['fv']),self.stats(b['ft']),self.stats(b['fs'])],1)  # [N,6]
        w=F.softplus(self.net(s))  # [N,3]
        return w[:,0:1]*b['fv']+w[:,1:2]*b['ft']+w[:,2:3]*b['fs']
class PerBitMLP(nn.Module):   # L2: per-codeword-bit MLP over 3 aligned LLRs (crypto-safe, position-wise)
    def __init__(self,H=32):
        super().__init__(); self.f=nn.Sequential(nn.Linear(3,H),nn.ReLU(),nn.Linear(H,H),nn.ReLU(),nn.Linear(H,1))
    def forward(self,b):
        x=torch.stack([b['fv'],b['ft'],b['fs']],-1)  # [N,100,3]
        return self.f(x).squeeze(-1)
class HiddenResid(nn.Module):  # H1: hidden re-readout residual (VINE+TM) + VideoSeal LLR, eq combine
    def __init__(self,dv=HV.shape[1],dt=HT.shape[1],H=32):
        super().__init__()
        self.rv=nn.Sequential(nn.Dropout(0.3),nn.Linear(dv,H),nn.ReLU(),nn.Linear(H,m))
        self.rt=nn.Sequential(nn.Dropout(0.3),nn.Linear(dt,H),nn.ReLU(),nn.Linear(H,m))
        for r in (self.rv,self.rt): nn.init.zeros_(r[-1].weight); nn.init.zeros_(r[-1].bias)
    def frags(self,b):
        fv=talign(b['rawv']+self.rv(b['hv']),b['sv'],b['Mv']); ft=talign(b['rawt']+self.rt(b['ht']),b['st'],b['Mt'])
        return fv,ft
    def forward(self,b):
        fv,ft=self.frags(b); return fv+ft+b['fs']
class HiddenResidMLP(HiddenResid):  # H2: hidden residual frags -> per-bit MLP combine
    def __init__(self,**k):
        super().__init__(**k); self.c=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,b):
        fv,ft=self.frags(b); x=torch.stack([fv,ft,b['fs']],-1); return self.c(x).squeeze(-1)
class ImgCNN(nn.Module):
    def __init__(self,out):
        super().__init__()
        self.c=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
                             nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(32*16,out))
    def forward(self,x): return self.c(x)
class ImgGate(nn.Module):   # I1: image CNN -> 3 per-frag reliability weights
    def __init__(self):
        super().__init__(); self.cnn=ImgCNN(3)
    def forward(self,b):
        w=F.softplus(self.cnn(b['img'])); return w[:,0:1]*b['fv']+w[:,1:2]*b['ft']+w[:,2:3]*b['fs']
class Full(HiddenResid):    # F1: hidden residual frags + image-conditioned per-bit combine (kitchen sink)
    def __init__(self,**k):
        super().__init__(**k); self.cnn=ImgCNN(16)
        self.c=nn.Sequential(nn.Linear(3+16,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,b):
        fv,ft=self.frags(b); g=self.cnn(b['img'])[:,None,:].expand(-1,m,-1)  # [N,100,16]
        x=torch.cat([torch.stack([fv,ft,b['fs']],-1),g],-1); return self.c(x).squeeze(-1)
# ================= train/eval harness =================
def pack(mask):
    b=dict(fv=T(FV[mask]),ft=T(FT[mask]),fs=T(FS[mask]),hv=T(HV[mask]),ht=T(HT[mask]),
           rawv=T(RAWV[mask]),rawt=T(RAWT[mask]),sv=T(SV[mask]),Mv=T(MV[mask]),st=T(ST[mask]),Mt=T(MT[mask]),tx=T(TX[mask]))
    if IMGpix is not None: b['img']=T(IMGpix[mask])
    return b
def run(name,make,needs_img=False,epochs=120,bs=4096):
    if needs_img and IMGpix is None: print(f"  {name:16s}  (skip: no image)"); return None
    torch.manual_seed(0); h=make().to(dev); opt=torch.optim.Adam(h.parameters(),1e-3,weight_decay=1e-4); lf=nn.BCEWithLogitsLoss()
    tr=pack(trm); N=tr['fv'].shape[0]
    for ep in range(epochs):
        h.train(); perm=torch.randperm(N,device=dev)
        for s in range(0,N,bs):
            idx=perm[s:s+bs]; bb={k:v[idx] for k,v in tr.items()}
            opt.zero_grad(); out=h(bb); loss=lf(out,bb['tx']); loss.backward(); opt.step()
    h.eval()
    with torch.no_grad():
        L=np.zeros_like(TX); te=pack(tem); L[tem]=h(te).cpu().numpy()
    npar=sum(p.numel() for p in h.parameters())
    print(f"  {name:16s} params={npar:>7d} overall={ba(L):.4f}",flush=True)
    return L
# ================= baselines =================
eq=FV+FT+FS
oracle=np.take_along_axis(np.stack([FV,FT,FS],0),
    np.argmax(np.stack([((FV>0)==TX).mean(1),((FT>0)==TX).mean(1),((FS>0)==TX).mean(1)],0),0)[None,:,None],0)[0]
print(f"=== {len(HV)} samples | test {len(te)} imgs | img={'ON' if IMGpix is not None else 'off'} ===")
print(f"  {'eq-3frag':16s} params=      0 overall={ba(eq):.4f}")
print(f"  {'oracle-3frag':16s} params=      0 overall={ba(oracle):.4f}  <-- ceiling")
res={'eq-3frag':eq,'oracle-3frag':oracle}
for nm,mk,ni in [("L1_gatedWsum",GatedWSum,False),("L2_perbitMLP",PerBitMLP,False),
                 ("H1_hiddenResid",HiddenResid,False),("H2_hiddenMLP",HiddenResidMLP,False),
                 ("I1_imgGate",ImgGate,True),("F1_full",Full,True)]:
    L=run(nm,mk,needs_img=ni)
    if L is not None: res[nm]=L
# ================= per-attack table =================
print("\n=== per-attack bit-acc (test) ===")
methods=list(res.keys()); print("  attack     "+"".join(f"{n[:11]:>13s}" for n in methods))
for k in np.unique(ATK):
    mk=ATK==k; print(f"  {k:9s}  "+"".join(f"{ba(res[n],mk):>13.4f}" for n in methods))
best=max([n for n in res if n not in('oracle-3frag',)],key=lambda n:ba(res[n]))
print(f"\nBEST(non-oracle): {best} = {ba(res[best]):.4f}  vs eq={ba(eq):.4f}  vs oracle={ba(oracle):.4f}")
print("ARCH_DONE")
