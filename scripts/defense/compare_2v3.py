"""2-frag (VINE+TM) vs 3-frag (VINE+TM+VideoSeal) on 18K data, per-attack.
Both equal-MRC and reliability-gated-Wsum; + VINE-only reference. Does the 3rd frag help or dilute?"""
import os,sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"; sys.path.insert(0,REPO)
dev="cuda"; torch.manual_seed(0); np.random.seed(0); CLAMP=15.0
d=np.load(sys.argv[1],allow_pickle=True)
def prob2llr(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
PV=d["PV"].astype(np.float32); PT=d["PT"].astype(np.float32); AS=d["AS"].astype(np.float32)
SV=d["SV"].astype(np.int64); MV=d["MV"].astype(np.float32); ST=d["ST"].astype(np.int64); MT=d["MT"].astype(np.float32)
TX=d["TX"].astype(np.float32); ATK=d["ATK"].astype(str); IMG=d["IMG"].astype(int); m=int(d["n"])
def npalign(raw,sig,M): return np.take_along_axis(M,sig,1)*np.take_along_axis(raw,sig,1)
FV=npalign(np.clip(prob2llr(PV),-CLAMP,CLAMP),SV,MV).astype(np.float32)
FT=npalign(np.clip(PT,-CLAMP,CLAMP),ST,MT).astype(np.float32)
FS=np.clip(AS,-CLAMP,CLAMP).astype(np.float32)
uimg=np.unique(IMG); rng=np.random.RandomState(0); rng.shuffle(uimg)
te=set(uimg[:len(uimg)//5].tolist()); tem=np.array([i in te for i in IMG]); trm=~tem
def ba(L,mask=None): return float(((L>0).astype(np.float32)==TX)[(tem if mask is None else (tem&mask))].mean())
# reliability-gated weighted sum over given frag list (per-frag scalar weight from |LLR| stats)
class Gate(nn.Module):
    def __init__(self,k): super().__init__(); self.k=k; self.net=nn.Sequential(nn.Linear(2*k,16),nn.ReLU(),nn.Linear(16,k))
    def st(self,Fx): a=Fx.abs(); return torch.stack([a.mean(1),(a>1).float().mean(1)],1)
    def forward(self,Fs):
        s=torch.cat([self.st(f) for f in Fs],1); w=F.softplus(self.net(s))
        return sum(w[:,i:i+1]*Fs[i] for i in range(self.k))
def train_gate(frags):
    T=lambda a: torch.tensor(a,device=dev); k=len(frags)
    Ftr=[T(f[trm]) for f in frags]; Fte=[T(f[tem]) for f in frags]; y=T(TX[trm])
    h=Gate(k).to(dev); opt=torch.optim.Adam(h.parameters(),1e-3,weight_decay=1e-4); lf=nn.BCEWithLogitsLoss()
    N=Ftr[0].shape[0]
    for ep in range(150):
        perm=torch.randperm(N,device=dev)
        for s in range(0,N,4096):
            idx=perm[s:s+4096]; opt.zero_grad(); out=h([f[idx] for f in Ftr]); lf(out,y[idx]).backward(); opt.step()
    h.eval()
    with torch.no_grad(): L=np.zeros_like(TX); L[tem]=h(Fte).cpu().numpy()
    return L
VINE=FV; eq2=FV+FT; eq3=FV+FT+FS
g2=train_gate([FV,FT]); g3=train_gate([FV,FT,FS])
cols=[("VINE-only",VINE),("2f eq",eq2),("2f GATE",g2),("3f eq",eq3),("3f GATE",g3)]
print(f"=== 18K, test {len(te)} imgs ===")
print("  attack     "+"".join(f"{n:>10s}" for n,_ in cols))
print("  OVERALL    "+"".join(f"{ba(L):>10.4f}" for _,L in cols))
for k in np.unique(ATK):
    mk=ATK==k; print(f"  {k:9s}  "+"".join(f"{ba(L,mk):>10.4f}" for _,L in cols))
print("CMP_DONE")
