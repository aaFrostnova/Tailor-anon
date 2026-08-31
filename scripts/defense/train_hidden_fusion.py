"""Hidden-state fusion vs LLR fusion. Per fragment: hidden -> learned re-readout -> raw logits ->
crypto-align (M[sigma[j]]*raw[sigma[j]]) -> codeword LLR. Fusion head sees BOTH frozen LLRs
(current ceiling) AND hidden-derived LLRs -> if hidden has extra info, exceeds LLR-oracle.
Output stays codeword LLR (Chase-BCH + 37bit -> crypto-ID FPR preserved)."""
import os,sys,numpy as np,torch,torch.nn as nn
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"; sys.path.insert(0,REPO)
dev="cuda"; torch.manual_seed(0); np.random.seed(0); CLAMP=15.0
d=np.load(sys.argv[1] if len(sys.argv)>1 else os.path.join(REPO,"results/defense/hidden_data.npz"),allow_pickle=True)
HV=d["HV"].astype(np.float32); HT=d["HT"].astype(np.float32)
SV=d["SV"].astype(np.int64); MV=d["MV"].astype(np.float32); ST=d["ST"].astype(np.int64); MT=d["MT"].astype(np.float32)
PV=d["PV"].astype(np.float32); PT=d["PT"].astype(np.float32); TX=d["TX"].astype(np.float32)
ATK=d["ATK"].astype(str); IMG=d["IMG"].astype(int); m=int(d["n"])
def align(raw,sig,M): return torch.gather(M,1,sig)*torch.gather(raw,1,sig)  # codeword LLR
# frozen codeword LLRs (VINE prob->LLR, TM logit)
def prob2llr(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
class HFuse(nn.Module):
    def __init__(self,dv=HV.shape[1],dt=HT.shape[1],hid=128):
        super().__init__()
        H=32
        self.rv=nn.Sequential(nn.Dropout(0.3),nn.Linear(dv,H),nn.ReLU(),nn.Linear(H,m))
        self.rt=nn.Sequential(nn.Dropout(0.3),nn.Linear(dt,H),nn.ReLU(),nn.Linear(H,m))
        nn.init.zeros_(self.rv[-1].weight); nn.init.zeros_(self.rv[-1].bias)   # residual starts at 0
        nn.init.zeros_(self.rt[-1].weight); nn.init.zeros_(self.rt[-1].bias)
    def forward(self,hv,ht,sv,Mv,st,Mt,fv,ft,rawv,rawt):  # residual on frozen RAW logits
        lv=align(rawv+self.rv(hv),sv,Mv); lt=align(rawt+self.rt(ht),st,Mt)   # hidden-corrected -> cw LLR
        return lv+lt   # equal-MRC of hidden-corrected LLRs (starts = frozen eq at init)
# frozen aligned LLRs (numpy->then torch)
FV=np.take_along_axis(MV,SV,1)*np.take_along_axis(np.clip(prob2llr(PV),-CLAMP,CLAMP),SV,1)
FT=np.take_along_axis(MT,ST,1)*np.take_along_axis(np.clip(PT,-CLAMP,CLAMP),ST,1)
uimg=np.unique(IMG); rng=np.random.RandomState(0); rng.shuffle(uimg); te=set(uimg[:max(1,len(uimg)//5)].tolist())
trm=np.array([i not in te for i in IMG]); tem=~trm
def ba(L,mask=None): P=(L>0).astype(np.float32); acc=(P==TX); return float(acc[tem if mask is None else (tem&mask)].mean())
T=lambda a: torch.tensor(a,device=dev)
h=HFuse().to(dev); opt=torch.optim.Adam(h.parameters(),1e-3,weight_decay=1e-4); lf=nn.BCEWithLogitsLoss()
RAWV=np.clip(prob2llr(PV),-CLAMP,CLAMP); RAWT=np.clip(PT,-CLAMP,CLAMP)
tr=dict(hv=T(HV[trm]),ht=T(HT[trm]),sv=T(SV[trm]),Mv=T(MV[trm]),st=T(ST[trm]),Mt=T(MT[trm]),fv=T(FV[trm]),ft=T(FT[trm]),rv=T(RAWV[trm]),rt=T(RAWT[trm]),tx=T(TX[trm]))
for ep in range(400):
    h.train(); opt.zero_grad(); out=h(tr['hv'],tr['ht'],tr['sv'],tr['Mv'],tr['st'],tr['Mt'],tr['fv'],tr['ft'],tr['rv'],tr['rt']); loss=lf(out,tr['tx']); loss.backward(); opt.step()
h.eval()
with torch.no_grad():
    fte=h(T(HV[tem]),T(HT[tem]),T(SV[tem]),T(MV[tem]),T(ST[tem]),T(MT[tem]),T(FV[tem]),T(FT[tem]),T(RAWV[tem]),T(RAWT[tem])).cpu().numpy()
FUS=np.zeros_like(TX); FUS[tem]=fte
# LLR baselines
eq=FV+FT; orl=np.where((FV>0).astype(float).mean(1,keepdims=True)>= (FT>0).astype(float).mean(1,keepdims=True),FV,FT)
oracle=np.where(((FV>0)==TX).mean(1,keepdims=True)>=((FT>0)==TX).mean(1,keepdims=True),FV,FT)  # per-sample best frozen frag
print(f"=== {len(HV)} samples, test {len(te)} imgs, HFuse params={sum(p.numel() for p in h.parameters())} ===")
print(f"overall: LLR-eq={ba(eq):.3f}  LLR-oracle(best frozen frag)={ba(oracle):.3f}  HIDDEN-FUSE={ba(FUS):.3f}")
print("per-attack: LLR-eq / LLR-oracle / HIDDEN-FUSE")
for k in np.unique(ATK):
    mk=ATK==k; print(f"  {k:11s} eq={ba(eq,mk):.3f}  oracle={ba(oracle,mk):.3f}  HIDDEN={ba(FUS,mk):.3f}")
print("HFUSE_DONE")
