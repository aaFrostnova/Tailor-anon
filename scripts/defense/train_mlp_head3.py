"""Plain per-bit MLP fusion head (vs Head3's gated weighted-sum). Per bit j: input
[av_j, at_j, as_j] concatenated with a broadcast 15-dim global LLR context -> MLP(18->hid->hid->1)
-> fused codeword LLR_j. Codeword-agnostic (shared over bits), more expressive than weighted-MRC.
Compares eq-MRC / MLPHead3 / oracle(best-frag), per-attack. Output stays codeword LLR (crypto-ID FPR intact)."""
import os,sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"; sys.path.insert(0,REPO)
CLAMP=15.0; dev="cuda"; torch.manual_seed(0); np.random.seed(0)
DATA=sys.argv[1] if len(sys.argv)>1 else os.path.join(REPO,"results/defense/fusion_head_all3.npz")
d=np.load(DATA,allow_pickle=True)
AV=np.clip(d["AV"],-CLAMP,CLAMP).astype(np.float32); AT=np.clip(d["AT"],-CLAMP,CLAMP).astype(np.float32)
AS=np.clip(d["AS"],-CLAMP,CLAMP).astype(np.float32); TX=d["TX"].astype(np.float32)
IMG=d["IMG"].astype(int); ATK=d["ATK"].astype(str)
class MLPHead3(nn.Module):
    def __init__(self,hid=64):
        super().__init__(); self.net=nn.Sequential(nn.Linear(3+15,hid),nn.ReLU(),nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,1))
    @staticmethod
    def feats(a):
        aa=a.abs(); return torch.stack([aa.mean(1),aa.std(1),(aa<0.5).float().mean(1),torch.tanh(aa).mean(1)],1)
    def ctx(self,av,at,as_):
        ag=lambda x,y:(torch.sign(x)==torch.sign(y)).float().mean(1,keepdim=True)
        return torch.cat([self.feats(av),self.feats(at),self.feats(as_),ag(av,at),ag(av,as_),ag(at,as_)],1)  # [B,15]
    def forward(self,av,at,as_):
        B,m=av.shape; x=torch.stack([av,at,as_],-1)                       # [B,m,3]
        c=self.ctx(av,at,as_)[:,None,:].expand(B,m,15)                    # [B,m,15]
        return self.net(torch.cat([x,c],-1).reshape(-1,18)).reshape(B,m)  # [B,m]
uimg=np.unique(IMG); rng=np.random.RandomState(0); rng.shuffle(uimg)
te=set(uimg[:max(1,len(uimg)//5)].tolist()); trm=np.array([i not in te for i in IMG]); tem=~trm
def ba(P,m=None): 
    P=(P>0).astype(np.float32); acc=(P==TX); return float(acc[tem if m is None else (tem&m)].mean())
def to_t(*x): return [torch.tensor(a,device=dev) for a in x]
eq=AV+AT+AS
orl=np.take_along_axis(np.stack([AV,AT,AS],0),np.argmax(np.stack([(( (AV>0)==TX).mean(1)),(((AT>0)==TX).mean(1)),(((AS>0)==TX).mean(1))],0),0)[None,:,None],0)[0]
h=MLPHead3().to(dev); opt=torch.optim.Adam(h.parameters(),1e-3,weight_decay=1e-4); lossf=nn.BCEWithLogitsLoss()
av,at,as_,tx=to_t(AV[trm],AT[trm],AS[trm],TX[trm])
for ep in range(300):
    h.train(); opt.zero_grad(); loss=lossf(h(av,at,as_),tx); loss.backward(); opt.step()
h.eval()
with torch.no_grad():
    at_=to_t(AV[tem],AT[tem],AS[tem]); fte=h(*at_).cpu().numpy()
fmlp=np.zeros_like(TX); fmlp[tem]=fte
print(f"=== {len(AV)} samples, {len(uimg)} imgs (test {len(te)}), params(MLPHead3)={sum(p.numel() for p in h.parameters())} ===")
print(f"  overall:  eq-MRC={ba(eq):.3f}  MLPHead3={ba(fmlp):.3f}  oracle={ba(orl):.3f}")
print("  per-attack: eq / MLPHead3 / oracle")
for k in np.unique(ATK):
    m=ATK==k
    print(f"    {k:11s} eq={ba(eq,m):.3f}  MLP={ba(fmlp,m):.3f}  oracle={ba(orl,m):.3f}")
print("MLP_DONE")
