"""CAUSAL proof (v2): does positional encoding remove border-locking? Minimal hide-a-secret autoencoder,
3 decoder conditions, identical otherwise. FIX v1's dead training (L2 crushed the signal): unbounded residual +
moderate L2, 16-bit secret, longer training, print progress. Measure payload localization + relocate-cliff.
  A zeros+no-coord   (VINE-like: only position cue = zero-padded border)
  B zeros+CoordConv  (explicit positional encoding everywhere)
  C circular+no-coord(no border cue at all)"""
import sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
torch.manual_seed(0); np.random.seed(0)
dev="cuda"; IMG=64; SEC=16
paths=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:400]
X=torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((IMG,IMG)),np.float32)/255).permute(2,0,1) for p in paths]).to(dev)
yy,xx=np.mgrid[0:IMG,0:IMG]; Dedge=torch.tensor(np.minimum(np.minimum(yy,IMG-1-yy),np.minimum(xx,IMG-1-xx))).to(dev)
def coordch(b):
    ys=torch.linspace(-1,1,IMG,device=dev).view(1,1,IMG,1).expand(b,1,IMG,IMG)
    xs=torch.linspace(-1,1,IMG,device=dev).view(1,1,1,IMG).expand(b,1,IMG,IMG)
    return torch.cat([xs,ys],1)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.fc=nn.Linear(SEC,16*16)
        s.net=nn.Sequential(nn.Conv2d(4,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),
                            nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,3,3,padding=1))
    def forward(s,x,sec):
        f=F.interpolate(s.fc(sec).view(-1,1,16,16),size=(IMG,IMG),mode="nearest")
        res=torch.tanh(s.net(torch.cat([x,f],1)))          # unbounded-ish (±1), L2 controls amplitude
        return (x+res).clamp(0,1),res
class Dec(nn.Module):
    def __init__(s,coord=False,pad="zeros"):
        super().__init__(); s.coord=coord; cin=3+(2 if coord else 0)
        s.net=nn.Sequential(nn.Conv2d(cin,64,3,2,1,padding_mode=pad),nn.ReLU(),
                            nn.Conv2d(64,128,3,2,1,padding_mode=pad),nn.ReLU(),
                            nn.Conv2d(128,128,3,2,1,padding_mode=pad),nn.ReLU()); s.fc=nn.Linear(128,SEC)
    def forward(s,x):
        if s.coord: x=torch.cat([x,coordch(x.size(0))],1)
        return s.fc(s.net(x).mean([2,3]))
def train(coord,pad,steps=6000,lam=0.3,tag=""):
    torch.manual_seed(1)
    E=Enc().to(dev); Dm=Dec(coord,pad).to(dev)
    opt=torch.optim.Adam(list(E.parameters())+list(Dm.parameters()),2e-3)
    for it in range(steps):
        idx=torch.randint(0,X.size(0),(32,),device=dev); x=X[idx]
        sec=torch.randint(0,2,(32,SEC),device=dev).float()
        wm,res=E(x,sec); logit=Dm(wm)
        loss=F.binary_cross_entropy_with_logits(logit,sec)+lam*(res**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it+1) in (2000,6000):
            with torch.no_grad(): ba=((logit>0).float()==sec).float().mean().item()
            print(f"    [{tag}] step {it+1}: bitacc={ba:.3f} res_rms={res.pow(2).mean().sqrt().item():.3f}",flush=True)
    E.eval(); Dm.eval()
    with torch.no_grad():
        x=X[:64]; sec=torch.randint(0,2,(64,SEC),device=dev).float()
        wm,res=E(x,sec); ba=((Dm(wm)>0).float()==sec).float().mean().item()
        r=res.abs().mean(1); bden=r[:,Dedge<8].mean().item(); cden=r[:,Dedge>=8].mean().item()
        curve={sh:((Dm(x+torch.roll(res,(sh,sh),(2,3)))>0).float()==sec).float().mean().item() for sh in [0,1,2,4,8,16]}
    return ba,bden/max(cden,1e-6),curve
for name,(coord,pad) in {"A zeros+no-coord":(False,"zeros"),"B zeros+CoordConv":(True,"zeros"),"C circular+no-coord":(False,"circular")}.items():
    ba,ratio,curve=train(coord,pad,tag=name)
    cl=" ".join(f"{s}px={curve[s]:.2f}" for s in [0,1,2,4,8,16])
    print(f"[{name:20}] bitacc={ba:.3f}  border/center energy ratio={ratio:.2f}  | relocate: {cl}",flush=True)
print("POSENC_ABLATION_DONE",flush=True)
