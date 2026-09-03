"""Image-conditioned 3-way fusion head over VINE+TrustMark+VideoSeal aligned codeword LLRs.
Beyond Head3 (which sees only LLR statistics): a small CNN reads the 64x64 attacked image so
the gate can INFER the attack (jpeg blocks / blur / crop border / regen smoothing / rotation)
and route to the surviving fragment. Output stays a codeword LLR -> Chase-BCH + exact 37-bit
match -> crypto-ID 2^-37 FPR preserved by construction."""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
CLAMP=15.0
class ImgHead3(nn.Module):
    def __init__(self, hid:int=64, img_dim:int=32):
        super().__init__()
        # image encoder: 64x64x3 -> img_dim  (infers attack context)
        self.enc=nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(),    # 32
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),   # 16
            nn.Conv2d(32,48,3,2,1), nn.ReLU(),   # 8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48,img_dim), nn.ReLU())
        # per-fragment calibrators (deeper than Head3)
        self.cal=nn.ModuleList([nn.Sequential(nn.Linear(1,hid),nn.ReLU(),nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,1)) for _ in range(3)])
        # gate: [LLR ctx (15) + image emb] -> 3 weights (deeper MLP)
        self.gate=nn.Sequential(nn.Linear(15+img_dim,hid),nn.ReLU(),nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,3))
    @staticmethod
    def feats(a):
        aa=a.abs(); return torch.stack([aa.mean(1),aa.std(1),(aa<0.5).float().mean(1),torch.tanh(aa).mean(1)],1)
    def ctx(self,av,at,as_):
        agr=lambda x,y:(torch.sign(x)==torch.sign(y)).float().mean(1,keepdim=True)
        return torch.cat([self.feats(av),self.feats(at),self.feats(as_),agr(av,at),agr(av,as_),agr(at,as_)],1)
    def forward(self,av,at,as_,img):   # img: [B,3,64,64] in [0,1]
        B,m=av.shape
        e=self.enc(img)
        w=F.softplus(self.gate(torch.cat([self.ctx(av,at,as_),e],1)))
        cs=[self.cal[i](x.reshape(-1,1)).reshape(B,m) for i,x in enumerate([av,at,as_])]
        return w[:,0:1]*cs[0]+w[:,1:2]*cs[1]+w[:,2:3]*cs[2]
def load_img_head3(path,device="cuda"):
    h=ImgHead3().to(device); h.load_state_dict(torch.load(path,map_location=device)); h.eval(); return h
