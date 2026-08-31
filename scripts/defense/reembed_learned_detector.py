"""MVP: learned detector on re-embed observations — can a CNN on the re-embed residual decide
'does this image carry the target watermark?' BETTER than the RMS probe / direct detection,
especially where direct detection fails?

Per image, per attack: W=VINE.embed(C,id).
  POS: att+=attack(W);  delta+ = embed(att+,id) - att+    (image DOES carry target id)
  NEG: att-=attack(C);  delta- = embed(att-,id) - att-    (clean, does NOT carry id; same attack)
Feature = delta (96x96x3). Train small CNN; compare test AUROC vs RMS(delta) vs direct-detect ba.
attack-matched pos/neg so the classifier learns 'watermark presence', not 'attack type'.
"""
import glob, io, sys, numpy as np
from PIL import Image
import torch, torch.nn as nn
import scipy.ndimage as ndi
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts/defense")
from src.vine_crypto_wrapper import VineCryptoWrapper
from _regen_util import build_regen_pipe, stable_regen
RES = 512; FS = 96; dev = "cuda"
vine = VineCryptoWrapper(master_key=b"v5_key_encoder_master", method_name="vine", n_bits=100, device=dev)
pipe = None
def arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def emb(a, iid):
    W = vine.embed(Image.fromarray(np.clip(a,0,255).astype(np.uint8)), iid); return arr(W if W.size==(RES,RES) else W.resize((RES,RES)))
def ccr(a,r):
    img=Image.fromarray(a.astype(np.uint8)); cw=int(round(RES*r)); l=(RES-cw)//2
    return arr(img.crop((l,l,l+cw,l+cw)).resize((RES,RES),Image.BICUBIC))
def rot(a,deg):
    pad=RES//2; big=Image.fromarray(np.pad(a.astype(np.uint8),((pad,pad),(pad,pad),(0,0)),mode='reflect')).rotate(deg,resample=Image.BICUBIC)
    bw,bh=big.size; l=(bw-RES)//2; return arr(big.crop((l,l,l+RES,l+RES)))
def attack(a,name,seed):
    if name=='crop75': return ccr(a,0.75)
    if name=='rot9': return rot(a,9)
    if name=='regen':
        global pipe
        if pipe is None: pipe=build_regen_pipe()
        return arr(stable_regen(pipe,Image.fromarray(a.astype(np.uint8)),1234+seed,denoise_steps=8))
def feat(delta): # downsample residual to FSxFS
    return np.stack([ndi.zoom(delta[:,:,c],FS/RES,order=1) for c in range(3)],0)
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
                          nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
def auroc(score,lab):
    score=np.asarray(score); lab=np.asarray(lab); P=score[lab==1]; N=score[lab==0]; c=0.0
    for a in P:
        for b in N: c+=(a>b)+0.5*(a==b)
    return c/(len(P)*len(N)+1e-9)
def run(name,N,seed0):
    files=sorted(glob.glob('/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg'))[4000:4000+N]
    X,y,rms,ba=[],[],[],[]
    for i,fp in enumerate(files):
        C=arr(Image.open(fp).convert('RGB')); iid=f'det_{i:05d}'; W=emb(C,iid)
        ap=attack(W,name,seed0+i); an=attack(C,name,seed0+i)
        dp=emb(ap,iid)-ap; dn=emb(an,iid)-an
        X.append(feat(dp)); y.append(1); rms.append(float(np.sqrt((dp**2).mean())))
        ba.append(vine.detect(Image.fromarray(np.clip(ap,0,255).astype(np.uint8)),iid)['bit_accuracy'])
        X.append(feat(dn)); y.append(0); rms.append(float(np.sqrt((dn**2).mean())))
        ba.append(vine.detect(Image.fromarray(np.clip(an,0,255).astype(np.uint8)),iid)['bit_accuracy'])
    X=np.array(X,np.float32)/8.0; y=np.array(y); rms=np.array(rms); ba=np.array(ba)
    ntr=int(0.8*len(X))//2*2  # split by image pairs
    Xtr,ytr=torch.tensor(X[:ntr]).to(dev),torch.tensor(y[:ntr],dtype=torch.float32).to(dev)
    Xte,yte=X[ntr:],y[ntr:]
    net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),1e-3)
    for ep in range(60):
        net.train(); opt.zero_grad(); out=net(Xtr); loss=nn.functional.binary_cross_entropy_with_logits(out,ytr); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): sc=net(torch.tensor(Xte).to(dev)).cpu().numpy()
    # baselines on TEST split: RMS (smaller=watermarked => use -rms), direct ba (higher=watermarked)
    rms_te=rms[ntr:]; ba_te=ba[ntr:]
    print(f"  {name:<8} learned_AUROC={auroc(sc,yte):.3f}   RMS_probe_AUROC={auroc(-rms_te,yte):.3f}   direct_detect_AUROC={auroc(ba_te,yte):.3f}   (n_test={len(yte)}, ba_pos={ba_te[yte==1].mean():.2f}/ba_neg={ba_te[yte==0].mean():.2f})")
print("learned re-embed detector vs RMS-probe vs direct-detect (n=120 imgs/attack, 80/20 split):")
for nm,sd in [('crop75',0),('rot9',1000),('regen',5000)]:
    run(nm,120,sd)
print("DONE")
