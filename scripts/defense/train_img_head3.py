"""Train the image-conditioned ImgHead3 on fusion_head_data3.npz.
Split by image (no leakage). Compare eq-MRC vs ImgHead3 vs oracle(per-sample best single frag),
overall + per-attack. Also trains a NO-IMAGE ablation (image zeroed) to isolate the image's value.
crypto-ID FPR preserved by construction (output is codeword LLR)."""
import os,sys,numpy as np,torch,torch.nn as nn
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"; sys.path.insert(0,REPO)
from src.img_head3 import ImgHead3, CLAMP
dev="cuda"; torch.manual_seed(0); np.random.seed(0)
DATA=sys.argv[1] if len(sys.argv)>1 else os.path.join(REPO,"results/defense/fusion_head_data3.npz")
d=np.load(DATA,allow_pickle=True)
AV=np.clip(d["AV"],-CLAMP,CLAMP).astype(np.float32); AT=np.clip(d["AT"],-CLAMP,CLAMP).astype(np.float32)
AS=np.clip(d["AS"],-CLAMP,CLAMP).astype(np.float32); TX=d["TX"].astype(np.float32)
IMG=d["IMG"].astype(int); ATK=d["ATK"].astype(str); IMGD=(d["IMGD"].astype(np.float32)/255).transpose(0,3,1,2)  # [N,3,64,64]
uimg=np.unique(IMG); rng=np.random.RandomState(0); rng.shuffle(uimg)
te=set(uimg[:max(1,len(uimg)//5)].tolist()); tr_m=np.array([i not in te for i in IMG]); te_m=~tr_m
def bitacc(llr): return ((llr>0).astype(np.float32)==TX)
def report(tag, llr_fn):
    llr=llr_fn(); acc=bitacc(llr)
    o=acc[te_m].mean()
    print(f"  {tag:16s} overall={o:.3f}",flush=True)
    return o,acc
# baselines
eq=lambda: AV+AT+AS
oracle_llr=lambda: np.take_along_axis(np.stack([AV,AT,AS],0),  # per-sample pick frag w/ max bitacc
    np.argmax(np.stack([bitacc(AV).mean(1),bitacc(AT).mean(1),bitacc(AS).mean(1)],0),0)[None,:,None],0)[0]
def to_t(*x): return [torch.tensor(a,device=dev) for a in x]
def train(use_img=True, epochs=250):
    h=ImgHead3().to(dev); opt=torch.optim.Adam(h.parameters(),1e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss()
    av,at,as_,tx,im=to_t(AV[tr_m],AT[tr_m],AS[tr_m],TX[tr_m],IMGD[tr_m] if use_img else IMGD[tr_m]*0)
    for ep in range(epochs):
        h.train(); opt.zero_grad(); f=h(av,at,as_,im); loss=lossf(f,tx); loss.backward(); opt.step()
    h.eval()
    with torch.no_grad():
        ate=to_t(AV[te_m],AT[te_m],AS[te_m], (IMGD[te_m] if use_img else IMGD[te_m]*0))
        fte=h(ate[0],ate[1],ate[2],ate[3]).cpu().numpy()
    return h,fte
print(f"=== data: {len(AV)} samples, {len(uimg)} imgs (test {len(te)} imgs), attacks={list(np.unique(ATK))} ===")
print("=== overall bit-acc (test) ===")
eo,_=report("eq-MRC", eq)
oo,_=report("oracle(best-frag)", oracle_llr)
h_img,f_img=train(use_img=True); acc_img=((f_img>0).astype(np.float32)==TX[te_m]).mean(); print(f"  {'ImgHead3':16s} overall={acc_img:.3f}")
h_noimg,f_noimg=train(use_img=False); acc_no=((f_noimg>0).astype(np.float32)==TX[te_m]).mean(); print(f"  {'Head3(no-img)':16s} overall={acc_no:.3f}")
# per-attack
print("=== per-attack bit-acc (test): eq / no-img / ImgHead3 / oracle ===")
teA=ATK[te_m]; TXte=TX[te_m]; eqte=(eq()[te_m]>0).astype(np.float32); orte=(oracle_llr()[te_m]>0).astype(np.float32)
imte=(f_img>0).astype(np.float32); note=(f_noimg>0).astype(np.float32)
for k in np.unique(ATK):
    m=teA==k
    if m.sum()==0: continue
    ba=lambda P: float((P[m]==TXte[m]).mean())
    print(f"  {k:9s} eq={ba(eqte):.3f}  noimg={ba(note):.3f}  IMG={ba(imte):.3f}  oracle={ba(orte):.3f}")
torch.save(h_img.state_dict(), os.path.join(REPO,"results/defense/img_head3.pt"))
print("TRAIN_DONE")
