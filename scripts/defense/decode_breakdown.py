"""Decode an attacked dir for the 3-frag composite and report detection per fusion variant:
VINE-only / equal-MRC-3 / learned-HEAD (deployed) / best-path-oracle(=max(VINE,eq3)).
detvec(F) = crypto-verify(F) OR zero-bit ba(F)>=tau. Used to fill the advanced-attack table
at larger N. Parametric on embed_dir/attacked_dir so it works for ext_vtv200_ctrlregen_*."""
import os, sys, glob, json, argparse
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
KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15.0
ap=argparse.ArgumentParser()
ap.add_argument("--embed_dir",required=True); ap.add_argument("--attacked_dir",required=True)
ap.add_argument("--attack_name",default="adv"); ap.add_argument("--output",default="")
a=ap.parse_args()
sb=ShortenedBCH(); n=sb.n; tau=float(binom.ppf(0.99,n,0.5)+1)/n
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=n,device=dev)
head=load_head3(os.path.join(REPO,"results/defense/frag3_head.pt"),dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
meta=json.load(open(os.path.join(a.embed_dir,"meta.json")))
A,B,C,T,I=[],[],[],[],[]; miss=0
for it in meta["items"]:
    fp=os.path.join(a.attacked_dir,f"img_{it['i']:05d}.png")
    if not os.path.exists(fp): miss+=1; continue
    iid=it["image_id"]; att=to512(Image.open(fp).convert("RGB")); tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
    A.append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(att),pv,Mv,kind="prob", n_codeword=n),-CLAMP,CLAMP).astype(np.float32))
    B.append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(att),pt,Mt,kind="logit",n_codeword=n),-CLAMP,CLAMP).astype(np.float32))
    C.append(np.clip(method_soft_to_codeword_llr(vsf.raw_logits(att),ps,Ms,kind="logit",n_codeword=n),-CLAMP,CLAMP).astype(np.float32))
    T.append(tx.astype(np.uint8)); I.append(iid)
A,B,C,T=np.array(A),np.array(B),np.array(C),np.array(T)
with torch.no_grad():
    H=head(torch.tensor(A,device=dev),torch.tensor(B,device=dev),torch.tensor(C,device=dev)).cpu().numpy()
def detvec(F):
    d=np.zeros(len(F))
    for i in range(len(F)):
        ver=bool(decode_and_verify(F[i],I[i],codec=sb)["detected"]); d[i]=1.0 if (ver or ((F[i]>0).astype(np.uint8)==T[i]).mean()>=tau) else 0.0
    return d
dvV=detvec(A); dvE=detvec(A+B+C); dvH=detvec(H); best=np.maximum(dvV,dvE)
res={"attack":a.attack_name,"n":len(A),"missing":miss,
     "VINE_only":float(dvV.mean()),"eq3":float(dvE.mean()),"HEAD":float(dvH.mean()),"bestpath_oracle":float(best.mean()),
     "head_fused_ba":float(((H>0).astype(np.uint8)==T).mean())}
print(f"[{a.attack_name}] n={len(A)} miss={miss} | VINE={res['VINE_only']:.3f} eq3={res['eq3']:.3f} HEAD={res['HEAD']:.3f} best-path={res['bestpath_oracle']:.3f} | head_ba={res['head_fused_ba']:.3f}")
if a.output: json.dump(res,open(a.output,"w"),indent=2)
print("BREAKDOWN_DONE")
