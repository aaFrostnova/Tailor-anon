"""Oracle (reliability-weighted) fusion under CtrlRegen+, to show the 3-frag dilution dip
is a fusion artifact, not a fragment problem. Decode the already-attacked 3-frag images
(ext_vtv_ctrlregen_0X) into per-fragment codeword LLRs (VINE / TM / VideoSeal), then compare:
  VINE-only | equal-MRC 2-frag (V+TM) | equal-MRC 3-frag | ORACLE 3-frag (per-image
  reliability weights: w_m = max(0, 2*ba_m - 1) -> dead pixel frags -> ~0, trust VINE).
Detection = crypto-verify OR fused-ba>=tau.
"""
import os, sys, glob, json
import numpy as np
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts"), os.path.join(REPO,"scripts/defense")]:
    sys.path.insert(0, p)
from PIL import Image
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr

KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15.0
sb=ShortenedBCH(); TAU=float(binom.ppf(0.99,sb.n,0.5)+1)/sb.n
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
meta=json.load(open(os.path.join(REPO,"results/defense/ext_vtv/meta.json")))
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im

def decode_dir(d):
    AV,AT,AS,TX,IDS=[],[],[],[],[]
    for it in meta["items"]:
        fp=os.path.join(REPO,d,f"img_{it['i']:05d}.png")
        if not os.path.exists(fp): continue
        iid=it["image_id"]; att=to512(Image.open(fp).convert("RGB"))
        tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
        pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vsf.get_perm_M(iid)
        AV.append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(att),pv,Mv,kind="prob",n_codeword=sb.n),-CLAMP,CLAMP))
        AT.append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(att),pt,Mt,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP))
        AS.append(np.clip(method_soft_to_codeword_llr(vsf.raw_logits(att),ps,Ms,kind="logit",n_codeword=sb.n),-CLAMP,CLAMP))
        TX.append(tx.astype(np.uint8)); IDS.append(iid)
    return np.array(AV),np.array(AT),np.array(AS),np.array(TX),IDS

e=1e-6
def norm(a): return a/(a.std(1,keepdims=True)+e)
def oracle(tx,*F):
    f=0.0
    for fr in F:
        b=((fr>0).astype(np.uint8)==tx).mean(1)[:,None]; w=np.clip(2*b-1,0,None); f=f+w*norm(fr)
    return f
def det_vec(F,tx,ids):
    d=np.zeros(len(F))
    for i in range(len(F)):
        ver=bool(decode_and_verify(F[i],ids[i],codec=sb)["detected"])
        d[i]=1.0 if (ver or ((F[i]>0).astype(np.uint8)==tx[i]).mean()>=TAU) else 0.0
    return d
def metr(F,tx,ids):
    dv=det_vec(F,tx,ids); return float(dv.mean()),float(((F>0).astype(np.uint8)==tx).mean())

print(f"tau={TAU:.3f}  (det/ba)\n")
hdr=f"{'CtrlRegen+':10s}|{'VINE-only':>9s}|{'3frag eq':>9s}|{'3frag relW':>10s}|{'3frag BEST-path':>15s}"
print(hdr); print("-"*len(hdr))
rows=[]
for step,tag in [("0.3","03"),("0.5","05"),("0.7","07")]:
    av,at,asv,tx,ids=decode_dir(f"results/defense/ext_vtv_ctrlregen_{tag}")
    if len(av)==0: print(f"s={step}: no images"); continue
    V=metr(av,tx,ids); E3=metr(av+at+asv,tx,ids); O3=metr(oracle(tx,av,at,asv),tx,ids)
    # BEST-path oracle: per-image detect if EITHER the VINE-only path OR the 3-frag path fires
    dv_vine=det_vec(av,tx,ids); dv_3=det_vec(av+at+asv,tx,ids); best=np.maximum(dv_vine,dv_3)
    bav=((av>0).astype(np.uint8)==tx).mean(); bat=((at>0).astype(np.uint8)==tx).mean(); bas=((asv>0).astype(np.uint8)==tx).mean()
    def c(x): return f"{x[0]:.2f}/{x[1]:.2f}"
    print(f"s={step:7s}|{c(V):>9s}|{c(E3):>9s}|{c(O3):>10s}|{best.mean():>13.2f}    [frag ba V/T/VS={bav:.2f}/{bat:.2f}/{bas:.2f}]")
    rows.append({"step":step,"vine":V,"eq3":E3,"oracle_relW":O3,"best_path":float(best.mean()),"frag_ba":[bav,bat,bas]})
json.dump(rows,open(os.path.join(REPO,"results/defense/ctrlregen_oracle.json"),"w"),indent=2)
print("\nORACLE_DONE")
