"""top-k for CROP+RESIZE (a fairer geometric attack than rot30) with & without regen.
Central-VINE embed (crop-robust) + decode-side crop/scale SEARCH. Per search-scale compute
ID-INDEPENDENT metrics {LLR energy, high-conf count, consistency} + oracle bit-acc.
'true pose' = argmax oracle ba. Report P(true scale ranked <=k). Q: does the soft metric find
the right scale under crop+resize (±regen)? Expected: yes IF aligned ba stays >=~0.7 (contrast)."""
import os,sys,glob,numpy as np
from PIL import Image
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,CF); sys.path.insert(0,CF+"/scripts/defense"); sys.path.insert(0,CF+"/external/videoseal")
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
sb=ShortenedBCH(); vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
pipe=build_regen_pipe()
def embed_central(orig,iid,keep=0.75):
    tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)); p,M=vine.get_perm_M(iid)
    s=int(512*keep); o=(512-s)//2; center=orig.crop((o,o,o+s,o+s))
    marked=vine.embed_with_target(center,apply_crypto(tx,p,M))
    out=orig.copy(); out.paste(marked.resize((s,s)),(o,o)); return out,tx.astype(np.uint8)
def a_crop(x,keep): s=int(512*keep); o=(512-s)//2; return x.crop((o,o,o+s,o+s)).resize((512,512))
def regen(x,j): return to512(stable_regen(pipe,x,7000+j,noise_step=30))
def view_at(pil,k):
    if k>=0.999: return pil
    s=int(512*k); o=(512-s)//2; return pil.crop((o,o,o+s,o+s))
def metrics_at(view,iid,tx):
    p,M=vine.get_perm_M(iid)
    L=np.clip(method_soft_to_codeword_llr(vine.raw_probs(view),p,M,kind="prob",n_codeword=sb.n),-CLAMP,CLAMP)
    energy=float(np.sum(np.abs(L))); highconf=int(np.sum(np.abs(L)>2.0))
    hard=(L>0).astype(np.uint8); data,nerr=sb.decode(hard); consistency=int(nerr) if data is not None else 99
    oracle_ba=float((hard==tx).mean()); return energy,highconf,consistency,oracle_ba
def rank_of(true_idx,scores,higher_better):
    order=np.argsort(-np.asarray(scores) if higher_better else np.asarray(scores))
    return int(np.where(order==true_idx)[0][0])
KEEPS=[round(float(k),3) for k in np.arange(0.60,1.0001,0.02)]   # 21 candidate scales
CONDS={"clean_crop75": lambda wm,j: a_crop(wm,0.75),
       "crop75_regen": lambda wm,j: regen(a_crop(wm,0.75),j),
       "crop90_regen": lambda wm,j: regen(a_crop(wm,0.90),j),
       "regen_only":   lambda wm,j: regen(wm,j)}
N=int(sys.argv[1]) if len(sys.argv)>1 else 30
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
R={c:{m:[] for m in ["energy","highconf","consistency"]} for c in CONDS}; OBA={c:[] for c in CONDS}
for j,fp in enumerate(files):
    iid=f"tkc_{j:05d}"; orig=to512(Image.open(fp).convert("RGB")); wm,tx=embed_central(orig,iid,0.75)
    for c,fn in CONDS.items():
        atk=fn(wm,j); E=[];H=[];C=[];OB=[]
        for k in KEEPS:
            e,h,cs,ob=metrics_at(view_at(atk,k),iid,tx); E.append(e);H.append(h);C.append(cs);OB.append(ob)
        ti=int(np.argmax(OB)); OBA[c].append(OB[ti])
        R[c]["energy"].append(rank_of(ti,E,True)); R[c]["highconf"].append(rank_of(ti,H,True)); R[c]["consistency"].append(rank_of(ti,C,False))
    if (j+1)%10==0: print(f"  [{j+1}/{N}]",flush=True)
def pk(r,k): return float(np.mean(np.asarray(r)<k))
print(f"\n=== top-k CROP+RESIZE scale-search recall (n={N}, {len(KEEPS)} scales, central-VINE) ===")
print(f"{'condition':14s} {'metric':12s} {'top1':>6s} {'top4':>6s} {'top8':>6s}   {'oracle_best_ba':>14s}")
for c in CONDS:
    for m in ["energy","highconf","consistency"]:
        rr=R[c][m]; print(f"{c:14s} {m:12s} {pk(rr,1):6.2f} {pk(rr,4):6.2f} {pk(rr,8):6.2f}   {np.mean(OBA[c]):14.3f}")
    print()
print("TOPKCROP_DONE")
