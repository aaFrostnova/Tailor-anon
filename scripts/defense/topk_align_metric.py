"""Does an ID-INDEPENDENT alignment metric rank the true pose in top-k — including under REGEN?
Tests option (a)/top-k for the search-multiplicity FPR fix. VINE fragment (regen survivor).
For each image: embed VINE (crypto-whitened, known id), attack, grid-search rotation poses;
per pose compute id-INDEPENDENT metrics {LLR energy, high-conf count, codeword-consistency}
and (eval-only, uses id) the oracle bit-acc. 'true pose' = argmax oracle bit-acc.
Report P(true pose ranked <=k by metric) for k=1,4,8, clean vs regen. If regen keeps top-8 high,
metric-guided top-k retains accept-on-any recall while restoring FPR<=2^-37*k."""
import os,sys,glob,numpy as np
from PIL import Image
import torchvision.transforms.functional as TF, torch
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,CF); sys.path.insert(0,CF+"/scripts/defense"); sys.path.insert(0,CF+"/external/videoseal")
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
KEY=b"v5_key_encoder_master"; dev="cuda"; RES=512
def to(im): return im.resize((RES,RES)) if im.size!=(RES,RES) else im
def p2l(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
def align1d(raw,perm,M): return np.asarray(M)[perm]*np.asarray(raw)[perm]
def _t(pil): return TF.to_tensor(to(pil)).unsqueeze(0).to(dev)
def _p(t): return Image.fromarray((t[0].clamp(0,1).permute(1,2,0)*255).byte().cpu().numpy())
def rot(pil,a): return _p(TF.rotate(_t(pil),float(a)))
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
pipe=build_regen_pipe()
def metrics_at(cand_pil,perm,M,tx):
    raw=vine.raw_probs(cand_pil); Lraw=p2l(raw)                 # 100 raw LLRs
    L=align1d(Lraw,perm,M)                                       # un-whitened codeword LLR
    energy=float(np.sum(np.abs(Lraw)))                           # id-INDEP (perm/sign don't change |.|)
    highconf=int(np.sum(np.abs(Lraw)>2.0))                      # id-INDEP
    hard=(L>0).astype(np.uint8); data,nerr=sb.decode(hard)
    consistency=int(nerr) if data is not None else 99            # id-INDEP codeword membership (low=aligned)
    oracle_ba=float(((L>0).astype(int)==tx).mean())             # EVAL ONLY (uses tx)
    return energy,highconf,consistency,oracle_ba
def rank_of(true_idx, scores, higher_better):
    order=np.argsort(-np.asarray(scores) if higher_better else np.asarray(scores))
    return int(np.where(order==true_idx)[0][0])                 # 0-based rank
N=int(sys.argv[1]) if len(sys.argv)>1 else 30
ANG=list(range(-45,46,3))                                        # 31 candidate poses
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
CONDS=["clean_rot30","rot30_regen","regen_only"]
R={c:{m:[] for m in ["energy","highconf","consistency"]} for c in CONDS}
OBA={c:[] for c in CONDS}                                        # oracle best-pose bit-acc
for i,fp in enumerate(files):
    iid=f"tk_{i:05d}"; pay=image_id_to_payload(iid,n_bits=sb.data_bits); tx=sb.encode(pay)
    perm,M=vine.get_perm_M(iid)
    orig=to(Image.open(fp).convert("RGB"))
    wm=to(vine.embed_with_target(orig,apply_crypto(tx,perm,M)))
    atk={}
    atk["clean_rot30"]=rot(wm,30)
    atk["rot30_regen"]=to(stable_regen(pipe,rot(wm,30),1234+i))
    atk["regen_only"]=to(stable_regen(pipe,wm,1234+i))
    for c in CONDS:
        E=[];H=[];C=[];OB=[]
        for a in ANG:
            e,h,cs,ob=metrics_at(rot(atk[c],a),perm,M,tx)        # apply candidate rotation a
            E.append(e);H.append(h);C.append(cs);OB.append(ob)
        true_idx=int(np.argmax(OB))                              # best-aligned pose (oracle)
        OBA[c].append(OB[true_idx])
        R[c]["energy"].append(rank_of(true_idx,E,True))
        R[c]["highconf"].append(rank_of(true_idx,H,True))
        R[c]["consistency"].append(rank_of(true_idx,C,False))
    if (i+1)%10==0: print(f"  [{i+1}/{N}]",flush=True)
def pk(ranks,k): return float(np.mean(np.asarray(ranks)<k))
print(f"\n=== top-k alignment-metric recall (n={N}, {len(ANG)} poses) ===")
print("P(true pose ranked < k by an ID-INDEPENDENT metric):")
print(f"{'condition':13s} {'metric':12s} {'top1':>6s} {'top4':>6s} {'top8':>6s}   {'oracle_best_ba':>14s}")
for c in CONDS:
    for m in ["energy","highconf","consistency"]:
        rr=R[c][m]
        print(f"{c:13s} {m:12s} {pk(rr,1):6.2f} {pk(rr,4):6.2f} {pk(rr,8):6.2f}   {np.mean(OBA[c]):14.3f}")
    print()
print("TOPK_DONE")
