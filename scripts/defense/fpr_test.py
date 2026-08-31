"""Rigorous FPR test of the DEPLOYED 3-frag composite detector (VINE+TM+VideoSeal +
offline 3-way head + best-path), the exact OR rule in composite_external_eval.py:

    detect = fver  (head-fused codeword crypto-verify, 37-bit exact -> design 2^-37)
          OR fzb   (head-fused codeword zero-bit ba>=tau, tau=0.63 -> design ~0.6%/trial)
          OR bestpath (ANY single-fragment crypto-verify, 37-bit exact -> design ~0)

Threat unit = one (unwatermarked image, claimed image_id) pair. We use UNWATERMARKED COCO
images (disjoint slice) x K random claimed ids (cross-id amplification: decode each image's
raw soft ONCE, then re-align under each id's crypto cheaply) to get N*K trials.

Reports, per detection path, the empirical FPR + Wilson 95% CI, the null fused-ba
distribution (does the LEARNED head distort the null vs equal-MRC?), and a tau sweep.
"""
import os, sys, glob, json
import numpy as np, torch
from PIL import Image
from scipy.stats import binom
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts")]:
    sys.path.insert(0,p)
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.fusion_head3 import load_head3
np.random.seed(0); torch.manual_seed(0)
KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15.0
N=int(sys.argv[1]) if len(sys.argv)>1 else 300         # distinct unwm images
K=int(sys.argv[2]) if len(sys.argv)>2 else 200         # claimed ids per image (zero-bit trials)
KC=int(sys.argv[3]) if len(sys.argv)>3 else 20         # claimed ids per image used for crypto (Chase) trials
sb=ShortenedBCH(); n=sb.n; tau=float(binom.ppf(0.99,n,0.5)+1)/n
def wilson(k,N,z=1.96):
    if N==0: return (0,0)
    p=k/N; d=1+z*z/N; c=p+z*z/(2*N); m=z*((p*(1-p)/N+z*z/(4*N*N))**0.5)
    return ((c-m)/d,(c+m)/d)

vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=n,device=dev)
head=load_head3(os.path.join(REPO,"results/defense/frag3_head.pt"),dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im

# ---- claimed ids: precompute per-fragment (perm,M) + target codeword tx ----
ids=[f"claim_{j:06d}" for j in range(K)]
PM={"v":[],"t":[],"s":[]}; TX=[]
for iid in ids:
    PM["v"].append(vine.get_perm_M(iid)); PM["t"].append(tm.get_perm_M(iid)); PM["s"].append(vsf.get_perm_M(iid))
    TX.append(sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8))
TX=np.array(TX)  # [K,n]

# ---- unwatermarked images (disjoint COCO slice) ----
imgs=sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))[8000:8000+N]
print(f"N={len(imgs)} unwm imgs x K={K} ids = {len(imgs)*K} zero-bit trials; crypto on {len(imgs)*KC} trials. tau={tau:.3f}",flush=True)

# ---- per-image: decode RAW soft once per fragment, then align under all K ids ----
AV=np.zeros((N*K,n),np.float32); AT=np.zeros((N*K,n),np.float32); AS=np.zeros((N*K,n),np.float32)
row=0
for ii,fp in enumerate(imgs):
    im=to512(Image.open(fp).convert("RGB"))
    rv=vine.raw_probs(im); rt=tm.raw_logits(im); rs=vsf.raw_logits(im)   # raw model outputs (id-independent)
    for j in range(K):
        AV[row]=np.clip(method_soft_to_codeword_llr(rv,*PM["v"][j],kind="prob", n_codeword=n),-CLAMP,CLAMP)
        AT[row]=np.clip(method_soft_to_codeword_llr(rt,*PM["t"][j],kind="logit",n_codeword=n),-CLAMP,CLAMP)
        AS[row]=np.clip(method_soft_to_codeword_llr(rs,*PM["s"][j],kind="logit",n_codeword=n),-CLAMP,CLAMP)
        row+=1
    if (ii+1)%50==0: print(f"  decoded {ii+1}/{N}",flush=True)
TXrep=np.tile(TX,(N,1))  # [N*K,n] target per trial

# ---- head-fuse all trials + equal-MRC for comparison ----
with torch.no_grad():
    FUSED=head(torch.tensor(AV,device=dev),torch.tensor(AT,device=dev),torch.tensor(AS,device=dev)).cpu().numpy()
EQ=AV+AT+AS
def ba_of(F): return (((F>0).astype(np.uint8))==TXrep).mean(1)   # per-trial bit-acc vs claimed target
ba_head=ba_of(FUSED); ba_eq=ba_of(EQ); ba_v=ba_of(AV); ba_t=ba_of(AT); ba_s=ba_of(AS)

print("\n=== NULL fused bit-acc distribution (should be ~Binom(100,.5)/100: mean .500 std .050) ===")
for nm,b in [("HEAD",ba_head),("equal-MRC",ba_eq),("VINE",ba_v),("TM",ba_t),("VideoSeal",ba_s)]:
    print(f"  {nm:10s} mean={b.mean():.4f} std={b.std():.4f}  max={b.max():.3f}  P(ba>=0.63)={np.mean(b>=tau):.4f}")

print("\n=== ZERO-BIT FPR (ba>=tau) per path, N*K trials, Wilson 95% CI ===")
M=N*K
for nm,b in [("HEAD-fused (deployed)",ba_head),("equal-MRC",ba_eq),("VINE-only",ba_v),("TM-only",ba_t),("VideoSeal-only",ba_s)]:
    k=int(np.sum(b>=tau)); lo,hi=wilson(k,M)
    print(f"  {nm:24s} {k:5d}/{M} = {k/M:.4f}  CI[{lo:.4f},{hi:.4f}]  (analytic null 0.0060)")

print(f"\n=== tau SWEEP (HEAD-fused zero-bit) ===  {'tau':>6s} {'empFPR':>9s} {'analytic':>9s}")
for q in [0.99,0.999,0.9999]:
    t=(int(binom.ppf(q,n,0.5))+1)/n; emp=np.mean(ba_head>=t); ana=1-binom.cdf(int(binom.ppf(q,n,0.5)),n,0.5)
    print(f"   q={q:<7}  {t:>6.3f} {emp:>9.4f} {ana:>9.2e}")

# ---- crypto-ID FPR (Chase-verify) on subset, exact deployed rule ----
print(f"\n=== CRYPTO-ID FPR (Chase 37-bit exact verify) on N*KC={N*KC} trials ===",flush=True)
fver_k=bp_k=cor_k=0; M2=0
for ii in range(N):
    for j in range(KC):
        r=ii*K+j; iid=ids[j]
        fver=bool(decode_and_verify(FUSED[r],iid,codec=sb)["detected"])
        bp=False
        for F in (AV[r],AT[r],AS[r]):
            if bool(decode_and_verify(F,iid,codec=sb)["detected"]): bp=True; break
        fzb=bool(ba_head[r]>=tau)
        fver_k+=fver; bp_k+=bp; cor_k+=(fver or fzb or bp); M2+=1
    if (ii+1)%50==0: print(f"  crypto {ii+1}/{N}",flush=True)
for nm,k in [("fused crypto-verify (fver)",fver_k),("best-path any-frag crypto (bestpath)",bp_k),("COMPOSITE_OR (fver|fzb|bestpath)",cor_k)]:
    lo,hi=wilson(k,M2); print(f"  {nm:38s} {k:4d}/{M2} = {k/M2:.4f}  CI[{lo:.4f},{hi:.4f}]")

out={"N":N,"K":K,"KC":KC,"tau":tau,"zerobit_head_fpr":float(np.mean(ba_head>=tau)),
     "zerobit_eq_fpr":float(np.mean(ba_eq>=tau)),"null_head_mean":float(ba_head.mean()),
     "null_head_std":float(ba_head.std()),"fver_fpr":fver_k/M2,"bestpath_fpr":bp_k/M2,
     "composite_or_fpr":cor_k/M2,"crypto_trials":M2,"zerobit_trials":M}
json.dump(out,open(os.path.join(REPO,"results/defense/fpr_test.json"),"w"),indent=2)
print("\nFPR_TEST_DONE")
