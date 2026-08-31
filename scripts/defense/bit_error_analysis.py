"""Per-bit error analysis: which codeword/physical bits get destroyed most by attacks?
Two spaces: (A) CODEWORD space (after crypto-align + fusion) — should be UNIFORM if per-image perm whitens;
(B) PHYSICAL space (VINE raw_probs / TM raw_logits per output bit) — may have structure.
Reports per-position error rate + mean|LLR| across images x attacks; saves figure."""
import glob, io, os, sys, tempfile, numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,REPO+"/scripts/defense")
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
RES=512; dev="cuda"; N=50
sb=ShortenedBCH(); n=sb.n
frag={"vine":VineCryptoWrapper(master_key=b"v5_key_encoder_master",method_name="vine",n_bits=n,device=dev),
      "trustmark":TrustMarkFragment(master_key=b"v5_key_encoder_master",method_name="trustmark",n_bits=n,model_type="B",device=dev)}
pipe=build_regen_pipe()
def arr(p): return np.asarray(p.convert("RGB").resize((RES,RES)),np.float32)
def ccr(pil,r): cw=int(RES*r); l=(RES-cw)//2; return pil.crop((l,l,l+cw,l+cw)).resize((RES,RES),Image.BICUBIC)
def jpeg(pil): b=io.BytesIO(); pil.save(b,"JPEG",quality=25); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def attack(name,W,seed):
    if name=="clean": return W
    if name=="jpeg": return jpeg(W)
    if name=="crop75": return ccr(W,0.75)
    if name=="regen": return stable_regen(pipe,W,seed,denoise_steps=8)
ATKS=["clean","jpeg","crop75","regen"]
files=sorted(glob.glob(os.path.join(REPO,"results/defense/raven_gen","*.png")))[:N]
# accumulators per attack
CW=  {a:np.zeros(n) for a in ATKS}; CWabs={a:np.zeros(n) for a in ATKS}
VPH= {a:np.zeros(n) for a in ATKS}; TPH={a:np.zeros(n) for a in ATKS}
cnt={a:0 for a in ATKS}
for idx,fp in enumerate(files):
    C=Image.open(fp).convert("RGB").resize((RES,RES)); iid=f"be_{idx:05d}"
    tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
    pv,Mv=frag["vine"].get_perm_M(iid); pt,Mt=frag["trustmark"].get_perm_M(iid)
    tgt_v=apply_crypto(tx,pv,Mv).astype(int); tgt_t=apply_crypto(tx,pt,Mt).astype(int)
    img=C
    for nm,p_,M_ in [("vine",pv,Mv),("trustmark",pt,Mt)]:
        img=frag[nm].embed_with_target(img,apply_crypto(tx,p_,M_));  img=img if img.size==(RES,RES) else img.resize((RES,RES))
    for a in ATKS:
        At=attack(a,img,1234+idx);  At=At if At.size==(RES,RES) else At.resize((RES,RES))
        vp=frag["vine"].raw_probs(At); tl=frag["trustmark"].raw_logits(At)
        VPH[a]+=((vp>0.5).astype(int)!=tgt_v); TPH[a]+=((tl>0).astype(int)!=tgt_t)
        lv=method_soft_to_codeword_llr(vp,pv,Mv,kind="prob",n_codeword=n); lt=method_soft_to_codeword_llr(tl,pt,Mt,kind="logit",n_codeword=n)
        fused=fuse_llrs({"v":lv,"t":lt},weights=None,n_codeword=n)
        CW[a]+=(llr_to_bits(fused)!=tx); CWabs[a]+=np.abs(fused); cnt[a]+=1
    if (idx+1)%10==0: print(f"[{idx+1}/{N}]",flush=True)
print(f"\n=== per-bit error analysis (n={N}) ===")
print(f"{'attack':<8}  CODEWORD-space err  (mean/std/min/max)        PHYS VINE err(mean/std)   PHYS TM err(mean/std)")
for a in ATKS:
    cw=CW[a]/cnt[a]; vph=VPH[a]/cnt[a]; tph=TPH[a]/cnt[a]
    print(f"{a:<8}  {cw.mean():.3f}/{cw.std():.3f}/{cw.min():.3f}/{cw.max():.3f}     {vph.mean():.3f}/{vph.std():.3f}        {tph.mean():.3f}/{tph.std():.3f}")
# correlation of per-position error across attacks (codeword space): is the SAME bit always weak?
cwmat=np.stack([CW[a]/cnt[a] for a in ATKS if a!="clean"])  # attacks x 100
corr=np.corrcoef(cwmat)
print(f"\ncodeword per-position error correlation across attacks (off-diag mean): {corr[np.triu_indices(len(cwmat),1)].mean():.3f}")
print("  (high => same codeword bits consistently weak; ~0 => no consistent weak bit / crypto whitened)")
# physical: is the SAME physical bit consistently weak?
vmat=np.stack([VPH[a]/cnt[a] for a in ATKS if a!="clean"]); vcorr=np.corrcoef(vmat)
tmat=np.stack([TPH[a]/cnt[a] for a in ATKS if a!="clean"]); tcorr=np.corrcoef(tmat)
print(f"physical VINE per-position error corr across attacks: {vcorr[np.triu_indices(len(vmat),1)].mean():.3f}")
print(f"physical TM   per-position error corr across attacks: {tcorr[np.triu_indices(len(tmat),1)].mean():.3f}")
# figure
fig,ax=plt.subplots(3,1,figsize=(13,9),sharex=True)
for a in ATKS:
    ax[0].plot(CW[a]/cnt[a],label=a,alpha=0.8)
    ax[1].plot(VPH[a]/cnt[a],label=a,alpha=0.8); ax[2].plot(TPH[a]/cnt[a],label=a,alpha=0.8)
ax[0].set_title("CODEWORD-space per-position error (after crypto-align+fusion) — flat = whitened"); ax[0].axhline(0.5,ls=":",c="gray")
ax[1].set_title("PHYSICAL VINE per-bit error"); ax[2].set_title("PHYSICAL TrustMark per-bit error")
for x in ax: x.legend(fontsize=7,ncol=4); x.set_ylabel("error rate"); x.grid(alpha=0.3)
ax[2].set_xlabel("bit position"); fig.tight_layout(); fig.savefig(REPO+"/results/figures/fig_bit_error_analysis.png",dpi=130,bbox_inches="tight")
print("[fig] fig_bit_error_analysis.png\nBIT_ERR_DONE")
