"""Find the minimum scale K for a single nested-VINE layer to RESIST REGEN.
A layer at scale K embeds the 256-native mark upscaled to fill a central (K*512)^2 region.
Upscale factor = 2K (256 -> K*512). Larger K = lower spatial freq at regen's 512 -> more regen-robust.
Sweep K, regen at 512, decode at the KNOWN K (crop central K, resize, VINE decode), report ba + crypto-verify.
Threshold = min K where regen ba clears the identity floor (BCH t=10 -> ba>=~0.90) / presence (0.63)."""
import os,sys,glob,numpy as np
from PIL import Image
CF="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,CF); sys.path.insert(0,CF+"/scripts/defense")
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.soft_bch import decode_and_verify
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen
KEY=b"v5_key_encoder_master"; dev="cuda"
N=int(sys.argv[1]) if len(sys.argv)>1 else 20
sb=ShortenedBCH(); vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
pipe=build_regen_pipe()
def R512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def embed_K(cover,target,K):
    if K>=0.999: return R512(vine.embed_with_target(cover,target))
    s=int(round(512*K)); o=(512-s)//2
    wm=vine.embed_with_target(cover.crop((o,o,o+s,o+s)),target)
    out=cover.copy(); out.paste(wm.resize((s,s)),(o,o)); return out
def decode_K(att,K,pv,Mv,tx,iid):
    if K>=0.999: view=att
    else: s=int(round(512*K)); o=(512-s)//2; view=att.crop((o,o,o+s,o+s))
    llr=method_soft_to_codeword_llr(vine.raw_probs(view),pv,Mv,kind="prob",n_codeword=sb.n)
    ba=float(((llr>0).astype(int)==tx).mean()); det=int(bool(decode_and_verify(llr,iid,codec=sb)["detected"]))
    return ba,det
KS=[0.5,0.6,0.7,0.75,0.8,0.9,1.0]
files=sorted(glob.glob("/scratch/workspace/mingzhel_umass_edu-ablator/ultraedit_10k/source/*.png"))[:N]
res={K:{"clean_ba":[],"regen_ba":[],"regen_det":[]} for K in KS}
for i,fp in enumerate(files):
    iid=f"kt_{i:05d}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)); pv,Mv=vine.get_perm_M(iid); tgt=apply_crypto(tx,pv,Mv)
    cover=R512(Image.open(fp).convert("RGB"))
    for K in KS:
        wm=embed_K(cover,tgt,K)
        cba,_=decode_K(wm,K,pv,Mv,tx,iid)
        rg=stable_regen(pipe,wm,seed=1234+i)
        rba,rdet=decode_K(rg,K,pv,Mv,tx,iid)
        res[K]["clean_ba"].append(cba); res[K]["regen_ba"].append(rba); res[K]["regen_det"].append(rdet)
    if (i+1)%5==0: print(f"  [{i+1}/{N}]",flush=True)
m=lambda x:float(np.mean(x))
print(f"\n=== nested-VINE single-layer: min K to resist REGEN (n={N}) ===")
print(f"{'K':>5s} {'256->R':>8s} {'up x':>5s} {'clean ba':>9s} {'regen ba':>9s} {'regen det':>10s}")
for K in KS:
    print(f"{K:5.2f} {int(512*K):>6d}px {2*K:4.2f}x {m(res[K]['clean_ba']):9.3f} {m(res[K]['regen_ba']):9.3f} {m(res[K]['regen_det']):10.3f}")
print("KT_DONE")
