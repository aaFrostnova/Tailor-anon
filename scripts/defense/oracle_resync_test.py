"""Direction A — decisive UPPER-BOUND test for a geometric resync front-end.
If we KNOW the geometric attack params, invert them exactly, then decode: how much do the
alignment-locked fragments (VINE, TrustMark) recover? This is the CEILING any learned resync
could reach. High ceiling -> train the learned estimator; low -> resync won't help (inversion
artifacts / content loss dominate). Reuses ext_vtv100 composite-watermarked images."""
import os, sys, glob, json
import numpy as np, torch
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO,"scripts")]: sys.path.insert(0,p)
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
KEY=b"v5_key_encoder_master"; dev="cuda"; CLAMP=15.0
sb=ShortenedBCH(); n=sb.n
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=n,model_type="B",device=dev)
vsf=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=n,device=dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def ba(frag,getter,kind,att,iid,tx):
    perm,M=frag.get_perm_M(iid)
    llr=np.clip(method_soft_to_codeword_llr(getattr(frag,getter)(att),perm,M,kind=kind,n_codeword=n),-CLAMP,CLAMP)
    return float(np.mean((llr>0).astype(np.uint8)==tx))
def decode_all(att,iid,tx):
    return (ba(vine,"raw_probs","prob",att,iid,tx), ba(tm,"raw_logits","logit",att,iid,tx), ba(vsf,"raw_logits","logit",att,iid,tx))

# geometric attacks + their EXACT inverse (oracle resync)
def rot(x,a): return x.rotate(a,resample=Image.BILINEAR)
def rot_inv(x,a): return x.rotate(-a,resample=Image.BILINEAR)
def cropzoom(x,r): s=int(512*r); o=(512-s)//2; return x.crop((o,o,o+s,o+s)).resize((512,512))
def cropzoom_inv(x,r):  # un-zoom to crop size, center-pad lost border by edge-replication
    s=int(512*r); small=np.asarray(x.resize((s,s)),np.uint8); o=(512-s)//2
    padded=np.pad(small,((o,512-s-o),(o,512-s-o),(0,0)),mode="edge")
    return Image.fromarray(padded[:512,:512])
ATT={
 "rot9":  (lambda x:rot(x,9),  lambda x:rot_inv(x,9)),
 "rot30": (lambda x:rot(x,30), lambda x:rot_inv(x,30)),
 "crop75":(lambda x:cropzoom(x,0.75), lambda x:cropzoom_inv(x,0.75)),
 "crop50":(lambda x:cropzoom(x,0.50), lambda x:cropzoom_inv(x,0.50)),
}
meta=json.load(open(os.path.join(REPO,"results/defense/ext_vtv100/meta.json")))
items=meta["items"][:64]
print(f"oracle resync ceiling test, n={len(items)}\n")
print(f"{'attack':7s} | {'VINE att→resync':16s} | {'TM att→resync':16s} | {'VideoSeal att→resync':18s}")
print("-"*70)
for name,(fn,inv) in ATT.items():
    A=np.zeros((len(items),3)); R=np.zeros((len(items),3))
    for k,it in enumerate(items):
        iid=it["image_id"]; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
        wm=to512(Image.open(os.path.join(REPO,"results/defense/ext_vtv100",f"img_{it['i']:05d}.png")).convert("RGB"))
        att=to512(fn(wm)); res=to512(inv(att))
        A[k]=decode_all(att,iid,tx); R[k]=decode_all(res,iid,tx)
    a=A.mean(0); r=R.mean(0)
    print(f"{name:7s} | {a[0]:.3f} → {r[0]:.3f}      | {a[1]:.3f} → {r[1]:.3f}      | {a[2]:.3f} → {r[2]:.3f}")
print("\n(bit-acc; 0.50=dead. resync helps iff att→resync jumps up. This is the ORACLE ceiling.)")
print("ORACLE_RESYNC_DONE")
