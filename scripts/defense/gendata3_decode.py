"""Decode VINE+TM+VideoSeal on externally-attacked composite images -> aligned codeword LLRs
+ 64x64 image, appended into a strong-attack npz (merged with cheap npz later).
Usage: gendata3_decode.py --out X.npz --dirs DIR1:label1 DIR2:label2 ..."""
import os,sys,glob,argparse
import numpy as np
from PIL import Image
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"external/videoseal"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--dirs",nargs="+",required=True)
a=ap.parse_args(); KEY=b"v5_key_encoder_master"; dev="cuda"
sb=ShortenedBCH()
vine=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=sb.n,device=dev)
tm=TrustMarkFragment(master_key=KEY,method_name="trustmark",n_bits=sb.n,model_type="B",device=dev)
vs=VideoSealFragment(master_key=KEY,method_name="videoseal",n_bits=sb.n,device=dev)
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
AV,AT,AS,TX,ATK,IMG,IMGD=[],[],[],[],[],[],[]
for spec in a.dirs:
    dd,label=spec.rsplit(":",1)
    fs=sorted(glob.glob(os.path.join(dd,"*.png")))
    print(f"[{label}] {len(fs)} imgs from {dd}",flush=True)
    for fp in fs:
        i=int(os.path.basename(fp)[:5]); iid=f"fh3_{i:05d}"
        tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits))
        pv,Mv=vine.get_perm_M(iid); pt,Mt=tm.get_perm_M(iid); ps,Ms=vs.get_perm_M(iid)
        att=to512(Image.open(fp).convert("RGB"))
        av=method_soft_to_codeword_llr(vine.raw_probs(att),pv,Mv,kind="prob",n_codeword=sb.n)
        at=method_soft_to_codeword_llr(tm.raw_logits(att),pt,Mt,kind="logit",n_codeword=sb.n)
        as_=method_soft_to_codeword_llr(vs.raw_logits(att),ps,Ms,kind="logit",n_codeword=sb.n)
        AV.append(av.astype(np.float32)); AT.append(at.astype(np.float32)); AS.append(as_.astype(np.float32))
        TX.append(tx.astype(np.uint8)); ATK.append(label); IMG.append(i)
        IMGD.append(np.asarray(att.resize((64,64)),np.uint8))
np.savez_compressed(a.out,AV=np.array(AV),AT=np.array(AT),AS=np.array(AS),TX=np.array(TX),
                    ATK=np.array(ATK),IMG=np.array(IMG),IMGD=np.array(IMGD,dtype=np.uint8),n=sb.n)
print(f"[saved] {len(AV)} strong-attack samples -> {a.out}\nDECODE_DONE")
