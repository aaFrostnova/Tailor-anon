"""DECISIVE test: is the payload at the border because the DECODER can only read a content-independent message
from the zero-padded frame border (absolute-position anchor)? Isolate the pure message signal P = wm(msg)-wm(null),
then RELOCATE it inward (roll) while keeping the background wm(null) fixed, and decode. If bit-acc collapses as P
moves off the border -> the decoder reads the message specifically from the border position (position-locked to
the frame). Also tells us if a message placed at the CENTER is readable at all."""
import sys, glob, numpy as np
from PIL import Image
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0,"scripts/defense")
import torch
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.soft_fusion import method_soft_to_codeword_llr
KEY=b"v5_key_encoder_master"; sb=ShortenedBCH(); n=sb.n
V=VineCryptoWrapper(master_key=KEY,method_name="vine",n_bits=n,device="cuda")
def to512(im): return im.resize((512,512)) if im.size!=(512,512) else im
def emb(arr,tgt): return np.asarray(to512(V.embed_with_target(Image.fromarray(np.clip(arr,0,255).astype(np.uint8)),tgt)),np.float64)
def ba_of(arr,iid,tx):
    p,M=V.get_perm_M(iid)
    rl=method_soft_to_codeword_llr(V.raw_probs(Image.fromarray(np.clip(arr,0,255).astype(np.uint8))),p,M,kind="prob",n_codeword=n)
    return float(((rl>0).astype(np.uint8)==tx).mean())
def tgt(iid): p,M=V.get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)),p,M)
srcs=sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:8]
shifts=[0,8,16,32,64,128,200]
acc={s:[] for s in shifts}
for j,fp in enumerate(srcs):
    iid=f"wbd_{j}"; tx=sb.encode(image_id_to_payload(iid,n_bits=sb.data_bits)).astype(np.uint8)
    cov=np.asarray(to512(Image.open(fp).convert("RGB")),np.float64)
    wnull=emb(cov,np.zeros(n,dtype=np.uint8)); wmsg=emb(cov,tgt(iid)); P=wmsg-wnull   # pure message signal (mostly border)
    for s in shifts:
        Pr=np.roll(np.roll(P,s,axis=0),s,axis=1)      # relocate the message inward/diagonally by s px (circular)
        acc[s].append(ba_of(wnull+Pr,iid,tx))         # background fixed, message moved
print("RESULT: bit-acc as the message signal is RELOCATED off the border (background fixed)")
for s in shifts:
    print(f"  shift={s:3d}px : bit-acc={np.mean(acc[s]):.3f}   (message center dist-from-edge ~{s}px)")
print("DONE")
