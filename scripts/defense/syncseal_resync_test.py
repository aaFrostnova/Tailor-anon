"""Plug-and-play test of Meta SyncSeal (arXiv 2509.15208) as a LEARNED geometric resync front-end
for our composite (VINE+TM+VideoSeal). SyncSeal embeds its own sync watermark, detects the 4 image
corners under geometric attack, and unwarps (perspective-rectifies) the frame. Pipeline tested:
  composite-watermarked W  --embed sync-->  Ws  --geo attack-->  A  --detect corners + unwarp-->  R
then decode OUR 3 fragments on A (baseline) vs R (sync-rectified); detection = ANY fragment crypto-
verifies (2^-37). Measures: (1) does adding the sync mark poison the payload (clean row)? (2) does
one-shot unwarp reach the oracle ceiling on rotation AND crop (the scale gap the angle-search can't)?
(3) fidelity cost of the sync layer. Compares to oracle numbers (resync-oracle-ceiling.md)."""
import os, sys, json
import numpy as np, torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO, "scripts")]: sys.path.insert(0, p)
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.soft_bch import decode_and_verify

KEY = b"v5_key_encoder_master"; dev = "cuda"; RES = 512
JIT = "/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_ckpt/syncmodel.jit.pt"
sb = ShortenedBCH(); n = sb.n
tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
sync = torch.jit.load(JIT).to(dev).eval()
frag = {
 "vine":      (VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev), "raw_probs", "prob"),
 "trustmark": (TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev), "raw_logits", "logit"),
 "videoseal": (VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev), "raw_logits", "logit"),
}
def to512(im): return im.resize((RES, RES)) if im.size != (RES, RES) else im
def pil_of(t): return Image.fromarray((t[0].clamp(0,1).permute(1,2,0)*255).round().byte().cpu().numpy())
def pt_of(pil): return to_tensor(pil).unsqueeze(0).to(dev)

def frag_llr(name, pil, iid):
    fr, getter, kind = frag[name]; p, M = fr.get_perm_M(iid)
    return method_soft_to_codeword_llr(getattr(fr, getter)(pil), p, M, kind=kind, n_codeword=n)
def detect(pil, iid, tx):
    """(any-fragment crypto-verify?, dict of per-frag bit-acc)"""
    det = False; ba = {}
    for name in frag:
        llr = frag_llr(name, pil, iid); ba[name] = float(np.mean((llr > 0).astype(np.uint8) == tx))
        if bool(decode_and_verify(llr, iid, codec=sb)["detected"]): det = True
    return det, ba

# geometric attacks (torchvision operators, matching SyncSeal's training family)
def cropzoom(t, r): s = int(RES*r); o = (RES-s)//2; return TF.resize(t[:, :, o:o+s, o:o+s], [RES, RES], antialias=True)
ATT = {
 "clean":  lambda t: t,
 "rot9":   lambda t: TF.rotate(t, 9.0),
 "rot30":  lambda t: TF.rotate(t, 30.0),
 "crop75": lambda t: cropzoom(t, 0.75),
 "crop50": lambda t: cropzoom(t, 0.50),
}
meta = json.load(open(os.path.join(REPO, "results/defense/ext_vtv100/meta.json")))
items = meta["items"][:int(sys.argv[1]) if len(sys.argv) > 1 else 48]
print(f"SyncSeal plug-and-play resync test, n={len(items)}, tau={tau:.3f}", flush=True)
psnrs = []
R = {a: {"base_det": [], "sync_det": [], "score": [],
         "ba_att": {k: [] for k in frag}, "ba_rect": {k: [] for k in frag}} for a in ATT}
for k, it in enumerate(items):
    iid = it["image_id"]; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
    W = to512(Image.open(os.path.join(REPO, "results/defense/ext_vtv100", f"img_{it['i']:05d}.png")).convert("RGB"))
    xW = pt_of(W)
    with torch.no_grad():
        Ws = sync.embed(xW)["imgs_w"]                     # add sync mark on top of composite
    psnrs.append(10*np.log10(1.0/max(1e-10, torch.mean((Ws-xW)**2).item())))
    for a, fn in ATT.items():
        with torch.no_grad():
            A = fn(Ws)
            d = sync.detect(A); pts = d["preds_pts"]; score = float(d["preds"][0, 0])
            Rimg = sync.unwarp(A, pts, (RES, RES))
        pilA = pil_of(A); pilR = pil_of(Rimg)
        bd, baA = detect(pilA, iid, tx); sd, baR = detect(pilR, iid, tx)
        R[a]["base_det"].append(1.0*bd); R[a]["sync_det"].append(1.0*sd); R[a]["score"].append(score)
        for name in frag: R[a]["ba_att"][name].append(baA[name]); R[a]["ba_rect"][name].append(baR[name])
    if (k+1) % 8 == 0: print(f"  [{k+1}/{len(items)}]", flush=True)

print(f"\nsync-mark fidelity vs composite: PSNR {np.mean(psnrs):.2f} dB (n={len(psnrs)})")
print(f"\n{'attack':7s} | {'base_det':8s} | {'sync_det':8s} | {'syncScore':9s} | per-frag bit-acc  att -> sync-rectified (V / T / S)")
print("-"*104)
for a in ATT:
    baA = [np.mean(R[a]['ba_att'][k]) for k in frag]; baR = [np.mean(R[a]['ba_rect'][k]) for k in frag]
    print(f"{a:7s} | {np.mean(R[a]['base_det']):8.3f} | {np.mean(R[a]['sync_det']):8.3f} | {np.mean(R[a]['score']):9.3f} | "
          f"V {baA[0]:.2f}->{baR[0]:.2f}  T {baA[1]:.2f}->{baR[1]:.2f}  S {baA[2]:.2f}->{baR[2]:.2f}")
print("\n(base_det=decode attacked directly; sync_det=decode after SyncSeal detect+unwarp; both = ANY-fragment crypto-verify.")
print(" oracle refs: rot30 TM 0.97, crop75 TM 0.95/VS 0.88, crop50 TM 0.67/VS 0.72, VINE crop=info-loss floor 0.49.)")
print("SYNCSEAL_TEST_DONE")
