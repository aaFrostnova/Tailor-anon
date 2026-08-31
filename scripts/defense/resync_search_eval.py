"""Direction A step 2 — does a TRAINING-FREE crypto-verify-guided rotation search reach the
oracle-resync ceiling? For each rotation attack we compare three DETECTION outcomes on the
deployed 3-fragment stack (VINE+TM+VideoSeal), where a fragment 'detects' iff its per-image
crypto-verify (2^-37 false-accept) fires OR its zero-bit bit-acc >= tau:
  (1) baseline    : decode at angle 0 (= current default pipeline; relies on VideoSeal for rot).
  (2) oracle      : rotate back by the EXACT -theta (upper bound any resync could reach).
  (3) blind search: rotate back by d in [-30..30] step 3 deg, accept iff ANY fragment crypto-
                    verifies at ANY angle (zero false-accept => the search is self-validating).
If blind ~ oracle >> baseline, rotation resync needs NO learned estimator: crypto-verify search
suffices, wired as a fallback best-path. Reuses ext_vtv100 (already composite-watermarked)."""
import os, sys, json
import numpy as np
from PIL import Image
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
sb = ShortenedBCH(); n = sb.n
tau = float(binom.ppf(0.99, n, 0.5) + 1) / n          # zero-bit detection threshold (~0.63)
frag = {
 "vine":      (VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev), "raw_probs", "prob"),
 "trustmark": (TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev), "raw_logits", "logit"),
 "videoseal": (VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev), "raw_logits", "logit"),
}
def to512(im): return im.resize((RES, RES)) if im.size != (RES, RES) else im
def rot(img, deg):  # reflect-pad -> rotate (BICUBIC) -> center-crop  (matches RAVEN resync operator)
    a = np.array(img); pad = RES // 2
    big = Image.fromarray(np.pad(a, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    bw, bh = big.size; l = (bw - RES) // 2; return big.crop((l, l, l + RES, l + RES))

def frag_llr(name, img, iid):
    fr, getter, kind = frag[name]
    p, M = fr.get_perm_M(iid)
    return method_soft_to_codeword_llr(getattr(fr, getter)(img), p, M, kind=kind, n_codeword=n)

def detect_at(img, iid, tx):
    """Best-path detection at a fixed geometry: (detected?, which fragment fired). Detect iff any
    fragment crypto-verifies (preferred, zero-FPR) or any fragment's zero-bit bit-acc >= tau."""
    fired = None; det = False
    for name in frag:
        llr = frag_llr(name, img, iid)
        if bool(decode_and_verify(llr, iid, codec=sb)["detected"]):
            return True, name + ":crypto"
        if float(np.mean((llr > 0).astype(np.uint8) == tx)) >= tau:
            det = True; fired = fired or (name + ":zerobit")
    return det, fired

def detect_resync_search(att, iid, tx):
    """Blind angle search; accept iff any fragment crypto-verifies at any candidate angle."""
    for d in np.arange(-30, 30.01, 3.0):
        for name in frag:
            if bool(decode_and_verify(frag_llr(name, rot(att, float(d)), iid), iid, codec=sb)["detected"]):
                return True, f"{name}@{d:+.0f}"
    return False, None

meta = json.load(open(os.path.join(REPO, "results/defense/ext_vtv100/meta.json")))
items = meta["items"][:int(sys.argv[1]) if len(sys.argv) > 1 else 48]
ATTACKS = [9.0, 20.0, 30.0]
R = {a: {"base": [], "oracle": [], "search": [], "carry": {}} for a in ATTACKS}
print(f"resync search vs oracle ceiling, n={len(items)}, tau={tau:.3f}\n", flush=True)
for k, it in enumerate(items):
    iid = it["image_id"]; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
    wm = to512(Image.open(os.path.join(REPO, "results/defense/ext_vtv100", f"img_{it['i']:05d}.png")).convert("RGB"))
    for a in ATTACKS:
        att = rot(wm, a)
        b, _ = detect_at(att, iid, tx)
        o, ocarry = detect_at(rot(att, -a), iid, tx)           # oracle: exact inverse
        s, scarry = (True, None) if b else detect_resync_search(att, iid, tx)  # search only if baseline fails
        R[a]["base"].append(1.0 * b); R[a]["oracle"].append(1.0 * o); R[a]["search"].append(1.0 * s)
        tag = scarry or ocarry
        if tag:
            key = tag.split("@")[0].split(":")[0]; R[a]["carry"][key] = R[a]["carry"].get(key, 0) + 1
    if (k + 1) % 8 == 0: print(f"  [{k+1}/{len(items)}]", flush=True)
print(f"\n{'attack':8s} | {'baseline_det':12s} | {'oracle_det':10s} | {'search_det':10s} | carrier fragment")
print("-" * 78)
for a in ATTACKS:
    car = ", ".join(f"{kk}×{vv}" for kk, vv in sorted(R[a]["carry"].items(), key=lambda x: -x[1]))
    print(f"rot{int(a):<5d} | {np.mean(R[a]['base']):12.3f} | {np.mean(R[a]['oracle']):10.3f} | {np.mean(R[a]['search']):10.3f} | {car}")
print("\n(detection rate; baseline=deployed default @0deg; oracle=exact -theta; search=blind +-30/3 crypto-verify.")
print(" search~=oracle >> baseline  =>  crypto-verify rotation search recovers rotation with NO learned estimator.)")
print("RESYNC_SEARCH_DONE")
