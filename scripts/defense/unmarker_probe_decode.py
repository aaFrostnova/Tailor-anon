"""Decisive test: does the re-embed probe work on UnMarker (alignment ~preserved, mark destroyed)?

Reads UnMarked watermarked vs clean images, computes the round-1 re-embed probe + direct detection + surv.
Compares probe AUROC against the in-data references (vae_c 0.91 aligned/survived; regen 0.645 aligned/partly-killed;
geometric ~0.47 misaligned). Prediction (4 lenses): probe FAILS (~0.5) because the discount tracks mark-survival.
"""
import glob, os, sys
import numpy as np
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO)
from src.vine_crypto_wrapper import VineCryptoWrapper
KEY = b"v5_key_encoder_master"; RES = 512
WMA = os.path.join(REPO, "results/defense/unm_probe_wm_atk")
CLA = os.path.join(REPO, "results/defense/unm_probe_clean_atk")


def arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def rms(x): return float(np.sqrt(np.mean(x ** 2)))
def emb(vine, a, iid):
    W = vine.embed(Image.fromarray(a.astype(np.uint8)), iid)
    return arr(W if W.size == (RES, RES) else W.resize((RES, RES)))
def auroc(pos, neg):
    c = 0.0
    for a in pos:
        for b in neg: c += (a < b) + 0.5 * (a == b)
    return c / (len(pos) * len(neg) + 1e-9)


def main():
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=100, device="cuda")
    n = len(glob.glob(os.path.join(WMA, "*.png")))
    iwr, iww, icr, baw, bac, surv = [], [], [], [], [], []
    for i in range(n):
        rid = f"unm_{i:05d}"; wid = f"WR_{i:05d}"
        Aw = arr(Image.open(os.path.join(WMA, f"img_{i:05d}.png")))
        Ac = arr(Image.open(os.path.join(CLA, f"img_{i:05d}.png")))
        iwr.append(rms(emb(vine, Aw, rid) - Aw)); iww.append(rms(emb(vine, Aw, wid) - Aw))
        icr.append(rms(emb(vine, Ac, rid) - Ac))
        baw.append(vine.detect(Image.fromarray(Aw.astype(np.uint8)), rid)["bit_accuracy"])
        bac.append(vine.detect(Image.fromarray(Ac.astype(np.uint8)), rid)["bit_accuracy"])
        surv.append(rms(Aw - Ac))
        print(f"  img{i:05d}: ba_w={baw[-1]:.2f} iwr={iwr[-1]:.2f} iww={iww[-1]:.2f} icr={icr[-1]:.2f} surv={surv[-1]:.1f}", flush=True)

    au = auroc(iwr, icr)
    print(f"\n=== UnMarker re-embed probe (pure VINE, n={n}) ===")
    print(f"direct detect:  ba_w={np.mean(baw):.3f}  ba_c={np.mean(bac):.3f}")
    print(f"re-embed incr:  iwr={np.mean(iwr):.2f}  iww={np.mean(iww):.2f}  icr={np.mean(icr):.2f}")
    print(f"probe AUROC  =  {au:.3f}   surv_rms={np.sqrt(np.mean(np.array(surv)**2)):.1f}")
    print(f"\nreference: vae_c(aligned,survived)=0.907 | regen(aligned,partly-killed)=0.645 | geometric(misaligned)~0.47")
    verdict = ("PROBE WORKS on UnMarker (alignment alone suffices => probe has a niche)" if au >= 0.75
               else "PROBE FAILS on UnMarker (~regen/chance => mark-survival is the decisive variable; direction closed)"
               if au <= 0.66 else "AMBIGUOUS")
    print(f"VERDICT: {verdict}")
    print("UNMARKER_PROBE_DONE")


if __name__ == "__main__":
    main()
