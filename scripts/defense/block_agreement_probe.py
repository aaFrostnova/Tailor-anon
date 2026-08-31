"""DECISIVE TEST for the localization-weighted-decode idea.

Question: after an attack, is the BIT damage spatially structured enough that decoding from the
surviving regions beats decoding from the whole image? (Pixel-damage concentration was measured at
~2x uniform, but adversarial attacks perceptually mask into texture, which need NOT coincide with
where the watermark bits live -- so pixel concentration does not imply BIT concentration.)

Method: leave-one-block-out (LOBO) influence.
  For a 4x4 grid, neutralize block i (replace with a heavy local blur -> removes mark+detail there
  without injecting sharp structure) and recompute bit-agreement vs the KNOWN target codeword.
    influence_i = ba(neutralized_i) - ba(full)      ( >0  => block i was HURTING = damaged )
  Then greedily neutralize the top-k hurting blocks (k=1..8) and check whether ba rises and whether
  crypto-verify fires. This is the hard-selection upper bound on what a localization-weighted
  decoder could achieve; it deliberately ignores the FPR selection-bias (a feasibility ceiling)."""
import os, sys, glob, json, argparse, numpy as np
from PIL import Image, ImageFilter
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SC = "/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks"
F = {"trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev)}
SPEC = {"trustmark": ("logit", "raw_logits"), "vine": ("prob", "raw_probs")}
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def llr(nm, pil, iid):
    kind, getter = SPEC[nm]; p, M = F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm], getter)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP)
def ba_of(nm, pil, iid, tx):
    rl = llr(nm, pil, iid); return float(((rl > 0).astype(np.uint8) == tx).mean()), bool(decode_and_verify(rl, iid, codec=sb)["detected"])
G = 4; BS = 512 // G
def neutralize(arr, blocks):
    """replace the given blocks with a heavy local blur (kills mark+detail, adds no structure)."""
    out = Image.fromarray(arr.copy())
    blur = out.filter(ImageFilter.GaussianBlur(12)); ba_ = np.asarray(blur); o = np.asarray(out).copy()
    for (r, c) in blocks: o[r*BS:(r+1)*BS, c*BS:(c+1)*BS] = ba_[r*BS:(r+1)*BS, c*BS:(c+1)*BS]
    return Image.fromarray(o)

def load_set(name):
    if name == "unmarker":
        meta = json.load(open(f"{SC}/p3embed_Dum/meta.json"))["items"]
        return [(m["iid"], f"{SC}/p3um_D/{m['fname']}") for m in meta if os.path.exists(f"{SC}/p3um_D/{m['fname']}")]
    if name.startswith("ctrlregen"):
        tag = name.split("_")[1]; meta = json.load(open(f"{SC}/p3embed_D/meta.json"))["items"]
        d = f"{SC}/p3cr_D_s{tag}"
        return [(m["iid"], f"{d}/{m['fname']}") for m in meta if os.path.exists(f"{d}/{m['fname']}")]
    if name == "offcenter_c75":       # generate: corner-anchored 75% crop of the clean embed
        meta = json.load(open(f"{SC}/p3embed_D/meta.json"))["items"]
        out = f"{SC}/oc75_corner"; os.makedirs(out, exist_ok=True); res = []
        for m in meta:
            src = f"{SC}/p3embed_D/{m['fname']}"
            if not os.path.exists(src): continue
            im = to512(Image.open(src).convert("RGB")); s = 384; o = 128
            im.crop((o, o, o+s, o+s)).resize((512, 512)).save(f"{out}/{m['fname']}")
            res.append((m["iid"], f"{out}/{m['fname']}"))
        return res
    raise SystemExit("unknown set")

ap = argparse.ArgumentParser(); ap.add_argument("--sets", nargs="+", default=["unmarker", "ctrlregen_09", "offcenter_c75"])
a = ap.parse_args()
report = {}
for setname in a.sets:
    items = load_set(setname)
    for nm in ("trustmark", "vine"):
        base_ba, best_ba, gain, infl_cv, fired_base, fired_best = [], [], [], [], [], []
        for iid, path in items:
            tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
            arr = np.asarray(to512(Image.open(path).convert("RGB")))
            b0, f0 = ba_of(nm, Image.fromarray(arr), iid, tx)
            infl = np.zeros((G, G))
            for r in range(G):
                for c in range(G):
                    bi, _ = ba_of(nm, neutralize(arr, [(r, c)]), iid, tx); infl[r, c] = bi - b0
            order = [tuple(x) for x in np.dstack(np.unravel_index(np.argsort(-infl, axis=None), (G, G)))[0]]
            bb, ff = b0, f0
            for k in range(1, 9):                       # greedy occlusion of the top-k hurting blocks
                bk, fk = ba_of(nm, neutralize(arr, order[:k]), iid, tx)
                if bk > bb: bb, ff = bk, fk
            base_ba.append(b0); best_ba.append(bb); gain.append(bb - b0)
            infl_cv.append(float(infl.std() / (abs(infl).mean() + 1e-9)))
            fired_base.append(1.0 if f0 else 0.0); fired_best.append(1.0 if ff else 0.0)
        report[f"{setname}/{nm}"] = {
            "n": len(items), "ba_full": round(float(np.mean(base_ba)), 4), "ba_best_region": round(float(np.mean(best_ba)), 4),
            "mean_gain": round(float(np.mean(gain)), 4), "influence_map_cv": round(float(np.mean(infl_cv)), 3),
            "verify_full": round(float(np.mean(fired_base)), 3), "verify_after_occlusion": round(float(np.mean(fired_best)), 3)}
        r = report[f"{setname}/{nm}"]
        print(f"{setname:16} {nm:10} ba_full={r['ba_full']:.3f} -> ba_best={r['ba_best_region']:.3f} "
              f"(gain {r['mean_gain']:+.3f})  infl_CV={r['influence_map_cv']:.2f}  "
              f"verify {r['verify_full']:.2f} -> {r['verify_after_occlusion']:.2f}", flush=True)
json.dump(report, open("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/block_agreement.json", "w"), indent=2)
print("BLOCK_AGREEMENT_DONE")
