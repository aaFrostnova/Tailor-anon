"""Reference-free rot9 rescue: search un-rotation angle, pick by decoder confidence / crypto-verify.

For a rot9-attacked composite (VINE+TrustMark-B), try un-rotating by candidate angles δ; at each,
decode the fused codeword. Selection (no ground truth): pick δ* = argmax mean|fused LLR|; also check
if ANY candidate crypto-verifies (exact 37-bit BCH, ~0 FPR). Report recovered bit-acc + detection.
"""
import argparse, glob, os, sys
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
KEY = b"v5_key_encoder_master"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits")}


def rot(img, deg):
    W, H = img.size; arr = np.array(img); pad = max(W, H) // 2
    refl = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    big = Image.fromarray(refl).rotate(deg, resample=Image.BICUBIC, expand=False)
    bw, bh = big.size; l, t = (bw - W) // 2, (bh - H) // 2
    return big.crop((l, t, l + W, t + H))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_images", type=int, default=15); ap.add_argument("--deg", type=float, default=9.0)
    args = ap.parse_args(); dev = "cuda"; sb = ShortenedBCH(); tau = 0.63
    frag = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev),
            "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)}
    cand = list(np.arange(-13.0, 13.01, 1.0))   # candidate un-rotation angles
    imgs = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n_images]

    def fused_llr(att, iid):
        al = {}
        for name in ["vine", "trustmark"]:
            kind, getter = SPEC[name]; m = frag[name]; perm, M = m.get_perm_M(iid)
            al[name] = method_soft_to_codeword_llr(getattr(m, getter)(att), perm, M, kind=kind, n_codeword=sb.n)
        return fuse_llrs(al, weights=None, n_codeword=sb.n)

    res = {"rot9_ba": [], "search_ba": [], "search_ang": [], "verify": []}
    for i, fp in enumerate(imgs):
        iid = f"rs_{i:05d}"; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
        o = Image.open(fp).convert("RGB").resize((512, 512))
        pv, Mv = frag["vine"].get_perm_M(iid); img = frag["vine"].embed_with_target(o, apply_crypto(tx, pv, Mv)).resize((512, 512))
        pt, Mt = frag["trustmark"].get_perm_M(iid); img = frag["trustmark"].embed_with_target(img, apply_crypto(tx, pt, Mt)).resize((512, 512))
        att = rot(img, args.deg)
        # no-search baseline
        res["rot9_ba"].append(float(np.mean(llr_to_bits(fused_llr(att, iid)) == tx)))
        # search: un-rotate by each candidate, pick max mean|LLR|; also track any crypto-verify
        best_conf, best = -1, None; any_verify = False
        for d in cand:
            f = fused_llr(rot(att, d), iid)
            conf = float(np.abs(f).mean())
            if decode_and_verify(f, iid, codec=sb)["detected"]: any_verify = True
            if conf > best_conf: best_conf, best = conf, (np.mean(llr_to_bits(f) == tx), d)
        res["search_ba"].append(float(best[0])); res["search_ang"].append(float(best[1])); res["verify"].append(1.0 if any_verify else 0.0)
        print(f"  [{i+1}/{len(imgs)}] rot9={res['rot9_ba'][-1]:.2f} -> search={best[0]:.2f} @un-rot {best[1]:+.0f}° verify={any_verify}", flush=True)

    print(f"\n=== reference-free rot{args.deg:.0f}° rescue via angle-search (VINE+TrustMark-B, n={len(imgs)}) ===")
    print(f"rot9 no-search   fused_ba = {np.mean(res['rot9_ba']):.3f}")
    print(f"rot9 + search    fused_ba = {np.mean(res['search_ba']):.3f}   (picked un-rot {np.mean(res['search_ang']):+.1f}° avg)")
    print(f"rot9 + search    crypto-verify (any angle) detect = {np.mean(res['verify']):.3f}")
    print("ROT_SEARCH_DONE")


if __name__ == "__main__":
    main()
