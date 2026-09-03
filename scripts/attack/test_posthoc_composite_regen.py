"""Does a PhaseMark+VINE post-hoc composite detect under regeneration?

The old post-hoc fused tier (VINE+DFT+QIM) collapses under generative regeneration
(fused detect 0.07 at regen_30). VINE and PhaseMark each retain ~0.66-0.77 bit-acc there
-- individually too noisy to BCH-decode (t=10), but they live in DIFFERENT subspaces
(SDXL-latent residual vs SD-VAE-latent phase) and carry the SAME codeword under different
crypto keys, so soft-LLR fusion of the two independent regen-survivors may cross the
decode threshold where each alone fails.

Reports, per attack: per-method bit-acc + the FUSED bit-acc (after LLR alignment+fusion),
the BCH-verify detection (decode_and_verify, exact id match), and the zero-bit detection
(bit-acc >= 0.62 ~ FPR 1% for 100 bits). Writes a persistent JSON.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from src.shortened_bch import ShortenedBCH                                   # noqa: E402
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto          # noqa: E402
from src.phasemark import PhaseMarkWrapper                                   # noqa: E402
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits  # noqa: E402
from src.soft_bch import decode_and_verify                                   # noqa: E402
from src.payload import image_id_to_payload                                  # noqa: E402

ZB_TAU = 0.62   # zero-bit detection threshold for 100 bits at ~1% FPR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=10)
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--attacks", nargs="+", default=[
        "clean", "vae", "regen_10_sd15", "regen_20_sd15", "regen_30_sd15",
        "flux_i2i_20", "flux_i2i_30", "rinse_2"])
    p.add_argument("--output", default="results/attack/posthoc_composite_regen.json")
    args = p.parse_args()

    sb = ShortenedBCH()
    key = b"v5_key_encoder_master"
    vine = VineCryptoWrapper(master_key=key, method_name="vine", n_bits=sb.n, device="cuda")
    pm = PhaseMarkWrapper(master_key=key, method_name="phasemark", n_bits=sb.n, vae_key="sd21", device="cuda")
    from benchmark_fused import apply_attack

    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))
    imgs = imgs[args.start_idx:args.start_idx + args.n_images]
    print(f"[composite-regen] n={len(imgs)}, fusing VINE(SDXL-latent) + PhaseMark(VAE-latent-phase)", flush=True)

    def tx_for(image_id):
        return sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))

    R = {a: {"vine": [], "pm": [], "fused": [], "vine_det": [], "pm_det": [], "fused_det": [],
             "zerobit_or": []} for a in args.attacks}
    coexist = {"vine": [], "pm": []}

    for i, fp in enumerate(imgs):
        image_id = f"comp_{i:05d}"
        tx = tx_for(image_id)
        img = Image.open(fp).convert("RGB").resize((512, 512))
        # embed VINE then PhaseMark on top (cross-subspace)
        pv, Mv = vine.get_perm_M(image_id)
        w = vine.embed_with_target(img, apply_crypto(tx, pv, Mv))
        pp, Mp = pm.get_perm_M(image_id)
        w = pm.embed_with_target(w, apply_crypto(tx, pp, Mp))
        if w.size != (512, 512):
            w = w.resize((512, 512))
        # clean coexistence
        coexist["vine"].append(float(np.mean(llr_to_bits(method_soft_to_codeword_llr(
            vine.raw_probs(w), pv, Mv, kind="prob", n_codeword=sb.n)) == tx)))
        coexist["pm"].append(float(np.mean(llr_to_bits(method_soft_to_codeword_llr(
            pm.raw_scores(w), pp, Mp, kind="logit", n_codeword=sb.n)) == tx)))

        for a in args.attacks:
            att = apply_attack(a, w)
            if att.size != (512, 512):
                att = att.resize((512, 512))
            lv = method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n)
            lp = method_soft_to_codeword_llr(pm.raw_scores(att), pp, Mp, kind="logit", n_codeword=sb.n)
            fused = fuse_llrs({"vine": lv, "phasemark": lp}, weights=None, n_codeword=sb.n)
            bv = float(np.mean(llr_to_bits(lv) == tx))
            bp = float(np.mean(llr_to_bits(lp) == tx))
            bf = float(np.mean(llr_to_bits(fused) == tx))
            R[a]["vine"].append(bv); R[a]["pm"].append(bp); R[a]["fused"].append(bf)
            R[a]["vine_det"].append(float(decode_and_verify(lv, image_id, codec=sb)["detected"]))
            R[a]["pm_det"].append(float(decode_and_verify(lp, image_id, codec=sb)["detected"]))
            R[a]["fused_det"].append(float(decode_and_verify(fused, image_id, codec=sb)["detected"]))
            R[a]["zerobit_or"].append(1.0 if max(bv, bp, bf) >= ZB_TAU else 0.0)
        print(f"  [{i+1}/{len(imgs)}] coexist vine={coexist['vine'][-1]:.2f} pm={coexist['pm'][-1]:.2f}", flush=True)

    def m(x):
        return float(np.mean(x)) if x else None

    out = {"n_images": len(imgs),
           "clean_coexist": {"vine": m(coexist["vine"]), "phasemark": m(coexist["pm"])},
           "attacks": {a: {k: m(R[a][k]) for k in R[a]} for a in args.attacks}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)

    print(f"\nclean coexistence: VINE {out['clean_coexist']['vine']:.3f}  PhaseMark {out['clean_coexist']['phasemark']:.3f}")
    print(f"\n{'attack':<15}{'vine_ba':>8}{'pm_ba':>7}{'fuse_ba':>8}{'v_det':>6}{'p_det':>6}{'f_det':>6}{'OR0bit':>7}")
    for a in args.attacks:
        c = out["attacks"][a]
        print(f"  {a:<13}{c['vine']:>8.3f}{c['pm']:>7.3f}{c['fused']:>8.3f}"
              f"{c['vine_det']:>6.2f}{c['pm_det']:>6.2f}{c['fused_det']:>6.2f}{c['zerobit_or']:>7.2f}")
    print(f"\n[done] -> {args.output}")
    print("ba=bit-acc, det=BCH-verify detection, OR0bit=zero-bit detection (max bit-acc>=0.62).")


if __name__ == "__main__":
    main()
