"""False-positive rate of the multi-fragment COMPOSITE detector (parametrized by --fragments).

Detection = fused-BCH-verify OR fused-zerobit(bit-acc>=tau). Measures whether the high
detection on watermarked images is real signal or false positives, on:
  (A) UNWATERMARKED images decoded against a registered image_id (any detection = FP),
      clean and under the (hardest) regen attack;
  (B) WRONG-ID: images watermarked with id_A decoded against id_B (any detection = FP).
Critical for the 4-fused config (vine dft qim trustmark): confirms fusing TrustMark did not
inflate the geometric/zerobit detection via false positives.
"""
import argparse, glob, os, sys, tempfile, json
import numpy as np, torch
from PIL import Image
from scipy.stats import binom

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.phasemark import PhaseMarkWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.learned_fragment_methods import DFTKredMethod, QuantQIMMethod
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload

KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
SPEC = {"phasemark": ("logit", "raw_scores", "target"), "vine": ("prob", "raw_probs", "target"),
        "dft": ("logit", "raw_logits", "id_tx"), "qim": ("logit", "raw_logits", "id_tx"),
        "trustmark": ("logit", "raw_logits", "target")}


def build(fragments, dev, sb, tm_variant="B"):
    b = {
        "phasemark": lambda: PhaseMarkWrapper(master_key=KEY, method_name="phasemark", n_bits=sb.n, vae_key="sd21", device=dev),
        "vine":      lambda: VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev),
        "dft":       lambda: DFTKredMethod(os.path.join(REPO, "results/dft_fftaware_baseline/ckpt.pt"), KEY, "dft_kred", dev),
        "qim":       lambda: QuantQIMMethod(os.path.join(REPO, "results/quant_qim_frozen_d006/ckpt.pt"), KEY, "quant_qim", dev),
        "trustmark": lambda: TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type=tm_variant, device=dev),
    }
    return {n: b[n]() for n in fragments}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments", nargs="+", default=["vine", "dft", "qim", "trustmark"])
    ap.add_argument("--n_clean", type=int, default=200)
    ap.add_argument("--n_wrong", type=int, default=50)
    ap.add_argument("--n_regen", type=int, default=50)
    ap.add_argument("--tm_variant", default="B")
    ap.add_argument("--output", default="results/defense/composite_fpr_4fused.json")
    args = ap.parse_args()
    dev = "cuda"
    sb = ShortenedBCH(); tau = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
    frag = build(args.fragments, dev, sb, args.tm_variant)
    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))

    def detect(img, image_id):
        tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))
        al = {}
        for name in args.fragments:
            kind, getter, _ = SPEC[name]
            m = frag[name]; perm, M = m.get_perm_M(image_id)
            al[name] = method_soft_to_codeword_llr(getattr(m, getter)(img), perm, M, kind=kind, n_codeword=sb.n)
        fused = fuse_llrs(al, weights=None, n_codeword=sb.n)
        fba = float(np.mean(llr_to_bits(fused) == tx))
        fver = float(decode_and_verify(fused, image_id, codec=sb)["detected"])
        fzb = 1.0 if fba >= tau else 0.0
        return (1.0 if (fver or fzb) else 0.0), fver, fzb, fba

    def embed(cover, image_id):
        tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits)); img = cover
        for name in args.fragments:
            m = frag[name]
            if SPEC[name][2] == "id_tx":
                img = m.embed(img, image_id, tx)
            else:
                perm, M = m.get_perm_M(image_id); img = m.embed_with_target(img, apply_crypto(tx, perm, M))
            if img.size != (512, 512): img = img.resize((512, 512))
        return img

    out = {"tau": tau, "fragments": args.fragments}

    # (A) UNWATERMARKED clean
    co, fv, fz, fb = [], [], [], []
    for i in range(args.n_clean):
        cover = Image.open(imgs[5000 + i]).convert("RGB").resize((512, 512))
        c, v, z, b = detect(cover, f"reg_{i:05d}")
        co.append(c); fv.append(v); fz.append(z); fb.append(b)
    out["unwm_clean"] = {"composite_FPR": float(np.mean(co)), "fused_verify_FPR": float(np.mean(fv)),
                         "fused_zerobit_FPR": float(np.mean(fz)), "mean_fused_bitacc": float(np.mean(fb)), "n": args.n_clean}
    print(f"[FPR] unwm clean (n={args.n_clean}): composite={np.mean(co):.3f} fused_ver={np.mean(fv):.3f} "
          f"fused_0bit={np.mean(fz):.3f} mean_ba={np.mean(fb):.3f}", flush=True)

    # (B) UNWATERMARKED + regen
    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    tmp = tempfile.mkdtemp()
    co, fz = [], []
    for i in range(args.n_regen):
        cover = Image.open(imgs[5000 + i]).convert("RGB").resize((512, 512))
        ip = os.path.join(tmp, f"u_{i}.png"); op = os.path.join(tmp, f"u_{i}_o.png")
        cover.save(ip); regen.attack([ip], [op])
        att = Image.open(op).convert("RGB").resize((512, 512))
        c, v, z, b = detect(att, f"reg_{i:05d}"); co.append(c); fz.append(z)
    out["unwm_regen"] = {"composite_FPR": float(np.mean(co)), "fused_zerobit_FPR": float(np.mean(fz)), "n": args.n_regen}
    print(f"[FPR] unwm + regen (n={args.n_regen}): composite={np.mean(co):.3f} fused_0bit={np.mean(fz):.3f}", flush=True)

    # (C) WRONG-ID
    co, fz = [], []
    for i in range(args.n_wrong):
        cover = Image.open(imgs[4000 + i]).convert("RGB").resize((512, 512))
        ida, idb = f"def_{i:05d}", f"def_{(i+7) % args.n_wrong:05d}"
        img = embed(cover, ida)
        c, v, z, b = detect(img, idb); co.append(c); fz.append(z)
    out["wrong_id"] = {"composite_FPR": float(np.mean(co)), "fused_zerobit_FPR": float(np.mean(fz)), "n": args.n_wrong}
    print(f"[FPR] wrong-id (n={args.n_wrong}): composite={np.mean(co):.3f} fused_0bit={np.mean(fz):.3f}", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\n[done] -> {args.output}\nFPR_DONE")


if __name__ == "__main__":
    main()
