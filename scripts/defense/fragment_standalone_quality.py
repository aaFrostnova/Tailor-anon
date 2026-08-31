"""Per-fragment STANDALONE quality: embed ONLY that fragment, measure PSNR/SSIM/LPIPS vs clean.
Answers 'which watermark hurts image quality most' fairly (no embedding-order confound).
Includes PhaseMark (dropped) for context."""
import argparse, glob, os, sys
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.phasemark import PhaseMarkWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.learned_fragment_methods import DFTKredMethod, QuantQIMMethod
from src.payload import image_id_to_payload
KEY = b"v5_key_encoder_master"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"


def psnr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    m = np.mean((a - b) ** 2); return 10 * np.log10(255 ** 2 / m) if m > 1e-9 else 99.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_images", type=int, default=20)
    args = ap.parse_args(); dev = "cuda"; sb = ShortenedBCH()
    frag = {
        "VINE": (VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev), "target"),
        "DFT-Kred": (DFTKredMethod(os.path.join(REPO, "results/dft_fftaware_baseline/ckpt.pt"), KEY, "dft_kred", dev), "id_tx"),
        "Quant-QIM": (QuantQIMMethod(os.path.join(REPO, "results/quant_qim_frozen_d006/ckpt.pt"), KEY, "quant_qim", dev), "id_tx"),
        "TrustMark-B": (TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev), "target"),
        "PhaseMark(dropped)": (PhaseMarkWrapper(master_key=KEY, method_name="phasemark", n_bits=sb.n, vae_key="sd21", device=dev), "target"),
    }
    import lpips as L; from pytorch_msssim import ssim as SS
    lp = L.LPIPS(net="alex").to(dev).eval()
    def t(p): return torch.from_numpy(np.asarray(p.convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].to(dev)
    imgs = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n_images]
    acc = {k: {"p": [], "s": [], "l": []} for k in frag}
    for i, fp in enumerate(imgs):
        iid = f"q_{i:05d}"; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
        o = Image.open(fp).convert("RGB").resize((512, 512))
        for name, (m, mode) in frag.items():
            if mode == "id_tx":
                wm = m.embed(o, iid, tx)
            else:
                perm, M = m.get_perm_M(iid); wm = m.embed_with_target(o, apply_crypto(tx, perm, M))
            if wm.size != (512, 512): wm = wm.resize((512, 512))
            with torch.no_grad():
                acc[name]["p"].append(psnr(o, wm)); acc[name]["s"].append(float(SS(t(o), t(wm), data_range=1.0)))
                acc[name]["l"].append(float(lp(t(o) * 2 - 1, t(wm) * 2 - 1).item()))
        print(f"  [{i+1}/{len(imgs)}]", flush=True)
    print(f"\n=== STANDALONE per-fragment quality vs clean (n={len(imgs)}) ===")
    print(f"{'fragment':<20}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}")
    for name in frag:
        P, S, Lv = np.mean(acc[name]["p"]), np.mean(acc[name]["s"]), np.mean(acc[name]["l"])
        print(f"{name:<20}{P:>8.2f}{S:>8.3f}{Lv:>8.3f}")
    print("STANDALONE_DONE")


if __name__ == "__main__":
    main()
