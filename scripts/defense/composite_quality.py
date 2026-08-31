"""Image quality of the multi-fragment COMPOSITE (parametrized by --fragments).

Embeds the fragments in order, measures cumulative PSNR/SSIM/LPIPS vs the clean original
after each layer. For the final config pass: --fragments vine dft qim trustmark.
A watermark is only useful if quality stays high (papers expect PSNR>=~30, SSIM>=~0.9).
"""
import argparse, glob, json, os, sys
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
SPEC = {"phasemark": "target", "vine": "target", "dft": "id_tx", "qim": "id_tx", "trustmark": "target"}


def psnr(a, b):
    a = np.asarray(a, np.float64) / 255; b = np.asarray(b, np.float64) / 255
    m = np.mean((a - b) ** 2)
    return 10 * np.log10(1 / m) if m > 1e-12 else 99.0


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
    ap.add_argument("--n_images", type=int, default=20)
    ap.add_argument("--tm_variant", default="B")
    ap.add_argument("--output", default="results/defense/composite_quality_4fused.json")
    args = ap.parse_args()
    dev = "cuda"
    sb = ShortenedBCH()
    frag = build(args.fragments, dev, sb, args.tm_variant)
    import lpips as lpipsmod
    from pytorch_msssim import ssim as ssim_fn
    lp = lpipsmod.LPIPS(net="alex").to(dev).eval()

    def to_t(pil):
        return torch.from_numpy(np.asarray(pil.convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].to(dev)

    def metrics(orig, cur):
        a, b = to_t(orig), to_t(cur)
        with torch.no_grad():
            return psnr(orig, cur), float(ssim_fn(a, b, data_range=1.0)), float(lp(a * 2 - 1, b * 2 - 1).item())

    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))[4000:4000 + args.n_images]
    stages = list(args.fragments)
    acc = {st: {"psnr": [], "ssim": [], "lpips": []} for st in stages}

    for i, fp in enumerate(imgs):
        image_id = f"q_{i:05d}"; tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))
        orig = Image.open(fp).convert("RGB").resize((512, 512)); img = orig
        for name in args.fragments:
            m = frag[name]
            if SPEC[name] == "id_tx":
                img = m.embed(img, image_id, tx)
            else:
                perm, M = m.get_perm_M(image_id); img = m.embed_with_target(img, apply_crypto(tx, perm, M))
            if img.size != (512, 512): img = img.resize((512, 512))
            p, s, l = metrics(orig, img)
            acc[name]["psnr"].append(p); acc[name]["ssim"].append(s); acc[name]["lpips"].append(l)
        print(f"  [{i+1}/{len(imgs)}] composite PSNR={acc[stages[-1]]['psnr'][-1]:.1f} "
              f"SSIM={acc[stages[-1]]['ssim'][-1]:.3f} LPIPS={acc[stages[-1]]['lpips'][-1]:.3f}", flush=True)

    out = {"n_images": len(imgs), "fragments": args.fragments, "stages": {}}
    print(f"\n=== cumulative quality after each fragment (n={len(imgs)}) ===")
    print(f"{'stage':<14}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}")
    cum = ""
    for st in stages:
        P, S, L = np.mean(acc[st]["psnr"]), np.mean(acc[st]["ssim"]), np.mean(acc[st]["lpips"])
        out["stages"][st] = {"psnr": float(P), "ssim": float(S), "lpips": float(L)}
        cum = (cum + "+" + st) if cum else st
        print(f"{cum:<14}{P:>8.2f}{S:>8.3f}{L:>8.3f}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\n[done] -> {args.output}\nQUALITY_DONE")


if __name__ == "__main__":
    main()
