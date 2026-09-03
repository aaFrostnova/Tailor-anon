"""Fused vs hard-OR detection benchmark over the 3 bit-carrying methods.

Embeds ONE shared shortened-BCH(100,37) codeword with VINE + DFT-Kred + QIM,
applies the attack suite at the models' native 256x256, and compares:

  - per-method individual decode (each method's own ECC decode of its hard bits);
  - hard-OR baseline (detected if ANY single method decodes the payload);
  - fused: soft per-bit LLR combining (maximal-ratio) + Chase soft-BCH decode;
  - fused-noChase ablation (LLR fuse then plain hard BCH, p=0).

Detection = recovered 37-bit payload equals image_id_to_payload(image_id).
Also reports per-method bit accuracy and an FPR control on wrong image_id.
"""

import argparse
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.payload import image_id_to_payload
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.learned_fragment_methods import DFTKredMethod, QuantQIMMethod
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify, chase_decode

DEVICE = "cuda"
_SD = {}
SD_MODELS = {
    "sd15": "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5",
    "sd21": "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1",
}


def load_sd(key="sd15"):
    if key not in _SD:
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_MODELS[key], torch_dtype=torch.float16, safety_checker=None).to(DEVICE)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        _SD[key] = pipe
    return _SD[key]


_VAE = {}


def vae_roundtrip(pil, key="sd15"):
    """Continuous VAE encode->decode round-trip (no diffusion): a different
    reconstruction bottleneck than img2img regeneration."""
    from diffusers import AutoencoderKL
    if key not in _VAE:
        _VAE[key] = AutoencoderKL.from_pretrained(
            SD_MODELS[key], subfolder="vae", torch_dtype=torch.float16).to(DEVICE).eval()
    vae = _VAE[key]
    W, H = pil.size
    ew, eh = (W // 8) * 8, (H // 8) * 8
    x = torch.from_numpy(np.asarray(pil.resize((ew, eh), Image.LANCZOS), np.float32)
                         ).permute(2, 0, 1)[None].to(DEVICE).half() / 255.0 * 2 - 1
    with torch.no_grad():
        rec = vae.decode(vae.encode(x).latent_dist.sample()).sample
    rec = ((rec.float()[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    out = Image.fromarray(rec)
    return out.resize((W, H), Image.LANCZOS) if out.size != (W, H) else out


_FLUX = {}


def _flux(kind):
    """Lazy-load each FLUX regenerator ONCE (12B); strength is passed per call."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wbench.regen_archs import FluxVAERoundTrip, FluxImg2Img
    if kind not in _FLUX:
        _FLUX[kind] = FluxVAERoundTrip(device=DEVICE) if kind == "vae" else FluxImg2Img(device=DEVICE)
    return _FLUX[kind]


def apply_attack(name, pil):
    if name == "clean":
        return pil
    if name == "vae":
        return vae_roundtrip(pil)
    if name == "flux_vae":                       # 16-channel FLUX VAE round-trip (high fidelity)
        return _flux("vae").regen(pil)
    if name.startswith("flux_i2i_"):             # FLUX flow-matching img2img, e.g. flux_i2i_20
        return _flux("i2i").regen(pil, strength=int(name.split("_")[2]) / 100.0)
    if name == "rinse_2":  # two successive light regenerations
        return apply_attack("regen_10_sd15", apply_attack("regen_10_sd15", pil))
    if name.startswith("jpeg_"):
        q = int(name.split("_")[1]); buf = BytesIO()
        pil.save(buf, format="JPEG", quality=q); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name.startswith("blur_"):
        return pil.filter(ImageFilter.GaussianBlur(radius=float(name.split("_")[1])))
    if name.startswith("noise_"):
        s = int(name.split("_")[1]) / 1000.0
        arr = np.asarray(pil, np.float32) / 255.0
        arr += np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * s
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    if name.startswith("crop_"):
        r = int(name.split("_")[1]) / 100.0
        W, H = pil.size; cw, ch = int(W * r), int(H * r)
        l, t = (W - cw) // 2, (H - ch) // 2
        return pil.crop((l, t, l + cw, t + ch)).resize((W, H), Image.BILINEAR)
    if name.startswith("resize_"):
        r = int(name.split("_")[1]) / 100.0
        W, H = pil.size; Wb, Hb = int(W * r), int(H * r)
        big = pil.resize((Wb, Hb), Image.BILINEAR)
        l, t = (Wb - W) // 2, (Hb - H) // 2
        return big.crop((l, t, l + W, t + H))
    if name.startswith("regen_"):
        parts = name.split("_")
        strength = int(parts[1]) / 100.0
        key = parts[2] if len(parts) > 2 else "sd15"
        pipe = load_sd(key)
        g = torch.Generator(DEVICE).manual_seed(42)
        out = pipe(prompt="", image=pil, strength=strength, num_inference_steps=50,
                   guidance_scale=1.0, generator=g).images[0]
        return out.resize(pil.size, Image.BILINEAR) if out.size != pil.size else out
    raise ValueError(name)


def psnr(a, b):
    a = np.asarray(a, np.float64) / 255.0; b = np.asarray(b, np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    return 10.0 * np.log10(1.0 / mse) if mse > 1e-12 else 60.0


def decode_all(methods, sb, pil, image_id):
    """Return aligned LLRs and per-method individual detection."""
    aligned, per_method = {}, {}
    for name, m in methods.items():
        perm, M = m.get_perm_M(image_id)
        if name == "vine":
            soft = m.raw_probs(pil); kind = "prob"
        else:
            soft = m.raw_logits(pil); kind = "logit"
        a = method_soft_to_codeword_llr(soft, perm, M, kind=kind, n_codeword=sb.n)
        aligned[name] = a
        hard = llr_to_bits(a)
        data, n_err = sb.decode(hard)
        exp = image_id_to_payload(image_id, n_bits=sb.data_bits)
        det = bool(data is not None and np.array_equal(data, exp))
        per_method[name] = {"detected": det, "n_err": int(n_err),
                            "bit_acc": float(np.mean(hard == _tx_cache(sb, image_id)))}
    return aligned, per_method


def adaptive_weights(aligned):
    """Per-image self-calibrating weights: each method weighted by its own mean
    |LLR| (confidence) on THIS image. Down-weights collapsed methods with no
    knowledge of the attack."""
    return {m: float(np.mean(np.abs(a))) for m, a in aligned.items()}


_TXC = {}
def _tx_cache(sb, image_id):
    if image_id not in _TXC:
        _TXC[image_id] = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))
    return _TXC[image_id]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--n_images", type=int, default=50)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--resolution", type=int, default=512,
                   help="embed/attack resolution (512 = SD regen in-distribution)")
    p.add_argument("--dft_ckpt", default="results/dft_fftaware_baseline/ckpt.pt")
    p.add_argument("--qim_ckpt", default="results/quant_qim_frozen_d006/ckpt.pt")
    p.add_argument("--weights", default=None, help="json file of per-method weights")
    p.add_argument("--chase_p", type=int, default=8)
    p.add_argument("--attacks", nargs="+", default=[
        "clean", "jpeg_75", "jpeg_50", "blur_1.5", "blur_2.5", "noise_50",
        "crop_90", "crop_70", "resize_110",
        "regen_10_sd15", "regen_20_sd15", "regen_30_sd15",
    ])
    p.add_argument("--output", default="results/fused/benchmark.json")
    args = p.parse_args()

    key = args.master_key.encode("utf-8")
    sb = ShortenedBCH()
    methods = {
        "vine": VineCryptoWrapper(master_key=key, method_name="vine", n_bits=100, device=DEVICE),
        "dft_kred": DFTKredMethod(str(REPO / args.dft_ckpt), key, "dft_kred", DEVICE),
        "quant_qim": QuantQIMMethod(str(REPO / args.qim_ckpt), key, "quant_qim", DEVICE),
    }
    weights = None
    if args.weights:
        weights = json.load(open(args.weights))

    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]
    print(f"[fused-bench] {len(files)} imgs, {len(args.attacks)} attacks", flush=True)

    agg = {a: {"or": [], "fused_uni": [], "fused_adapt": [], "fused_uni_nochase": [],
               "per_method": {m: [] for m in methods},
               "bitacc": {m: [] for m in methods}, "fused_bitacc": []}
           for a in args.attacks}
    fpr = {"or": [], "fused_uni": [], "fused_adapt": []}
    psnrs = []

    for i, fp in enumerate(files):
        image_id = f"bench_{i:05d}"
        tx = _tx_cache(sb, image_id)
        pil = Image.open(fp).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
        pv, Mv = methods["vine"].get_perm_M(image_id)
        wm = methods["vine"].embed_with_target(pil, apply_crypto(tx, pv, Mv))
        wm = methods["dft_kred"].embed(wm, image_id, tx)
        wm = methods["quant_qim"].embed(wm, image_id, tx)
        psnrs.append(psnr(pil, wm))

        for atk in args.attacks:
            att = apply_attack(atk, wm)
            aligned, per_method = decode_all(methods, sb, att, image_id)
            fused_u = fuse_llrs(aligned, weights=None, n_codeword=sb.n)
            fused_a = fuse_llrs(aligned, weights=adaptive_weights(aligned), n_codeword=sb.n)
            agg[atk]["or"].append(float(any(v["detected"] for v in per_method.values())))
            agg[atk]["fused_uni"].append(
                float(decode_and_verify(fused_u, image_id, codec=sb, p=args.chase_p)["detected"]))
            agg[atk]["fused_adapt"].append(
                float(decode_and_verify(fused_a, image_id, codec=sb, p=args.chase_p)["detected"]))
            agg[atk]["fused_uni_nochase"].append(
                float(decode_and_verify(fused_u, image_id, codec=sb, p=0)["detected"]))
            agg[atk]["fused_bitacc"].append(float(np.mean(llr_to_bits(fused_a) == tx)))
            for m in methods:
                agg[atk]["per_method"][m].append(float(per_method[m]["detected"]))
                agg[atk]["bitacc"][m].append(per_method[m]["bit_acc"])

        # FPR control: decode clean wm against a WRONG image_id
        wrong_id = f"bench_{(i + 7) % len(files):05d}_wrong"
        aligned_w, pm_w = decode_all(methods, sb, wm, wrong_id)
        fused_uw = fuse_llrs(aligned_w, weights=None, n_codeword=sb.n)
        fused_aw = fuse_llrs(aligned_w, weights=adaptive_weights(aligned_w), n_codeword=sb.n)
        fpr["or"].append(float(any(v["detected"] for v in pm_w.values())))
        fpr["fused_uni"].append(float(decode_and_verify(fused_uw, wrong_id, codec=sb, p=args.chase_p)["detected"]))
        fpr["fused_adapt"].append(float(decode_and_verify(fused_aw, wrong_id, codec=sb, p=args.chase_p)["detected"]))
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(files)}] psnr={np.mean(psnrs):.1f}", flush=True)

    summary = {"n_images": len(files), "psnr_mean": float(np.mean(psnrs)),
               "chase_p": args.chase_p, "attacks": {},
               "fpr": {k: float(np.mean(v)) for k, v in fpr.items()}}
    for atk in args.attacks:
        d = agg[atk]
        summary["attacks"][atk] = {
            "or_detect": float(np.mean(d["or"])),
            "fused_uniform_detect": float(np.mean(d["fused_uni"])),
            "fused_adaptive_detect": float(np.mean(d["fused_adapt"])),
            "fused_uniform_nochase_detect": float(np.mean(d["fused_uni_nochase"])),
            "fused_bitacc": float(np.mean(d["fused_bitacc"])),
            "per_method_detect": {m: float(np.mean(d["per_method"][m])) for m in methods},
            "per_method_bitacc": {m: float(np.mean(d["bitacc"][m])) for m in methods},
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    # Table
    print(f"\n{'='*94}")
    print(f"PSNR={summary['psnr_mean']:.2f}dB  FPR or={summary['fpr']['or']:.3f} "
          f"fused_uni={summary['fpr']['fused_uni']:.3f} fused_adapt={summary['fpr']['fused_adapt']:.3f}")
    print(f"{'attack':<16}{'OR':>7}{'fUNI':>7}{'fADAPT':>8}{'fUni-noC':>10}{'fBitAcc':>9}"
          f"{'vine':>7}{'dft':>7}{'qim':>7}")
    print("-" * 94)
    for atk in args.attacks:
        s = summary["attacks"][atk]
        pm = s["per_method_detect"]
        print(f"{atk:<16}{s['or_detect']:>7.2f}{s['fused_uniform_detect']:>7.2f}"
              f"{s['fused_adaptive_detect']:>8.2f}{s['fused_uniform_nochase_detect']:>10.2f}"
              f"{s['fused_bitacc']:>9.3f}"
              f"{pm['vine']:>7.2f}{pm['dft_kred']:>7.2f}{pm['quant_qim']:>7.2f}")
    print("=" * 94)
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
