"""Evaluate trained KeyConditionedEncoder + SimpleDecoder.

For each test image:
  1. Embed with encoder (using key-derived S_lat)
  2. Apply attack (clean, regen_mild, regen_medium, regen_heavy)
  3. Decode with learned decoder → 127 logits → σ⁻¹ + M → BCH decode
  4. Report PSNR, bit_acc, BCH detection, TPR@1%FPR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.key_encoder import KeyConditionedEncoder, SimpleDecoder, s_lat_to_pixel, count_params  # noqa
from src.payload import BCHCodec, image_id_to_payload  # noqa
from src.sign_envelope import derive_keyed_constants  # noqa
from train_T_lat import make_latent_region_masks, build_id_pool  # noqa

DEVICE = "cuda"

# ---- SD regen attack (reuse from test_latent_prototype)
_SD_PIPE = None
def load_sd():
    global _SD_PIPE
    if _SD_PIPE is None:
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        SD_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_PATH, torch_dtype=torch.float16, safety_checker=None,
        ).to(DEVICE)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        _SD_PIPE = pipe
    return _SD_PIPE

def regen_attack(pil, strength=0.10):
    pipe = load_sd()
    g = torch.Generator(DEVICE).manual_seed(42)
    out = pipe(prompt="", image=pil, strength=strength,
               num_inference_steps=50, guidance_scale=1.0, generator=g).images[0]
    if out.size != pil.size:
        out = out.resize(pil.size, Image.BILINEAR)
    return out

# ---- helpers
def pil_to_tensor(pil):
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1) * 2 - 1).unsqueeze(0).to(DEVICE)

def tensor_to_pil(t):
    arr = ((t[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def tpr_threshold(n_bits=127, fpr=0.01):
    from scipy.stats import binom
    for k in range(n_bits, n_bits // 2, -1):
        if binom.sf(k - 1, n_bits, 0.5) > fpr:
            return (k + 1) / n_bits
    return 0.5


def decode_with_matched_filter(
    pil_orig, pil_suspect, master_key, image_id, codec,
):
    """Reference-needed decode using pixel-space matched filter."""
    x_orig = np.asarray(pil_orig.convert("RGB"), dtype=np.float32) / 255.0 * 2 - 1
    x_susp = np.asarray(pil_suspect.convert("RGB"), dtype=np.float32) / 255.0 * 2 - 1
    x_orig = x_orig.transpose(2, 0, 1)
    x_susp = x_susp.transpose(2, 0, 1)

    residual = (x_susp - x_orig).astype(np.float32)
    H, W = residual.shape[1], residual.shape[2]
    region_bounds = build_pixel_region_bounds(codec.n, H, W)

    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
    bits = np.zeros(codec.n, dtype=np.uint8)
    for j in range(codec.n):
        r = int(perm[j])
        y0, y1, x0, x1 = region_bounds[r]
        val = float(residual[:, y0:y1, x0:x1].mean())
        region_sign = 1 if val >= 0 else -1
        decoded_sign = region_sign * int(M[r])
        bits[j] = 0 if decoded_sign > 0 else 1

    expected_payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    expected_cw = codec.encode(expected_payload)
    raw_bit_acc = float(np.mean(bits == expected_cw))
    payload, n_err = codec.decode(bits)
    image_id_match = payload is not None and np.array_equal(payload, expected_payload)

    return {
        "raw_bit_acc": raw_bit_acc,
        "n_err": int(n_err),
        "detected": bool(image_id_match and 0 <= n_err <= codec.t),
    }


def build_pixel_region_bounds(n_bits, H=256, W=256):
    """Same 12×12 grid as matched filter."""
    grid = int(np.ceil(np.sqrt(n_bits)))
    cell_h, cell_w = H // grid, W // grid
    bounds = []
    for i in range(n_bits):
        r, c = i // grid, i % grid
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < grid - 1 else H
        x0 = c * cell_w
        x1 = (c + 1) * cell_w if c < grid - 1 else W
        bounds.append((y0, y1, x0, x1))
    return bounds


def build_s_pixel_aligned(sign_per_region, region_bounds, H=256, W=256):
    """Build S_pixel directly in pixel space matching matched-filter grid."""
    S = torch.zeros(1, 1, H, W, device=DEVICE)
    for i, (y0, y1, x0, x1) in enumerate(region_bounds):
        S[0, 0, y0:y1, x0:x1] = float(sign_per_region[i])
    return S


def embed_with_encoder(pil, encoder, master_key, image_id, codec, region_masks):
    """Embed using KeyConditionedEncoder."""
    x = pil_to_tensor(pil)

    payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    codeword = codec.encode(payload)
    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)

    # Build target sign per region (same as build_id_pool logic)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    bit_at_region = codeword[inv_perm]
    sign_per_region = M.astype(np.float32) * (1.0 - 2.0 * bit_at_region.astype(np.float32))

    # Build S_pixel aligned to pixel-space grid (NOT via latent upsample)
    H, W = 256, 256
    region_bounds = build_pixel_region_bounds(codec.n, H, W)
    S_pixel = build_s_pixel_aligned(sign_per_region, region_bounds, H, W)

    encoder.eval()
    with torch.no_grad():
        x_w = encoder(x, S_pixel)

    pil_w = tensor_to_pil(x_w)
    diff = x_w[0].cpu().numpy() - x[0].cpu().numpy()
    mse = float(np.mean(diff ** 2))
    psnr = 10 * np.log10(4.0 / max(mse, 1e-12))

    return pil_w, {"psnr": psnr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--master_key", default="v5_key_encoder_master")
    p.add_argument("--strengths", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    args = p.parse_args()

    codec = BCHCodec()
    thr = tpr_threshold(codec.n, fpr=0.01)
    master_key = args.master_key.encode("utf-8")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    base_ch = int(ckpt.get("base_ch", 64))
    n_blocks = int(ckpt.get("n_blocks", 8))

    encoder = KeyConditionedEncoder(base_ch=base_ch, n_blocks=n_blocks).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    print(f"[setup] encoder: {count_params(encoder):,} params (matched filter decode, no learned decoder)")
    print(f"[setup] scale={encoder.scale.item():.4f}")
    print(f"[setup] TPR threshold={thr:.4f}")

    # Region masks for building S_lat (used by embed only)
    region_masks_np = make_latent_region_masks(codec.n, (4, 32, 32))
    region_masks_list = [region_masks_np[i] for i in range(codec.n)]

    image_files = sorted((REPO / "images").glob("*.png"))[:args.n_images]
    print(f"[setup] {len(image_files)} test images")

    attacks = {"clean": None}
    for s in args.strengths:
        attacks[f"regen_{s}"] = s

    # Evaluate
    results = {atk: [] for atk in attacks}
    psnr_list = []

    for img_idx, fp in enumerate(image_files):
        pil = Image.open(fp).convert("RGB").resize((256, 256), Image.BILINEAR)
        image_id = f"eval_{img_idx:04d}"

        pil_w, info = embed_with_encoder(pil, encoder, master_key, image_id, codec, region_masks_list)
        psnr_list.append(info["psnr"])

        for atk_name, strength in attacks.items():
            if strength is None:
                pil_test = pil_w
            else:
                pil_test = regen_attack(pil_w, strength=strength)

            res = decode_with_matched_filter(
                pil, pil_test, master_key, image_id, codec,
            )
            results[atk_name].append(res)

        if (img_idx + 1) % 10 == 0:
            print(f"  [{img_idx+1}/{len(image_files)}] done", flush=True)

    # Report
    psnr_mean = float(np.mean(psnr_list))
    print(f"\n  PSNR={psnr_mean:.2f}dB (mean over {len(image_files)} images)")

    for atk_name in attacks:
        r_list = results[atk_name]
        acc = np.mean([r["raw_bit_acc"] for r in r_list])
        bch = np.mean([float(r["detected"]) for r in r_list])
        tpr = np.mean([float(r["raw_bit_acc"] >= thr) for r in r_list])
        print(f"  {atk_name:15s}: BCH={bch*100:5.1f}%  TPR@1%FPR={tpr*100:5.1f}%  raw_acc={acc:.3f}")


if __name__ == "__main__":
    main()
