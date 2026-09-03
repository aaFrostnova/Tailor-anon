"""Minimal latent-domain watermark prototype — sanity check.

Goal: test whether a key-derived sign envelope embedded in SD VAE latent space
survives (a) the VAE encode-decode round trip and (b) SD img2img regeneration.

This is a smoke test, not a full benchmark. T_lat is a fixed random Gaussian
(no training). We measure raw bit accuracy of the recovered BCH codeword.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.payload import BCHCodec, image_id_to_payload  # noqa: E402
from src.sign_envelope import (  # noqa: E402
    derive_keyed_constants,
    build_sign_pattern,
    make_region_masks,
    recover_codeword_from_residual,
)

SD_PATH = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5"
DEVICE = "cuda"

# Cached pipelines
_VAE = None
_PIPE = None


def load_vae():
    global _VAE
    if _VAE is None:
        from diffusers import AutoencoderKL
        _VAE = AutoencoderKL.from_pretrained(SD_PATH, subfolder="vae").to(DEVICE).eval()
        for p in _VAE.parameters():
            p.requires_grad_(False)
    return _VAE


def load_sd():
    global _PIPE
    if _PIPE is None:
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        _PIPE = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD_PATH, torch_dtype=torch.float16, safety_checker=None,
        ).to(DEVICE)
        _PIPE.scheduler = DDIMScheduler.from_config(_PIPE.scheduler.config)
        _PIPE.set_progress_bar_config(disable=True)
    return _PIPE


# ------------------------------------------------------------ image io

def pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    """PIL → tensor in [-1, 1], (1, 3, H, W) on DEVICE."""
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1) * 2.0 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """tensor in [-1, 1], (1, 3, H, W) → PIL."""
    arr = ((t[0].clamp(-1, 1).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0), "RGB")


# ------------------------------------------------------------ latent ops

def encode_latent(x: torch.Tensor) -> torch.Tensor:
    """x in [-1, 1] (1,3,256,256) → latent (1,4,32,32)."""
    vae = load_vae()
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
        return z * vae.config.scaling_factor


def decode_latent(z: torch.Tensor) -> torch.Tensor:
    """latent (1,4,32,32) → x in [-1, 1] (1,3,256,256)."""
    vae = load_vae()
    with torch.no_grad():
        x = vae.decode(z / vae.config.scaling_factor).sample
        return x


def make_latent_region_masks(n_bits: int = 127, latent_shape=(4, 32, 32)):
    """Tile (4, 32, 32) latent into n_bits regions.

    Strategy: 12×12 grid on the 32×32 spatial plane × all 4 channels,
    take first n_bits regions.
    Returns list of bool arrays of shape latent_shape.
    """
    C, H, W = latent_shape
    grid = int(np.ceil(np.sqrt(n_bits)))  # 12 for n_bits=127
    cell_h = H // grid
    cell_w = W // grid
    masks = []
    for i in range(n_bits):
        r, c = i // grid, i % grid
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < grid - 1 else H
        x0 = c * cell_w
        x1 = (c + 1) * cell_w if c < grid - 1 else W
        m = np.zeros(latent_shape, dtype=bool)
        m[:, y0:y1, x0:x1] = True
        masks.append(m)
    return masks


def make_T_lat(master_key: bytes, latent_shape=(4, 32, 32), seed: int = 0):
    """Generate a fixed random latent template (no training).

    Use a key-derived seed so this is reproducible per master_key.
    """
    import hashlib
    h = hashlib.sha256(master_key + b"/T_lat_seed").digest()
    rng_seed = int.from_bytes(h[:8], "big") % (2**31 - 1) + seed
    rng = np.random.RandomState(rng_seed)
    T = rng.randn(*latent_shape).astype(np.float32)
    # Normalize to unit RMS so alpha controls magnitude
    T = T / (np.sqrt(np.mean(T**2)) + 1e-8)
    return T


# ------------------------------------------------------------ embed/verify

def _jnd_mask_from_pil(pil: Image.Image, jnd_min: float = 0.4, jnd_max: float = 1.0) -> np.ndarray:
    """Per-pixel JND mask from 5×5 local std of grayscale image.
    Returns (1, H, W) array of mask values in [jnd_min, jnd_max]. Smooth regions → small.
    """
    from scipy.ndimage import uniform_filter
    arr = np.asarray(pil.convert("L"), dtype=np.float32) / 255.0
    mean = uniform_filter(arr, size=5)
    var = uniform_filter(arr ** 2, size=5) - mean ** 2
    std = np.sqrt(np.clip(var, 0.0, None))
    # normalize: std ~ [0, 0.25], map to [jnd_min, jnd_max] linearly
    mask = jnd_min + (jnd_max - jnd_min) * np.clip(std / 0.15, 0.0, 1.0)
    return mask[None, :, :]   # (1, H, W)


def embed_latent(
    pil: Image.Image,
    master_key: bytes,
    image_id: str,
    T_lat: np.ndarray,
    alpha: float = 0.5,
    codec: BCHCodec = None,
    pixel_jnd: bool = False,
):
    """Embed a latent-domain watermark and return (watermarked_pil, info).
    If pixel_jnd=True, multiply the decoded perturbation by a per-pixel JND
    mask to suppress visible distortion in smooth regions.
    """
    if codec is None:
        codec = BCHCodec()

    # 1. encode to latent
    x = pil_to_tensor(pil)
    z = encode_latent(x)  # (1, 4, 32, 32)
    z_np = z[0].cpu().numpy()
    latent_shape = z_np.shape

    # 2. crypto envelope (same as v3 but in latent shape)
    payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    codeword = codec.encode(payload)
    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
    region_masks = make_latent_region_masks(codec.n, latent_shape)
    S_lat = build_sign_pattern(codeword, perm, M, region_masks, latent_shape)

    # 3. add to latent
    delta = alpha * T_lat * S_lat
    z_w_np = z_np + delta
    z_w = torch.from_numpy(z_w_np).unsqueeze(0).to(DEVICE)

    # 4. decode back
    x_w = decode_latent(z_w)

    # 4b. (optional) pixel-domain JND post-mask
    if pixel_jnd:
        jnd_np = _jnd_mask_from_pil(pil)   # (1, H, W) in [jnd_min, jnd_max]
        jnd_t = torch.from_numpy(jnd_np).to(DEVICE).float()
        delta_pix = x_w[0] - x[0]
        delta_pix_masked = delta_pix * jnd_t            # broadcast (1, H, W) → (3, H, W)
        x_w = (x[0] + delta_pix_masked).unsqueeze(0).clamp(-1.0, 1.0)

    pil_w = tensor_to_pil(x_w)

    # PSNR vs original
    diff = x_w[0].cpu().numpy() - x[0].cpu().numpy()
    mse = float(np.mean(diff**2))
    psnr = 10 * np.log10((2.0**2) / max(mse, 1e-12))  # range is [-1,1]

    return pil_w, {
        "codeword": codeword,
        "perm": perm,
        "M": M,
        "psnr": psnr,
        "delta_rms": float(np.sqrt(np.mean(delta**2))),
    }


def verify_latent(
    orig_pil: Image.Image,
    suspect_pil: Image.Image,
    master_key: bytes,
    image_id: str,
    T_lat: np.ndarray,
    codec: BCHCodec = None,
):
    """Decode the watermark from suspect_pil; return info dict."""
    if codec is None:
        codec = BCHCodec()

    # 1. encode both to latent
    z_orig = encode_latent(pil_to_tensor(orig_pil))[0].cpu().numpy()
    z_susp = encode_latent(pil_to_tensor(suspect_pil))[0].cpu().numpy()

    # 2. latent residual
    r_lat = z_susp - z_orig

    # 3. region-wise matched filter
    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
    region_masks = make_latent_region_masks(codec.n, z_orig.shape)
    recovered_bits, scores = recover_codeword_from_residual(
        r_lat.astype(np.float32),
        T_lat.astype(np.float32),
        perm, M, region_masks,
    )

    expected_payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    expected_cw = codec.encode(expected_payload)

    raw_bit_acc = float(np.mean(recovered_bits == expected_cw))

    payload, n_err = codec.decode(recovered_bits)
    image_id_match = payload is not None and np.array_equal(payload, expected_payload)

    return {
        "raw_bit_acc": raw_bit_acc,
        "n_err": int(n_err),
        "image_id_match": bool(image_id_match),
        "detected": bool(image_id_match and 0 <= n_err <= codec.t),
        "region_score_mean": float(np.mean(scores)),
    }


def verify_latent_ensemble(
    orig_pil: Image.Image,
    suspect_pil: Image.Image,
    master_key: bytes,
    image_id: str,
    T_lat: np.ndarray,
    decoder,
    codec: BCHCodec = None,
    decoder_weight: float = 0.5,
):
    """Ensemble of matched filter + learned decoder.

    Combines per-region signed scores from both detectors:
        score_i = (1 - w) * mf_score_i_normalized + w * decoder_logit_i_normalized

    where each detector's per-region scores are normalised by their own std so
    the two streams are on comparable scales before the weighted sum. The sign
    of the combined score is then descrambled via sigma^-1 + XOR M and BCH-decoded.
    """
    if codec is None:
        codec = BCHCodec()

    # 1. encode both latents (torch + numpy reuse)
    z_orig_t = encode_latent(pil_to_tensor(orig_pil))
    z_susp_t = encode_latent(pil_to_tensor(suspect_pil))
    r_lat_t = z_susp_t - z_orig_t
    r_lat_np = r_lat_t[0].cpu().numpy().astype(np.float32)

    # 2a. matched filter (numpy)
    region_masks = make_latent_region_masks(codec.n, r_lat_np.shape)
    T_lat_np = T_lat.astype(np.float32)
    mf_scores = np.zeros(codec.n, dtype=np.float32)
    for r in range(codec.n):
        rmask = region_masks[r]
        mf_scores[r] = float(np.sum(r_lat_np[rmask] * T_lat_np[rmask]))

    # 2b. learned decoder (torch)
    T_lat_t = torch.from_numpy(T_lat_np).to(DEVICE)
    decoder.eval()
    with torch.no_grad():
        decoder_logits = decoder(r_lat_t, T_lat_t)[0].cpu().numpy().astype(np.float32)

    # 3. normalise each stream by its std, then weighted sum
    def _z(x):
        s = float(np.std(x)) + 1e-8
        return x / s
    combined = (1.0 - decoder_weight) * _z(mf_scores) + decoder_weight * _z(decoder_logits)
    # Convention: matched filter ip>=0 -> sign +1 -> bit 0 (after mask XOR)
    # decoder logit>=0 -> bit 1 (target_bits=1 when sign_per_region=+1).
    # The mf score has the SAME sign convention as +sign_per_region, so combining
    # straight (decoder logit's positive = sign_per_region=+1; mf score positive =
    # sign_per_region=+1) is consistent.
    region_sign = np.where(combined >= 0, 1, -1).astype(np.int8)

    # 4. descramble + BCH
    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
    bits = np.zeros(codec.n, dtype=np.uint8)
    for j in range(codec.n):
        r = int(perm[j])
        decoded_sign = int(region_sign[r]) * int(M[r])
        bits[j] = 0 if decoded_sign > 0 else 1

    expected_payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    expected_cw = codec.encode(expected_payload)
    raw_bit_acc = float(np.mean(bits == expected_cw))
    payload, n_err = codec.decode(bits)
    image_id_match = payload is not None and np.array_equal(payload, expected_payload)

    return {
        "raw_bit_acc": raw_bit_acc,
        "n_err": int(n_err),
        "image_id_match": bool(image_id_match),
        "detected": bool(image_id_match and 0 <= n_err <= codec.t),
        "region_score_mean": float(np.mean(np.abs(combined))),
    }


def verify_latent_learned(
    orig_pil: Image.Image,
    suspect_pil: Image.Image,
    master_key: bytes,
    image_id: str,
    T_lat: np.ndarray,
    decoder,
    codec: BCHCodec = None,
):
    """Same as verify_latent but uses a learned CNN decoder for region-sign
    extraction instead of matched filter. The crypto envelope descrambling
    (sigma^-1 + XOR M) and BCH decoding stay external.
    """
    if codec is None:
        codec = BCHCodec()

    # 1. encode both to latent (torch tensors, GPU)
    z_orig_t = encode_latent(pil_to_tensor(orig_pil))   # (1, 4, 32, 32)
    z_susp_t = encode_latent(pil_to_tensor(suspect_pil))
    r_lat_t = z_susp_t - z_orig_t                       # (1, 4, 32, 32)

    # 2. learned region-sign logits (region-aware decoder)
    T_lat_t = torch.from_numpy(T_lat.astype(np.float32)).to(DEVICE)
    decoder.eval()
    with torch.no_grad():
        logits = decoder(r_lat_t, T_lat_t)               # (1, 127)
    # convert logits to region-sign predictions {-1, +1}
    # convention: logit > 0 -> bit=1 in target_bits -> sign_per_region > 0 -> +1
    region_sign_pred = torch.where(logits >= 0,
                                   torch.tensor(1, device=logits.device, dtype=torch.int8),
                                   torch.tensor(-1, device=logits.device, dtype=torch.int8))
    region_sign_np = region_sign_pred[0].cpu().numpy().astype(np.int8)

    # 3. descramble via (perm, M): inv-perm + multiply by M to recover codeword bits
    perm, M = derive_keyed_constants(master_key, image_id, n_bits=codec.n)
    # recover bits in codeword order (match recover_codeword_from_residual convention)
    bits = np.zeros(codec.n, dtype=np.uint8)
    for j in range(codec.n):
        r = int(perm[j])
        decoded_sign = int(region_sign_np[r]) * int(M[r])
        bits[j] = 0 if decoded_sign > 0 else 1
    recovered_bits = bits

    expected_payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
    expected_cw = codec.encode(expected_payload)

    raw_bit_acc = float(np.mean(recovered_bits == expected_cw))
    payload, n_err = codec.decode(recovered_bits)
    image_id_match = payload is not None and np.array_equal(payload, expected_payload)

    return {
        "raw_bit_acc": raw_bit_acc,
        "n_err": int(n_err),
        "image_id_match": bool(image_id_match),
        "detected": bool(image_id_match and 0 <= n_err <= codec.t),
        "region_score_mean": float(logits.abs().mean().item()),
    }


# ------------------------------------------------------------ regen attack

def regen_attack(pil: Image.Image, strength: float = 0.10) -> Image.Image:
    pipe = load_sd()
    g = torch.Generator(DEVICE).manual_seed(42)
    out = pipe(prompt="", image=pil, strength=strength,
               num_inference_steps=50, guidance_scale=1.0, generator=g).images[0]
    if out.size != pil.size:
        out = out.resize(pil.size, Image.BILINEAR)
    return out


# ------------------------------------------------------------ main test

def tpr_threshold(n_bits: int = 127, fpr: float = 0.01) -> float:
    """Inverse-Binomial threshold: smallest k such that P(X >= k | X~Bin(n, 0.5)) <= fpr.
    Returns the bit-accuracy threshold k/n.
    """
    from scipy.stats import binom
    # find smallest k with sf(k-1) <= fpr
    for k in range(n_bits, n_bits // 2, -1):
        if binom.sf(k - 1, n_bits, 0.5) > fpr:
            return (k + 1) / n_bits
    return 0.5


def main():
    master_key = b"latent_proto_master_key_2025"
    codec = BCHCodec()
    T_lat = make_T_lat(master_key)
    thr = tpr_threshold(codec.n, fpr=0.01)
    print(f"[setup] T_lat shape={T_lat.shape}, RMS={np.sqrt(np.mean(T_lat**2)):.4f}")
    print(f"[setup] BCH(n={codec.n}, k={codec.data_bits}, t={codec.t})")
    print(f"[setup] TPR@1%FPR threshold (bit acc): {thr:.4f}")

    image_dir = REPO / "images"
    image_files = sorted(image_dir.glob("*.png"))[:20]
    print(f"[setup] {len(image_files)} test images")

    # Sweep alpha across PSNR/robustness frontier.
    alphas_to_test = [0.03, 0.05, 0.08, 0.12, 0.18]
    strengths = [0.10, 0.20, 0.30]   # regen_mild / medium / heavy

    for alpha in alphas_to_test:
        print(f"\n=== alpha = {alpha} ===")
        rows = []
        for img_idx, fp in enumerate(image_files):
            pil = Image.open(fp).convert("RGB").resize((256, 256), Image.BILINEAR)
            image_id = f"proto_{img_idx:04d}"

            pil_w, info = embed_latent(pil, master_key, image_id, T_lat, alpha=alpha, codec=codec)
            res_clean = verify_latent(pil, pil_w, master_key, image_id, T_lat, codec)

            regen_results = {}
            for s in strengths:
                pil_regen = regen_attack(pil_w, strength=s)
                regen_results[s] = verify_latent(pil, pil_regen, master_key, image_id, T_lat, codec)

            rows.append({
                "img": fp.name, "psnr": info["psnr"],
                "clean": res_clean,
                "regen": regen_results,
            })

        psnr = float(np.mean([r["psnr"] for r in rows]))
        psnr_std = float(np.std([r["psnr"] for r in rows]))
        clean_acc = np.mean([r["clean"]["raw_bit_acc"] for r in rows])
        clean_det = np.mean([float(r["clean"]["detected"]) for r in rows])
        clean_tpr = np.mean([float(r["clean"]["raw_bit_acc"] >= thr) for r in rows])
        print(f"  α={alpha} PSNR={psnr:5.2f}±{psnr_std:4.2f}dB | "
              f"clean: BCHdet={clean_det*100:3.0f}% TPR={clean_tpr*100:3.0f}% acc={clean_acc:.3f}", end="")
        for s in strengths:
            acc = np.mean([r["regen"][s]["raw_bit_acc"] for r in rows])
            det = np.mean([float(r["regen"][s]["detected"]) for r in rows])
            tpr = np.mean([float(r["regen"][s]["raw_bit_acc"] >= thr) for r in rows])
            print(f" | s={s}: BCHdet={det*100:3.0f}% TPR={tpr*100:3.0f}% acc={acc:.3f}", end="")
        print()


if __name__ == "__main__":
    main()
