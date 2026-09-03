#!/usr/bin/env python3
"""W-Bench regeneration benchmark.

Cross-method comparison of watermark/fingerprint survival under SD-based image
regeneration (img2img attacks). Methods compared in this script:
  - Ours: cryptographic_fingerprint with verify_multi_domain detection
  - TrustMark (Adobe, ICCV 2025)
  - WAM (Watermark Anything, ICLR 2025, Meta)

VINE runs in a separate conda env; see scripts/benchmark_regen_vine.py.

Attacks (W-Bench-style stochastic regeneration via SD-v1-5 img2img):
  - no_attack:       passthrough
  - regen_mild:      strength=0.10  (light noise + denoise)
  - regen_medium:    strength=0.20
  - regen_heavy:     strength=0.40

Usage:
  python scripts/benchmark_regen.py \
      --src_dir /project/.../coco_fp_global_200/images_clean \
      --n_images 20 \
      --out_dir results/wbench_regen
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.keygen import generate_master_key, derive_all_subkeys, image_id_from_path
from src.verify import verify_multi_domain, verify_multi_domain_v2
from src.fragment_config import get_default_configs, get_v2_configs
from src.embed import embed_multi_domain, embed_multi_domain_signed
from src.fragment import generate_fragment
from src.pipeline import load_image
from src.payload import BCHCodec, image_id_to_payload
from src.bit_assignment import assign_bits_to_regions
from src.jnd import compute_jnd_mask
from src.template_v3 import (
    construct_template_T_via_vae,
    LearnableTemplate,
)
from src.v3_pipeline import embed_v3, verify_v3


# ------------------------------------------------------------------------- IO

def pil_to_chw(pil: Image.Image) -> np.ndarray:
    """PIL RGB → (C,H,W) float32 in [0,1]."""
    return np.array(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0


def chw_to_pil(arr: np.ndarray) -> Image.Image:
    hwc = np.clip(arr.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(hwc, "RGB")


# ------------------------------------------------------------------------ SD regen

_SD_PIPE = None


def load_sd(model_path: str):
    global _SD_PIPE
    if _SD_PIPE is None:
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        print(f"[regen] loading SD pipeline from {model_path}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, safety_checker=None,
        ).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        _SD_PIPE = pipe
    return _SD_PIPE


def regen_stoch(pil: Image.Image, strength: float, seed: int = 42) -> Image.Image:
    """W-Bench stochastic regeneration via SD img2img.

    Higher strength = stronger attack (more noise added before denoising).
    """
    pipe = load_sd("/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5")
    g = torch.Generator("cuda").manual_seed(seed)
    out = pipe(prompt="", image=pil, strength=strength,
               num_inference_steps=50, guidance_scale=1.0,
               generator=g).images[0]
    if out.size != pil.size:
        out = out.resize(pil.size, Image.BILINEAR)
    return out


# --------------------------------------------------------------------- Ours method

class OursWatermarker:
    name = "ours"
    n_bits = 0   # v1 carries no payload bits; TPR@FPR not meaningful

    def expected_bits(self, image_id: str = None) -> np.ndarray:
        return np.array([], dtype=np.uint8)

    def decode_raw_bits(self, pil: Image.Image, orig_pil: Image.Image = None,
                        image_id: str = None) -> np.ndarray:
        return np.array([], dtype=np.uint8)

    def __init__(self, master_key: bytes, eps: float = 16 / 255.0, K: int = 8):
        self.master_key = master_key
        self.eps = eps
        self.K = K
        self.configs = get_default_configs(K)
        self.weights = [1.0 / K] * K
        self.image_id = "global"

    def embed(self, pil: Image.Image, **_) -> Image.Image:
        arr = pil_to_chw(pil)
        subkeys = derive_all_subkeys(self.master_key, self.image_id, self.K)
        frags = [generate_fragment(sk, arr.shape, self.eps, k)
                 for k, sk in enumerate(subkeys)]
        fp_arr = embed_multi_domain(arr, frags, self.configs, self.weights)
        return chw_to_pil(fp_arr)

    def detect(self, attacked_pil: Image.Image, orig_pil: Image.Image, **_) -> dict:
        orig = pil_to_chw(orig_pil)
        att = pil_to_chw(attacked_pil)
        if att.shape != orig.shape:
            att_pil_resized = attacked_pil.resize(orig_pil.size, Image.BILINEAR)
            att = pil_to_chw(att_pil_resized)
        res = verify_multi_domain(
            orig, att, self.master_key, self.image_id,
            configs=self.configs, epsilon=self.eps,
            weights=self.weights, alpha=0.001,
        )
        return {
            "detected": bool(res["detected"]),
            "combined_p": float(res["combined_p_value"]),
            "mean_correlation": float(res["mean_correlation"]),
            "fragment_survived": [float(p < 0.001) for p in res["per_fragment"]["p_values"]],
        }


# ---------------------------------------------------------------- Ours v2 (payload-carrying)

class _SkipBitsMixin:
    """Default no-op bits API for methods that don't support TPR@FPR."""
    n_bits = 0

    def expected_bits(self, image_id: str = None) -> np.ndarray:
        return np.array([], dtype=np.uint8)

    def decode_raw_bits(self, pil: Image.Image, orig_pil: Image.Image = None,
                        image_id: str = None) -> np.ndarray:
        return np.array([], dtype=np.uint8)


class OursV2Watermarker:
    """v2 image-provenance: 127-bit BCH payload via signed K=13 fragments.

    Each test image is assigned its own image_id (the file stem). Embedding
    encodes image_id_to_payload(image_id) into 127 bits via BCH(127,64,t=10),
    then routes the signed K=13 fragments through embed_multi_domain_signed.
    Detection (reference-needed, forensic) re-derives the assignment and
    decodes the BCH codeword back to the recovered image_id.
    """

    name = "ours_v2"

    def __init__(self, master_key: bytes, eps: float = 16 / 255.0):
        self.master_key = master_key
        self.eps = eps
        self.configs = get_v2_configs()
        self.K = len(self.configs)
        self.weights = [1.0 / self.K] * self.K
        self.codec = BCHCodec()
        # populated per-image via kwargs at embed/detect
        self.current_image_id = "global"

    def embed(self, pil: Image.Image, image_id: str = None, **_) -> Image.Image:
        if image_id is None:
            image_id = self.current_image_id
        self.current_image_id = image_id
        arr = pil_to_chw(pil)
        payload = image_id_to_payload(image_id, self.codec.data_bits)
        codeword = self.codec.encode(payload)
        subkeys = derive_all_subkeys(self.master_key, image_id, self.K)
        fragments = [
            generate_fragment(sk, arr.shape, self.eps, fragment_index=k)
            for k, sk in enumerate(subkeys)
        ]
        sign_masks, bit_idx = assign_bits_to_regions(
            self.configs, arr.shape, self.codec.n, self.master_key, image_id,
        )
        jnd = compute_jnd_mask(arr)
        fp = embed_multi_domain_signed(
            arr, fragments, sign_masks, bit_idx, codeword,
            self.configs, weights=self.weights, jnd_mask=jnd,
        )
        return chw_to_pil(fp)

    def detect(
        self, attacked_pil: Image.Image, orig_pil: Image.Image,
        image_id: str = None, **_,
    ) -> dict:
        if image_id is None:
            image_id = self.current_image_id
        orig = pil_to_chw(orig_pil)
        att = pil_to_chw(attacked_pil)
        if att.shape != orig.shape:
            att = pil_to_chw(attacked_pil.resize(orig_pil.size, Image.BILINEAR))
        res = verify_multi_domain_v2(
            orig, att, self.master_key, image_id,
            configs=self.configs, epsilon=self.eps,
            weights=self.weights, alpha=0.001, n_bits=self.codec.n,
        )
        # Raw codeword bit accuracy = fraction of 127 bits recovered correctly
        # against the expected (pre-BCH-decode) codeword.
        from src.payload import image_id_to_payload as _id_to_payload
        expected_payload = _id_to_payload(image_id, self.codec.data_bits)
        expected_cw = self.codec.encode(expected_payload)
        raw_cw = np.array(res["raw_recovered_codeword"], dtype=np.uint8)
        raw_bit_acc = float(np.mean(raw_cw == expected_cw))

        n_err = int(res["n_err"])
        # When BCH succeeds we report 1.0 - n_err/n; when it fails we report
        # the raw bit accuracy (which is what TrustMark/WAM report too).
        if n_err >= 0:
            bit_acc = 1.0 - n_err / float(self.codec.n)
        else:
            bit_acc = raw_bit_acc

        return {
            "detected": bool(res["detected"]),
            "image_id_match": bool(res["image_id_match"]),
            "combined_p": float(res["combined_p_value"]),
            "mean_correlation": float(res["mean_correlation"]),
            "n_err": n_err,
            "bit_accuracy": float(bit_acc),
            "raw_bit_accuracy": float(raw_bit_acc),
            "recovered_id_hex": res.get("recovered_image_id_hex"),
            "expected_id_hex": res.get("expected_image_id_hex"),
        }


# ---------------------------------------------------------------- Ours v3 (hybrid template)

class OursV3Watermarker:
    """v3 hybrid: shared template T × keyed sign envelope + BCH payload."""

    name = "ours_v3"
    # n_bits set in __init__ from codec.n (127 default)

    def expected_bits(self, image_id: str) -> np.ndarray:
        from src.payload import image_id_to_payload
        payload = image_id_to_payload(image_id, self.codec.data_bits)
        return self.codec.encode(payload)

    def decode_raw_bits(self, pil: Image.Image, orig_pil: Image.Image = None,
                        image_id: str = None) -> np.ndarray:
        """Return 127-bit codeword as predicted by region-sign decoder
        (before BCH error correction)."""
        from src.v3_pipeline import verify_v3
        if image_id is None:
            image_id = self.current_image_id
        orig = pil_to_chw(orig_pil)
        att = pil_to_chw(pil)
        if att.shape != orig.shape:
            att = pil_to_chw(pil.resize(orig_pil.size, Image.BILINEAR))
        if orig.shape != self.template_T.shape:
            from PIL import Image as _I
            T_pil = chw_to_pil(np.clip(self.template_T / self.eps * 0.5 + 0.5, 0, 1))
            T_pil = T_pil.resize((orig.shape[2], orig.shape[1]), _I.BILINEAR)
            T = (np.array(T_pil).transpose(2, 0, 1) / 255.0 - 0.5) * 2 * self.eps
            T = T.astype(np.float32)
        else:
            T = self.template_T
        res = verify_v3(orig, att, T, self.master_key, image_id, codec=self.codec)
        return np.array(res["recovered_codeword"], dtype=np.uint8)

    def __init__(
        self,
        master_key: bytes,
        eps: float = 16 / 255.0,
        resolution: int = 256,
        template_ckpt: str = None,
        device: str = "cuda",
        bch_m: int = 7,
        bch_t: int = 10,
    ):
        self.master_key = master_key
        self.eps = eps
        self.codec = BCHCodec(m=bch_m, t=bch_t)
        self.current_image_id = "global"
        self.n_bits = self.codec.n

        if template_ckpt is not None:
            # Plan B: load trained template
            tmpl = LearnableTemplate.load(template_ckpt)
            self.template_T = tmpl.detach_numpy()
        else:
            # Plan A: VAE-projected template
            self.template_T = construct_template_T_via_vae(
                master_key=master_key,
                shape=(3, resolution, resolution),
                eps=eps,
                device=device,
            )
        # Free VAE if not needed elsewhere
        self.resolution = resolution

    def embed(self, pil: Image.Image, image_id: str = None, **_) -> Image.Image:
        if image_id is None:
            image_id = self.current_image_id
        self.current_image_id = image_id
        arr = pil_to_chw(pil)
        if arr.shape != self.template_T.shape:
            # Resize template to image resolution (rare path)
            from PIL import Image as _I
            T_pil = chw_to_pil(np.clip(self.template_T / self.eps * 0.5 + 0.5, 0, 1))
            T_pil = T_pil.resize((arr.shape[2], arr.shape[1]), _I.BILINEAR)
            T = (np.array(T_pil).transpose(2, 0, 1) / 255.0 - 0.5) * 2 * self.eps
            T = T.astype(np.float32)
        else:
            T = self.template_T
        wm, _info = embed_v3(arr, T, self.master_key, image_id,
                             codec=self.codec, use_jnd=True)
        return chw_to_pil(wm)

    def detect(
        self, attacked_pil: Image.Image, orig_pil: Image.Image,
        image_id: str = None, **_,
    ) -> dict:
        if image_id is None:
            image_id = self.current_image_id
        orig = pil_to_chw(orig_pil)
        att = pil_to_chw(attacked_pil)
        if att.shape != orig.shape:
            att = pil_to_chw(attacked_pil.resize(orig_pil.size, Image.BILINEAR))
        # Resize T if needed
        if orig.shape != self.template_T.shape:
            from PIL import Image as _I
            T_pil = chw_to_pil(np.clip(self.template_T / self.eps * 0.5 + 0.5, 0, 1))
            T_pil = T_pil.resize((orig.shape[2], orig.shape[1]), _I.BILINEAR)
            T = (np.array(T_pil).transpose(2, 0, 1) / 255.0 - 0.5) * 2 * self.eps
            T = T.astype(np.float32)
        else:
            T = self.template_T
        res = verify_v3(orig, att, T, self.master_key, image_id, codec=self.codec)
        return {
            "detected": bool(res["detected"]),
            "image_id_match": bool(res["image_id_match"]),
            "n_err": int(res["n_err"]),
            "bit_accuracy": float(res["raw_bit_accuracy"]),
            "raw_bit_accuracy": float(res["raw_bit_accuracy"]),
            "recovered_id_hex": res.get("recovered_payload_hex"),
            "expected_id_hex": res.get("expected_payload_hex"),
        }


# ---------------------------------------------------------------- TrustMark method

class TrustMarkWatermarker:
    name = "trustmark"
    # TrustMark's internal pre-BCH bits are not exposed; we compare the
    # post-BCH 56-bit decoded message and report bit-level accuracy from that.
    n_bits = 56

    def __init__(self):
        from trustmark import TrustMark
        self.tm = TrustMark(model_type="Q", verbose=False)
        # TrustMark uses 7-bit ASCII chars; we encode 8 ASCII chars = 56 bits
        # plus version bits ⇒ fits in default 100-bit payload with BCH_5 ECC
        self.message = "WBenchEv"  # 8 chars

    def expected_bits(self, image_id: str = None) -> np.ndarray:
        s = self._bits(self.message)
        return np.array([int(c) for c in s], dtype=np.uint8)

    def decode_raw_bits(self, pil: Image.Image, orig_pil: Image.Image = None,
                        image_id: str = None) -> np.ndarray:
        try:
            secret, present, schema = self.tm.decode(pil)
        except Exception:
            secret, present = "", False
        n = len(self.expected_bits())
        if not present or not secret:
            # No bits recovered → simulate "random" output with zeros.
            # bit_acc against zeros is ~half (since expected has roughly equal 0/1).
            return np.zeros(n, dtype=np.uint8)
        bits_pred = self._bits(secret)
        bits_pred = (bits_pred + "0" * n)[:n]
        return np.array([int(c) for c in bits_pred], dtype=np.uint8)

    def _bits(self, text: str) -> str:
        return "".join(format(ord(c), "08b") for c in text)

    def embed(self, pil: Image.Image, **_) -> Image.Image:
        return self.tm.encode(pil, self.message)

    def detect(self, attacked_pil: Image.Image, **_) -> dict:
        try:
            secret, present, schema = self.tm.decode(attacked_pil)
        except Exception:
            secret, present, schema = "", False, -1
        bits_gt = self._bits(self.message)
        bits_pred = self._bits(secret) if secret else ""
        bits_pred = (bits_pred + "0" * len(bits_gt))[: len(bits_gt)]
        bit_acc = sum(a == b for a, b in zip(bits_gt, bits_pred)) / len(bits_gt)
        return {
            "detected": bool(present and secret == self.message),
            "decoded_secret_matches": bool(secret == self.message),
            "watermark_present": bool(present),
            "bit_accuracy": float(bit_acc),
        }


# --------------------------------------------------------------------- WAM method

class WamWatermarker:
    name = "wam"
    # n_bits set in __init__ from msg_bits

    def expected_bits(self, image_id: str = None) -> np.ndarray:
        return self.msg.cpu().numpy().astype(np.uint8).flatten()

    def decode_raw_bits(self, pil: Image.Image, orig_pil: Image.Image = None,
                        image_id: str = None) -> np.ndarray:
        img_pt = self.transform(pil).unsqueeze(0).to("cuda")
        preds = self.wam.detect(img_pt)["preds"]
        mask_preds = self.F.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]
        pred_msg = self.predict(bit_preds, mask_preds).cpu().float()
        return pred_msg.numpy().astype(np.uint8).flatten()

    def __init__(self, msg_bits: int = 32):
        self.n_bits = msg_bits
        wam_root = "/project/pi_shiqingma_umass_edu/mingzheli/watermark/wam/repo"
        sys.path.insert(0, wam_root)
        from notebooks.inference_utils import load_model_from_checkpoint, default_transform, unnormalize_img
        from watermark_anything.data.metrics import msg_predict_inference
        import torch.nn.functional as F
        # WAM's params.json references relative paths (configs/embedder.yaml etc.)
        # so we must chdir into the WAM repo for model load.
        cwd0 = os.getcwd()
        os.chdir(wam_root)
        try:
            self.wam = load_model_from_checkpoint(
                "checkpoints/params.json",
                "/project/pi_shiqingma_umass_edu/mingzheli/watermark/wam/models/checkpoint.pth",
            ).to("cuda").eval()
        finally:
            os.chdir(cwd0)
        self.transform = default_transform
        self.unnormalize = unnormalize_img
        self.predict = msg_predict_inference
        self.F = F
        torch.manual_seed(0)
        self.msg = torch.randint(0, 2, (1, msg_bits)).float().to("cuda")
        self.msg_bits = msg_bits

    def embed(self, pil: Image.Image, **_) -> Image.Image:
        img_pt = self.transform(pil).unsqueeze(0).to("cuda")
        out = self.wam.embed(img_pt, self.msg)
        wm = self.unnormalize(out["imgs_w"]).clamp(0, 1)[0]
        wm = (wm.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        wm_pil = Image.fromarray(wm, "RGB")
        if wm_pil.size != pil.size:
            wm_pil = wm_pil.resize(pil.size, Image.BILINEAR)
        return wm_pil

    def detect(self, attacked_pil: Image.Image, **_) -> dict:
        img_pt = self.transform(attacked_pil).unsqueeze(0).to("cuda")
        preds = self.wam.detect(img_pt)["preds"]
        mask_preds = self.F.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]
        pred_msg = self.predict(bit_preds, mask_preds).cpu().float()
        bit_acc = (pred_msg == self.msg.cpu()).float().mean().item()
        # WAM's detected definition: bit_acc > random (>=0.55 is conservative)
        return {
            "detected": bool(bit_acc >= 0.80),  # 80%+ bit accuracy
            "bit_accuracy": float(bit_acc),
        }


# ------------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True, help="Clean source images dir")
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_images", action="store_true",
                   help="Save embedded + attacked images for VINE phase-2")
    p.add_argument("--methods", nargs="+",
                   default=["ours", "trustmark", "wam"])
    p.add_argument("--attacks", nargs="+",
                   default=["no_attack", "regen_mild", "regen_medium", "regen_heavy"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--v3_template_ckpt", default=None,
                   help="Optional trained template .pt for ours_v3 (Plan B)")
    p.add_argument("--bch_m", type=int, default=7,
                   help="BCH Galois field exponent m (n = 2^m - 1)")
    p.add_argument("--bch_t", type=int, default=10,
                   help="BCH error-correction capability t")
    return p.parse_args()


def _apply_np_transform(pil: Image.Image, fn) -> Image.Image:
    """Bridge: PIL → np (C,H,W) [0,1] → fn → PIL. Resizes back to input dim if changed."""
    arr = np.array(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
    out = fn(arr)
    if out.ndim == 3 and out.shape != arr.shape:
        # transform changed resolution (e.g. center_crop without resize). Resize back.
        hwc = np.clip(out.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        out_pil = Image.fromarray(hwc, "RGB").resize(pil.size, Image.BILINEAR)
        return out_pil
    hwc = np.clip(out.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(hwc, "RGB")


def run_attack(name: str, pil: Image.Image) -> Image.Image:
    """Dispatch attacks. SD regen attacks go through diffusers; others use
    the model-agnostic transforms from src/transforms.py.
    """
    from src import transforms as _T

    if name == "no_attack":
        return pil
    if name == "regen_mild":
        return regen_stoch(pil, strength=0.10)
    if name == "regen_medium":
        return regen_stoch(pil, strength=0.20)
    if name == "regen_heavy":
        return regen_stoch(pil, strength=0.40)
    # ---- non-regen transforms (4 representatives) ----
    if name == "jpeg_q40":
        return _apply_np_transform(pil, lambda x: _T.jpeg_compress(x, quality=40))
    if name == "crop_resize_60":
        return _apply_np_transform(pil, lambda x: _T.center_crop_resize(x, ratio=0.60))
    if name == "blur_2.0":
        return _apply_np_transform(pil, lambda x: _T.gaussian_blur(x, radius=2.0))
    if name == "jpeg40+crop60":
        return _apply_np_transform(
            pil, lambda x: _T.jpeg_compress(_T.random_crop_resize(x, ratio=0.6), quality=40),
        )
    raise ValueError(f"unknown attack {name}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------ load source images
    src_dir = Path(args.src_dir)
    image_files = sorted([f for f in src_dir.iterdir()
                          if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])[: args.n_images]
    print(f"[setup] loaded {len(image_files)} test images from {src_dir}")

    pils = []
    for f in image_files:
        pil = Image.open(f).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
        pils.append((f.stem, pil))

    # ------ instantiate methods
    methods = {}
    if "ours" in args.methods:
        # Use a fixed master key so detect uses the same key
        master_key = generate_master_key(32)
        np.random.seed(0)
        methods["ours"] = OursWatermarker(master_key=master_key)
    if "ours_v2" in args.methods:
        # v2 uses its own fixed master key (independent of v1's)
        master_key_v2 = generate_master_key(32)
        methods["ours_v2"] = OursV2Watermarker(master_key=master_key_v2)
    if "ours_v3" in args.methods:
        # v3 hybrid: VAE-projected T (Plan A) or trained T (Plan B via --v3_template_ckpt)
        master_key_v3 = generate_master_key(32)
        methods["ours_v3"] = OursV3Watermarker(
            master_key=master_key_v3,
            resolution=args.resolution,
            template_ckpt=getattr(args, "v3_template_ckpt", None),
            bch_m=getattr(args, "bch_m", 7),
            bch_t=getattr(args, "bch_t", 10),
        )
    if "trustmark" in args.methods:
        methods["trustmark"] = TrustMarkWatermarker()
    if "wam" in args.methods:
        methods["wam"] = WamWatermarker()
    print(f"[setup] methods: {list(methods.keys())}")

    # ------ run benchmark
    results = {m: {a: [] for a in args.attacks} for m in methods}
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(exist_ok=True)

    for idx, (img_name, orig_pil) in enumerate(pils):
        print(f"\n[image {idx+1}/{len(pils)}] {img_name}")
        for method_name, method in methods.items():
            t0 = time.time()
            wm_pil = method.embed(orig_pil, image_id=img_name)
            t_embed = time.time() - t0
            if args.save_images:
                wm_pil.save(img_dir / f"{img_name}__{method_name}__embedded.png")

            for attack_name in args.attacks:
                t0 = time.time()
                attacked_pil = run_attack(attack_name, wm_pil)
                t_attack = time.time() - t0
                # Also attack the CLEAN (un-watermarked) image — same attack —
                # so we get the FPR baseline for this method+attack combination.
                attacked_clean_pil = run_attack(attack_name, orig_pil)

                t0 = time.time()
                if method_name in ("ours", "ours_v2", "ours_v3"):
                    det = method.detect(attacked_pil, orig_pil=orig_pil, image_id=img_name)
                else:
                    det = method.detect(attacked_pil)
                t_detect = time.time() - t0

                # Raw-bit decode for TPR@FPR: needs (image, optional orig, image_id)
                expected = method.expected_bits(image_id=img_name)
                if method.n_bits > 0:
                    bits_wm = method.decode_raw_bits(
                        attacked_pil, orig_pil=orig_pil, image_id=img_name,
                    )
                    bits_cln = method.decode_raw_bits(
                        attacked_clean_pil, orig_pil=orig_pil, image_id=img_name,
                    )
                    bit_acc_wm = float(np.mean(bits_wm == expected))
                    bit_acc_clean = float(np.mean(bits_cln == expected))
                else:
                    bit_acc_wm = 0.0
                    bit_acc_clean = 0.0

                det["image"] = img_name
                det["t_embed"] = t_embed
                det["t_attack"] = t_attack
                det["t_detect"] = t_detect
                det["bit_acc_wm_raw"] = bit_acc_wm
                det["bit_acc_clean_raw"] = bit_acc_clean
                det["n_bits"] = int(method.n_bits)
                results[method_name][attack_name].append(det)

                if method_name == "ours":
                    key_metric = f"r={det.get('mean_correlation', 0):.4f}"
                elif method_name in ("ours_v2", "ours_v3"):
                    key_metric = (
                        f"id_match={'Y' if det.get('image_id_match') else 'N'} "
                        f"raw_bit={det.get('raw_bit_accuracy', 0):.3f} "
                        f"n_err={det.get('n_err', -1)}"
                    )
                else:
                    key_metric = f"bit_acc={det.get('bit_accuracy', 0):.3f}"
                print(f"  [{method_name:9s}] {attack_name:13s} | "
                      f"detected={'Y' if det['detected'] else 'N'} | {key_metric} | "
                      f"t={t_attack:.1f}s")

                if args.save_images:
                    attacked_pil.save(img_dir / f"{img_name}__{method_name}__{attack_name}.png")

    # ------ aggregate
    from scipy.stats import binom
    summary = {}
    for method_name, attacks in results.items():
        summary[method_name] = {}
        for attack_name, runs in attacks.items():
            n = len(runs)
            if n == 0:
                continue
            det_rate = sum(r["detected"] for r in runs) / n
            entry = {"detection_rate": det_rate, "n": n}
            if "bit_accuracy" in runs[0]:
                entry["bit_accuracy_mean"] = float(np.mean([r["bit_accuracy"] for r in runs]))
            if "mean_correlation" in runs[0]:
                entry["correlation_mean"] = float(np.mean([r["mean_correlation"] for r in runs]))
                entry["combined_p_mean"] = float(np.mean([r["combined_p"] for r in runs]))
            # ----- TPR@FPR (parametric Binomial null on raw bits) -----
            n_bits = int(runs[0].get("n_bits", 0))
            if n_bits > 0:
                bit_acc_wm = np.array([r["bit_acc_wm_raw"] for r in runs])
                bit_acc_cln = np.array([r["bit_acc_clean_raw"] for r in runs])
                entry["bit_acc_wm_mean"] = float(bit_acc_wm.mean())
                entry["bit_acc_clean_mean"] = float(bit_acc_cln.mean())
                # Parametric thresholds via Binomial(n_bits, 0.5) survival
                for fpr in (0.10, 0.01, 0.001):
                    k_thr = binom.isf(fpr, n_bits, 0.5)
                    tau = k_thr / n_bits
                    tpr = float(np.mean(bit_acc_wm >= tau))
                    fpr_empirical = float(np.mean(bit_acc_cln >= tau))
                    entry[f"tau_{fpr*100:.1f}pct"] = float(tau)
                    entry[f"tpr_at_fpr_{fpr*100:.1f}pct"] = tpr
                    entry[f"fpr_empirical_{fpr*100:.1f}pct"] = fpr_empirical
            summary[method_name][attack_name] = entry

    # ------ output
    full_path = out_dir / "results_full.json"
    summary_path = out_dir / "summary.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ----- print three side-by-side tables -----
    def _print_table(label: str, get_cell):
        print(f"\n=== {label} ===")
        header = f"{'method':>11s} | " + " | ".join(f"{a:>13s}" for a in args.attacks)
        print(header)
        print("-" * len(header))
        for method_name in methods:
            cells = []
            for attack in args.attacks:
                s = summary[method_name].get(attack, {})
                cells.append(f"{get_cell(method_name, s):>13s}")
            print(f"{method_name:>11s} | " + " | ".join(cells))

    _print_table(
        "(a) Detection rate — BCH-style binary",
        lambda m, s: f"{s.get('detection_rate', 0)*100:.0f}%",
    )
    _print_table(
        "(b) Raw bit accuracy mean (on watermarked)",
        lambda m, s: (
            f"{s['bit_acc_wm_mean']:.3f}"
            if "bit_acc_wm_mean" in s else "n/a"
        ),
    )
    _print_table(
        "(c) TPR @ 1% FPR (parametric Binomial null on raw bits)",
        lambda m, s: (
            f"{s['tpr_at_fpr_1.0pct']*100:.0f}%"
            if "tpr_at_fpr_1.0pct" in s else "n/a"
        ),
    )

    print(f"\n[done] full → {full_path}\n[done] summary → {summary_path}")


if __name__ == "__main__":
    main()
