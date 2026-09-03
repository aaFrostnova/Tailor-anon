"""FusedDetector: embed + soft-fusion-decode over the 3 bit-carrying methods.

Centralizes the shared-codeword embed and the LLR-fusion + Chase soft-BCH decode
so the benchmark, the attack-characterization harness, tamper detection, and
MultiMethodFingerprint.detect_fused all share one implementation.

All bit methods operate at 256x256. Detection = recovered payload identity.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image

from src.payload import image_id_to_payload
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.learned_fragment_methods import DFTKredMethod, QuantQIMMethod
from src.soft_fusion import (
    method_soft_to_codeword_llr, fuse_llrs, llr_to_bits, erasure_positions,
)
from src.soft_bch import decode_and_verify

DEFAULT_DFT_CKPT = "results/dft_fftaware_baseline/ckpt.pt"
DEFAULT_QIM_CKPT = "results/quant_qim_frozen_d006/ckpt.pt"


class FusedDetector:
    def __init__(
        self,
        master_key: bytes = b"v5_key_encoder_master",
        dft_ckpt: str = DEFAULT_DFT_CKPT,
        qim_ckpt: str = DEFAULT_QIM_CKPT,
        device: str = "cuda",
        use_vine: bool = True,
        use_dft: bool = True,
        work_res: int = 512,
    ):
        self.master_key = master_key
        self.device = device
        self.sb = ShortenedBCH()
        self.work_res = work_res  # embedding/attack resolution (512 = SD in-distribution)
        self.methods: Dict[str, object] = {}
        if use_vine:
            self.methods["vine"] = VineCryptoWrapper(
                master_key=master_key, method_name="vine", n_bits=100, device=device)
        if use_dft:  # drop DFT-Kred when the geometric scale-aware fragment owns the DFT domain
            self.methods["dft_kred"] = DFTKredMethod(dft_ckpt, master_key, "dft_kred", device)
        self.methods["quant_qim"] = QuantQIMMethod(qim_ckpt, master_key, "quant_qim", device)
        # Only these expose carrier locations (for characterization / tamper-loc).
        self.location_aware = [m for m in ("dft_kred", "quant_qim") if m in self.methods]

    # --------------------------------------------------------------- embed
    def codeword(self, image_id: str) -> np.ndarray:
        return self.sb.encode(image_id_to_payload(image_id, n_bits=self.sb.data_bits))

    def _to_res(self, pil: Image.Image) -> Image.Image:
        pil = pil.convert("RGB")
        if pil.size != (self.work_res, self.work_res):
            pil = pil.resize((self.work_res, self.work_res), Image.LANCZOS)
        return pil

    def embed(self, pil: Image.Image, image_id: str) -> Image.Image:
        tx = self.codeword(image_id)
        pil = self._to_res(pil)
        if "vine" in self.methods:
            pv, Mv = self.methods["vine"].get_perm_M(image_id)
            pil = self.methods["vine"].embed_with_target(pil, apply_crypto(tx, pv, Mv))
        if "dft_kred" in self.methods:
            pil = self.methods["dft_kred"].embed(pil, image_id, tx)
        pil = self.methods["quant_qim"].embed(pil, image_id, tx)
        return pil

    # --------------------------------------------------------------- decode
    def aligned_llrs(self, pil: Image.Image, image_id: str):
        """Return {method: aligned LLR over the codeword frame} and raw soft outputs."""
        aligned, raw = {}, {}
        for name, m in self.methods.items():
            perm, M = m.get_perm_M(image_id)
            if name == "vine":
                soft = m.raw_probs(pil); kind = "prob"
            else:
                soft = m.raw_logits(pil); kind = "logit"
            aligned[name] = method_soft_to_codeword_llr(
                soft, perm, M, kind=kind, n_codeword=self.sb.n)
            raw[name] = soft
        return aligned, raw

    def detect(
        self,
        pil: Image.Image,
        image_id: str,
        weights: Optional[Dict[str, float]] = None,
        chase_p: int = 8,
        erasure_idx: Optional[Sequence[int]] = None,
        erasure_abs_thresh: Optional[float] = None,
    ) -> dict:
        aligned, raw = self.aligned_llrs(pil, image_id)
        fused = fuse_llrs(aligned, weights=weights, n_codeword=self.sb.n)
        if erasure_idx is None and erasure_abs_thresh is not None:
            erasure_idx = erasure_positions(fused, erasure_abs_thresh).tolist()
        res = decode_and_verify(fused, image_id, codec=self.sb, p=chase_p,
                                erasure_idx=erasure_idx)
        # per-method individual decode (the hard-OR baseline components)
        exp = image_id_to_payload(image_id, n_bits=self.sb.data_bits)
        per_method = {}
        for name, a in aligned.items():
            hard = llr_to_bits(a)
            data, n_err = self.sb.decode(hard)
            per_method[name] = {
                "detected": bool(data is not None and np.array_equal(data, exp)),
                "n_err": int(n_err),
            }
        res["per_method"] = per_method
        res["or_detected"] = bool(any(v["detected"] for v in per_method.values()))
        res["fused_llr"] = fused
        res["aligned"] = aligned
        res["raw_soft"] = raw
        return res
