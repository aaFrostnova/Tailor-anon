"""Reference-free geometric tier: the scale-aware DFT fragment + pilot synchronization,
wrapped as a method (embed / identity-detect) for the combined system.

Carries 30 known pilot bits (for sync) + payload bits (= image_id identity). Detection
searches (scale, rotation) to maximize pilot agreement, then checks the payload. Embeds
in the DFT magnitude of the green channel, so it REPLACES DFT-Kred (same domain) rather
than stacking with it. Operates with a 256x256 carrier; for larger inputs the residual
is computed at 256 and upscaled (same strategy as the other fragments).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.dft_kred_modules import DFTKredEncoder
from src.logpolar_fragment import ScaleAwareFFTDecoder
from src.payload import image_id_to_payload


class ScaleAwareGeoFragment:
    name = "scaleaware_geo"

    def __init__(self, ckpt_path, master_key=b"v5_key_encoder_master", device="cuda",
                 n_pilots=30, scales=None, angles_deg=(0.0,), pilot_seed=12345):
        self.device = device
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.n_bits = int(ck["n_bits"])
        self.n_pilots = n_pilots
        self.n_payload = self.n_bits - n_pilots
        self.res = 256
        cfg = ck.get("config", {})
        self.enc = DFTKredEncoder(
            n_bits=self.n_bits, K=ck["K"], M=ck["M"], resolution=256,
            r_lo=ck["r_lo"], r_hi=ck["r_hi"], init_delta=ck["init_delta"],
            canonical_id=ck["canonical_id"]).to(device).eval()
        self.dec = ScaleAwareFFTDecoder(
            self.enc.carriers, self.enc.bit_flip, resolution=256,
            init_delta=ck["init_delta"], hidden=cfg.get("hidden", 64)).to(device).eval()
        self.dec.load_state_dict(ck["decoder_state_dict"])
        self.pilot = np.random.RandomState(pilot_seed).randint(0, 2, n_pilots).astype(np.uint8)
        self.scales = np.round(np.arange(0.80, 1.251, 0.01), 3) if scales is None else np.asarray(scales)
        self.angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))

    def codeword(self, image_id):
        payload = image_id_to_payload(image_id, n_bits=self.n_payload)
        return np.concatenate([self.pilot, payload]).astype(np.uint8)

    def _to256(self, pil):
        return transforms.ToTensor()(
            pil.convert("RGB").resize((256, 256), Image.LANCZOS)).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed(self, pil, image_id):
        tx = self.codeword(image_id)
        pil = pil.convert("RGB"); W, H = pil.size
        img256 = self._to256(pil)
        secret = torch.tensor(tx, dtype=torch.float32).unsqueeze(0).to(self.device)
        wm256 = self.enc(img256, secret)
        if (W, H) == (256, 256):
            return transforms.ToPILImage()(wm256[0].clamp(0, 1).cpu())
        import torch.nn.functional as F
        resid = F.interpolate(wm256 - img256, size=(H, W), mode="bilinear", align_corners=False)
        orig = transforms.ToTensor()(pil).unsqueeze(0).to(self.device)
        return transforms.ToPILImage()((orig + resid).clamp(0, 1)[0].cpu())

    @torch.no_grad()
    def detect(self, pil, image_id):
        expected = image_id_to_payload(image_id, n_bits=self.n_payload)
        x = self._to256(pil)
        P = self.n_pilots
        ns = len(self.scales)
        scales_t = torch.tensor(self.scales, dtype=torch.float32, device=self.device)
        x_rep = x.expand(ns, -1, -1, -1)
        best_agree, best_lg = -1.0, None
        for a in self.angles:
            ang_t = torch.full((ns,), float(a), device=self.device)
            logits = self.dec(x_rep, scale=scales_t, angle=ang_t).cpu().numpy()  # (ns, n_bits)
            pilot_hat = (logits[:, :P] > 0).astype(np.uint8)
            agree = (pilot_hat == self.pilot[None, :]).mean(axis=1)               # (ns,)
            j = int(np.argmax(agree))
            if agree[j] > best_agree:
                best_agree, best_lg = float(agree[j]), logits[j]
        payload = (best_lg[P:] > 0).astype(np.uint8)
        n = min(len(payload), len(expected))
        return {"detected": bool(np.array_equal(payload[:n], expected[:n])),
                "pilot_agree": best_agree,
                "payload_bit_acc": float(np.mean(payload[:n] == expected[:n])),
                "name": self.name}
