"""Evaluate the scale-aware decoder with PILOT-based synchronization.

The training curve showed the decoder reads the watermark at ~0.95 when the
geometry is KNOWN; blind payload-confidence cannot find the geometry. So we embed
P known pilot bits: at decode, the (scale, rotation) that maximizes pilot agreement
is the recovered geometry (an unambiguous sync signal), and the payload is read
there. Reports payload bit-acc per attack vs a fixed (no-sync) decode.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from src.dft_kred_modules import DFTKredEncoder
from src.logpolar_fragment import ScaleAwareFFTDecoder
from benchmark_fused import apply_attack

DEVICE = "cuda"
IMG_DIR = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"


def rot(pil, deg):
    return pil.rotate(deg, resample=Image.BILINEAR, expand=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/logpolar/scaleaware_C.pt")
    p.add_argument("--n_images", type=int, default=12)
    p.add_argument("--n_pilots", type=int, default=30)
    args = p.parse_args()

    ck = torch.load(REPO / args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    enc = DFTKredEncoder(n_bits=ck["n_bits"], K=ck["K"], M=ck["M"], resolution=256,
                         r_lo=ck["r_lo"], r_hi=ck["r_hi"], init_delta=ck["init_delta"],
                         canonical_id=ck["canonical_id"]).to(DEVICE).eval()
    dec = ScaleAwareFFTDecoder(enc.carriers, enc.bit_flip, resolution=256,
                               init_delta=ck["init_delta"], hidden=cfg.get("hidden", 64)).to(DEVICE).eval()
    dec.load_state_dict(ck["decoder_state_dict"])
    n_bits = ck["n_bits"]; P = args.n_pilots
    pilot = np.random.RandomState(12345).randint(0, 2, P).astype(np.uint8)  # known pattern

    scales = np.round(np.arange(0.80, 1.251, 0.01), 3)
    angles = np.deg2rad(np.arange(-8, 8.1, 1.0))
    t = transforms.ToTensor()

    def to_t(pil):
        return t(pil.convert("RGB").resize((256, 256), Image.LANCZOS)).unsqueeze(0).to(DEVICE)

    @torch.no_grad()
    def pilot_sync_decode(att, payload, search_ang):
        x = to_t(att)
        best = (-1, None)
        angs = angles if search_ang else [0.0]
        for a in angs:
            for s in scales:
                lg = dec(x, scale=float(s), angle=float(a))[0].cpu().numpy()
                agree = float(np.mean((lg[:P] > 0).astype(np.uint8) == pilot))
                if agree > best[0]:
                    best = (agree, lg)
        lg = best[1]
        pay_acc = float(np.mean((lg[P:] > 0).astype(np.uint8) == payload))
        return pay_acc, best[0]

    @torch.no_grad()
    def fixed_decode(att, payload):
        lg = dec(to_t(att), 1.0, 0.0)[0].cpu().numpy()
        return float(np.mean((lg[P:] > 0).astype(np.uint8) == payload))

    attacks = {"clean": (lambda p: p, False), "resize_105": (lambda p: apply_attack("resize_105", p), False),
               "resize_110": (lambda p: apply_attack("resize_110", p), False),
               "resize_120": (lambda p: apply_attack("resize_120", p), False),
               "rot_+5": (lambda p: rot(p, 5), True), "jpeg_50": (lambda p: apply_attack("jpeg_50", p), False),
               "crop_90": (lambda p: apply_attack("crop_90", p), False),
               # compound (geometric + photometric), the realistic deployment case
               "resize110+jpeg": (lambda p: apply_attack("jpeg_50", apply_attack("resize_110", p)), False),
               "rot5+jpeg": (lambda p: apply_attack("jpeg_50", rot(p, 5)), True),
               "resize110+noise": (lambda p: apply_attack("noise_50", apply_attack("resize_110", p)), False)}
    agg = {a: {"fixed": [], "sync": [], "pilot": []} for a in attacks}

    files = sorted([f for f in Path(IMG_DIR).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[4000:4000 + args.n_images]
    for fp in files:
        img = to_t(Image.open(fp))
        payload = np.random.RandomState(abs(hash(fp.name)) % (2**31)).randint(0, 2, n_bits - P).astype(np.uint8)
        secret = np.concatenate([pilot, payload]).astype(np.uint8)
        st = torch.tensor(secret, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            wm = enc(img, st)
        wm_pil = transforms.ToPILImage()(wm[0].clamp(0, 1).cpu())
        for aname, (fn, sa) in attacks.items():
            att = fn(wm_pil)
            agg[aname]["fixed"].append(fixed_decode(att, payload))
            pa, pil_ag = pilot_sync_decode(att, payload, sa)
            agg[aname]["sync"].append(pa); agg[aname]["pilot"].append(pil_ag)

    print(f"\n=== scale-aware + pilot-sync ({args.ckpt}, {len(files)} imgs, {P} pilots / {n_bits-P} payload) ===")
    print(f"{'attack':<12}{'fixed':>8}{'pilot-sync':>12}{'pilot-agree':>13}")
    for a in attacks:
        d = agg[a]
        print(f"{a:<12}{np.mean(d['fixed']):>8.3f}{np.mean(d['sync']):>12.3f}{np.mean(d['pilot']):>13.3f}", flush=True)


if __name__ == "__main__":
    main()
