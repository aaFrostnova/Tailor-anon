"""Sweep carrier density and measure the analytical Fourier-Mellin resize/rotation
ceiling with redundancy-consensus sync. No trained decoder needed (encoder is
analytical). Compares fixed-bin vs Mellin-consensus decode across geometric attacks."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from src.dft_kred_modules import DFTKredEncoder
from src.logpolar_reffree import mellin_analytical
from benchmark_fused import apply_attack

DEVICE = "cuda"
IMG_DIR = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"

CONFIGS = [
    dict(name="K5M2_r20-60", K=5, M=2, r_lo=20, r_hi=60),     # the existing baseline density
    dict(name="K12M4_r8-110", K=12, M=4, r_lo=8, r_hi=110),
    dict(name="K20M8_r8-120", K=20, M=8, r_lo=8, r_hi=120),
    dict(name="K32M10_r6-120", K=32, M=10, r_lo=6, r_hi=120),
]


def rot(pil, deg):
    return pil.rotate(deg, resample=Image.BILINEAR, expand=False)


def to_t(pil):
    return transforms.ToTensor()(pil.convert("RGB").resize((256, 256), Image.LANCZOS)).unsqueeze(0)


def main():
    files = sorted([f for f in Path(IMG_DIR).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[4000:4010]
    scales = np.round(np.arange(0.82, 1.221, 0.01), 3)
    angles = np.deg2rad(np.arange(-8, 8.1, 0.5))
    attacks = {
        "clean": (lambda p: p, [0.0]),
        "resize_110": (lambda p: apply_attack("resize_110", p), [0.0]),
        "resize_120": (lambda p: apply_attack("resize_120", p), [0.0]),
        "rot_+5": (lambda p: rot(p, 5), angles),
        "jpeg_50": (lambda p: apply_attack("jpeg_50", p), [0.0]),
    }

    for cfg in CONFIGS:
        enc = DFTKredEncoder(n_bits=100, K=cfg["K"], M=cfg["M"], resolution=256,
                             r_lo=cfg["r_lo"], r_hi=cfg["r_hi"], init_delta=50.0,
                             canonical_id=f"mellin_{cfg['name']}").to(DEVICE).eval()
        delta = float(F.softplus(enc.base_delta) + 1.0)
        carriers = enc.carriers.cpu().numpy(); bit_flip = enc.bit_flip.cpu().numpy()
        res = {a: {"fixed": [], "mellin": [], "scale": []} for a in attacks}
        for fp in files:
            img = to_t(Image.open(fp)).to(DEVICE)
            secret = np.random.RandomState(abs(hash(fp.name)) % (2**31)).randint(0, 2, 100).astype(np.uint8)
            st = torch.tensor(secret, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                wm = enc(img, st)
            wm_pil = transforms.ToPILImage()(wm[0].clamp(0, 1).cpu())
            for aname, (fn, angs) in attacks.items():
                att = fn(wm_pil)
                green = transforms.ToTensor()(att.convert("RGB").resize((256, 256), Image.LANCZOS))[1].numpy()
                # fixed (no search)
                bf = mellin_analytical(carriers, bit_flip, delta, green, [1.0], [0.0])
                res[aname]["fixed"].append(float(np.mean(bf["bits"] == secret)))
                # mellin consensus search
                bm = mellin_analytical(carriers, bit_flip, delta, green, scales, angs)
                res[aname]["mellin"].append(float(np.mean(bm["bits"] == secret)))
                res[aname]["scale"].append(bm["scale"])
        print(f"\n[{cfg['name']}] delta={delta:.1f} carriers/bit={cfg['K']*cfg['M']}")
        print(f"  {'attack':<12}{'fixed':>8}{'mellin':>8}{'rec.scale':>11}")
        for a in attacks:
            d = res[a]
            print(f"  {a:<12}{np.mean(d['fixed']):>8.3f}{np.mean(d['mellin']):>8.3f}{np.mean(d['scale']):>11.3f}", flush=True)


if __name__ == "__main__":
    main()
