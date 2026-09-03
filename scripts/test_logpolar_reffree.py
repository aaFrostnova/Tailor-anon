"""Demonstrate ref-free resize/rotation robustness via Fourier-Mellin scale-search,
reusing the trained DFT-Kred carriers (no retrain). Compares fixed-bin decode vs
scale-search decode under resize and rotation."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from src.learned_fragment_methods import DFTKredMethod
from src.logpolar_reffree import scale_search_logits
from benchmark_fused import apply_attack

DEVICE = "cuda"
KEY = b"v5_key_encoder_master"
DFT_CKPT = str(REPO / "results/dft_fftaware_baseline/ckpt.pt")
IMG_DIR = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"


def rot(pil, deg):
    return pil.rotate(deg, resample=Image.BILINEAR, expand=False)


def green256(method, pil):
    return method._img_to_tensor(pil)[0, 1].cpu().numpy()


def main():
    method = DFTKredMethod(DFT_CKPT, KEY, "dft_kred", DEVICE)
    n_bits = method.n_bits
    files = sorted([f for f in Path(IMG_DIR).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[4000:4012]
    scales = np.round(np.arange(0.85, 1.211, 0.01), 3)
    angles_deg = np.arange(-8, 8.1, 1.0)
    angles = np.deg2rad(angles_deg)

    # attack -> (function, whether to also search rotation)
    attacks = {
        "clean": (lambda p: p, False),
        "resize_105": (lambda p: apply_attack("resize_105", p), False),
        "resize_110": (lambda p: apply_attack("resize_110", p), False),
        "resize_120": (lambda p: apply_attack("resize_120", p), False),
        "rot_+5": (lambda p: rot(p, 5), True),
        "rot_-5": (lambda p: rot(p, -5), True),
        "jpeg_50": (lambda p: apply_attack("jpeg_50", p), False),
    }
    agg = {a: {"fixed": [], "search": [], "scale": [], "angle": [], "oracle": []} for a in attacks}

    for fp in files:
        pil = Image.open(fp).convert("RGB").resize((256, 256), Image.LANCZOS)
        img = method._img_to_tensor(pil)
        secret = np.random.RandomState(hash(fp.name) % (2**31)).randint(0, 2, n_bits).astype(np.uint8)
        st = torch.tensor(secret, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            wm = method.enc(img, st)
        wm_pil = transforms.ToPILImage()(wm[0].clamp(0, 1).cpu())

        for aname, (fn, do_rot) in attacks.items():
            att = fn(wm_pil)
            # fixed-bin decode
            fixed_logits = method.raw_logits(att)
            agg[aname]["fixed"].append(float(np.mean((fixed_logits > 0).astype(np.uint8) == secret)))
            # scale(+angle)-search decode
            ang = angles if do_rot else (0.0,)
            logits, a_hat, th_hat, conf = scale_search_logits(
                method.dec, green256(method, att), scales, ang, DEVICE)
            agg[aname]["search"].append(float(np.mean((logits > 0).astype(np.uint8) == secret)))
            agg[aname]["scale"].append(a_hat); agg[aname]["angle"].append(np.rad2deg(th_hat))
            # ORACLE: best bit-acc over the scale grid (cheats by using true bits) ->
            # tells whether the signal survives at ANY scale (vs is destroyed).
            gnp = green256(method, att)
            best_acc = 0.0
            for a_try in scales:
                lg, _, _, _ = scale_search_logits(method.dec, gnp, [a_try], (0.0,), DEVICE)
                best_acc = max(best_acc, float(np.mean((lg > 0).astype(np.uint8) == secret)))
            agg[aname]["oracle"].append(best_acc)

    print(f"\n{'='*78}")
    print(f"DFT-Kred fixed-bin vs Fourier-Mellin scale-search (ref-free), {len(files)} imgs @256")
    print(f"{'attack':<14}{'fixed':>8}{'search':>8}{'oracle':>8}{'rec.scale':>11}{'rec.ang':>9}")
    print("-" * 78)
    for a in attacks:
        d = agg[a]
        print(f"{a:<14}{np.mean(d['fixed']):>8.3f}{np.mean(d['search']):>8.3f}"
              f"{np.mean(d['oracle']):>8.3f}{np.mean(d['scale']):>11.3f}{np.mean(d['angle']):>9.1f}")
    print("=" * 78)
    print("oracle = best bit-acc over the scale grid (signal-survival upper bound).")


if __name__ == "__main__":
    main()
