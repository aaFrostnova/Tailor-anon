"""Isolate WHY the frequency carrier fails 'resize': pure affine scale/rotation
(reflection-padded, no crop) vs the harness resize_110 (enlarge + center crop).
If Mellin recovers pure scale/rotation but not resize_110, the CROP is the killer."""
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


def affine(x, scale=1.0, rot_deg=0.0):
    B, dev = x.shape[0], x.device
    a = torch.tensor(rot_deg * np.pi / 180.0)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = torch.cos(a) / scale; th[:, 0, 1] = -torch.sin(a) / scale
    th[:, 1, 0] = torch.sin(a) / scale; th[:, 1, 1] = torch.cos(a) / scale
    grid = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


def main():
    files = sorted([f for f in Path(IMG_DIR).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[4000:4008]
    enc = DFTKredEncoder(n_bits=100, K=12, M=4, resolution=256, r_lo=8, r_hi=110,
                         init_delta=50.0, canonical_id="mellin_isolate").to(DEVICE).eval()
    delta = float(F.softplus(enc.base_delta) + 1.0)
    carr = enc.carriers.cpu().numpy(); bf = enc.bit_flip.cpu().numpy()
    scales = np.round(np.arange(0.82, 1.221, 0.01), 3)
    angs = np.deg2rad(np.arange(-8, 8.1, 0.5))
    t = transforms.ToTensor()

    def decode(att_t, secret, ang_grid):
        green = att_t[0, 1].cpu().numpy()
        b = mellin_analytical(carr, bf, delta, green, scales, ang_grid)
        return float(np.mean(b["bits"] == secret)), b["scale"], np.rad2deg(b["angle"])

    conds = ["clean", "affine_scale_1.12", "affine_rot_5", "resize_110(crop)", "crop_90"]
    agg = {c: [] for c in conds}
    for fp in files:
        img = t(Image.open(fp).convert("RGB").resize((256, 256), Image.LANCZOS)).unsqueeze(0).to(DEVICE)
        secret = np.random.RandomState(abs(hash(fp.name)) % (2**31)).randint(0, 2, 100).astype(np.uint8)
        st = torch.tensor(secret, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            wm = enc(img, st)
        wm_pil = transforms.ToPILImage()(wm[0].clamp(0, 1).cpu())
        agg["clean"].append(decode(wm, secret, [0.0]))
        agg["affine_scale_1.12"].append(decode(affine(wm, scale=1.12), secret, [0.0]))
        agg["affine_rot_5"].append(decode(affine(wm, rot_deg=5.0), secret, angs))
        rz = t(apply_attack("resize_110", wm_pil).convert("RGB").resize((256, 256), Image.LANCZOS)).unsqueeze(0)
        agg["resize_110(crop)"].append(decode(rz, secret, [0.0]))
        cr = t(apply_attack("crop_90", wm_pil).convert("RGB").resize((256, 256), Image.LANCZOS)).unsqueeze(0)
        agg["crop_90"].append(decode(cr, secret, [0.0]))

    print(f"\nMellin analytical (K12M4), {len(files)} imgs")
    print(f"{'condition':<20}{'bit-acc':>9}{'rec.scale':>11}{'rec.ang':>9}")
    for c in conds:
        accs = [x[0] for x in agg[c]]; sc = [x[1] for x in agg[c]]; an = [x[2] for x in agg[c]]
        print(f"{c:<20}{np.mean(accs):>9.3f}{np.mean(sc):>11.3f}{np.mean(an):>9.1f}", flush=True)


if __name__ == "__main__":
    main()
