"""Can Fourier-Mellin (logpolar_reffree.mellin_analytical) rescue rot9 for the DFT-Kred carrier?

Embed DFT-Kred (analytic FFT-magnitude QIM), rotate 9°, then decode three ways:
  (a) clean, angle=0           -> sanity (should be ~1.0)
  (b) rot9, angle=0 (no sync)  -> should collapse to ~0.5 (carriers moved)
  (c) rot9, angle-SEARCH       -> should recover if Fourier-Mellin works
Reports mean bit-acc + the recovered rotation angle.
"""
import argparse, glob, os, sys
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.learned_fragment_methods import DFTKredMethod
from src.logpolar_reffree import mellin_analytical
KEY = b"v5_key_encoder_master"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"


def rotate_reflect(img, deg):
    W, H = img.size; arr = np.array(img); pad = max(W, H) // 2
    refl = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    big = Image.fromarray(refl).rotate(deg, resample=Image.BICUBIC, expand=False)
    bw, bh = big.size; l, t = (bw - W) // 2, (bh - H) // 2
    return big.crop((l, t, l + W, t + H))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_images", type=int, default=15)
    ap.add_argument("--deg", type=float, default=9.0); args = ap.parse_args()
    dev = "cuda"
    dft = DFTKredMethod(os.path.join(REPO, "results/dft_fftaware_baseline/ckpt.pt"), KEY, "dft_kred", dev)
    res = dft.resolution
    enc = dft.enc
    carriers = enc.carriers.cpu().numpy()            # (n_bits, K*M, 2)
    bit_flip = enc.bit_flip.cpu().numpy()
    delta = float(F.softplus(enc.base_delta) + 1.0)
    n_bits = dft.n_bits
    print(f"[rot9-mellin] res={res} n_bits={n_bits} n_pos={carriers.shape[1]} delta={delta:.2f}", flush=True)

    def green256(pil):
        g = np.asarray(pil.convert("RGB").resize((res, res)), np.float64)[:, :, 1] / 255.0
        return g

    imgs = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n_images]
    angles_search = np.deg2rad(np.arange(-14.0, 14.01, 0.5))
    scales = [1.0]
    A = {"clean": [], "rot_nosync": [], "rot_search": []}; rec_ang = []
    for i, fp in enumerate(imgs):
        o = Image.open(fp).convert("RGB").resize((res, res))
        bits = np.random.RandomState(i).randint(0, 2, n_bits).astype(np.float32)
        xt = torch.from_numpy(np.asarray(o, np.float32) / 255).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            wm = enc(xt, torch.from_numpy(bits)[None].to(dev))[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        wm_pil = Image.fromarray((wm * 255).astype(np.uint8))
        rot_pil = rotate_reflect(wm_pil, args.deg)
        bi = bits.astype(np.uint8)
        # (a) clean angle 0
        r = mellin_analytical(carriers, bit_flip, delta, green256(wm_pil), scales, angles=(0.0,))
        A["clean"].append(np.mean(r["bits"] == bi))
        # (b) rot, angle 0
        r = mellin_analytical(carriers, bit_flip, delta, green256(rot_pil), scales, angles=(0.0,))
        A["rot_nosync"].append(np.mean(r["bits"] == bi))
        # (c) rot, angle search
        r = mellin_analytical(carriers, bit_flip, delta, green256(rot_pil), scales, angles=angles_search)
        A["rot_search"].append(np.mean(r["bits"] == bi)); rec_ang.append(np.rad2deg(r["angle"]))
        print(f"  [{i+1}/{len(imgs)}] clean={A['clean'][-1]:.2f} rot0={A['rot_nosync'][-1]:.2f} "
              f"search={A['rot_search'][-1]:.2f} @{np.rad2deg(r['angle']):+.1f}°", flush=True)

    print(f"\n=== Fourier-Mellin rot{args.deg:.0f}° rescue (DFT-Kred, n={len(imgs)}) ===")
    print(f"clean (sanity)      bit-acc = {np.mean(A['clean']):.3f}")
    print(f"rot{args.deg:.0f}, no sync       bit-acc = {np.mean(A['rot_nosync']):.3f}")
    print(f"rot{args.deg:.0f}, angle-SEARCH  bit-acc = {np.mean(A['rot_search']):.3f}   (recovered angle mean {np.mean(rec_ang):+.1f}°)")
    print("ROT9_MELLIN_DONE")


if __name__ == "__main__":
    main()
