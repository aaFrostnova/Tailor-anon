"""Decisive pre-test for CDBW content-binding: is a robust perceptual hash H_img stable enough?
Content-binding works iff, for the attacks under which VideoSeal (the binding carrier) survives,
the SAME-image hash drift (clean vs attacked) is << the CROSS-image distance. Measures a
64-bit DCT pHash on real images: per-attack same-image Hamming drift vs the cross-image
Hamming distribution -> tells us which attacks admit binding and what tolerance tau_ham to use.
"""
import os, io, glob, argparse
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy.fft import dct

def phash(pil, hsz=8, imsz=32):
    g = np.asarray(pil.convert("L").resize((imsz, imsz)), np.float32)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:hsz, :hsz]
    med = np.median(d[1:].flatten())          # exclude DC
    bits = (d >= med).flatten()
    bits[0] = 0
    return bits.astype(np.uint8)               # 64-bit hash
def ham(a, b): return int(np.sum(a != b))

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def a_jpeg(x, q): b = io.BytesIO(); x.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def a_blur(x, s): return x.filter(ImageFilter.GaussianBlur(s))
def a_noise(x, s):
    return Image.fromarray(np.clip(np.asarray(x, np.float64) + np.random.RandomState(0).normal(0, s*255, (512,512,3)), 0, 255).astype(np.uint8))
def a_bright(x, f): return ImageEnhance.Brightness(x).enhance(f)
def a_crop(x, r): s = int(512*r); o = (512-s)//2; return x.crop((o,o,o+s,o+s)).resize((512,512))
def a_rot(x, a): return x.rotate(a, resample=Image.BILINEAR)
ATT = {"jpeg25": lambda x: a_jpeg(x,25), "jpeg10": lambda x: a_jpeg(x,10),
       "blur2": lambda x: a_blur(x,2), "noise0.05": lambda x: a_noise(x,0.05),
       "bright1.4": lambda x: a_bright(x,1.4), "crop90": lambda x: a_crop(x,0.9),
       "crop75": lambda x: a_crop(x,0.75), "rot5": lambda x: a_rot(x,5), "rot9": lambda x: a_rot(x,9)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/defense/ext_vtv100")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(a.src, a.glob)))[:a.n]
    imgs = [to512(Image.open(f).convert("RGB")) for f in fs]
    clean_h = [phash(im) for im in imgs]
    # cross-image distance distribution (different images)
    cross = [ham(clean_h[i], clean_h[j]) for i in range(len(imgs)) for j in range(i+1, min(i+11, len(imgs)))]
    cross = np.array(cross)
    print(f"pHash robustness (64-bit), n={len(imgs)}")
    print(f"CROSS-image Hamming: mean={cross.mean():.1f}  p05={np.percentile(cross,5):.0f}  min={cross.min()}")
    print(f"\n{'attack':10s} | same-img drift (mean / p95 / max) | separated from cross-img?")
    print("-"*70)
    for name, fn in ATT.items():
        drift = np.array([ham(clean_h[i], phash(to512(fn(imgs[i])))) for i in range(len(imgs))])
        # a tolerance tau exists iff max same-img drift < p05 cross-img (clean separation)
        sep = drift.max() < np.percentile(cross, 5)
        marg = int(np.percentile(cross, 5) - np.percentile(drift, 95))
        print(f"{name:10s} | {drift.mean():5.1f} / {np.percentile(drift,95):4.0f} / {drift.max():3d}          | {'YES' if sep else 'NO ':3s}  (p95→p05cross margin {marg:+d} bits)")
    print("\nBinding viable for an attack iff same-image drift stays well below cross-image distance.")
    print("PHASH_DONE")

if __name__ == "__main__":
    main()

def tau_analysis():
    import glob as _g, os as _o
    fs = sorted(_g.glob("results/defense/ext_vtv100/*.png"))[:100]
    imgs = [to512(Image.open(f).convert("RGB")) for f in fs]
    H = [phash(im) for im in imgs]
    cross = np.array([ham(H[i], H[j]) for i in range(len(imgs)) for j in range(i+1, len(imgs))])
    drift = {name: np.array([ham(H[i], phash(to512(fn(imgs[i])))) for i in range(len(imgs))]) for name, fn in ATT.items()}
    print("\n=== end-to-end binding rates at tolerance tau_ham (genuine PASS / copy CATCH) ===")
    print(f"{'tau':>4} | copy-catch | " + " ".join(f"{n[:7]:>7s}" for n in ATT))
    for tau in [10, 12, 14, 16, 18]:
        catch = float(np.mean(cross > tau))
        row = " ".join(f"{np.mean(drift[n] <= tau):7.2f}" for n in ATT)
        print(f"{tau:>4} | {catch:10.3f} | {row}")
    print("(genuine PASS = P(same-img drift<=tau) [want ~1]; copy CATCH = P(cross-img>tau) [want ~1])")

if __name__ != "__main__":
    pass
