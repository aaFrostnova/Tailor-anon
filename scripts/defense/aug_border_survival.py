"""Test the user's hypothesis: do VINE's training augmentations (blur/jpeg/noise) degrade a signal LESS at the
BORDER than in the interior? Place the SAME random pattern at a corner / edge-mid / center patch, apply each aug,
and measure survival = corr(delta, aug(img+delta)-aug(img)) in the patch. If border survival > center survival,
the augs are gentler at the border -> the encoder would put the payload there (survival-driven placement)."""
import glob, io, numpy as np
from PIL import Image, ImageFilter
rng = np.random.RandomState(0); IMG = 256; P = 48; A = 0.12
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:12]
imgs = [np.asarray(Image.open(s).convert("RGB").resize((IMG, IMG)), np.float64) / 255 for s in srcs]
delta = rng.randn(P, P, 3) * A                                   # fixed pattern, same for all placements
places = {"corner": (0, 0), "edge-mid": (0, IMG // 2 - P // 2), "center": (IMG // 2 - P // 2, IMG // 2 - P // 2)}
def blur(a): return np.asarray(Image.fromarray((a * 255).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.0)), np.float64) / 255
def jpeg(a):
    b = io.BytesIO(); Image.fromarray((a * 255).clip(0, 255).astype(np.uint8)).save(b, "JPEG", quality=40)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float64) / 255
def noise(a): return a + rng.randn(IMG, IMG, 3) * 0.05
def resize(a):  # downscale-upscale (VINE's resize aug)
    im = Image.fromarray((a * 255).clip(0, 255).astype(np.uint8)); s = int(IMG * 0.6)
    return np.asarray(im.resize((s, s)).resize((IMG, IMG)), np.float64) / 255
augs = {"blur_s2": blur, "jpeg_q40": jpeg, "noise.05": noise, "resize.6": resize}
def surv(aug, img, r0, c0):
    prop = (aug(np.clip(img + place_delta(img, r0, c0), 0, 1)) - aug(img))[r0:r0+P, c0:c0+P]  # propagated delta in patch
    d = delta.ravel() - delta.mean(); p = prop.ravel() - prop.mean()
    return float((d * p).sum() / (np.sqrt((d**2).sum() * (p**2).sum()) + 1e-9))
def place_delta(img, r0, c0):
    z = np.zeros_like(img); z[r0:r0+P, c0:c0+P] = delta; return z
print(f"survival = corr(delta, propagated delta) in the {P}x{P} patch, averaged over {len(imgs)} images")
print(f"{'aug':10} | " + " | ".join(f"{k:>9}" for k in places))
for aname, aug in augs.items():
    row = {pl: np.mean([surv(aug, img, r0, c0) for img in imgs]) for pl, (r0, c0) in places.items()}
    print(f"{aname:10} | " + " | ".join(f"{row[k]:9.3f}" for k in places) +
          f"   border/center = {(0.5*(row['corner']+row['edge-mid'])/max(row['center'],1e-6)):.2f}")
print("AUG_SURVIVAL_DONE")
