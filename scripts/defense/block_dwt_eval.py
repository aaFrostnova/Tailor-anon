"""Decisive test for the block-DWT zero-bit watermark (arXiv:2512.14994 reimpl):
(A) FAITHFULNESS — reproduce the paper's robustness (JPEG/blur/noise/rotation/crop/
    brightness) as a global watermark-detection rate (frac of watermarked imgs whose
    global score >= P) and confirm FPR on unwatermarked imgs.
(B) DECISIVE REGEN — does it survive diffusion regeneration (the gap our composite
    needs)? Classical low-freq watermark -> predicted to DIE like TrustMark/MaskWM.
Detection rule: global_score >= P (paper p>=0.8 for FPR~0.01). Reports detection rate
+ mean global score per attack, and FPR.
"""
import os, sys, glob, io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.block_dwt_wm import BlockDWTWatermark
from _regen_util import build_regen_pipe, stable_regen

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
P = 0.8                       # global detection threshold (paper: FPR ~0.01)
L = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0   # paper uses l=14 for robustness expts
wm = BlockDWTWatermark(l=L, entropy_adaptive=True)   # paper's robustness table uses entropy-adaptive
pipe = build_regen_pipe()

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def arr(im): return np.asarray(to512(im).convert("RGB"), np.float64)
def im_of(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def a_jpeg(a, q):
    b = io.BytesIO(); im_of(a).save(b, "JPEG", quality=q); b.seek(0); return arr(Image.open(b))
def a_blur(a, r): return arr(im_of(a).filter(ImageFilter.GaussianBlur(r)))
def a_noise(a, s): return a + np.random.RandomState(7).normal(0, s * 255, a.shape)
def a_bright(a, f): return arr(ImageEnhance.Brightness(im_of(a)).enhance(f))
def a_rot(a, d): return arr(im_of(a).rotate(d, resample=Image.BILINEAR, expand=False))
def a_crop(a, area):
    # paper semantics: REMOVE 50% (zero the border), keep size + block alignment so
    # surviving centre blocks still detect (NOT crop+resize, which misaligns every block)
    s = int(round(512 * np.sqrt(area))); o = (512 - s) // 2
    out = np.zeros_like(a); out[o:o + s, o:o + s] = a[o:o + s, o:o + s]; return out
def a_regen(a, n):
    cur = im_of(a)
    for i in range(n): cur = to512(stable_regen(pipe, cur, seed=11 + i))
    return arr(cur)

ATTACKS = [("clean", lambda a: a), ("jpeg50", lambda a: a_jpeg(a, 50)), ("jpeg25", lambda a: a_jpeg(a, 25)),
           ("blur2", lambda a: a_blur(a, 2.0)), ("noise.05", lambda a: a_noise(a, 0.05)),
           ("bright0.5", lambda a: a_bright(a, 0.5)), ("rot90", lambda a: a_rot(a, 90)),
           ("crop0.5", lambda a: a_crop(a, 0.5)), ("regen_x1", lambda a: a_regen(a, 1)),
           ("regen_x2", lambda a: a_regen(a, 2))]

imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:180 + N]

# pre-embed once per image
embedded = []; ps = []
for fp in imgs:
    o = arr(Image.open(fp)); w = wm.embed(o); embedded.append((o, w))
    e = np.mean((o - w) ** 2); ps.append(99.0 if e < 1e-9 else 10 * np.log10(255 * 255 / e))
print(f"n={N} l={L} P={P} cstar={wm.cstar:.1f}  embed PSNR={np.mean(ps):.1f}dB\n", flush=True)

print(f"{'attack':10s} | {'det-rate':>8s} {'mean-score':>10s}")
print("-" * 34)
res = {}
for nm, fn in ATTACKS:
    scores = [wm.detect(fn(w))[0] for (o, w) in embedded]
    dr = float(np.mean([s >= P for s in scores])); ms = float(np.mean(scores))
    res[nm] = (dr, ms); print(f"{nm:10s} | {dr:8.2f} {ms:10.3f}", flush=True)

# FPR on UNwatermarked held-out images
neg = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
       sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[230:230 + N]
fpr = float(np.mean([wm.detect(arr(Image.open(fp)))[0] >= P for fp in neg]))
print(f"\n[FPR] unwatermarked detection rate = {fpr:.3f} (n={len(neg)})")
import json
json.dump({"n": N, "l": L, "P": P, "psnr": float(np.mean(ps)), "fpr": fpr,
           "rows": [{"attack": k, "det": v[0], "score": v[1]} for k, v in res.items()]},
          open(os.path.join(REPO, "results/defense/block_dwt_eval.json"), "w"), indent=2)
print("BLOCKDWT_EVAL_DONE")
