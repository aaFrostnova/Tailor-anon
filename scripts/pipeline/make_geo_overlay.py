"""Measure the geometric / resampling attack columns straight from eval_matrix.py's OWN definitions.

Why this exists. The in-process surrogate campaign re-implemented crop and rotation locally, and the
two implementations had drifted apart from the ones the reported matrix uses:
  * crop75/crop50 -- the campaign read the fraction as an AREA fraction (side = sqrt(k)), the matrix
    reads it as a SIDE fraction. "crop75" therefore named a visibly milder attack in the table the
    solver reasons over than in the table we report.
  * rot9 -- the campaign rotated with the default fill, leaving black corners; the matrix pads by
    reflection first and crops back, so no corner content is lost. VINE carries its payload in a
    border ring, so which of the two is used changes VINE's rotation number outright.
This overlay imports GEO / _crop_then_jpeg from eval_matrix.py instead of restating them, so there is
exactly one definition of each attack. It re-measures crop75/crop50/rot9 under the canonical
definition and adds the three columns the surrogate never had: rs256, hflip, crop_jpeg.

Emits base/delta/cap for those six columns; merged over the older values by make_canonical_surrogate.

Usage: python make_geo_overlay.py [N=100]
"""
import sys, os, json, glob, time, subprocess, io
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense"):
    sys.path.insert(0, p)
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
from src.image_pool import sample as _pool_sample, composition_of
from eval_matrix import GEO, _center_crop_resize          # the SAME callables the matrix uses

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SLICE_START = 0
sb = ShortenedBCH(); NB = sb.n; KEY = b"v5_key_encoder_master"; dev = "cuda"
GRID = {f: [round(float(v), 4) for v in np.linspace(lo, hi, 9)]
        for f, (lo, hi) in {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6),
                            "VideoSeal": (0.5, 1.5)}.items()}
RANGES = {f: (GRID[f][0], GRID[f][-1]) for f in GRID}
MID = {f: GRID[f][len(GRID[f]) // 2] for f in GRID}
FRAGS = ["VINE", "TrustMark", "VideoSeal"]
ORDERED_PAIRS = [(a, b) for a in FRAGS for b in FRAGS if a != b]

def _crop_jpeg_pil(im, frac=0.75, quality=25):
    """eval_matrix._crop_then_jpeg operates on file paths; same operation, in memory."""
    img = _center_crop_resize(im.convert("RGB"), frac)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

ATT = {"crop75": GEO["crop75"], "crop50": GEO["crop50"], "rot9": GEO["rot9"],
       "rs256": GEO["rs256"], "hflip": GEO["hflip"], "crop_jpeg": _crop_jpeg_pil,
       # A translation crop, unlike every other crop here: the content moves off the embedding grid
       # rather than staying centred and rescaled. Without this column a front-end built for it can
       # only ever measure as zero, so it could never be selected.
       "border20": GEO["border20"]}
print(f"N={N} attacks={list(ATT)} (crop75/crop50/rot9 re-measured under the matrix's definition)", flush=True)

print("loading fragments...", flush=True)
FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
      "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
      "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}

def target_bits(i): return np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
def r512(im): return im if im.size == (512, 512) else im.resize((512, 512))
def soft_llr(fn, pil):
    f = FR[fn]
    if fn == "VINE":
        p = np.asarray(f.raw_probs(pil)).ravel()[:NB]
        return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    return np.asarray(f.raw_logits(pil)).ravel()[:NB].astype(np.float64)
def hard(fn, pil): return (soft_llr(fn, pil) > 0).astype(np.uint8)

# capacity: same non-parametric I(bit;llr) estimator the main campaign uses
def _Hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
def _mi_per_bit(truth, llr, K=25):
    b = np.asarray(truth).ravel().astype(np.float64); s = np.asarray(llr).ravel()
    Hprior = _Hb(b.mean())
    qs = np.quantile(s, np.linspace(0, 1, K + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    idx = np.clip(np.digitize(s, qs[1:-1]), 0, K - 1)
    H = 0.0
    for k in range(K):
        m = idx == k
        if not m.any(): continue
        H += (m.mean()) * _Hb(b[m].mean())
    return float(max(0.0, Hprior - H))

files = _pool_sample(N, offset=0)   # across all five sources, in the evaluation set's proportions
assert len(files) == N, f"pool sample short: {len(files)}"
print('sources:', composition_of(files), flush=True)
solo = {(a, f, s): [] for a in ATT for f in FRAGS for s in GRID[f]}
pair = {(a, f1, f2): [] for a in ATT for (f1, f2) in ORDERED_PAIRS}
capT = {(a, f, s): [] for a in ATT for f in FRAGS for s in GRID[f]}
capL = {(a, f, s): [] for a in ATT for f in FRAGS for s in GRID[f]}
t0 = time.time()
for i, fp in enumerate(files):
    cover = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC)
    t = target_bits(SLICE_START + i)
    emb = {f: {s: r512(FR[f].embed_with_target(cover, t, strength=s)) for s in GRID[f]} for f in FRAGS}
    comp = {(f1, f2): r512(FR[f2].embed_with_target(emb[f1][MID[f1]], t, strength=MID[f2]))
            for (f1, f2) in ORDERED_PAIRS}
    for a, fn in ATT.items():
        for f in FRAGS:
            for s in GRID[f]:
                v = r512(fn(emb[f][s]))
                llr = soft_llr(f, v)
                solo[(a, f, s)].append(float(np.mean((llr > 0).astype(np.uint8) == t)))
                capT[(a, f, s)].append(t.copy()); capL[(a, f, s)].append(llr)
        for (f1, f2) in ORDERED_PAIRS:
            pair[(a, f1, f2)].append(float(np.mean(hard(f1, r512(fn(comp[(f1, f2)]))) == t)))
    if (i + 1) % 5 == 0: print(f"  ...{i+1}/{N} [{time.time()-t0:.0f}s]", flush=True)

mean = lambda L: float(np.mean(L)) if L else float("nan")
base, delta, cap = {}, {}, {}
for a in ATT:
    for f in FRAGS:
        ys = [mean(solo[(a, f, s)]) for s in GRID[f]]
        base[f"{f}|{a}"] = {"xs": list(GRID[f]), "ys": [round(y, 4) for y in ys]}
        cy = [round(NB * _mi_per_bit(np.concatenate(capT[(a, f, s)]),
                                     np.concatenate(capL[(a, f, s)])), 3) for s in GRID[f]]
        cap[f"{f}|{a}"] = {"xs": list(GRID[f]), "ys": cy}
        print(f"[{a}] {f:10s} ba=" + ",".join(f"{y:.3f}" for y in ys) + "  cap=" +
              ",".join(f"{y:.1f}" for y in cy), flush=True)
    for (f1, f2) in ORDERED_PAIRS:
        dv = max(0.0, mean(solo[(a, f1, MID[f1])]) - mean(pair[(a, f1, f2)]))
        lo, hi = RANGES[f2]
        delta[f"{f2}|{f1}|{a}"] = {"xs": [lo, hi], "ys": [round(dv, 4), round(dv, 4)]}
    print(f"[{a}] deltas: " + ", ".join(f"{f2}->{f1}:{delta[f'{f2}|{f1}|{a}']['ys'][0]:.4f}"
                                        for (f1, f2) in ORDERED_PAIRS), flush=True)
try:
    git_rev = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
except Exception:
    git_rev = "unknown"
json.dump({"attacks_added": list(ATT), "base": base, "delta": delta, "cap": cap,
           "source": {"n": N, "pool": composition_of(files), "slice": [SLICE_START, SLICE_START + N],
                      "measured_at": os.environ.get("TS", "unknown"), "git": git_rev,
                      "attack": "imported verbatim from eval_matrix.GEO / _crop_then_jpeg so the "
                                "surrogate and the reported matrix name the same operations; this "
                                "OVERRIDES the campaign's earlier crop75/crop50/rot9, which used a "
                                "different crop convention (area vs side) and a black-corner rotation.",
                      "grids": GRID, "mid": MID}},
          open(f"{SC}/surrogate_geo_overlay.json", "w"), indent=2)
print(f"wrote {SC}/surrogate_geo_overlay.json", flush=True)
print("GEO_OVERLAY_DONE", flush=True)
