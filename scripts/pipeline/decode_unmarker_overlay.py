"""Decode the UnMarker-attacked images and emit a surrogate overlay for the `unmarker` column.

Three strength settings per fragment. Three is not an arbitrary economy: it is the smallest design
that can TEST the earlier probe's claim that this attack is strength-independent rather than assume
it. If the three agree within noise the curve is flat and that is a measured result; if they do not,
the probe was wrong and a denser sweep is warranted.
"""
import sys, os, json, glob, subprocess
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BASE = os.environ.get("UNMK_BASE", f"{SC}/unmarker_overlay")
sb = ShortenedBCH(); NB = sb.n; KEY = b"v5_key_encoder_master"; dev = "cuda"
FR = {"vine": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
      "trustmark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
      "videoseal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}
NAME = {"vine": "VINE", "trustmark": "TrustMark", "videoseal": "VideoSeal"}
# strengths are read from the directories that exist, so the same decoder serves the 3-knot probe and
# the full 9-knot sweep without a second copy that could drift from it
def _strengths(m):
    out = []
    for d in glob.glob(f"{BASE}/att_{m}_*"):
        try: out.append(float(os.path.basename(d).rsplit("_", 1)[1]))
        except ValueError: pass
    return sorted(out)
STR = {m: _strengths(m) for m in ("vine", "trustmark", "videoseal")}
print("strength grids read from disk:", STR, flush=True)

def hard(fn, pil):
    f = FR[fn]; v = f.raw_probs(pil) if fn == "vine" else f.raw_logits(pil)
    v = np.asarray(v).ravel()[:NB]
    return (v > (0.5 if fn == "vine" else 0.0)).astype(np.uint8)

base, cap, per = {}, {}, {}
for m, ss in STR.items():
    ys, sds = [], []
    for s in ss:
        att = sorted(glob.glob(f"{BASE}/att_{m}_{s}/*.png"))[:N]
        bas = []
        for fp in att:
            i = int(os.path.basename(fp)[1:6])
            t = np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
            img = Image.open(fp).convert("RGB")
            if img.size != (512, 512): img = img.resize((512, 512))
            bas.append(float(np.mean(hard(m, img) == t)))
        ys.append(float(np.mean(bas)) if bas else float("nan"))
        sds.append(float(np.std(bas, ddof=1)) if len(bas) > 1 else float("nan"))
        per[f"{NAME[m]}@{s}"] = {"mean": ys[-1], "sd": sds[-1], "n": len(bas)}
        print(f"  {NAME[m]:10s} s={s:<5} ba={ys[-1]:.4f} +/- {sds[-1]:.4f}  (n={len(bas)})", flush=True)
    base[f"{NAME[m]}|unmarker"] = {"xs": list(ss), "ys": [round(y, 4) for y in ys]}
    spread = float(np.nanmax(ys) - np.nanmin(ys)); noise = float(np.nanmean(sds) / np.sqrt(max(N, 1)))
    # a spread can exceed the noise by chance when many knots are compared, so the trend is also tested:
    # a real strength dependence shows up as a slope, not as one knot sitting apart from the rest
    xs_a = np.asarray(ss, float); ys_a = np.asarray(ys, float)
    ok = ~np.isnan(ys_a)
    slope = float(np.polyfit(xs_a[ok], ys_a[ok], 1)[0]) if ok.sum() >= 3 else float("nan")
    rng = float(xs_a.max() - xs_a.min())
    rise = slope * rng                                   # what the trend predicts end-to-end
    verdict = "STRENGTH-DEPENDENT" if (spread > 2 * noise and abs(rise) > 2 * noise) else "STRENGTH-INDEPENDENT"
    print(f"  -> {NAME[m]}: spread {spread:.4f}, SE {noise:.4f}, trend over the range {rise:+.4f}"
          f"  => {verdict}", flush=True)
    per[f"{NAME[m]}|verdict"] = {"spread": spread, "se": noise, "trend": rise, "verdict": verdict}

# every ordered pair needs a delta; not measured here, so declared zero and listed as such
delta = {}
FR_N = ["VINE", "TrustMark", "VideoSeal"]
RANGES = {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6), "VideoSeal": (0.5, 1.5)}
for g in FR_N:
    for f in FR_N:
        if g == f: continue
        lo, hi = RANGES[g]
        delta[f"{g}|{f}|unmarker"] = {"xs": [lo, hi], "ys": [0.0, 0.0]}
try:
    git = subprocess.run(["git","-C",CF,"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
except Exception: git = "unknown"
json.dump({"attacks_added": ["unmarker"], "base": base, "delta": delta, "cap": {},
           "per_cell": per,
           # the embeds (embed_at_strength.py) draw across the five sources at per-source offset 0, so the
           # stamp records that composition rather than a hard-coded pool name
           "source": {"n": N, "pool": __import__("src.image_pool", fromlist=["composition_of", "sample"]).composition_of(
                          __import__("src.image_pool", fromlist=["sample"]).sample(N, offset=0)),
                      "knots": {NAME[m]: len(STR[m]) for m in STR},
                      "git": git, "measured_at": os.environ.get("TS","unknown"),
                      "attack": "UnMarker (arXiv 2405.08363), two-stage CW optimiser with a purely "
                                "perceptual + spectral loss, run in its own environment via "
                                "scripts/attack/unmarker_batch.py with attack_configs/Vine.yaml. "
                                "Three strength settings per fragment, chosen to test rather than "
                                "assume strength-independence. Pairwise interference was not measured "
                                "and is declared zero.",
                      "caveat": "the solver treats this attack as ADVERSARIAL, so a request naming it "
                                "still forces a live measurement; this curve bounds the search, it "
                                "does not settle the verdict."}},
          open(f"{SC}/surrogate_unmarker_overlay.json", "w"), indent=2)
print(f"\nwrote {SC}/surrogate_unmarker_overlay.json"); print("UNMARKER_OVERLAY_DONE")
