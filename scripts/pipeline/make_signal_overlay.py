"""Re-measure the signal / photometric / neural-compression columns under the SHARED attack definitions.

The in-process campaign had restated these operators locally and the restatements were milder than the
ones the reported matrix runs: measured against the cover, campaign brightness came out at 23.9 dB where
the reported operator gives 11.5 dB, contrast 28.8 against 16.3, and noise 34.2 against 26.2. Only JPEG
agreed. The solver was consequently planning against a weaker signal family than it was scored on, which
is why signal-only requests were satisfied unconditionally.

This overlay imports src.attacks -- the one definition -- and re-measures base / delta / capacity for
that family, overriding the campaign's values in the canonical merge.

Usage: python make_signal_overlay.py [N=100]
"""
import sys, os, json, glob, time, subprocess
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker"):
    sys.path.insert(0, p)
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
from src.image_pool import sample as _pool_sample, composition_of
from src.attacks import attack_pil                            # the ONE definition

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
sb = ShortenedBCH(); NB = sb.n; KEY = b"v5_key_encoder_master"; dev = "cuda"
GRID = {f: [round(float(v), 4) for v in np.linspace(lo, hi, 9)]
        for f, (lo, hi) in {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6),
                            "VideoSeal": (0.5, 1.5)}.items()}
RANGES = {f: (GRID[f][0], GRID[f][-1]) for f in GRID}
MID = {f: GRID[f][len(GRID[f]) // 2] for f in GRID}
FRAGS = ["VINE", "TrustMark", "VideoSeal"]
PAIRS = [(a, b) for a in FRAGS for b in FRAGS if a != b]
# names as the surrogate keys them; attack_pil resolves the aliases to the shared operators
ATT = ["jpeg25", "blur", "noise", "bright", "contrast", "vaeB", "vaeC"]
print(f"N={N} attacks={ATT} (shared definitions from src.attacks)", flush=True)

FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
      "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
      "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}

def tb(i): return np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
def r512(im): return im if im.size == (512, 512) else im.resize((512, 512))
def llr(fn, pil):
    f = FR[fn]
    if fn == "VINE":
        p = np.asarray(f.raw_probs(pil)).ravel()[:NB]
        return np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1-1e-6))
    return np.asarray(f.raw_logits(pil)).ravel()[:NB].astype(np.float64)
def hard(fn, pil): return (llr(fn, pil) > 0).astype(np.uint8)
def _Hb(p):
    p = np.clip(p, 1e-12, 1-1e-12); return -(p*np.log2(p) + (1-p)*np.log2(1-p))
def _mi(truth, l, K=25):
    b = np.asarray(truth).ravel().astype(np.float64); s = np.asarray(l).ravel()
    H0 = _Hb(b.mean()); qs = np.quantile(s, np.linspace(0, 1, K+1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    idx = np.clip(np.digitize(s, qs[1:-1]), 0, K-1); H = 0.0
    for k in range(K):
        m = idx == k
        if m.any(): H += m.mean() * _Hb(b[m].mean())
    return float(max(0.0, H0 - H))

files = _pool_sample(N, offset=0)   # across all five sources, in the evaluation set's proportions
assert len(files) == N, f"pool sample short: {len(files)}"
print('sources:', composition_of(files), flush=True)
solo = {(a, f, s): [] for a in ATT for f in FRAGS for s in GRID[f]}
pair = {(a, f1, f2): [] for a in ATT for (f1, f2) in PAIRS}
cT = {(a, f, s): [] for a in ATT for f in FRAGS for s in GRID[f]}
cL = {(a, f, s): [] for a in ATT for f in FRAGS for s in GRID[f]}
t0 = time.time()
for i, fp in enumerate(files):
    cover = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC)
    t = tb(i)
    emb = {f: {s: r512(FR[f].embed_with_target(cover, t, strength=s)) for s in GRID[f]} for f in FRAGS}
    comp = {(f1, f2): r512(FR[f2].embed_with_target(emb[f1][MID[f1]], t, strength=MID[f2]))
            for (f1, f2) in PAIRS}
    for a in ATT:
        for f in FRAGS:
            for s in GRID[f]:
                v = r512(attack_pil(a, emb[f][s], dev=dev))
                L = llr(f, v)
                solo[(a, f, s)].append(float(np.mean((L > 0).astype(np.uint8) == t)))
                cT[(a, f, s)].append(t.copy()); cL[(a, f, s)].append(L)
        for (f1, f2) in PAIRS:
            pair[(a, f1, f2)].append(float(np.mean(hard(f1, r512(attack_pil(a, comp[(f1, f2)], dev=dev))) == t)))
    if (i+1) % 5 == 0: print(f"  ...{i+1}/{N} [{time.time()-t0:.0f}s]", flush=True)

mean = lambda L: float(np.mean(L)) if L else float("nan")
base, delta, cap = {}, {}, {}
for a in ATT:
    for f in FRAGS:
        ys = [mean(solo[(a, f, s)]) for s in GRID[f]]
        base[f"{f}|{a}"] = {"xs": list(GRID[f]), "ys": [round(y, 4) for y in ys]}
        cap[f"{f}|{a}"] = {"xs": list(GRID[f]),
                           "ys": [round(NB*_mi(np.concatenate(cT[(a,f,s)]), np.concatenate(cL[(a,f,s)])), 3)
                                  for s in GRID[f]]}
        print(f"[{a}] {f:10s} ba=" + ",".join(f"{y:.3f}" for y in ys), flush=True)
    for (f1, f2) in PAIRS:
        dv = max(0.0, mean(solo[(a, f1, MID[f1])]) - mean(pair[(a, f1, f2)]))
        lo, hi = RANGES[f2]
        delta[f"{f2}|{f1}|{a}"] = {"xs": [lo, hi], "ys": [round(dv, 4), round(dv, 4)]}
try:
    git_rev = subprocess.run(["git","-C",CF,"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
except Exception: git_rev = "unknown"
json.dump({"attacks_added": ATT, "base": base, "delta": delta, "cap": cap,
           "source": {"n": N, "pool": composition_of(files), "slice": [0, N], "git": git_rev,
                      "measured_at": os.environ.get("TS", "unknown"),
                      "attack": "src.attacks.attack_pil -- the same operators and parameters the "
                                "reported matrix runs; OVERRIDES the campaign's milder restatements"}},
          open(f"{SC}/surrogate_signal_overlay.json", "w"), indent=2)
print(f"wrote {SC}/surrogate_signal_overlay.json"); print("SIGNAL_OVERLAY_DONE")
