"""Same-type overwrite: does interference peak when the 2nd payload is the
COMPLEMENT of the 1st? Sweep Hamming distance d(A,B) and measure how much the
first watermark A survives a second embed B on top.

Embed A -> A-marked; embed B (d-far from A) on A-marked -> AB-marked.
Decode A on AB-marked = retention of first mark. Decode B = does overwriter win.
Linear/spread-spectrum theory: ba(A) = 1 - 0.5*d, minimized at d=1 (complement).
DwtDct = linear control; VINE-R / TrustMark-B = learned (test if they obey it).
"""
import os, sys, glob
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from wbench.methods import TrustMarkMethod, VineMethod, InvisibleWMMethod

N = 10
DVALS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
imgs = sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png")))[:N]

def ba(rec, bits):
    n = min(len(rec), len(bits)); return float(np.mean(rec[:n] == bits[:n]))

def flip_to_distance(A, d, rng):
    """Return B = A with exactly round(d*n) bits flipped (random positions)."""
    n = len(A); k = int(round(d * n)); B = A.copy()
    if k > 0:
        idx = rng.choice(n, size=k, replace=False); B[idx] ^= 1
    return B

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im

def run(m, label):
    retA = {d: [] for d in DVALS}; survB = {d: [] for d in DVALS}
    for fp in imgs:
        orig = to512(Image.open(fp).convert("RGB"))
        rng = np.random.RandomState(hash((fp, label)) % (2**31))
        A = rng.randint(0, 2, m.n_bits).astype(np.uint8)
        Amark = to512(m.embed(orig, A))
        for d in DVALS:
            B = flip_to_distance(A, d, rng)
            AB = to512(m.embed(Amark, B))
            retA[d].append(ba(m.decode(AB), A))
            survB[d].append(ba(m.decode(AB), B))
        print(f"  [{label}] {os.path.basename(fp)} done", flush=True)
    rA = [float(np.mean(retA[d])) for d in DVALS]
    sB = [float(np.mean(survB[d])) for d in DVALS]
    print(f"=== {label} (n={N}) ===")
    print("   d      :  " + "  ".join(f"{d:.3f}" for d in DVALS))
    print("   ba(A)  :  " + "  ".join(f"{v:.3f}" for v in rA) + "   <- first-mark retention")
    print("   ba(B)  :  " + "  ".join(f"{v:.3f}" for v in sB) + "   <- overwriter survival")
    return rA, sB

results = {}
print("Loading DwtDct (linear control)..."); results["DwtDct(32b,linear)"] = run(InvisibleWMMethod("dwtDct"), "DwtDct(32b,linear)")
print("Loading TrustMark-B...");            results["TrustMark-B"]        = run(TrustMarkMethod("B"), "TrustMark-B")
print("Loading VINE-R...");                 results["VINE-R"]             = run(VineMethod("R"), "VINE-R")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for lab, (rA, sB) in results.items():
    ax[0].plot(DVALS, rA, "-o", label=lab, lw=2)
    ax[1].plot(DVALS, sB, "-o", label=lab, lw=2)
lin = [1 - 0.5 * d for d in DVALS]
ax[0].plot(DVALS, lin, "k--", lw=1.5, label="linear theory 1-0.5d")
ax[0].axhline(0.5, color="grey", ls=":", lw=1)
ax[0].set_title("First-mark retention ba(A) vs payload Hamming distance d(A,B)")
ax[0].set_xlabel("d(A,B)  (0=same payload, 1=complement)"); ax[0].set_ylabel("bit-acc of A after overwrite")
ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_ylim(0.45, 1.02)
ax[1].axhline(0.5, color="grey", ls=":", lw=1)
ax[1].set_title("Overwriter B survival"); ax[1].set_xlabel("d(A,B)"); ax[1].set_ylabel("bit-acc of B")
ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_ylim(0.45, 1.02)
plt.tight_layout()
out = os.path.join(REPO, "results/defense/overwrite_hamming.png")
plt.savefig(out, dpi=120); print(f"[plot] -> {out}")
print("HAMMING_DONE")
