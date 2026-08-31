"""Visualize complement-overwrite residuals to show REPLACEMENT (not cancellation).

For one image, per method, show 4 residual maps:
  W(A)              = embed(orig, A) - orig
  W(~A)             = embed(orig, complement(A)) - orig
  overwrite (real)  = embed(embed(orig,A), ~A) - orig    <- what actually happens
  W(A)+W(~A)        = linear sum (what pure cancellation would look like ~ 0)
+ correlations: corr(A,~A) (expect <0), corr(stack,~A) (expect ~+1 => 2nd wins).
"""
import os, sys, glob
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from wbench.methods import TrustMarkMethod, VineMethod, InvisibleWMMethod

fp = sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png")))[0]
orig = Image.open(fp).convert("RGB").resize((512, 512))
o = np.asarray(orig, np.float64)

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def gray(r): return r.mean(axis=2)  # HxWx3 -> HxW
def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum() / d) if d > 0 else 0.0
def rms(r): return float(np.sqrt(np.mean(r**2)))

methods = [("DwtDct (32b, linear)", InvisibleWMMethod("dwtDct")),
           ("TrustMark-B", TrustMarkMethod("B")),
           ("VINE-R", VineMethod("R"))]

rows = []
for name, m in methods:
    rs = np.random.RandomState(42)
    A = rs.randint(0, 2, m.n_bits).astype(np.uint8)
    Bc = 1 - A                                            # complement
    rA = np.asarray(to512(m.embed(orig, A)), np.float64) - o
    rB = np.asarray(to512(m.embed(orig, Bc)), np.float64) - o
    Amark = to512(m.embed(orig, A))
    rStack = np.asarray(to512(m.embed(Amark, Bc)), np.float64) - o
    rSum = rA + rB
    gA, gB, gS, gSum = gray(rA), gray(rB), gray(rStack), gray(rSum)
    info = dict(name=name, gA=gA, gB=gB, gS=gS, gSum=gSum,
                cAB=corr(gA, gB), cSB=corr(gS, gB), cSA=corr(gS, gA),
                rmsA=rms(gA), rmsSum=rms(gSum), rmsS=rms(gS))
    rows.append(info)
    print(f"[{name}] corr(A,~A)={info['cAB']:+.2f}  corr(stack,~A)={info['cSB']:+.2f}  "
          f"corr(stack,A)={info['cSA']:+.2f}  RMS: A={info['rmsA']:.2f} sum={info['rmsSum']:.2f} stack={info['rmsS']:.2f}", flush=True)

fig, ax = plt.subplots(3, 4, figsize=(16, 12))
col_titles = ["W(A)", "W(¬A)  complement", "overwrite A→¬A  (REAL)", "W(A)+W(¬A)  (cancellation ref)"]
for i, r in enumerate(rows):
    v = np.percentile(np.abs(np.concatenate([r['gA'], r['gB'], r['gS'], r['gSum']])), 99)
    panels = [(r['gA'], f"RMS={r['rmsA']:.2f}"),
              (r['gB'], f"corr(A,¬A)={r['cAB']:+.2f}"),
              (r['gS'], f"corr(stack,¬A)={r['cSB']:+.2f}\ncorr(stack,A)={r['cSA']:+.2f}"),
              (r['gSum'], f"RMS={r['rmsSum']:.2f}  (vs A {r['rmsA']:.2f})")]
    for j, (g, sub) in enumerate(panels):
        a = ax[i, j]
        a.imshow(g, cmap="seismic", vmin=-v, vmax=v)
        a.set_xticks([]); a.set_yticks([])
        if i == 0: a.set_title(col_titles[j], fontsize=12, fontweight="bold")
        a.set_xlabel(sub, fontsize=10)
    ax[i, 0].set_ylabel(r['name'], fontsize=13, fontweight="bold")
fig.suptitle("Complement overwrite residuals: 2nd mark REPLACES 1st (overwrite≈W(¬A)), it does NOT cancel to ~0",
             fontsize=14, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(REPO, "results/defense/complement_residual.png")
plt.savefig(out, dpi=110); print(f"[plot] -> {out}")
print("VIS_DONE")
