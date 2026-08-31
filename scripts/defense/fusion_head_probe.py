"""Phase 2: headroom probe. On the held-out split, compare fusion strategies:
  equal-MRC (current)  : f = a_v + a_t
  calib-MRC (no train) : per-sample std-normalize each fragment, then sum
  oracle-weight (UB)   : weight each fragment by its TRUE bit-acc (downweight dead)
  oracle-select (UB)   : use only the more-reliable fragment
Metric: codeword bit-acc + detection (crypto-ID via Chase OR zerobit>=tau) per attack.
If oracle >> equal on regen/crop, a learned head has room. No training here.
"""
import os, sys
import numpy as np
from scipy.stats import binom

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify

d = np.load(os.path.join(REPO, "results/defense/fusion_head_data.npz"), allow_pickle=True)
AV, AT, TX, ATK, IMG = d["AV"], d["AT"], d["TX"], d["ATK"], d["IMG"]
n = int(d["n"]); sb = ShortenedBCH(); tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
attacks = list(d["attacks"])
cut = int(np.quantile(np.unique(IMG), 0.7))           # train imgs < cut, test >= cut
te = IMG >= cut
print(f"n={n} tau={tau:.3f} | test samples={te.sum()} (imgs>= {cut})", flush=True)

e = 1e-6
def norm(a): return a / (a.std(axis=1, keepdims=True) + e)
def ba(f, tx): return (np.sign(f) > 0).astype(np.uint8) == tx

def fuse(method, av, at, tx):
    if method == "equal": return av + at
    if method == "calib": return norm(av) + norm(at)
    avn, atn = norm(av), norm(at)
    bav = ((av > 0).astype(np.uint8) == tx).mean(1)     # per-sample true bit-acc
    bat = ((at > 0).astype(np.uint8) == tx).mean(1)
    if method == "oracle_w":
        wv = np.clip(2 * bav - 1, 0, None)[:, None]; wt = np.clip(2 * bat - 1, 0, None)[:, None]
        f = wv * avn + wt * atn
        flat = (np.abs(wv) + np.abs(wt) < e)[:, 0]      # both dead -> fallback equal
        f[flat] = (avn + atn)[flat]; return f
    if method == "oracle_sel":
        pick_v = (bav >= bat)[:, None]
        return np.where(pick_v, avn, atn)

METHODS = ["equal", "calib", "oracle_w", "oracle_sel"]
avg = {m: [] for m in METHODS}
av, at, tx, atk, img = AV[te], AT[te], TX[te], ATK[te], IMG[te]
print(f"\n{'attack':9s} | " + " | ".join(f"{m:>20s}" for m in METHODS))
print(f"{'':9s} | " + " | ".join(f"{'det / ba':>20s}" for m in METHODS))
for k in attacks:
    sel = atk == k
    if sel.sum() == 0: continue
    row = []
    for m in METHODS:
        F = fuse(m, av[sel], at[sel], tx[sel])
        baa = np.mean([ba(F[i], tx[sel][i]).mean() for i in range(sel.sum())])
        dets = []
        for i in np.where(sel)[0]:
            Fi = fuse(m, av[i:i+1], at[i:i+1], tx[i:i+1])[0]
            iid = f"fh_{int(img[i]):05d}"
            ver = bool(decode_and_verify(Fi, iid, codec=sb)["detected"])
            zb = (np.sign(Fi) > 0).astype(np.uint8) == tx[i]
            dets.append(1.0 if (ver or zb.mean() >= tau) else 0.0)
        det = float(np.mean(dets)); avg[m].append(det)
        row.append(f"{det:.2f} / {baa:.2f}")
    print(f"{k:9s} | " + " | ".join(f"{c:>20s}" for c in row))
print(f"{'OVERALL':9s} | " + " | ".join(f"{np.mean(avg[m]):.3f}{'':>14s}" for m in METHODS))
print("PROBE_DONE")
