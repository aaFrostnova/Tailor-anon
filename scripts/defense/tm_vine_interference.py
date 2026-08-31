"""Measure mutual interference between VINE-R and TrustMark-B (both low-freq).

For the same images, compare each decoder's clean bit-acc:
  solo:  embed only that method, decode it.
  comp:  embed BOTH (real pipeline order VINE->TM), decode each.
Also reverse order (TM->VINE) + cross-talk (decode A on B-only image = chance?)
+ PSNR of solo vs composite (does stacking two low-freq marks cost extra quality?).
"""
import os, sys, glob
import numpy as np
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from wbench.methods import TrustMarkMethod, VineMethod

N = 12
imgs = sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png")))[:N]

def psnr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)

def ba(rec, bits):
    n = min(len(rec), len(bits)); return float(np.mean(rec[:n] == bits[:n]))

print("Loading TrustMark-B + VINE-R..."); tm = TrustMarkMethod("B"); vn = VineMethod("R")

acc = {k: [] for k in ["vine_solo", "vine_comp", "vine_xtalk",
                        "tm_solo", "tm_comp", "tm_xtalk",
                        "vine_comp_rev", "tm_comp_rev"]}
ps = {k: [] for k in ["vine_solo", "tm_solo", "comp", "comp_rev"]}

for fp in imgs:
    orig = Image.open(fp).convert("RGB").resize((512, 512))
    rs = np.random.RandomState(hash(fp) % (2**31))
    vb = rs.randint(0, 2, vn.n_bits).astype(np.uint8)
    tb = rs.randint(0, 2, tm.n_bits).astype(np.uint8)

    # --- solo ---
    v_only = vn.embed(orig, vb);   v_only = v_only.resize((512, 512)) if v_only.size != (512, 512) else v_only
    t_only = tm.embed(orig, tb);   t_only = t_only.resize((512, 512)) if t_only.size != (512, 512) else t_only
    acc["vine_solo"].append(ba(vn.decode(v_only), vb))
    acc["tm_solo"].append(ba(tm.decode(t_only), tb))
    ps["vine_solo"].append(psnr(orig, v_only)); ps["tm_solo"].append(psnr(orig, t_only))

    # --- composite, real order VINE -> TM ---
    comp = tm.embed(v_only, tb); comp = comp.resize((512, 512)) if comp.size != (512, 512) else comp
    acc["vine_comp"].append(ba(vn.decode(comp), vb))
    acc["tm_comp"].append(ba(tm.decode(comp), tb))
    ps["comp"].append(psnr(orig, comp))

    # --- reverse order TM -> VINE ---
    comp_rev = vn.embed(t_only, vb); comp_rev = comp_rev.resize((512, 512)) if comp_rev.size != (512, 512) else comp_rev
    acc["vine_comp_rev"].append(ba(vn.decode(comp_rev), vb))
    acc["tm_comp_rev"].append(ba(tm.decode(comp_rev), tb))
    ps["comp_rev"].append(psnr(orig, comp_rev))

    # --- cross-talk: decode A on B-only image (should be ~0.5 chance) ---
    acc["vine_xtalk"].append(ba(vn.decode(t_only), vb))   # VINE decode on TM-only
    acc["tm_xtalk"].append(ba(tm.decode(v_only), tb))     # TM decode on VINE-only
    print(f"  {os.path.basename(fp)} done", flush=True)

def m(k): return float(np.mean(acc[k]))
def mp(k): return float(np.mean(ps[k]))

print("\n=== bit-accuracy (clean, n=%d) ===" % N)
print(f"  VINE:  solo {m('vine_solo'):.3f}  | in-comp(V->T) {m('vine_comp'):.3f}  | rev(T->V) {m('vine_comp_rev'):.3f}  | xtalk(on TM-only) {m('vine_xtalk'):.3f}")
print(f"  TM:    solo {m('tm_solo'):.3f}  | in-comp(V->T) {m('tm_comp'):.3f}  | rev(T->V) {m('tm_comp_rev'):.3f}  | xtalk(on VINE-only) {m('tm_xtalk'):.3f}")
print(f"  >> VINE interference drop (V->T): {m('vine_solo')-m('vine_comp'):+.3f}   TM drop (V->T): {m('tm_solo')-m('tm_comp'):+.3f}")
print("\n=== PSNR vs orig ===")
print(f"  VINE-solo {mp('vine_solo'):.2f}  TM-solo {mp('tm_solo'):.2f}  composite(V->T) {mp('comp'):.2f}  composite(T->V) {mp('comp_rev'):.2f}")
print("INTERFERENCE_DONE")
