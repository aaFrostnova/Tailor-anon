"""Joint null of the deployed presence tests: does the budget split over k+1 tests hold the budget, and by how much
is the union bound conservative?

Unwatermarked pool images (offset 600, disjoint from the fitting slices and the certification slice at 300) are
decoded once per fragment; K random keys per image give N*K null draws of every fragment's bit accuracy against
the keyed codeword and of the fused codeword of every fragment subset. For each budget alpha the union false-positive
rate of a k-fragment configuration's k+1 zero-bit tests (one per fragment plus the fused one) is measured with all
tests at the single-fragment threshold tau_1 = beta_from_fpr(alpha) and with the split threshold tau_k =
presence_threshold(tau_1, k) the solver and the decoder use, against alpha, the union bound (k+1)*P(X >= tau_1)
and the independence prediction. Cluster bootstrap over images for the intervals.
Usage: python null_presence_union.py [N=300] [K=500]
"""
import sys, os, json, time
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300; K = int(sys.argv[2]) if len(sys.argv) > 2 else 500
sys.argv = [sys.argv[0]]
from scipy.stats import binom
import watermark_smt_v2 as W
from eval_matrix import OursComposite, _id_payload
from src.image_pool import sample as _pool_sample, composition_of
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}; FR = ["VINE", "TrustMark", "VideoSeal"]
dev = "cuda"
fe = {k: False for k in ("resync", "scale", "angle", "tile")}
comp = OursComposite(dev, tm_variant="B", vine_variant="R",
                     config={"frags": [FKEY[f] for f in FR], "order": [FKEY[f] for f in FR], "strengths": {}, **W.frontend_config(fe)})
n_cw = comp.sb.n
files = _pool_sample(N, offset=600); print(f"N={N} unwatermarked images {composition_of(files)}, K={K} keys, codeword n={n_cw}", flush=True)
SUBSETS = [("VINE",), ("TrustMark",), ("VideoSeal",), ("VINE", "TrustMark"), ("VINE", "VideoSeal"), ("TrustMark", "VideoSeal"), ("VINE", "TrustMark", "VideoSeal")]
ba = {f: np.zeros((N, K)) for f in FR}; fba = {S: np.zeros((N, K)) for S in SUBSETS if len(S) > 1}
keys = [f"null_{j:05d}" for j in range(K)]
tx = {iid: comp.sb.encode(_id_payload(iid, comp.sb.data_bits)) for iid in keys}
pm = {f: {iid: comp.frag[FKEY[f]].get_perm_M(iid) for iid in keys} for f in FR}
t0 = time.time()
for i, fp in enumerate(files):
    pil = Image.open(fp).convert("RGB"); pil = pil if pil.size == (512, 512) else pil.resize((512, 512), Image.BICUBIC)
    raw = {}
    for f in FR:
        kind, getter = comp.SPEC[FKEY[f]]; raw[f] = (kind, np.asarray(getattr(comp.frag[FKEY[f]], getter)(pil)))
    for j, iid in enumerate(keys):
        L = {}
        for f in FR:
            kind, r = raw[f]; perm, M = pm[f][iid]
            L[f] = method_soft_to_codeword_llr(r, perm, M, kind=kind, n_codeword=n_cw)
            ba[f][i, j] = np.mean((L[f] > 0).astype(np.uint8) == tx[iid])
        for S in fba:
            fused = fuse_llrs({FKEY[f]: L[f] for f in S}, weights=None, n_codeword=n_cw)
            fba[S][i, j] = np.mean(llr_to_bits(fused) == tx[iid])
    if (i + 1) % 25 == 0: print(f"  {i+1}/{N} images [{time.time()-t0:.0f}s]", flush=True)
np.savez(f"{SC}/null_presence_union_samples.npz", **{f"ba_{f}": ba[f] for f in FR}, **{"fba_" + "+".join(S): fba[S] for S in fba})
# ---- analysis ----
ALPHAS = [0.1, 0.05, 0.01, 1e-3, 1e-4]
rng = np.random.default_rng(0); B = 300; boot_idx = rng.integers(0, N, size=(B, N))
def fires(S, tau):
    m = np.zeros((N, K), bool)
    for f in S: m |= ba[f] >= tau - 1e-12
    if len(S) > 1: m |= fba[S] >= tau - 1e-12
    return m
def rate_ci(m):
    r = m.mean(); bs = np.array([m[idx].mean() for idx in boot_idx])
    return float(r), [float(x) for x in np.percentile(bs, [2.5, 97.5])]
out = {"N": N, "K": K, "images": composition_of(files), "codeword_n": n_cw, "null_mean_ba": {f: float(ba[f].mean()) for f in FR},
       "null_sd_ba": {f: float(ba[f].std()) for f in FR}, "binomial_sd": float(np.sqrt(0.25 / n_cw)),
       "corr_fragment_fused": {"+".join(S): {f: float(np.corrcoef(ba[f].ravel(), fba[S].ravel())[0, 1]) for f in S} for S in fba},
       "corr_between_fragments": {f"{a}|{b}": float(np.corrcoef(ba[a].ravel(), ba[b].ravel())[0, 1]) for a in FR for b in FR if a < b},
       "rows": []}
print(f"\nnull bit accuracy: mean {out['null_mean_ba']}  sd {out['null_sd_ba']}  (binomial sd {out['binomial_sd']:.4f})")
print("correlation fragment vs fused:", {k: {f: round(v, 3) for f, v in d.items()} for k, d in out["corr_fragment_fused"].items()})
print("correlation between fragments:", {k: round(v, 3) for k, v in out["corr_between_fragments"].items()})
print(f"\n{'alpha':>7s} {'subset':24s} {'k':>2s} {'tau1':>5s} {'tauk':>5s} | {'FPR all@tau1':>13s} {'95% CI':>17s} | {'FPR split@tauk':>14s} {'95% CI':>17s} | {'bound':>7s} {'indep':>7s} {'single@tau1':>11s}")
for alpha in ALPHAS:
    tau1 = W.beta_from_fpr(alpha); p1 = float(binom.sf(int(round(tau1 * n_cw)) - 1, n_cw, 0.5))
    for S in SUBSETS:
        k = len(S); tauk = W.presence_threshold(tau1, k); nt = 1 if k == 1 else k + 1
        r1, c1 = rate_ci(fires(S, tau1)); rk, ck = rate_ci(fires(S, tauk))
        single = float(np.mean([ (ba[f] >= tau1 - 1e-12).mean() for f in S ]))
        row = {"alpha": alpha, "subset": "+".join(S), "k": k, "tau1": tau1, "tauk": tauk, "n_tests": nt, "fpr_all_at_tau1": r1, "ci_all_at_tau1": c1,
               "fpr_split_at_tauk": rk, "ci_split_at_tauk": ck, "union_bound": min(1.0, nt * p1), "independence": 1 - (1 - p1) ** nt, "single_test_at_tau1": single}
        out["rows"].append(row)
        print(f"{alpha:7.0e} {row['subset']:24s} {k:2d} {tau1:5.2f} {tauk:5.2f} | {r1:13.4f} [{c1[0]:.4f},{c1[1]:.4f}] | {rk:14.4f} [{ck[0]:.4f},{ck[1]:.4f}] | {row['union_bound']:7.4f} {row['independence']:7.4f} {single:11.4f}")
json.dump(out, open(f"{SC}/null_presence_union.json", "w"), indent=1)
print("NULL_UNION_DONE", flush=True)
