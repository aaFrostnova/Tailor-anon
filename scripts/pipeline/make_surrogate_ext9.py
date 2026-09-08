"""EXTENDED GPU measurement campaign -> $SC/surrogate_table_ext9.json. Copy of make_surrogate.py
(Track A of the Phase-4 "close three modeling gaps" plan; see
$CF/.superpowers/sdd/2026-08-28-phase4-surrogate-completion/progress.md) that keeps EVERYTHING
make_surrogate.py measures (base_f/delta/d/e/additivity, identical methodology, so this file is a
drop-in superset) and, piggybacking on the SAME embed/attack/decode passes, adds three things:

GAP 1 -- capacity as a function of strength.
  make_surrogate.py's hard() computes a soft value (VINE prob / TrustMark,VideoSeal logit) then
  thresholds it and throws it away. We keep it, convert to LLR (VINE via _prob2llr; TM/VideoSeal
  logits already ARE LLR-like) and accumulate (truth, llr) per (fragment, strength, attack) over
  all N images, then run the non-parametric quantile-binned mi_per_bit estimator (same formula as
  $CF/scripts/defense/capacity_mi.py -- that module can't be imported directly, its top level does
  `np.load(sys.argv[1])`, so _prob2llr/_Hb/_mi_per_bit are inlined verbatim, exactly the way
  $SC/make_baseline.py's stage3 already does it). capacity C_f(s,a) = NB * mi_per_bit, stored as a
  PWL over the fragment's own strength grid under top-level key "cap", keyed "F|A" exactly like
  "base" -- same {xs,ys} PWL dict shape.

GAP 3a -- front-ends as functions of strength.
  nested-VINE: at each VINE strength knot, embed nested rings (scales 1.0/0.75/0.5, via the
  deployed composite_external_eval.nested_vine_embed(..., strength=s)) instead of plain VINE, and
  record (i) the extra MSE cost vs plain VINE at the same strength -> PWL nested_penalty(s_VINE),
  (ii) bit-acc under crop75/crop50 -> PWL base_nested_VINE|crop75, |crop50.
  CAUGHT BUG (pre-submission review of the first full-run attempt): decoding the crop-attacked
  nested image with the plain single-scale decoder reads at chance -- the ring survives the crop
  but sits at a NON-canonical scale (a layer embedded at K under an attack that kept fraction c
  lands at K/c), so a scale-blind decode is looking in the wrong place, not measuring "no signal".
  Fixed by mirroring the deployed geo_cascade stage-B2 blind scale search exactly (see
  composite_external_eval.py's `for f in np.arange(0.34, 1.0001, args.vine_scale_step): ... view =
  att.crop(...)` -- crop WITHOUT resizing back to 512, feed straight to raw_probs, same fine
  step=0.005 since VINE's scale-capture width is <1%) and taking the MAX bit-acc over that grid --
  the faithful surrogate for "deployment accepts as soon as ANY searched scale crypto-verifies".
  ~133 decodes/image/cell is too expensive at N=100x5 strengths x2 attacks, so ONLY these two
  cells subsample to N_NESTED_SCALE images (honestly stamped as "n" on those two cells); the old
  plain-decode number is kept alongside as a "nested_VINE_no_scale_search|*" diagnostic so the
  wrong-scale-vs-right-scale contrast stays visible in the table.
  resync: SyncSeal (src/syncseal_frontend.py, TorchScript, already used by
  syncseal_resync_test.py/syncseal_vine_compound_test.py) IS wired in-process -- confirmed
  importable + checkpoint present before this run, so it is NOT skipped. For every fragment x
  strength knot: add the SyncSeal sync mark on top of the cached composite, rot9-attack, then
  SyncSeal detect+unwarp before decoding that fragment -> PWL base_resync_F|rot9 per fragment.
  A "raw_synced_noresync" companion PWL (sync mark added, rot9'd, but NOT rectified) is also kept
  so the report can separate "does the sync layer itself poison the payload" from "does rectify
  recover bit-acc" -- both cheap byproducts of the same forward passes.
  Both stored under top-level key "frontend".

GAP 3b -- 3rd-order additivity at multiple strength points.
  The original stage-5 validation checked one off-knot strength triple. Extended to 4 strength
  combinations spanning the grids (all-low = each fragment's grid min, all-mid = grid midpoint,
  all-high = grid max, mixed = the ORIGINAL off-knot triple 0.5/0.85/1.1), each over the full 3
  embed-order permutations, reporting max|dba| and max|dPSNR| per combo plus an explicit
  low->high trend verdict. Stored under top-level key "additivity_multi"; the ORIGINAL top-level
  "additivity" key is kept and set to the "mixed" combo's result so anything reading the old
  schema (Surrogate-adjacent tooling) still finds it unchanged in shape.

Canonical embed order everywhere: VINE -> TrustMark -> VideoSeal.
Attacks: the Phase-1 in-process family (global_constraints.md #6):
  jpeg25, blur, noise, bright, contrast, crop75, crop50, rot9, vaeB, vaeC.
Usage: python make_surrogate_ext.py [N=20]
"""
import sys, os, io, json, glob, time, subprocess
from collections import defaultdict
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense"):
    sys.path.insert(0, p)
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
from src.image_pool import sample as _pool_sample, composition_of
from surrogate_model import Surrogate, PWL, RANGES, PHASE1_ATTACKS
from composite_external_eval import nested_vine_embed        # GAP 3a (nested), strength= kwarg

T0 = time.time()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
dev = "cuda"
KEY = b"v5_key_encoder_master"
sb = ShortenedBCH(); NB = sb.n
ORDER = ["VINE", "TrustMark", "VideoSeal"]      # canonical embed order everywhere
np.random.seed(0)

def mse(a, b):
    # distortion surrogate lives in the MSE domain: MSE is (approximately) additive across
    # near-independent per-fragment residuals and INCREASES monotonically with embed strength,
    # unlike PSNR (log-domain, decreases with strength) which broke the D = sum(d)+sum(e) model.
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(((a - b) ** 2).mean())

def psnr_from_mse(m):
    # only used to report/gate the additivity check in human-readable dB; never fed back into
    # the additive model itself.
    return 10 * np.log10(255.0 ** 2 / max(m, 1e-9))

def target_bits(i):
    return np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)

# ---------------- Phase-1 in-process attack family (identical to make_surrogate.py) ----------------
# Attacks come from src.attacks -- the single definition the reported matrix also uses. They used to
# be restated here, and the restatements had drifted: crop read its fraction as an area rather than a
# side, rotation filled the corners with black instead of reflecting, and brightness/contrast/noise
# were materially milder. The surrogate therefore described an easier threat than the evaluation
# scored against, which inflated every feasibility verdict built on it.
from src.attacks import attack_pil, IN_PROCESS

ATT = {a: (lambda _a: (lambda p: attack_pil(_a, p, dev=dev)))(a)
       for a in ("jpeg25", "blur", "noise", "bright", "contrast",
                 "crop75", "crop50", "rot9", "vaeB", "vaeC")}
assert all(__import__("src.attacks", fromlist=["ALIASES"]).ALIASES.get(a, a) in IN_PROCESS for a in ATT)
assert list(ATT) == PHASE1_ATTACKS, (list(ATT), PHASE1_ATTACKS)

# ---------------- GAP 1 helpers: inlined from capacity_mi.py via make_baseline.py's pattern ----------------
# (capacity_mi.py's module level does `np.load(sys.argv[1])` so it cannot be imported; make_baseline.py
# already solved this by inlining the same three functions -- reproduced verbatim here.)
def _prob2llr(p):
    return np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1-1e-6))
def _Hb(p):
    p = np.clip(p, 1e-12, 1-1e-12); return -(p*np.log2(p) + (1-p)*np.log2(1-p))
def _mi_per_bit(truth, llr, K=25):
    """Non-parametric I(bit;llr) in bits/position, pooled over all (img,pos). Calibration-free via
    quantile bins. Identical formula to capacity_mi.py's mi_per_bit / make_baseline.py's _mi_per_bit."""
    b = truth.ravel().astype(np.float64); s = llr.ravel()
    Hprior = _Hb(b.mean())
    edges = np.quantile(s, np.linspace(0, 1, K+1)); edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.clip(np.digitize(s, edges)-1, 0, K-1)
    Hcond = 0.0
    for k in range(K):
        m = idx == k
        if m.any(): Hcond += m.mean() * _Hb(b[m].mean())
    return max(Hprior - Hcond, 0.0), Hprior

# ---------------- load fragments ----------------
print("loading fragments...", flush=True)
FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
      "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
      "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}

# ---------------- Step 2: probe VideoSeal's scaling_w band ----------------
_vs_model = FR["VideoSeal"].model
_blender = getattr(_vs_model, "blender", None)
if _blender is not None and hasattr(_blender, "scaling_w"):
    scaling_w_default = float(_blender.scaling_w); scaling_w_path = "model.blender.scaling_w"
elif hasattr(_vs_model, "scaling_w"):
    scaling_w_default = float(_vs_model.scaling_w); scaling_w_path = "model.scaling_w"
else:
    scaling_w_default = None; scaling_w_path = None
print(f"VideoSeal scaling_w probe: path={scaling_w_path} default={scaling_w_default}", flush=True)
if scaling_w_default is None:
    print("ERROR: could not resolve VideoSeal scaling_w on model or model.blender "
          "(Task 5's embed_with_target handle may be wrong) -- aborting.", flush=True)
    sys.exit(1)

# 9 levels instead of 5: a leave-one-out check on the 5-level curves put the interpolation error
# at about 3.8x the measurement noise for this (cheap) attack family, i.e. the limit here is how
# coarsely the curve is sampled rather than how noisily. Halving the spacing is what that calls for.
import numpy as _np
GRID = {f: [round(float(v), 4) for v in _np.linspace(lo, hi, 9)]
        for f, (lo, hi) in {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6),
                            "VideoSeal": (0.5, 1.5)}.items()}
for f in ORDER:
    assert abs(GRID[f][0] - RANGES[f][0]) < 1e-9 and abs(GRID[f][-1] - RANGES[f][1]) < 1e-9, \
        (f, GRID[f], RANGES[f])
MID = {f: GRID[f][len(GRID[f])//2] for f in ORDER}            # middle knot of each 5-point grid
COARSE_IDX = [0, len(GRID['VINE'])//2, len(GRID['VINE'])-1]                          # lo/mid/hi subset used for the pairwise e_fg grid
PAIRS = [("VINE", "TrustMark"), ("VINE", "VideoSeal"), ("TrustMark", "VideoSeal")]  # embed order f1->f2
print("grids:", GRID, flush=True)
print("mid strengths:", MID, flush=True)

# ---------------- GAP 3a resync: try to wire SyncSeal in-process; skip honestly if it fails ----------------
RESYNC_ENABLED = True
RESYNC_SKIP_REASON = None
sync = None
DEFAULT_JIT = None
try:
    from src.syncseal_frontend import load_sync, sync_embed, sync_rectify, DEFAULT_JIT
    sync = load_sync(DEFAULT_JIT, dev)
    print(f"SyncSeal resync front-end loaded OK from {DEFAULT_JIT}", flush=True)
except Exception as ex:
    RESYNC_ENABLED = False
    RESYNC_SKIP_REASON = f"{type(ex).__name__}: {ex}"
    print(f"SyncSeal resync front-end UNAVAILABLE, skipping the resync half of GAP 3a "
          f"(nested half still runs): {RESYNC_SKIP_REASON}", flush=True)

# ---------------- GAP 3a nested-crop FIX: same blind scale grid as composite_external_eval.py's
# geo_cascade stage B2 ("for f in np.arange(0.34, 1.0001, args.vine_scale_step)"). Step must stay
# 0.005 -- VINE's scale-capture width is <1%, a coarser grid would step over the peak entirely. ----
VINE_SCALE_GRID = np.arange(0.34, 1.0001, 0.005)
print(f"VINE nested-crop scale-search grid: {len(VINE_SCALE_GRID)} points, "
      f"[{VINE_SCALE_GRID[0]:.3f}, {VINE_SCALE_GRID[-1]:.3f}] step 0.005", flush=True)

# ---------------- soft-value-preserving decode (GAP 1 piggyback point) ----------------
def soft_llr(fn, pil):
    """Same soft readout make_surrogate.py's hard() computes and discards -- kept here as an LLR.
    VINE emits a probability -> converted via _prob2llr; TrustMark/VideoSeal emit logits, which are
    already LLR-like (per-task instruction: 'use directly')."""
    f = FR[fn]
    v = f.raw_probs(pil) if fn == "VINE" else f.raw_logits(pil)
    v = np.asarray(v, dtype=np.float64).ravel()[:NB]
    return _prob2llr(v) if fn == "VINE" else v

def hard(fn, pil):
    """Unchanged external behavior vs make_surrogate.py's hard(): (v>0.5) for VINE prob and (v>0.0)
    for TM/VideoSeal logit are exactly (llr>0) since _prob2llr(0.5)==0.0 -- verified boundary-exact.
    Routed through soft_llr so stage1+2 need only ONE decoder forward pass per (fragment,attack)."""
    return (soft_llr(fn, pil) > 0).astype(np.uint8)

files = _pool_sample(N, offset=0)   # across all five sources, in the evaluation set's proportions
print(f"n={len(files)} covers", flush=True)
covers = [Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC) for fp in files]
targets = [target_bits(i) for i in range(len(covers))]

# coordinator-approved cost mitigation: the blind scale search is ~133 decodes/image/cell, so at
# full N=100 x 5 VINE strengths x 2 crop attacks that's ~133k extra VINE decodes -- too expensive.
# Reduce N for ONLY the two scale-searched cells (base_nested_VINE|crop75, |crop50); every other
# measurement (capacity, resync, additivity, base/delta/e, nested_penalty MSE, the naive-decode
# diagnostic) still uses the full N. The reduced n is stamped honestly on those two cells below.
N_NESTED_SCALE = min(25, len(covers))
print(f"nested-crop blind scale search subsamples to N_NESTED_SCALE={N_NESTED_SCALE} "
      f"(of {len(covers)}) for cost -- everything else uses the full N", flush=True)

# ================= accumulators (stages 1-4, unchanged) =================
d_sum = {f: {s: 0.0 for s in GRID[f]} for f in ORDER}
d_cnt = {f: {s: 0 for s in GRID[f]} for f in ORDER}
base_sum = {(f, a): {s: 0.0 for s in GRID[f]} for f in ORDER for a in ATT}
base_cnt = {(f, a): {s: 0 for s in GRID[f]} for f in ORDER for a in ATT}
delta_sum = {(g, f, a): {s: 0.0 for s in GRID[g]} for g in ORDER for f in ORDER if g != f for a in ATT}
delta_cnt = {(g, f, a): {s: 0 for s in GRID[g]} for g in ORDER for f in ORDER if g != f for a in ATT}
e_samples = {p: [] for p in PAIRS}   # embed-order-keyed; re-keyed to sorted() when stored into Surrogate

# ================= GAP 1 accumulators: (truth, llr) per (fragment, strength, attack) =================
cap_T = {(f, a): {s: [] for s in GRID[f]} for f in ORDER for a in ATT}
cap_L = {(f, a): {s: [] for s in GRID[f]} for f in ORDER for a in ATT}

# ================= GAP 3a accumulators =================
nested_pen_sum = {s: 0.0 for s in GRID["VINE"]}       # MSE(nested) - MSE(plain) at same strength
nested_pen_cnt = {s: 0 for s in GRID["VINE"]}
# diagnostic: naive single-scale decode of the crop-attacked nested image (full N). Expected to
# read near chance -- the ring is intact but the decoder is looking at the wrong scale.
nested_crop_naive_sum = {a: {s: 0.0 for s in GRID["VINE"]} for a in ("crop75", "crop50")}
nested_crop_naive_cnt = {a: {s: 0 for s in GRID["VINE"]} for a in ("crop75", "crop50")}
# deliverable: MAX bit-acc over the blind VINE scale grid (N_NESTED_SCALE images only)
nested_crop_scaled_sum = {a: {s: 0.0 for s in GRID["VINE"]} for a in ("crop75", "crop50")}
nested_crop_scaled_cnt = {a: {s: 0 for s in GRID["VINE"]} for a in ("crop75", "crop50")}

resync_sum = {f: {s: 0.0 for s in GRID[f]} for f in ORDER}       # rot9 + sync-embed + sync-rectify
resync_cnt = {f: {s: 0 for s in GRID[f]} for f in ORDER}
resync_raw_sum = {f: {s: 0.0 for s in GRID[f]} for f in ORDER}   # rot9 + sync-embed, NO rectify (diagnostic)
resync_raw_cnt = {f: {s: 0 for s in GRID[f]} for f in ORDER}
sync_psnr_list = []                                              # fidelity cost of the sync layer itself

n_ok = 0
for i, cover in enumerate(covers):
    t = targets[i]
    try:
        # ---- stage 1+2: base_f(s,a), d_f(s); cache every per-fragment per-strength embed ----
        # + GAP 1 piggyback: keep the LLR before it gets thresholded away.
        E = {f: {} for f in ORDER}
        for f in ORDER:
            for s in GRID[f]:
                emb = FR[f].embed_with_target(cover, t, strength=s)
                E[f][s] = emb
                d_sum[f][s] += mse(cover, emb); d_cnt[f][s] += 1
                for a, af in ATT.items():
                    att = af(emb)
                    if att is None:
                        continue
                    llr = soft_llr(f, att)
                    ba = float(np.mean((llr > 0).astype(np.uint8) == t))
                    base_sum[(f, a)][s] += ba; base_cnt[(f, a)][s] += 1
                    cap_T[(f, a)][s].append(t); cap_L[(f, a)][s].append(llr)

        # ---- stage 2b (GAP 3a, nested half): nested-VINE penalty (MSE) + crop75/crop50 bit-acc,
        #      at every VINE strength knot, reusing the plain-VINE embed E["VINE"][s] just cached ----
        for s in GRID["VINE"]:
            nested = nested_vine_embed(FR["VINE"], cover, t, strength=s)
            if nested.size != (512, 512): nested = nested.resize((512, 512))
            nested_pen_sum[s] += mse(cover, nested) - mse(cover, E["VINE"][s]); nested_pen_cnt[s] += 1
            for a in ("crop75", "crop50"):
                att = ATT[a](nested)
                # diagnostic: single-scale decode (the bug the coordinator caught) -- ring intact,
                # wrong scale, expected near chance. Kept for contrast, NOT the deliverable cell.
                ba_naive = float(np.mean(hard("VINE", att) == t))
                nested_crop_naive_sum[a][s] += ba_naive; nested_crop_naive_cnt[a][s] += 1
                # deliverable: blind scale search, MAX bit-acc over the grid -- mirrors
                # composite_external_eval.py geo_cascade stage B2 exactly (crop WITHOUT resizing
                # back to 512, straight into raw_probs; deployment accepts as soon as ANY searched
                # scale crypto-verifies, so max-over-scales is the faithful surrogate). Expensive
                # (~133 decodes/image/cell) -> only the first N_NESTED_SCALE images.
                if i < N_NESTED_SCALE:
                    best_ba = 0.0
                    for frac in VINE_SCALE_GRID:
                        if frac >= 0.999:
                            view = att
                        else:
                            sc = int(round(512 * float(frac))); o = (512 - sc) // 2
                            view = att.crop((o, o, o + sc, o + sc))
                        llr = soft_llr("VINE", view)
                        ba_f = float(np.mean((llr > 0).astype(np.uint8) == t))
                        if ba_f > best_ba: best_ba = ba_f
                    nested_crop_scaled_sum[a][s] += best_ba; nested_crop_scaled_cnt[a][s] += 1

        # ---- stage 2c (GAP 3a, resync half, skipped cleanly if RESYNC_ENABLED is False): for every
        #      fragment x strength, add the SyncSeal sync mark on the cached composite, rot9-attack,
        #      then decode both WITHOUT (raw diagnostic) and WITH (real deliverable) rectify ----
        if RESYNC_ENABLED:
            for f in ORDER:
                for s in GRID[f]:
                    comp = E[f][s]
                    Ws = sync_embed(sync, comp, dev)
                    sync_psnr_list.append(psnr_from_mse(mse(comp, Ws)))
                    # the shared definition, not a local one: this call site kept referring to a
                    # `rot9` that the move to src/attacks.py removed, and an earlier import
                    # failure masked it -- every image in the run died here
                    A = ATT["rot9"](Ws)
                    resync_raw_sum[f][s] += float(np.mean(hard(f, A) == t)); resync_raw_cnt[f][s] += 1
                    R, _score = sync_rectify(sync, A, dev)
                    if R.size != (512, 512): R = R.resize((512, 512))
                    resync_sum[f][s] += float(np.mean(hard(f, R) == t)); resync_cnt[f][s] += 1

        # ---- stage 3: delta_{g->f}(s_g,a) -- f alone@MID, then g on top at each s_g ----
        for f in ORDER:
            base_img = E[f][MID[f]]
            for g in ORDER:
                if g == f:
                    continue
                for s_g in GRID[g]:
                    comp = FR[g].embed_with_target(base_img, t, strength=s_g)
                    for a, af in ATT.items():
                        att = af(comp)
                        if att is None:
                            continue
                        ba = float(np.mean(hard(f, att) == t))
                        delta_sum[(g, f, a)][s_g] += ba; delta_cnt[(g, f, a)][s_g] += 1

        # ---- stage 4: e_{fg}(s_f+s_g) on the coarse lo/mid/hi x lo/mid/hi grid ----
        for (f1, f2) in PAIRS:
            for k1 in COARSE_IDX:
                s1 = GRID[f1][k1]
                for k2 in COARSE_IDX:
                    s2 = GRID[f2][k2]
                    comp = FR[f2].embed_with_target(E[f1][s1], t, strength=s2)
                    MSE_comp = mse(cover, comp)
                    e_val = MSE_comp - mse(cover, E[f1][s1]) - mse(cover, E[f2][s2])
                    e_samples[(f1, f2)].append((s1 + s2, e_val))
        n_ok += 1
    except Exception as ex:
        print(f"  !! image {i} FAILED main loop: {type(ex).__name__}: {ex}", flush=True)
    if (i + 1) % max(1, len(covers) // 10) == 0 or i + 1 == len(covers):
        print(f"  ...{i+1}/{len(covers)} images done ({n_ok} ok) [{time.time()-T0:.0f}s]", flush=True)

print(f"measurement pass done: {n_ok}/{len(covers)} images ok", flush=True)

# ================= fit PWLs (stages 1-4, unchanged) =================
d_pwl = {f: PWL(GRID[f], [d_sum[f][s] / max(1, d_cnt[f][s]) for s in GRID[f]]) for f in ORDER}

base_pwl, base_mean = {}, {}
for (f, a), sdict in base_sum.items():
    xs = GRID[f]; ys = []
    for s in xs:
        c = base_cnt[(f, a)][s]
        m = (sdict[s] / c) if c > 0 else 0.5     # attack unavailable (e.g. no compressai) -> chance
        ys.append(m); base_mean[(f, a, s)] = m
    base_pwl[(f, a)] = PWL(xs, ys)

delta_pwl = {}
for (g, f, a), sdict in delta_sum.items():
    xs = GRID[g]
    ba_alone = base_mean[(f, a, MID[f])]
    ys = []
    for s_g in xs:
        c = delta_cnt[(g, f, a)][s_g]
        ba_after = (sdict[s_g] / c) if c > 0 else ba_alone
        ys.append(max(0.0, ba_alone - ba_after))
    delta_pwl[(g, f, a)] = PWL(xs, ys)

e_pwl = {}
for pair, samples in e_samples.items():
    groups = defaultdict(list)
    for s_sum, e_val in samples:
        groups[round(s_sum, 6)].append(e_val)
    xs = sorted(groups); ys = [float(np.mean(groups[x])) for x in xs]
    e_pwl[tuple(sorted(pair))] = PWL(xs, ys)     # Surrogate.e() looks up by sorted(f,g)

sg = Surrogate(fragments=ORDER, attacks=list(ATT),
               ranges={f: tuple(RANGES[f]) for f in ORDER},
               base=base_pwl, delta=delta_pwl, d=d_pwl, e=e_pwl)
print("fitted surrogate PWLs "
      f"(base={len(base_pwl)} delta={len(delta_pwl)} d={len(d_pwl)} e={len(e_pwl)})", flush=True)

# ================= GAP 1: fit capacity PWLs =================
cap_pwl = {}
for f in ORDER:
    for a in ATT:
        xs = GRID[f]; ys = []
        for s in xs:
            Tl, Ll = cap_T[(f, a)][s], cap_L[(f, a)][s]
            if not Tl:
                ys.append(0.0)      # attack unavailable this cell -> no information, not "chance"
                continue
            mi, _ = _mi_per_bit(np.array(Tl), np.array(Ll), K=25)
            ys.append(float(mi * NB))
        cap_pwl[(f, a)] = PWL(xs, ys)
print("fitted capacity PWLs "
      f"(cap={len(cap_pwl)}); e.g. VINE|clean-like sanity samples below", flush=True)
for a in ("jpeg25", "crop75", "vaeC"):
    for f in ORDER:
        print(f"  cap[{f}|{a}] = {[round(y,1) for y in cap_pwl[(f,a)].ys]} bits over s={GRID[f]}",
              flush=True)

# ================= GAP 3a: fit front-end PWLs =================
nested_penalty_pwl = PWL(GRID["VINE"],
                          [nested_pen_sum[s] / max(1, nested_pen_cnt[s]) for s in GRID["VINE"]])
nested_crop_naive_pwl = {a: PWL(GRID["VINE"], [nested_crop_naive_sum[a][s] / max(1, nested_crop_naive_cnt[a][s])
                                                for s in GRID["VINE"]])
                          for a in ("crop75", "crop50")}
nested_crop_scaled_pwl = {a: PWL(GRID["VINE"], [nested_crop_scaled_sum[a][s] / max(1, nested_crop_scaled_cnt[a][s])
                                                 for s in GRID["VINE"]])
                           for a in ("crop75", "crop50")}
print(f"nested_penalty(s_VINE) MSE = {[round(y,2) for y in nested_penalty_pwl.ys]} over s={GRID['VINE']}",
      flush=True)
for a in ("crop75", "crop50"):
    print(f"  base_nested_VINE|{a} (scale-searched, n={N_NESTED_SCALE}) = "
          f"{[round(y,4) for y in nested_crop_scaled_pwl[a].ys]}  "
          f"naive single-scale (n={n_ok}) = {[round(y,4) for y in nested_crop_naive_pwl[a].ys]}  "
          f"plain-VINE-no-nesting|{a} = {[round(y,4) for y in base_pwl[('VINE', a)].ys]}", flush=True)

frontend = {
    "nested_penalty": {"xs": nested_penalty_pwl.xs, "ys": nested_penalty_pwl.ys},
    "base_nested_VINE|crop75": {"xs": nested_crop_scaled_pwl["crop75"].xs,
                                 "ys": nested_crop_scaled_pwl["crop75"].ys, "n": N_NESTED_SCALE},
    "base_nested_VINE|crop50": {"xs": nested_crop_scaled_pwl["crop50"].xs,
                                 "ys": nested_crop_scaled_pwl["crop50"].ys, "n": N_NESTED_SCALE},
    "nested_VINE_no_scale_search|crop75": {"xs": nested_crop_naive_pwl["crop75"].xs,
                                            "ys": nested_crop_naive_pwl["crop75"].ys, "n": n_ok},
    "nested_VINE_no_scale_search|crop50": {"xs": nested_crop_naive_pwl["crop50"].xs,
                                            "ys": nested_crop_naive_pwl["crop50"].ys, "n": n_ok},
    "nested_scale_search_meta": {
        "note": "base_nested_VINE|crop75,|crop50 decode the crop-attacked nested image at the SAME "
                "blind scale grid the deployed geo_cascade stage B2 uses (composite_external_eval.py), "
                "taking the MAX bit-acc over the grid -- the faithful surrogate for 'deployment "
                "accepts as soon as ANY searched scale crypto-verifies'. nested_VINE_no_scale_search|* "
                "is the single-canonical-scale decode (what a scale-blind decoder sees: ring intact, "
                "wrong scale) kept alongside for contrast.",
        "grid_range": [float(VINE_SCALE_GRID[0]), float(VINE_SCALE_GRID[-1])],
        "grid_step": 0.005, "n_scales": int(len(VINE_SCALE_GRID)), "n_images": N_NESTED_SCALE,
    },
    "resync": {"status": "wired" if RESYNC_ENABLED else "skipped",
               "skip_reason": RESYNC_SKIP_REASON,
               "sync_jit": DEFAULT_JIT},
}
if RESYNC_ENABLED:
    resync_pwl = {f: PWL(GRID[f], [resync_sum[f][s] / max(1, resync_cnt[f][s]) for s in GRID[f]])
                  for f in ORDER}
    resync_raw_pwl = {f: PWL(GRID[f], [resync_raw_sum[f][s] / max(1, resync_raw_cnt[f][s]) for s in GRID[f]])
                       for f in ORDER}
    frontend["resync"]["clean_sync_psnr_db"] = round(float(np.mean(sync_psnr_list)), 2) if sync_psnr_list else None
    for f in ORDER:
        frontend[f"base_resync_{f}|rot9"] = {"xs": resync_pwl[f].xs, "ys": resync_pwl[f].ys}
        frontend[f"raw_synced_noresync_{f}|rot9"] = {"xs": resync_raw_pwl[f].xs, "ys": resync_raw_pwl[f].ys}
        print(f"  base_resync_{f}|rot9 = {[round(y,4) for y in resync_pwl[f].ys]}  "
              f"(no-rectify {[round(y,4) for y in resync_raw_pwl[f].ys]}, "
              f"plain-no-sync {[round(y,4) for y in base_pwl[(f,'rot9')].ys]})", flush=True)

# ================= GAP 3b: additivity validation at MULTIPLE strength combinations =================
# deliberately spans the grids: all-low/all-mid/all-high are exact grid knots, "mixed" reprises the
# ORIGINAL off-knot triple from make_surrogate.py so both PWL interpolation AND the additive-order
# assumption are stress-tested at each point, not just exact-knot lookups at one place.
PERMS = [("VINE", "TrustMark", "VideoSeal"),
         ("TrustMark", "VideoSeal", "VINE"),
         ("VideoSeal", "VINE", "TrustMark")]

STRENGTH_COMBOS = {
    # indices must follow the grid LENGTH, not the 5-point layout these were first written for:
    # on a 9-point grid a hardcoded [4] is the midpoint, so "all_high" silently re-measured "all_mid"
    # and the true top of the range went untested.
    "all_low":  {f: GRID[f][0]                 for f in ORDER},
    "all_qtr":  {f: GRID[f][len(GRID[f]) // 4] for f in ORDER},
    "all_mid":  {f: GRID[f][len(GRID[f]) // 2] for f in ORDER},
    "all_high": {f: GRID[f][-1]                for f in ORDER},
    "mixed":    {"VINE": 0.5, "TrustMark": 0.85, "VideoSeal": 1.1},
}

def predicted_ba(VS, perm, f, a):
    val = sg.base(f, a).eval(VS[f])
    idx = perm.index(f)
    for g in perm[idx + 1:]:                     # only fragments embedded AFTER f interfere with f
        val -= sg.delta(g, f, a).eval(VS[g])
    return val

def predicted_MSE(VS):
    total = sum(sg.d(f).eval(VS[f]) for f in ORDER)
    for (f1, f2) in PAIRS:
        total += sg.e(f1, f2).eval(VS[f1] + VS[f2])
    return total

def run_additivity(VS, label):
    val_true_ba = defaultdict(list)      # (perm_idx, f, a) -> [ba,...]
    val_true_MSE = defaultdict(list)     # perm_idx -> [mse,...]
    for pi, perm in enumerate(PERMS):
        for i, cover in enumerate(covers):
            t = targets[i]
            try:
                img = cover
                for f in perm:
                    img = FR[f].embed_with_target(img, t, strength=VS[f])
                val_true_MSE[pi].append(mse(cover, img))
                for a, af in ATT.items():
                    att = af(img)
                    if att is None:
                        continue
                    for f in ORDER:
                        ba = float(np.mean(hard(f, att) == t))
                        val_true_ba[(pi, f, a)].append(ba)
            except Exception as ex:
                print(f"  !! validation[{label}] perm {perm} image {i} FAILED: {type(ex).__name__}: {ex}",
                      flush=True)
        print(f"  validation[{label}] perm {perm} done [{time.time()-T0:.0f}s]", flush=True)

    pred_MSE = predicted_MSE(VS); pred_psnr = psnr_from_mse(pred_MSE)
    max_dba, max_dpsnr = 0.0, 0.0
    detail, per_perm = [], []
    for pi, perm in enumerate(PERMS):
        true_MSE = float(np.mean(val_true_MSE[pi])) if val_true_MSE[pi] else float("nan")
        true_psnr = psnr_from_mse(true_MSE)
        dpsnr = abs(true_psnr - pred_psnr); max_dpsnr = max(max_dpsnr, dpsnr)
        for a in ATT:
            for f in ORDER:
                vals = val_true_ba.get((pi, f, a))
                if not vals:
                    continue
                true_ba = float(np.mean(vals))
                pba = predicted_ba(VS, perm, f, a)
                dba = abs(true_ba - pba); max_dba = max(max_dba, dba)
                detail.append({"perm": list(perm), "f": f, "a": a,
                                "true_ba": round(true_ba, 4), "pred_ba": round(pba, 4), "dba": round(dba, 4)})
        per_perm.append({"perm": list(perm), "true_MSE": round(true_MSE, 3), "true_psnr": round(true_psnr, 3),
                          "dpsnr": round(dpsnr, 3)})
        print(f"  [{label}] perm {perm}: true_MSE={true_MSE:.3f} pred_MSE={pred_MSE:.3f} "
              f"true_psnr={true_psnr:.3f} pred_psnr={pred_psnr:.3f} dpsnr={dpsnr:.3f}", flush=True)

    result = {
        "label": label, "strengths": dict(VS),
        "max_dba": round(max_dba, 4), "max_dpsnr": round(max_dpsnr, 4),
        "pass": bool(max_dba <= 0.02 and max_dpsnr <= 0.3),
        "gate": {"dba_le": 0.02, "dpsnr_le": 0.3},
        "pred_MSE": round(pred_MSE, 3), "pred_psnr": round(pred_psnr, 3),
        "perms": [list(p) for p in PERMS], "per_perm": per_perm,
        "worst10": sorted(detail, key=lambda r: -r["dba"])[:10],
    }
    print(f"ADDITIVITY[{label}]: max_dba={result['max_dba']} max_dpsnr={result['max_dpsnr']} "
          f"pass={result['pass']}", flush=True)
    return result

additivity_multi = {}
for label, VScombo in STRENGTH_COMBOS.items():
    additivity_multi[label] = run_additivity(VScombo, label)

dba_seq = [additivity_multi[l]["max_dba"] for l in ("all_low", "all_qtr", "all_mid", "all_high")
           if l in additivity_multi]
dpsnr_seq = [additivity_multi[l]["max_dpsnr"] for l in ("all_low", "all_mid", "all_high")]
degrades = bool(dba_seq[-1] > dba_seq[0] + 0.01 or dpsnr_seq[-1] > dpsnr_seq[0] + 0.1)
additivity_multi["trend"] = {
    "order": ["all_low", "all_mid", "all_high"],
    "max_dba_sequence": dba_seq, "max_dpsnr_sequence": dpsnr_seq,
    "degrades_at_high_strength": degrades,
    "note": "compares max|dba| and max|dPSNR| at each fragment's grid-min vs grid-mid vs grid-max "
            "triple (the 'mixed' off-knot combo is reported separately, not part of this monotone "
            "sequence). An increase from low->high means the pairwise-additive model gets LESS "
            "accurate as fragments are embedded more strongly (more nonlinear cross-fragment "
            "interaction) -- reported honestly whether or not it happens.",
}
# backward-compatible: keep the ORIGINAL single-point "additivity" key, same VS triple/schema as
# make_surrogate.py, so anything reading the old shape still finds it unchanged.
additivity = additivity_multi["mixed"]

# ================= assemble + write =================
try:
    git_rev = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    git_rev = "unknown"

out = sg.to_dict()
out["additivity"] = additivity
out["additivity_multi"] = additivity_multi
out["cap"] = {f"{f}|{a}": {"xs": cap_pwl[(f, a)].xs, "ys": cap_pwl[(f, a)].ys} for (f, a) in cap_pwl}
out["frontend"] = frontend
out["source"] = {"pool": composition_of(files),   # the real composition, not a placeholder
                 
    "n": n_ok, "n_requested": N,
    "measured_at": os.environ.get("TS", "unknown"),
    "git": git_rev,
    "distortion_domain": "mse_255",     # d_f/e_fg below are MSE over 0-255 pixels, NOT dB
    "grids": GRID, "mid": MID, "coarse_idx": COARSE_IDX,
    "scaling_w_default": scaling_w_default,
    "videoseal_scaling_w_default": scaling_w_default, "videoseal_scaling_w_path": scaling_w_path,
    "canonical_order": ORDER, "images_dir": f"{SC}/pool/E/img",
    "elapsed_sec": round(time.time() - T0, 1),
    "extends": "make_surrogate.py / surrogate_table.json (same fragments/attacks/grids/methodology)",
    "gaps_added": ["cap (GAP1: capacity-vs-strength via soft-MI)",
                   "frontend (GAP3a: nested-VINE + resync as fn of strength)",
                   "additivity_multi (GAP3b: 4 strength combos x 3 orders)"],
    "resync_status": "wired" if RESYNC_ENABLED else "skipped",
    "resync_skip_reason": RESYNC_SKIP_REASON,
    "fix_log": ["2026-08-28: coordinator caught base_nested_VINE|crop75,|crop50 reading at chance "
                "in the first full-run smoke because the crop-attacked nested image was decoded at "
                "a single canonical scale (ring intact, wrong scale). Fixed to match the deployed "
                "geo_cascade blind scale search (max bit-acc over np.arange(0.34,1.0001,0.005), "
                "crop-without-resize into raw_probs); those two cells subsample to N_NESTED_SCALE "
                "for cost and the old single-scale number is kept as nested_VINE_no_scale_search|*."],
}
out_path = f"{SC}/surrogate_table_ext9.json"
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote", out_path, flush=True)
print("MAKE_SURROGATE_EXT_DONE", flush=True)
