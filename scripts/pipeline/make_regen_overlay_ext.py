"""EXT regen overlay -- Phase 4 gap-closure (item B): all 3 fragments swept over their full native
strength grids under regen, ALL 6 ordered-pair regen deltas, and rinse (regen applied a 2nd time in
sequence -- this repo's existing rinse definition, see measure_vine_rinse.py / disentangle_rinse_res.py
/ measure_regen_rinse_compare.py: "rinse reuses regen's 1st round") giving base_f(s,rinse) on all 3
fragments plus the VINE<->VideoSeal ordered-pair deltas under rinse.

Measured on the cross-source pool sample (src/image_pool.sample) at per-source offset SLICE_START=100,
disjoint from the in-process fit slice (offset 0, n=100) and identical to the mild-regen column's images.
Nine strength knots per fragment (see GRID), matching the rest of the table.

CROSS-ENV FLOW ([A]/[B]/[C] as in make_regen_overlay.py, plus an extra [B2] second ctrlregen pass
for rinse), driven by make_regen_overlay_ext.sbatch:
  [fingerprint env] python make_regen_overlay_ext.py embed  N   -> PNGs in $WORK/embed
  [ctrlregen env]   ctrlregen_batch.py --in_dir embed --out_dir attacked --step 0.5 --seed 1   (regen, 1x)
  [shell]           symlink the rinse-eligible subset of attacked/ into attacked_for_rinse/
  [ctrlregen env]   ctrlregen_batch.py --in_dir attacked_for_rinse --out_dir rinsed --step 0.5 --seed 2  (rinse=2x)
  [fingerprint env] python make_regen_overlay_ext.py decode N   -> $SC/surrogate_regen_overlay_ext.json

Usage: python make_regen_overlay_ext.py {embed|decode} [N=50]
"""
import sys, os, io, json, glob, time, subprocess
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

STAGE = sys.argv[1] if len(sys.argv) > 1 else "embed"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
assert STAGE in ("embed", "decode"), f"unknown stage {STAGE!r}, expected embed|decode"

# --- env parameterization (defaults reproduce the original regen/rinse run exactly) ---
ATT_LABEL   = os.environ.get("ATT_LABEL", "regen")          # surrogate attack column name
RINSE_LABEL = os.environ.get("RINSE_LABEL", "rinse")
DO_RINSE    = os.environ.get("DO_RINSE", "1") == "1"
WORK_SUFFIX = os.environ.get("WORK_SUFFIX", "")
OUT_NAME    = os.environ.get("OUT_NAME", "surrogate_regen_overlay_ext.json")
WORK = f"{SC}/regen_overlay_ext_n{N}{WORK_SUFFIX}"
EMBED_DIR = f"{WORK}/embed"
ATT_DIR = f"{WORK}/attacked"          # regen (1x, ctrlregen pass 1) output
RIN_DIR = f"{WORK}/rinsed"            # rinse (2x, ctrlregen pass 2 on pass-1's own output) output
REGEN_STEP = float(os.environ.get("REGEN_STEP", "0.5"))   # ctrlregen_batch.py default --step (canonical "regen", not the s03/05/07 sweep)
SLICE_START = 100  # per-source offset into the cross-source pool sample: disjoint from the in-process
                    # fit slice (offset 0, n=100) and the same images the mild-regen column uses

sb = ShortenedBCH(); NB = sb.n           # 100 transmitted bits
KEY = b"v5_key_encoder_master"
dev = "cuda"

# Nine knots, matching the rest of the table. The diffusion columns were left at five when the
# cheap columns were densified, because the leave-one-out check that justified the move scoped
# itself to "this (cheap) attack family". Measured since: thinning the nine-knot columns back to
# five costs a median 0.0066 and up to 0.0488 bit accuracy at the dropped knots, and the diffusion
# curves carry 1.09x the curvature of the cheap ones at the same knot count -- so the penalty
# applies here too rather than being absorbed by flatter curves.
import numpy as _np9
GRID = {f: [round(float(v), 4) for v in _np9.linspace(lo, hi, 9)]
        for f, (lo, hi) in {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6),
                            "VideoSeal": (0.5, 1.5)}.items()}
RANGES = {f: (GRID[f][0], GRID[f][-1]) for f in GRID}
MID = {f: GRID[f][len(GRID[f]) // 2] for f in GRID}   # the middle knot, as make_surrogate_ext9.py takes it
assert MID == {"VINE": 0.6, "TrustMark": 1.0, "VideoSeal": 1.0}, MID   # index 2 of a 9-knot grid is NOT the middle

print(f"[{STAGE}] N={N} WORK={WORK} NB={NB} REGEN_STEP={REGEN_STEP} SLICE=[{SLICE_START}:{SLICE_START+N})", flush=True)

print("loading fragments...", flush=True)
FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
      "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
      "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}


def target_bits(i):
    return np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)


def hard(fn, pil):
    f = FR[fn]
    v = f.raw_probs(pil) if fn == "VINE" else f.raw_logits(pil)
    v = np.asarray(v).ravel()[:NB]
    return (v > (0.5 if fn == "VINE" else 0.0)).astype(np.uint8)


from src.attacks import GEO
rot9 = GEO["rot9"]   # single definition

# Across all five sources in the evaluation set's proportions, like every other overlay. Deterministic
# in (N, offset), so embed and decode -- separate processes -- see the same list.
from src.image_pool import sample as _pool_sample, composition_of as _composition_of
files = _pool_sample(N, offset=SLICE_START)
assert len(files) == N, f"pool sample at offset {SLICE_START} has {len(files)} images, need N={N}"


def mse(a, b):
    return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))


def psnr_from_mse(m):
    return 10 * np.log10(255.0 ** 2 / max(m, 1e-9))


def r512(img):
    """Canvas normalisation, NOT an attack -- it restores the working size after an embed. The old
    name `resize512` read like the rs256 resample attack, which is a different thing entirely."""
    return img if img.size == (512, 512) else img.resize((512, 512))

# =====================================================================================
# EXT: every fragment swept over its full native-strength grid, and ALL 6 ordered pairs.
# Filenames encode the cell:  solo_{code}_{s}_{i}.png   pair_{first}{second}_{i}.png
CODE = {"VINE": "V", "TrustMark": "T", "VideoSeal": "S"}
UNCODE = {v: k for k, v in CODE.items()}
FRAGS = ["VINE", "TrustMark", "VideoSeal"]
ORDERED_PAIRS = [(a, b) for a in FRAGS for b in FRAGS if a != b]      # 6 ordered pairs
CELLS_PER_IMAGE = sum(len(GRID[f]) for f in FRAGS) + len(ORDERED_PAIRS)   # 15 + 6 = 21

# rinse (2nd ctrlregen pass) only needs: all 3 fragments' solo grids (15) + the VINE<->VideoSeal
# pair in both orders (2) = 17 cells/image -- NOT the 4 TrustMark cross pairs (out of task scope
# for rinse; would need ~another whole pass of ctrlregen compute for numbers nobody asked for).
RINSE_PAIR_CODES = {CODE["VINE"] + CODE["VideoSeal"], CODE["VideoSeal"] + CODE["VINE"]}   # {"VS","SV"}
RINSE_CELLS_PER_IMAGE = sum(len(GRID[f]) for f in FRAGS) + len(RINSE_PAIR_CODES)          # 15 + 2 = 17

if STAGE == "embed":
    os.makedirs(EMBED_DIR, exist_ok=True)
    print(f"[embed] pool sample offset {SLICE_START}, n={len(files)} sources={_composition_of(files)} -> {EMBED_DIR} "
          f"({CELLS_PER_IMAGE} cells/image, {RINSE_CELLS_PER_IMAGE} of them rinse-eligible)", flush=True)

    clean, psnr_log = {}, {}
    t0 = time.time()
    for i, fp in enumerate(files):
        cover = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC)
        t = target_bits(i)
        solo = {}
        # ---- solo: every fragment over its full grid ----
        for f in FRAGS:
            solo[f] = {}
            for s in GRID[f]:
                emb = r512(FR[f].embed_with_target(cover, t, strength=s))
                solo[f][s] = emb
                emb.save(f"{EMBED_DIR}/solo_{CODE[f]}_{s:.3f}_{i:05d}.png")
                clean.setdefault(f"{f}@{s}", []).append(float(np.mean(hard(f, emb) == t)))
                psnr_log.setdefault(f"{f}@{s}", []).append(psnr_from_mse(mse(cover, emb)))
        # ---- all 6 ordered pairs at reference (mid) strengths: first -> second ----
        for (f1, f2) in ORDERED_PAIRS:
            comp = r512(FR[f2].embed_with_target(solo[f1][MID[f1]], t, strength=MID[f2]))
            comp.save(f"{EMBED_DIR}/pair_{CODE[f1]}{CODE[f2]}_{i:05d}.png")
            clean.setdefault(f"pair_{f1}->{f2}|{f1}", []).append(float(np.mean(hard(f1, comp) == t)))
        if (i + 1) % max(1, N // 10) == 0 or i + 1 == N:
            print(f"  ...{i+1}/{N} embedded [{time.time()-t0:.0f}s]", flush=True)

    n_png = len(glob.glob(f"{EMBED_DIR}/*.png"))
    print(f"[embed] wrote {n_png} PNGs (expect {CELLS_PER_IMAGE*N}) for ctrlregen", flush=True)
    print("[embed] clean-decode sanity (should be ~1.0):", flush=True)
    for k in sorted(clean):
        print(f"    {k:34s} {float(np.mean(clean[k])):.3f}", flush=True)
    json.dump({"clean_sanity": {k: round(float(np.mean(v)), 4) for k, v in clean.items()},
               "embed_psnr": {k: round(float(np.mean(v)), 2) for k, v in psnr_log.items()},
               "cells_per_image": CELLS_PER_IMAGE, "n": N,
               "slice": [SLICE_START, SLICE_START + N]},
              open(f"{WORK}/embed_stage_results.json", "w"), indent=2)
    print("REGEN_OVERLAY_EXT_EMBED_DONE", flush=True)

# =====================================================================================
elif STAGE == "decode":
    def scan_dir(dirpath):
        """Decode every recognized cell in dirpath (one attack condition's worth of PNGs) and
        return (solo_ba, pair_ba, n_files). Shared by the regen (ATT_DIR) and rinse (RIN_DIR) scans."""
        files = sorted(glob.glob(f"{dirpath}/*.png"))
        solo_ba = {(f, s): [] for f in FRAGS for s in GRID[f]}
        pair_ba = {(f1, f2): [] for (f1, f2) in ORDERED_PAIRS}
        unknown = []
        t0 = time.time()
        for k, fp in enumerate(files):
            bn = os.path.basename(fp)[:-4]
            img = Image.open(fp).convert("RGB").resize((512, 512))
            try:
                if bn.startswith("solo_"):
                    _, c, s_str, i_str = bn.split("_")
                    f = UNCODE[c]; s = float(s_str); i = int(i_str)
                    key = min(GRID[f], key=lambda g: abs(g - s))       # snap to the grid knot
                    solo_ba[(f, key)].append(float(np.mean(hard(f, img) == target_bits(i))))
                elif bn.startswith("pair_"):
                    _, cc, i_str = bn.split("_")
                    f1, f2 = UNCODE[cc[0]], UNCODE[cc[1]]; i = int(i_str)
                    pair_ba[(f1, f2)].append(float(np.mean(hard(f1, img) == target_bits(i))))
                else:
                    unknown.append(bn)
            except Exception as ex:
                print(f"  !! parse {bn}: {type(ex).__name__}: {ex}", flush=True); unknown.append(bn)
            if (k + 1) % max(1, len(files) // 10) == 0 or k + 1 == len(files):
                print(f"  ...{k+1}/{len(files)} decoded from {os.path.basename(dirpath)} [{time.time()-t0:.0f}s]", flush=True)
        if unknown:
            print(f"[decode] WARNING {len(unknown)} unparsed in {dirpath}, e.g. {unknown[:4]}", flush=True)
        return solo_ba, pair_ba, len(files)

    mean = lambda L: (float(np.mean(L)) if L else float("nan"))

    # ---- regen (1x, pass 1) ----
    solo_ba, pair_ba, n_att = scan_dir(ATT_DIR)
    print(f"[decode] {n_att} attacked(regen) images in {ATT_DIR} (expect {CELLS_PER_IMAGE*N})", flush=True)
    if n_att == 0:
        print("[decode] FATAL: ctrlregen produced nothing", flush=True); sys.exit(1)

    base = {}
    for f in FRAGS:
        ys = [mean(solo_ba[(f, s)]) for s in GRID[f]]
        base[f"{f}|{ATT_LABEL}"] = {"xs": list(GRID[f]), "ys": [round(y, 4) for y in ys]}
        print(f"[decode] base_{f}(s,regen) = " +
              ", ".join(f"{s}:{y:.3f}" for s, y in zip(GRID[f], ys)), flush=True)

    delta = {}
    for (f1, f2) in ORDERED_PAIRS:                    # composite f1 -> f2 measures delta_{f2->f1}
        alone = mean(solo_ba[(f1, MID[f1])])
        after = mean(pair_ba[(f1, f2)])
        dv = max(0.0, alone - after)
        lo, hi = RANGES[f2]
        delta[f"{f2}|{f1}|{ATT_LABEL}"] = {"xs": [lo, hi], "ys": [round(dv, 4), round(dv, 4)]}
        print(f"[decode] delta_{f2}->{f1}(regen) = {dv:.4f}   "
              f"(alone {alone:.3f} -> after {after:.3f})", flush=True)

    # ---- rinse (2x: ctrlregen applied a 2nd time to its own pass-1 output) ----
    rin_files_probe = sorted(glob.glob(f"{RIN_DIR}/*.png"))
    rinse_skip_reason = None
    base_rinse, delta_rinse = {}, {}
    if not rin_files_probe:
        rinse_skip_reason = f"no images found in {RIN_DIR} -- rinse pass (B2) did not run or wrote elsewhere"
        print(f"[decode] WARNING: rinse SKIPPED -- {rinse_skip_reason}", flush=True)
    else:
        rsolo_ba, rpair_ba, n_rin = scan_dir(RIN_DIR)
        print(f"[decode] {n_rin} rinsed(2x) images in {RIN_DIR} (expect {RINSE_CELLS_PER_IMAGE*N})", flush=True)
        for f in FRAGS:
            ys = [mean(rsolo_ba[(f, s)]) for s in GRID[f]]
            base_rinse[f"{f}|{RINSE_LABEL}"] = {"xs": list(GRID[f]), "ys": [round(y, 4) for y in ys]}
            print(f"[decode] base_{f}(s,rinse)  = " +
                  ", ".join(f"{s}:{y:.3f}" for s, y in zip(GRID[f], ys)), flush=True)
        for (f1, f2) in [("VINE", "VideoSeal"), ("VideoSeal", "VINE")]:
            alone = mean(rsolo_ba[(f1, MID[f1])])
            after = mean(rpair_ba[(f1, f2)])
            dv = max(0.0, alone - after)
            lo, hi = RANGES[f2]
            delta_rinse[f"{f2}|{f1}|{RINSE_LABEL}"] = {"xs": [lo, hi], "ys": [round(dv, 4), round(dv, 4)]}
            print(f"[decode] delta_{f2}->{f1}(rinse) = {dv:.4f}   "
                  f"(alone {alone:.3f} -> after {after:.3f})", flush=True)

    base.update(base_rinse)
    delta.update(delta_rinse)

    try:
        git_rev = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_rev = "unknown"

    attacks_added = [ATT_LABEL] + ([] if (rinse_skip_reason or not DO_RINSE) else [RINSE_LABEL])
    out = {"attacks_added": attacks_added, "base": base, "delta": delta,
           "source": {"n": N, "pool": _composition_of(files), "knots": 9,
                      "images_slice": [SLICE_START, SLICE_START + N],
                      "slice_note": (f"cross-source pool sample at per-source offset {SLICE_START} "
                                     "(src/image_pool.sample): disjoint from the in-process fit slice "
                                     "(offset 0, n=100); the same images as the mild-regen column."),
                      "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git_rev,
                      "regen_mechanism": {"script": "scripts/attack/ctrlregen_batch.py", "step": REGEN_STEP,
                                          "seed": 1, "note": "same single canonical regen strength as "
                                          "make_regen_overlay.py (ctrlregen_batch.py's own --step default)"},
                      "rinse_mechanism": {
                          "wired": rinse_skip_reason is None,
                          "skipped_reason": rinse_skip_reason,
                          "definition": ("ctrlregen_batch.py applied a SECOND time to the first pass's own "
                                         "output (regen composed with itself) -- matches this repo's "
                                         "existing rinse definition (measure_vine_rinse.py, "
                                         "disentangle_rinse_res.py, measure_regen_rinse_compare.py: "
                                         "'rinse reuses regen's 1st round' / stable_regen applied twice "
                                         "sequentially with different seeds), just substituting THIS "
                                         "overlay's canonical regen operator (ctrlregen_batch.py) for "
                                         "those scripts' _regen_util.py stable-diffusion operator."),
                          "step": REGEN_STEP, "pass2_seed": 2,
                          "seed_note": ("pass 2 uses --seed 2 (pass 1 uses ctrlregen_batch.py's default "
                                        "--seed 1) so the two rounds are not seed-identical, mirroring "
                                        "measure_vine_rinse.py's per-round seed increment."),
                          "coverage": ("base_f(s,rinse) for ALL 3 fragments over their full 9-knot grids; "
                                       "delta only for the VINE<->VideoSeal ordered pair (both directions) "
                                       "-- NOT extended to the 4 TrustMark cross pairs under rinse (out of "
                                       "task scope; would double the already-expensive 2nd ctrlregen pass "
                                       "for numbers nobody asked for).")},
                      "grids": GRID, "mid": MID, "cells_per_image": CELLS_PER_IMAGE,
                      "rinse_cells_per_image": RINSE_CELLS_PER_IMAGE,
                      "coverage": ("ALL 3 fragments swept over their full 9-knot native grids under regen "
                                   "(TrustMark|regen, VideoSeal|regen were single-point in the original "
                                   "overlay); ALL 6 ordered pairs measured at reference (mid) strengths "
                                   "under regen (the original overlay only had VINE<->VideoSeal, 2 of 6); "
                                   "rinse per rinse_mechanism above."),
                      "supersedes": ("surrogate_regen_overlay.json's base/delta cells -- this file's base/"
                                     "delta are measured on a disjoint slice and are the more complete "
                                     "coverage; merge by preferring THIS file's keys on overlap.")}}
    json.dump(out, open(f"{SC}/{OUT_NAME}", "w"), indent=2)
    json.dump(out, open(f"{WORK}/{OUT_NAME}", "w"), indent=2)
    print(f"[decode] wrote {SC}/{OUT_NAME}  (attack column: {ATT_LABEL})", flush=True)
    print("REGEN_OVERLAY_EXT_DECODE_DONE", flush=True)
