"""Interference over the overwriting fragment's strength on the CtrlRegen+ columns (cross-environment).

The protocol is make_delta_curves.py's: the host f is embedded at its mid strength with random target
bits, g is embedded over it at each knot of g's grid, f is decoded after the attack, and
delta(s_g) = ba_f(alone) - ba_f(with g at s_g), clipped at zero. The only difference is that CtrlRegen+
runs in its own conda environment, so the campaign is split around it (the sbatch runs the attack
between the two calls, `ctrlregen_batch.py --step_list 0.3,0.5,0.7`, which writes att_ctrlregen_s03,
_s05, _s07 next to the embed directory):
  embed  <g> <f> <n> <cell_dir>   -> cell_dir/embed/a_i%05d.png (f alone) and w%02d_i%05d.png (g at knot k)
  decode <g> <f> <cell_dir>       -> surrogate_delta_sweep_xenv_<g>_<f>_3of3.json (one curve per step)
"""
import sys, os, json, glob, subprocess
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
MODE, G, F = sys.argv[1], sys.argv[2], sys.argv[3]; REST = sys.argv[4:]; sys.argv = [sys.argv[0]]
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
from src.image_pool import sample as _pool_sample, composition_of

# g's grid spans the fragment's whole (extended) range, since the solver reads delta at g's strength
# anywhere in it; the host's reference strength is the original mid, as in every delta measurement.
GRID = {"VINE": [0.14, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "TrustMark": [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0],
        "VideoSeal": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.2]}
MID = {"VINE": 0.6, "TrustMark": 1.0, "VideoSeal": 1.0}
STEPS = ("s03", "s05", "s07")
sb = ShortenedBCH(); NB = sb.n; KEY = b"v5_key_encoder_master"; dev = "cuda"
FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
      "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
      "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}
def tb(i): return np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
def r512(im): return im if im.size == (512, 512) else im.resize((512, 512))
def hard(fn, pil):
    fr = FR[fn]; v = fr.raw_probs(pil) if fn == "VINE" else fr.raw_logits(pil)
    v = np.asarray(v).ravel()[:NB]
    return (v > (0.5 if fn == "VINE" else 0.0)).astype(np.uint8)

if MODE == "embed":
    n, cd = int(REST[0]), REST[1]; ed = f"{cd}/embed"; os.makedirs(ed, exist_ok=True)
    files = _pool_sample(n, offset=0)
    for i, fp in enumerate(files):
        cover = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC); t = tb(i)
        base = r512(FR[F].embed_with_target(cover, t, strength=MID[F]))
        base.save(f"{ed}/a_i{i:05d}.png")
        for k, s in enumerate(GRID[G]):
            r512(FR[G].embed_with_target(base, t, strength=s)).save(f"{ed}/w{k:02d}_i{i:05d}.png")
    json.dump({"g": G, "f": F, "grid": GRID[G], "host_strength": MID[F], "n": n, "pool": composition_of(files)},
              open(f"{cd}/meta.json", "w"), indent=1)
    print(f"embedded {n} covers x (1 + {len(GRID[G])}) sets for delta_{G}->{F} into {ed}  STAGE_DONE", flush=True)

elif MODE == "decode":
    cd = REST[0]; meta = json.load(open(f"{cd}/meta.json")); assert meta["g"] == G and meta["f"] == F
    grid, n = meta["grid"], int(meta["n"]); curves = {}; reference = {}
    for step in STEPS:
        ad = f"{cd}/att_ctrlregen_{step}"
        files = sorted(glob.glob(f"{ad}/*.png"))
        if not files:
            raise RuntimeError(f"{ad}: no attacked images; the CtrlRegen+ step did not run")
        alone, with_k = [], {k: [] for k in range(len(grid))}
        for fp in files:
            name = os.path.basename(fp)[:-4]; tag, istr = name.split("_i"); i = int(istr); t = tb(i)
            ba = float(np.mean(hard(F, r512(Image.open(fp).convert("RGB"))) == t))
            if tag == "a": alone.append(ba)
            else: with_k[int(tag[1:])].append(ba)
        short = [k for k in with_k if len(with_k[k]) < n] + (["a"] if len(alone) < n else [])
        if short:
            raise RuntimeError(f"{ad}: incomplete sets {short} (expected {n} images each)")
        a0 = float(np.mean(alone))
        ys = [round(max(0.0, a0 - float(np.mean(with_k[k]))), 4) for k in range(len(grid))]
        curves[f"{G}|{F}|ctrlregen_{step}"] = {"xs": grid, "ys": ys}
        reference[f"{F}@{MID[F]}|ctrlregen_{step}|alone"] = round(a0, 4)
        reference[f"{F}@{MID[F]}|ctrlregen_{step}|with_{G}"] = {str(s): round(float(np.mean(with_k[k])), 4) for k, s in enumerate(grid)}
        print(f"  ctrlregen_{step}: {F} alone {a0:.4f}; delta_{G}->{F}(s_{G}) = " + ", ".join(f"{s}:{y:.3f}" for s, y in zip(grid, ys)), flush=True)
    try: git = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception: git = "unknown"
    out = f"{os.environ.get('DELTA_OUT_DIR') or SC}/surrogate_delta_sweep_xenv_{G}_{F}_3of3.json"   # DELTA_OUT_DIR: smoke runs
    json.dump({"delta": curves, "base": {}, "reference": reference,
               "source": {"n": n, "pool": meta["pool"], "git": git, "measured_at": os.environ.get("TS", "unknown"),
                          "cells": [[G, F, f"ctrlregen_{s}"] for s in STEPS], "host_strength": MID[F],
                          "note": "f at mid, g swept over its knots after f; CtrlRegen+ in its own env; delta = alone - with"}},
              open(out, "w"), indent=1)
    print("wrote", out, flush=True); print("DELTA_XENV_DONE", flush=True)
else:
    raise SystemExit(f"unknown mode {MODE}")
