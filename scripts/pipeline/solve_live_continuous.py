"""The CEGAR loop on the CONTINUOUS solver, as a driver over live_calibration.solve_with_live.

Requests are sampled from the same distribution as the solver-level evaluation; the user's images are
a cross-source held-out slice (image_pool offset 400, disjoint from the fit slice at 0, the diffusion
campaign at 100 and the curve validation at 100), N per request; the patch rule is the gated, shrunk,
asymmetric offset with the live standard error and the measured prior width.

Usage: python solve_live_continuous.py [n_requests=8] [n_images=50]
Env: OUT_NAME (default solve_live_continuous.json), MARGIN (request safety allowance, default W.DEFAULT_MARGIN)
"""
import sys, os, json, time, random
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC):
    sys.path.insert(0, p)
import watermark_smt_v2 as W
from surrogate_model import Surrogate
from src.attacks import attack_pil_any          # live measurement runs every attack, diffusion included, through the full dispatcher
from live_calibration import solve_with_live, composite_measurer
from src.image_pool import sample as _pool_sample, composition_of

NREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NIMG = int(sys.argv[2]) if len(sys.argv) > 2 else 50
MARGIN = float(os.environ.get("MARGIN", W.DEFAULT_MARGIN))
SG0 = Surrogate.from_dict(json.load(open(f"{SC}/surrogate_canonical.json")))
LIVE_OK = {"jpeg25", "blur", "noise", "bright", "contrast", "crop75", "crop50", "rot9", "rs256", "hflip",
           "crop_jpeg", "vaeB", "vaeC", "regen", "rinse2x", "rinse4x"}
pw = json.load(open(f"{SC}/prior_width.json")) if os.path.exists(f"{SC}/prior_width.json") else {}
PRIOR_SD = float(pw.get("prior_sd_p90") or pw.get("prior_sd_median") or 0.02)

sys.argv = [sys.argv[0]]
import solver_eval_continuous as SEC
random.seed(7)
pool = [s for s in SEC.SCEN if all(a in SG0.attacks for a in s["attacks"])]
sample = random.sample(pool, min(NREQ, len(pool)))
files = _pool_sample(NIMG, offset=400)
covers = [Image.open(f).convert("RGB").resize((512, 512), Image.BICUBIC) for f in files]
print(f"{len(sample)} requests x {NIMG} held-out images {composition_of(files)}, margin {MARGIN}, prior sd {PRIOR_SD:.4f}\n", flush=True)
measure = composite_measurer(covers, dev="cuda")
# The columns the offline table is not allowed to settle (UnMarker always, and any column whose table
# value sits within 2 image-to-image sd of the threshold) need an executor or the request comes back
# "SAT (pending live: ...)". XENV=1 turns on the cross-environment one; it costs about 3.5 GPU-minutes
# per image on UnMarker, so that column runs on UNMK_N images rather than all of them.
XENV = os.environ.get("XENV", "0") == "1"
xmeasure, xok = None, ()
if XENV:
    from xenv_measure import xenv_measurer, XENV_OK
    xmeasure = xenv_measurer(covers, dev="cuda", n_unmarker=int(os.environ.get("UNMK_N", 8)))
    xok = XENV_OK
    print(f"cross-environment executor ON for {xok} (UnMarker on {os.environ.get('UNMK_N', 8)} images)", flush=True)

rows = []
for qi, sc in enumerate(sample):
    scen = dict(min_psnr=sc["min_psnr"], max_ms=sc["max_ms"], attacks=list(sc["attacks"]), min_ba=sc["min_ba"],
                allow_resync=sc["allow_resync"], allow_nested=sc["allow_nested"], min_bits=sc["min_bits"], resolution=512, margin=MARGIN)
    print(f"  [{qi+1}/{len(sample)}] {sc['_aset']} beta={sc['min_ba']:.2f}", flush=True)
    r = solve_with_live(scen, measure, SG0, LIVE_OK, max_rounds=4, prior_sd=PRIOR_SD,
                        log=lambda s: print(s, flush=True), xenv_measure=xmeasure, xenv_ok=xok)
    rows.append({"i": qi, "aset": sc["_aset"], "beta": sc["min_ba"], **r})
    print(f"      -> {r['verdict']} after {r['rounds']} round(s), {len(r['patched'])} cell(s) patched [{r['sec']:.0f}s]\n", flush=True)

n_pend = sum(1 for r in rows if r.get("pending_live"))
n_cert = sum(1 for r in rows if r["verdict"].startswith("SAT (live-cert"))
n_incl = sum(1 for r in rows if r["verdict"].startswith("SAT (live-inconclusive"))
n_uns = sum(1 for r in rows if r["verdict"] == "UNSAT")
n_pat = sum(1 for r in rows if any(p["offset"] != 0 for p in r["patched"]))
print(f"=== {len(rows)} requests ===")
print(f"  live-certified SAT        : {n_cert}")
print(f"  SAT, disagreement in noise: {n_incl}")
print(f"  UNSAT                     : {n_uns}")
print(f"  needed >=1 patch+resolve  : {n_pat}")
print(f"  left owing a live column  : {n_pend}" + ("" if XENV else "  (run with XENV=1 to close them)"))
print(f"  mean rounds               : {np.mean([r['rounds'] for r in rows]):.2f}")
allp = [p for r in rows for p in r["patched"] if p["offset"] != 0]
if allp:
    d = np.array([p["table"] - p["live"] for p in allp])
    print(f"  patched cells             : {len(allp)}, table minus live: mean {d.mean():+.4f}, max {d.max():+.4f}")
json.dump({"rows": rows, "n_img": NIMG, "margin": MARGIN, "prior_sd": PRIOR_SD, "images": composition_of(files),
           "table_assembled_at": json.load(open(f"{SC}/surrogate_canonical.json"))["source"].get("assembled_at")},
          open(f"{SC}/{os.environ.get('OUT_NAME', 'solve_live_continuous.json')}", "w"), indent=2, default=str)
print("\nwrote", os.environ.get("OUT_NAME", "solve_live_continuous.json")); print("SOLVE_LIVE_CONTINUOUS_DONE")
