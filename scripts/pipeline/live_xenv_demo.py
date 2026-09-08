"""End-to-end demonstration that the live gate closes: a C4-style request, solved and then measured.

Two runs of the same request. The first has no cross-environment executor, so UnMarker (adversarial)
and the sampled CtrlRegen+ step come back in `pending_live` and the verdict says so. The second passes
xenv_measurer, which embeds the user's images with the configuration the solve returned, hands them to
the attack's own conda environment, and reads them back with the deployed best-path decoder; the loop
then either certifies the request or patches the curve it read and re-solves.
Usage: python live_xenv_demo.py [n_requests=2] [n_img=10] [n_unmarker=8]
"""
import sys, os, json, time, random
from PIL import Image
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
NREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NIMG = int(sys.argv[2]) if len(sys.argv) > 2 else 10
NUNM = int(sys.argv[3]) if len(sys.argv) > 3 else 8
sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
from surrogate_model import Surrogate
from live_calibration import solve_with_live, composite_measurer
from xenv_measure import xenv_measurer, XENV_OK
from class_defs import classes, sample_class
from src.image_pool import sample as _pool_sample, composition_of
SG = Surrogate.from_dict(json.load(open(f"{SC}/surrogate_canonical.json")))
LIVE_OK = {"jpeg25","blur","noise","bright","contrast","crop75","crop50","rot9","rs256","hflip","crop_jpeg","border20","vaeB","vaeC","regen","rinse2x"}
C = classes(SG.attacks)[3]
SCEN = sample_class(C, 3, 2000, W.beta_from_fpr)
# the requests the class evaluation found FEASIBLE, so the demonstration is about the gate rather than
# about infeasibility; sampled from the certified set so the configuration is one already measured offline
cert = [r["i"] for r in json.load(open(f"{SC}/certify_classes_C4.json"))["rows"] if r["verdict"] == "SAT"]
random.seed(11); pick = random.sample(cert, min(NREQ, len(cert)))
files = _pool_sample(NIMG, offset=700)          # a slice no fit, certification or full-pool run touched
covers = [Image.open(f).convert("RGB").resize((512, 512), Image.BICUBIC) for f in files]
print(f"{len(pick)} C4 requests x {NIMG} held-out images {composition_of(files)}; UnMarker on {NUNM}", flush=True)
inproc = composite_measurer(covers, dev="cuda")
xmeas = xenv_measurer(covers, dev="cuda", n_unmarker=NUNM)
rows = []
for i in pick:
    sc = SCEN[i]
    scen = dict(min_psnr=sc["min_psnr"], max_ms=sc["max_ms"], attacks=list(sc["attacks"]), min_ba=sc["min_ba"],
                allow_resync=True, allow_nested=True, min_bits=sc["min_bits"], resolution=512, margin=W.DEFAULT_MARGIN)
    out = {"i": i, "cr": sc["_cr"], "beta": sc["min_ba"], "fpr": sc["fpr"]}
    for tag, xm, xok in (("no executor", None, ()), ("xenv executor", xmeas, XENV_OK)):
        t0 = time.time()
        print(f"\n=== q{i:04d} beta={sc['min_ba']:.2f} {sc['_cr']}  [{tag}]", flush=True)
        r = solve_with_live(scen, inproc, SG, LIVE_OK, max_rounds=2, prior_sd=0.023,
                            log=lambda s: print(s, flush=True), xenv_measure=xm, xenv_ok=xok)
        print(f"    verdict  : {r['verdict']}", flush=True)
        print(f"    pending  : {r.get('pending_live')}", flush=True)
        print(f"    patched  : {[(p['attack'], p['table'], p['live'], p['offset']) for p in r['patched']]}", flush=True)
        print(f"    took {time.time()-t0:.0f}s", flush=True)
        out[tag] = {"verdict": r["verdict"], "pending": r.get("pending_live"), "patched": r["patched"],
                    "provenance": r["provenance"], "live_reason": r.get("live_reason"), "sec": r["sec"]}
    rows.append(out)
json.dump({"rows": rows, "n_img": NIMG, "n_unmarker": NUNM}, open(f"{SC}/live_xenv_demo.json", "w"), indent=1)
print("\nwrote live_xenv_demo.json  DEMO_DONE", flush=True)
