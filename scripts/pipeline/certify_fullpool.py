"""Full-pool certification: one certified request per class, embedded on ALL 10,000 pool images (five
sources), every in-scope attack run live, judged against the request's base threshold, with per-source
breakdown and the delivered PSNR on the whole pool.

The representative is the class's certified request at the modal false-positive budget (1e-2) with the
fewest optional attacks (deterministic; falls back to the first certified request). Family "inprocess" runs
the class's core + optional attacks that the in-process attack layer implements, minus the diffusion pair;
family "diffusion" runs regen and rinse2x on the first N images. Attacks the representative's own request
did not name are still run and flagged out of scope. Usage:
  python certify_fullpool.py <class 0-4> <shard> <n_shards> [N=10000] [family=inprocess]
"""
import sys, os, json, time
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
CLS = int(sys.argv[1]); SHARD = int(sys.argv[2]); NSHARD = int(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
FAMILY = sys.argv[5] if len(sys.argv) > 5 else "inprocess"
sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
from class_defs import classes
from eval_matrix import OursComposite
from src.attacks import attack_pil_any
from src.image_pool import sample as _pool_sample, composition_of, POOL
sg_attacks = json.load(open(f"{SC}/surrogate_canonical.json"))["attacks"]
C = classes(sg_attacks)[CLS]; dev = "cuda"
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}
INPROCESS = {"jpeg25", "blur", "noise", "bright", "contrast", "crop75", "crop50", "rot9", "rs256", "hflip", "crop_jpeg", "border20", "vaeB", "vaeC"}
DIFFUSION = {"regen", "rinse2x"}
R = json.load(open(f"{SC}/certify_classes_{C['key']}.json"))
sat = [r for r in R["rows"] if r.get("verdict") == "SAT" and r.get("live_feasible_inprocess")]
assert sat, f"{C['key']}: no certified request to represent the class"
core = set(C["core"])
cand = sorted(sat, key=lambda r: (abs(np.log10(r["fpr"]) + 2), len(set(r["attacks"]) - core), r["i"]))
rep = cand[0]; cfg = dict(rep["cfg"]); cfg["S"] = list(cfg["order"])
fam = INPROCESS if FAMILY == "inprocess" else DIFFUSION
attacks = [a for a in C["core"] + C["optional"] if a in fam]
in_scope = {a: (a in rep["attacks"]) for a in attacks}
thr_base = rep["threshold_base"]
comp = OursComposite(dev, tm_variant="B", vine_variant="R", config={"frags": [FKEY[f] for f in cfg["order"]], "order": [FKEY[f] for f in cfg["order"]],
                                                                   "strengths": {}, **W.frontend_config(cfg["fe"])})
comp.strength = dict(comp.DEFAULT_STRENGTH); comp.strength.update({FKEY[f]: float(cfg["s"][f]) for f in cfg["S"]})
def live_bestpath(v, sec, beta):
    iid, tx = sec
    fba, det = comp.decode(v, sec)
    def _per(img): return {n: float(np.mean((comp._frag_llr(n, img, iid) > 0).astype(np.uint8) == tx)) for n in comp.order}
    per = _per(v)
    if comp.geo and max(per.values()) < beta:
        ok, vw = comp.geo_cascade(v, iid, tx, return_view=True)
        if ok and vw is not None:
            per_c = _per(vw)
            if max(per_c.values()) > max(per.values()): per = per_c
    return max(per.values()), float(det)
def source_of(path):
    """pool/<source>/<file>: the source is the first path component under the pool root"""
    rel = os.path.relpath(path, POOL)
    return rel.split(os.sep)[0] if not rel.startswith("..") else "?"
files = _pool_sample(N, offset=0)
idx = list(range(SHARD, len(files), NSHARD))
print(f"{C['key']} {C['name']}: representative request {rep['i']} fpr={rep['fpr']:.0e} cfg={'+'.join(cfg['order'])} s={ {f: round(v, 3) for f, v in cfg['s'].items()} } "
      f"fe={[k for k, v in cfg['fe'].items() if v]} thr_base={thr_base:.3f}; family={FAMILY} attacks={attacks}; shard {SHARD}/{NSHARD}: {len(idx)} of {len(files)} images", flush=True)
def psnr(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64); m = np.mean((a - b) ** 2)
    return 99.0 if m == 0 else float(10 * np.log10(255.0 ** 2 / m))
out = {"cls": C["key"], "rep": rep["i"], "fpr": rep["fpr"], "cfg": cfg, "thr_base": thr_base, "family": FAMILY, "attacks": attacks, "in_scope": in_scope,
       "N": len(files), "shard": SHARD, "n_shards": NSHARD, "idx": idx, "src": [], "psnr": [], "ba": {a: [] for a in attacks}, "det": {a: [] for a in attacks}}
t0 = time.time()
for n, i in enumerate(idx):
    cov = Image.open(files[i]).convert("RGB").resize((512, 512), Image.BICUBIC)
    emb, sec = comp.embed(cov, i)
    out["src"].append(source_of(files[i])); out["psnr"].append(psnr(emb, cov))
    for a in attacks:
        v = attack_pil_any(a, emb, dev=dev)
        if v.size != (512, 512): v = v.resize((512, 512))
        bp, det = live_bestpath(v, sec, thr_base); out["ba"][a].append(bp); out["det"][a].append(det)
    if (n + 1) % 25 == 0:
        print(f"  {n+1}/{len(idx)} psnr {np.mean(out['psnr']):.2f} " + " ".join(f"{a}={np.mean(out['ba'][a]):.3f}" for a in attacks) + f" [{time.time()-t0:.0f}s]", flush=True)
od = f"{SC}/fullpool/{C['key']}"; os.makedirs(od, exist_ok=True)
json.dump(out, open(f"{od}/{FAMILY}_shard{SHARD:02d}of{NSHARD:02d}.json", "w"))
print(f"wrote {od}/{FAMILY}_shard{SHARD:02d}of{NSHARD:02d}.json  FULLPOOL_SHARD_DONE", flush=True)
