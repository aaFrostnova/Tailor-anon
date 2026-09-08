"""Front-end replacement cells on the columns that run in their own environment (CtrlRegen+ x3, UnMarker).

The deployed solo composite is embedded with exactly ONE stage on, the images are handed to the attacks
(the sbatch runs them between the two calls here), and every attacked image is read back three ways:
  * the deployed read with the stage on (per-fragment bit accuracy, keyed verification, acceptance;
    the same rule as the in-process stage cells, perimage_campaign._deployed_read), which feeds the
    per-image block the solver's acceptance floor reads at each request's own threshold;
  * the composite's own decode with the stage on and with the cascade suppressed, the pair
    make_frontend_components.py records as base_fe / ctrl_fe (bit accuracy) and det_fe / detctrl_fe
    (detection), which feeds the replacement curves;
  * the wall time of both decodes, which feeds latency_ms_<stage>|<column>.
  embed  <stage> <frag> <strength> <n> <cell_dir>  -> cell_dir/embed/i%05d.png
  decode <stage> <frag> <strength> <cell_dir>      -> perimage_ext/stage_<stage>_xenv/<frag>_<strength>.json
FEX_OUT overrides the output directory (smoke runs). fe_xenv_merge.py folds the cells into overlays.
"""
import sys, os, json, glob, time
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
MODE, STAGE, F, s = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); REST = sys.argv[5:]; sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
from eval_matrix import OursComposite, presence_detected
from src.soft_fusion import fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.image_pool import sample as _pool_sample, composition_of
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}
COLS = ("ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07", "unmarker")
dev = "cuda"
assert STAGE in ("resync", "scale", "angle", "tile"), STAGE
comp = OursComposite(dev, tm_variant="B", vine_variant="R",
                     config={"frags": [FKEY[F]], "order": [FKEY[F]], "strengths": {},
                             **W.frontend_config({STAGE: True})})
comp.strength = dict(comp.DEFAULT_STRENGTH); comp.strength[FKEY[F]] = float(s)
def r512(im): return im if im.size == (512, 512) else im.resize((512, 512))


def _deployed_read(comp, order, img, iid, tx):
    """Copy of perimage_campaign._deployed_read (that module runs its mode at import time)."""
    def _read(v):
        L = {f: comp._frag_llr(FKEY[f], v, iid) for f in order}
        ba = {f: float(np.mean((L[f] > 0).astype(np.uint8) == tx)) for f in order}
        ver = {f: bool(decode_and_verify(L[f], iid, codec=comp.sb)["detected"]) for f in order}
        fused = fuse_llrs({FKEY[f]: L[f] for f in order}, weights=None, n_codeword=comp.sb.n)
        fba = float(np.mean(llr_to_bits(fused) == tx))
        fver = bool(decode_and_verify(fused, iid, codec=comp.sb)["detected"])
        det = fver or any(ver.values()) or presence_detected(ba, fba, len(order))
        return ba, ver, det
    ba, ver, det = _read(img)
    if comp.geo and not det:
        ok, vw = comp.geo_cascade(img, iid, tx, return_view=True)
        if ok and vw is not None:
            ba2, ver2, _ = _read(vw)
            ba = ba2; ver = {f: ver[f] or ver2[f] for f in order}; det = True
    return ba, ver, float(det)


if MODE == "embed":
    n, cd = int(REST[0]), REST[1]; ed = f"{cd}/embed"; os.makedirs(ed, exist_ok=True)
    files = _pool_sample(n, offset=0)                     # the slice the in-process stage cells used
    for i, fp in enumerate(files):
        cov = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC)
        emb, _ = comp.embed(cov, i); r512(emb).save(f"{ed}/i{i:05d}.png")
    json.dump({"stage": STAGE, "frag": F, "strength": s, "n": n, "pool": composition_of(files)}, open(f"{cd}/meta.json", "w"), indent=1)
    print(f"embedded {n} images of {F}@{s} with stage {STAGE} on into {ed}  STAGE_DONE", flush=True)

elif MODE == "decode":
    cd = REST[0]; meta = json.load(open(f"{cd}/meta.json"))
    out_dir = os.environ.get("FEX_OUT") or f"{SC}/perimage_ext/stage_{STAGE}_xenv"
    ba, ver, det, fe = {}, {}, {}, {}
    for col in COLS:
        files = sorted(glob.glob(f"{cd}/att_{col}/i*.png"))
        if not files:
            if col == "unmarker":
                print(f"  {col}: no attacked images (not requested for this knot)", flush=True); continue
            raise RuntimeError(f"{cd}/att_{col}: no attacked images; the CtrlRegen+ step did not run")
        b, v, d = [], [], []; on, off, don, doff, lon, loff = [], [], [], [], [], []
        t0 = time.time()
        for fp in files:
            i = int(os.path.basename(fp)[1:6]); iid, tx = comp.secret_for(i)
            img = r512(Image.open(fp).convert("RGB"))
            b_, v_, d_ = _deployed_read(comp, [F], img, iid, tx)
            b.append(b_[F]); v.append(v_[F]); d.append(d_)
            _t = time.time(); ba_off, det_off = comp.decode_no_cascade(img, (iid, tx)); loff.append((time.time() - _t) * 1000.0)
            _t = time.time(); ba_on, det_on = comp.decode(img, (iid, tx)); lon.append((time.time() - _t) * 1000.0)
            on.append(float(ba_on)); off.append(float(ba_off)); don.append(float(det_on)); doff.append(float(det_off))
        ba[col], ver[col], det[col] = b, v, d
        fe[col] = {"on": on, "off": off, "don": don, "doff": doff, "lat_on_ms": lon, "lat_off_ms": loff, "n": len(files)}
        print(f"  {STAGE:6s} {F}@{s} {col:14s} n={len(files)} deployed mean {np.mean(b):.3f} verify {np.mean(v):.2f} det {np.mean(d):.2f} | "
              f"decode on {np.mean(on):.3f} off {np.mean(off):.3f} | latency on {np.mean(lon):.0f} off {np.mean(loff):.0f} ms [{time.time()-t0:.0f}s]", flush=True)
    if not ba:
        raise RuntimeError("nothing decoded")
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{F}_{s}.json"
    json.dump({"frag": F, "stage": STAGE, "strength": s, "n": min(len(x) for x in ba.values()), "images": "pool offset 0 (ext9 slice)",
               "embed": f"composite crypto codeword, stage {STAGE} on", "pool": meta.get("pool"),
               "ba": ba, "ver": ver, "det": det,
               "means": {a: round(float(np.mean(ba[a])), 4) for a in ba}, "det_rate": {a: round(float(np.mean(det[a])), 4) for a in ba},
               "fe": fe}, open(path, "w"))
    print(f"  wrote {os.path.relpath(path, SC)}", flush=True); print("FEX_DECODE_DONE", flush=True)
else:
    raise SystemExit(f"unknown mode {MODE}")
