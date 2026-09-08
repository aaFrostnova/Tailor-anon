"""Decode the attacked ring-embedded VINE images with the deployed cascade (scale search on) and write
one knot of the stage's replacement curves: base_fe_scale_VINE|unmarker and |ctrlregen_s03/s05/s07.
Usage: python unmk_ring_decode.py <strength> <knot_dir>
"""
import sys, os, json, glob
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
from eval_matrix import OursComposite
import watermark_smt_v2 as W
s = float(sys.argv[1]); kd = sys.argv[2]
comp = OursComposite("cuda", tm_variant="B", vine_variant="R", config={"frags": ["vine"], "order": ["vine"], "strengths": {"vine": s},
                     **W.frontend_config({"resync": False, "scale": True, "angle": False, "tile": False})})
comp.strength = dict(comp.DEFAULT_STRENGTH); comp.strength["vine"] = s
def bestpath(img, sec):
    iid, tx = sec
    fba, det = comp.decode(img, sec)
    per = float(np.mean((comp._frag_llr("vine", img, iid) > 0).astype(np.uint8) == tx))
    if per < 0.9 and comp.geo:
        ok, vw = comp.geo_cascade(img, iid, tx, return_view=True)
        if ok and vw is not None: per = max(per, float(np.mean((comp._frag_llr("vine", vw, iid) > 0).astype(np.uint8) == tx)))
    return per, float(det)
out = {}
for col, sub in (("unmarker", "att_unmarker"), ("ctrlregen_s03", "att_ctrlregen_s03"), ("ctrlregen_s05", "att_ctrlregen_s05"), ("ctrlregen_s07", "att_ctrlregen_s07")):
    files = sorted(glob.glob(f"{kd}/{sub}/i*.png"))
    if not files: print(f"  {col}: no attacked images", flush=True); continue
    bas, dets = [], []
    for fp in files:
        idx = int(os.path.basename(fp)[1:6]); img = Image.open(fp).convert("RGB")
        if img.size != (512, 512): img = img.resize((512, 512))
        b, d = bestpath(img, comp.secret_for(idx)); bas.append(b); dets.append(d)
    out[col] = {"mean": float(np.mean(bas)), "sd": float(np.std(bas, ddof=1)) if len(bas) > 1 else 0.0, "det": float(np.mean(dets)), "n": len(bas)}
    print(f"  VINE+ring@{s} {col:14s} n={len(bas)} best-path {out[col]['mean']:.3f} +/- {out[col]['sd']:.3f} det {out[col]['det']:.2f}", flush=True)
json.dump({"strength": s, "cols": out}, open(f"{kd}/knot.json", "w"), indent=1)
print("KNOT_DECODE_DONE", flush=True)
