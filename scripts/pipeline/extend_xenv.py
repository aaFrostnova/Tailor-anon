"""Range extension, adversarial columns: embed one fragment at an EXTENSION knot with the deployed
composite, hand the images to CtrlRegen+ (three steps) and UnMarker in their own environments (the
sbatch does that between the two calls here), then read every attacked image back per image.
Usage: python extend_xenv.py embed  <frag> <strength> <n> <cell_dir>
       python extend_xenv.py decode <frag> <strength> <cell_dir>
"""
import sys, os, json, glob
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
MODE, F, s = sys.argv[1], sys.argv[2], float(sys.argv[3]); REST = sys.argv[4:]; sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
from eval_matrix import OursComposite
from src.soft_bch import decode_and_verify
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}
comp = OursComposite("cuda", tm_variant="B", vine_variant="R",
                     config={"frags": [FKEY[F]], "order": [FKEY[F]], "strengths": {},
                             **W.frontend_config({"resync": False, "scale": False, "angle": False, "tile": False})})
comp.strength = dict(comp.DEFAULT_STRENGTH); comp.strength[FKEY[F]] = float(s)
def r512(img): return img if img.size == (512, 512) else img.resize((512, 512))
if MODE == "embed":
    from src.image_pool import sample as _pool_sample
    n, cd = int(REST[0]), REST[1]; os.makedirs(f"{cd}/embed", exist_ok=True)
    files = _pool_sample(n, offset=0)
    for i, fp in enumerate(files):
        cov = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC)
        emb, _ = comp.embed(cov, i); r512(emb).save(f"{cd}/embed/i{i:05d}.png")
    print(f"embedded {n} images of {F}@{s} into {cd}/embed  STAGE_DONE", flush=True)
else:
    cd = REST[0]; OUT = f"{SC}/perimage_ext"
    for col, sub in (("ctrlregen_s03", "att_ctrlregen_s03"), ("ctrlregen_s05", "att_ctrlregen_s05"),
                     ("ctrlregen_s07", "att_ctrlregen_s07"), ("unmarker", "att_unmarker")):
        files = sorted(glob.glob(f"{cd}/{sub}/i*.png"))
        if not files: print(f"  {col}: no attacked images", flush=True); continue
        idx, ba, ver = [], [], []
        for fp in files:
            i = int(os.path.basename(fp)[1:6]); iid, tx = comp.secret_for(i)
            L = comp._frag_llr(FKEY[F], r512(Image.open(fp).convert("RGB")), iid)
            idx.append(i); ba.append(float(np.mean((L > 0).astype(np.uint8) == tx)))
            ver.append(bool(decode_and_verify(L, iid, codec=comp.sb)["detected"]))
        os.makedirs(f"{OUT}/{col}", exist_ok=True)
        json.dump({"frag": F, "strength": s, "attack": col, "n": len(ba), "embed": "composite crypto codeword (range extension)",
                   "idx": idx, "ba": ba, "ver": ver, "mean": round(float(np.mean(ba)), 4)}, open(f"{OUT}/{col}/{F}_{s}.json", "w"))
        print(f"  {F}@{s} {col:14s} n={len(ba)} mean {np.mean(ba):.3f} verify {np.mean(ver):.2f}", flush=True)
    print("XENV_DECODE_DONE", flush=True)
