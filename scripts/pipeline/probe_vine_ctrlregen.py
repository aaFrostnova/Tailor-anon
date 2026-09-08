"""A clean, direct measurement of VINE under CtrlRegen+, to check the table against a fresh run.

For each VINE strength and each front-end setting (plain, and the nested ring with the scale search),
embed N images with the DEPLOYED composite, attack them with CtrlRegen+ at three steps in that attack's
own environment, and read every attacked image with the deployed decoder: per-image bit accuracy on the
view it accepts, the keyed verification flag, and its verdict. Images come from pool offset 1100, a
slice no campaign has used.
  embed  <n> <workdir>     write embeds for every (front-end, strength) cell
  decode <workdir>         read back whatever the attack environment produced
"""
import sys, os, json, glob
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
MODE = sys.argv[1]; sys.argv2 = sys.argv[2:]; ARGS = sys.argv[2:]; sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
from eval_matrix import OursComposite, presence_detected
from src.soft_bch import decode_and_verify
from src.soft_fusion import fuse_llrs, llr_to_bits
STRENGTHS = [0.3, 0.5, 0.7, 1.0]
FES = {"plain": {"resync": False, "scale": False, "angle": False, "tile": False},
       "ring":  {"resync": False, "scale": True,  "angle": False, "tile": False}}
def comp_for(fe, s):
    c = OursComposite("cuda", tm_variant="B", vine_variant="R",
                      config={"frags": ["vine"], "order": ["vine"], "strengths": {}, **W.frontend_config(FES[fe])})
    c.strength = dict(c.DEFAULT_STRENGTH); c.strength["vine"] = float(s); return c
def r512(im): return im if im.size == (512, 512) else im.resize((512, 512))
if MODE == "embed":
    from src.image_pool import sample as _pool_sample, composition_of
    N = int(ARGS[0]); WD = ARGS[1]
    files = _pool_sample(N, offset=1100)
    covers = [Image.open(f).convert("RGB").resize((512, 512), Image.BICUBIC) for f in files]
    print("embedding %d images %s" % (N, composition_of(files)), flush=True)
    for fe in FES:
        for s in STRENGTHS:
            d = f"{WD}/{fe}_{s}/embed"; os.makedirs(d, exist_ok=True)
            comp = comp_for(fe, s)
            for i, cov in enumerate(covers):
                emb, _ = comp.embed(cov, i); r512(emb).save(f"{d}/i{i:05d}.png")
            print("  %s s=%.1f -> %s" % (fe, s, d), flush=True)
    print("EMBED_DONE", flush=True)
else:
    WD = ARGS[0]; out = {}
    for fe in FES:
        for s in STRENGTHS:
            comp = comp_for(fe, s)
            for step in ("s03", "s05", "s07"):
                files = sorted(glob.glob(f"{WD}/{fe}_{s}/att_ctrlregen_{step}/i*.png"))
                if not files: continue
                ba, ver, det = [], [], []
                for fp in files:
                    i = int(os.path.basename(fp)[1:6]); iid, tx = comp.secret_for(i)
                    v = r512(Image.open(fp).convert("RGB"))
                    def rd(img):
                        L = comp._frag_llr("vine", img, iid)
                        b = float(np.mean((L > 0).astype(np.uint8) == tx))
                        vr = bool(decode_and_verify(L, iid, codec=comp.sb)["detected"])
                        return b, vr, (vr or presence_detected({"vine": b}, b, 1))
                    b, vr, d_ = rd(v)
                    if comp.geo and not d_:
                        ok, vw = comp.geo_cascade(v, iid, tx, return_view=True)
                        if ok and vw is not None: b, vr, d_ = rd(vw); d_ = True
                    ba.append(b); ver.append(vr); det.append(bool(d_))
                out[f"{fe}|{s}|ctrlregen_{step}"] = {"ba": ba, "ver": ver, "det": det, "n": len(ba)}
                print("  %-6s s=%.1f %-14s n=%-3d mean %.3f  sd %.3f  det %.2f  verified %.2f" % (
                    fe, s, "ctrlregen_" + step, len(ba), np.mean(ba), np.std(ba), np.mean(det), np.mean(ver)), flush=True)
    json.dump(out, open(f"{SC}/probe_vine_ctrlregen.json", "w"))
    print("DECODE_DONE", flush=True)
