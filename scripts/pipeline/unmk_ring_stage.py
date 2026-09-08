"""Embed VINE WITH the nested ring at one strength knot into the UnMarker campaign's 30 images.

The scale stage the solver may switch on is the ring embed plus the decode-side scale search. The
unmarker column was measured on the plain VINE embed and a decode-only probe showed the search cannot
rescue that embed (bare 0.49 = searched 0.49 at every knot); the recovery on record (0.53 -> 0.98) was
measured on the ring. The replacement curve for the stage under UnMarker (and, so the stage stays
admissible on the adversarial requests, under CtrlRegen+ at its three strengths) therefore has to be
measured on ring-embedded images. Secrets come from the composite (secret_for), as in certification.
Usage: python unmk_ring_stage.py <strength> <n> <out_dir>
"""
import sys, os
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
from eval_matrix import OursComposite
import watermark_smt_v2 as W
from src.image_pool import sample as _pool_sample, composition_of
s = float(sys.argv[1]); n = int(sys.argv[2]); out = sys.argv[3]; os.makedirs(out, exist_ok=True)
comp = OursComposite("cuda", tm_variant="B", vine_variant="R", config={"frags": ["vine"], "order": ["vine"], "strengths": {"vine": s},
                     **W.frontend_config({"resync": False, "scale": True, "angle": False, "tile": False})})
comp.strength = dict(comp.DEFAULT_STRENGTH); comp.strength["vine"] = s
files = _pool_sample(n, offset=0)
for i, fp in enumerate(files):
    cov = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC)
    emb, sec = comp.embed(cov, i)
    if emb.size != (512, 512): emb = emb.resize((512, 512))
    emb.save(os.path.join(out, f"i{i:05d}.png"))
print(f"ring-embedded {len(files)} @ VINE s={s} -> {out} sources={composition_of(files)}  STAGE_DONE", flush=True)
