"""Does the best-path test ever decide a case the fused test does not, and the other way round?

The deployed decoder accepts an image if EITHER
  bestpath : some single fragment's own LLR vector Chase-decodes and the payload verifies, or
  fver     : the equal-weight (maximal-ratio) sum of the fragments' LLRs does,
(then the zero-bit presence test, then the geometric cascade on a miss).

Fusion adds evidence when both fragments carry some, and dilutes it when one is dead: a dead fragment
contributes noise LLRs to the sum. This records, per image, both tests and the per-fragment payload
verification, so the two directions can be counted rather than argued.
Usage: python bestpath_vs_fused.py [n=60]
"""
import sys, os, json
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
from eval_matrix import OursComposite
from src.attacks import attack_pil_any
from src.image_pool import sample as _pool_sample, composition_of
from src.soft_fusion import fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
dev = "cuda"
files = _pool_sample(N, offset=900)                      # a slice no campaign has used
covers = [Image.open(f).convert("RGB").resize((512, 512), Image.BICUBIC) for f in files]
print(f"{N} images {composition_of(files)}", flush=True)
CFGS = [("VINE+VideoSeal", ["vine", "videoseal"], {"vine": 0.7, "videoseal": 1.0}),
        ("VINE+TrustMark", ["vine", "trustmark"], {"vine": 0.7, "trustmark": 1.0})]
ATT = ["crop_jpeg", "crop50", "border20", "regen", "vaeC", "jpeg25", "rot9"]
out = []
for cname, frags, st in CFGS:
    comp = OursComposite(dev, tm_variant="B", vine_variant="R",
                         config={"frags": frags, "order": frags, "strengths": {},
                                 **W.frontend_config({"resync": False, "scale": False, "angle": False, "tile": False})})
    comp.strength = dict(comp.DEFAULT_STRENGTH); comp.strength.update(st)
    for a in ATT:
        rec = []
        for i, cov in enumerate(covers):
            emb, sec = comp.embed(cov, i); iid, tx = sec
            v = attack_pil_any(a, emb, dev=dev)
            if v.size != (512, 512): v = v.resize((512, 512))
            L = {n: comp._frag_llr(n, v, iid) for n in comp.order}
            per_ver = {n: bool(decode_and_verify(L[n], iid, codec=comp.sb)["detected"]) for n in comp.order}
            per_ba = {n: float(np.mean(llr_to_bits(L[n]) == tx)) for n in comp.order}
            fused = fuse_llrs(L, weights=None, n_codeword=comp.sb.n)
            fver = bool(decode_and_verify(fused, iid, codec=comp.sb)["detected"])
            fba = float(np.mean(llr_to_bits(fused) == tx))
            rec.append({"i": i, "per_ver": per_ver, "per_ba": per_ba, "fver": fver, "fba": fba,
                        "bestpath": any(per_ver.values())})
        bp_only = [r for r in rec if r["bestpath"] and not r["fver"]]
        fv_only = [r for r in rec if r["fver"] and not r["bestpath"]]
        both = [r for r in rec if r["fver"] and r["bestpath"]]
        none = [r for r in rec if not r["fver"] and not r["bestpath"]]
        print(f"{cname:16s} {a:10s}  bestpath-only {len(bp_only):3d}   fused-only {len(fv_only):3d}   both {len(both):3d}   neither {len(none):3d}", flush=True)
        for r in (bp_only[:2] + fv_only[:2]):
            tag = "bestpath only" if r["bestpath"] and not r["fver"] else "fused only"
            print(f"     img {r['i']:3d} {tag:14s} per-frag ba " +
                  " ".join(f"{n}={r['per_ba'][n]:.2f}{'V' if r['per_ver'][n] else '.'}" for n in comp.order) +
                  f"   fused ba {r['fba']:.2f}{'V' if r['fver'] else '.'}", flush=True)
        out.append({"cfg": cname, "attack": a, "bestpath_only": len(bp_only), "fused_only": len(fv_only),
                    "both": len(both), "neither": len(none), "records": rec})
json.dump({"n": N, "rows": out}, open(f"{SC}/bestpath_vs_fused.json", "w"), indent=1)
print("\nwrote bestpath_vs_fused.json  BPVF_DONE", flush=True)
