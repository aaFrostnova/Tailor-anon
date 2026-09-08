"""Live check of the budget-split ablation: for chosen requests, embed the recorded (split-aware) configuration
and the ablated one (every zero-bit test at the single-fragment threshold) on the certification images (pool
offset 300, N=100), apply the request's in-process attacks, and read every fragment on the primary view plus,
when no fragment verifies there, the crypto-verify-gated cascade view. Acceptance at a threshold tau is then
budget-consistent: verified on the primary view, or primary best-path >= tau, or verified on the cascade view;
the mean is taken over the view the decoder accepts at that tau. Reported at tau_1 (what the ablated solver
assumed) and at tau_k (what the deployed decoder uses for k fragments), with the per-image rate floor 0.9.
Usage: python measure_ablation_live.py <request ids from ablate_presence_split.json, e.g. 146 1118 378> [N=100]
"""
import sys, os, json, time
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker", SC): sys.path.insert(0, p)
IDS = [int(x) for x in sys.argv[1:] if x.isdigit()]; N = 100
sys.argv = [sys.argv[0]]
import watermark_smt_v2 as W
from eval_matrix import OursComposite
from src.attacks import attack_pil_any
from src.image_pool import sample as _pool_sample, composition_of
from src.soft_bch import decode_and_verify
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}
LIVE_OK = {"jpeg25", "blur", "noise", "bright", "contrast", "crop75", "crop50", "rot9", "rs256", "hflip", "crop_jpeg", "border20", "vaeB", "vaeC", "regen", "rinse2x"}
dev = "cuda"
def composite_for(order, fe, strengths):
    c = OursComposite(dev, tm_variant="B", vine_variant="R",
                      config={"frags": [FKEY[f] for f in order], "order": [FKEY[f] for f in order], "strengths": {}, **W.frontend_config({k: bool(fe.get(k)) for k in ("resync", "scale", "angle", "tile")})})
    c.strength = dict(c.DEFAULT_STRENGTH); c.strength.update({FKEY[f]: float(strengths[f]) for f in order}); return c
def r512(img): return img if img.size == (512, 512) else img.resize((512, 512))
def read_views(comp, order, img, iid, tx):
    """primary view (best-path ba, any verified) and, if nothing verifies there, the cascade view (or None)."""
    def _read(v):
        L = {f: comp._frag_llr(FKEY[f], v, iid) for f in order}
        ba = max(float(np.mean((L[f] > 0).astype(np.uint8) == tx)) for f in order)
        ver = any(bool(decode_and_verify(L[f], iid, codec=comp.sb)["detected"]) for f in order)
        return ba, ver
    bp, vp = _read(img); bc, vc = None, False
    if comp.geo and not vp:
        ok, vw = comp.geo_cascade(img, iid, tx, return_view=True)
        if ok and vw is not None: bc, vc = _read(vw)
    return bp, vp, bc, vc
def stats(rows, tau):
    acc = np.array([vp or bp >= tau - 1e-12 or vc for bp, vp, bc, vc in rows])
    mean_view = np.array([bp if (vp or bp >= tau - 1e-12) else (bc if vc else bp) for bp, vp, bc, vc in rows])
    return float(mean_view.mean()), float(acc.mean())
abl = json.load(open(f"{SC}/ablate_presence_split.json"))["rows"]; rows_by = {r["i"]: r for r in abl}
files = _pool_sample(N, offset=300); covers = [Image.open(f).convert("RGB").resize((512, 512), Image.BICUBIC) for f in files]
print(f"N={N} certification images {composition_of(files)}", flush=True)
out = []
for i in IDS:
    r = rows_by[i]; atts = [a for a in r["attacks"] if a in LIVE_OK]; k_rec, k_abl = len(r["recorded"]["frags"]), len(r["ablated"]["frags"])
    tau1 = W.presence_threshold(W.beta_from_fpr(r["fpr"]), 1)
    print(f"\n=== {r['cls']} q{i:04d} fpr={r['fpr']:.0e} attacks={atts}  tau1={tau1:.2f} tau_k(rec)={W.presence_threshold(W.beta_from_fpr(r['fpr']), k_rec):.2f} tau_k(abl)={W.presence_threshold(W.beta_from_fpr(r['fpr']), k_abl):.2f}", flush=True)
    res = {"i": i, "cls": r["cls"], "fpr": r["fpr"], "attacks": atts, "configs": {}}
    for lab in ("recorded", "ablated"):
        cfg = r[lab]; order = cfg["order"]; k = len(order); tauk = W.presence_threshold(W.beta_from_fpr(r["fpr"]), k)
        comp = composite_for(order, cfg["fe"], cfg["s"]); t0 = time.time()
        embeds = [comp.embed(cov, j) for j, cov in enumerate(covers)]
        psnr = float(np.mean([10 * np.log10(255.0 ** 2 / max(np.mean((np.asarray(c, np.float64) - np.asarray(r512(e), np.float64)) ** 2), 1e-9)) for c, (e, _) in zip(covers, embeds)]))
        desc = f"{'+'.join(order)}{'/' + '+'.join(x for x, v in cfg['fe'].items() if v) if any(cfg['fe'].values()) else ''} s={ {f: round(cfg['s'][f], 3) for f in order} }"
        print(f"  [{lab}] {desc}  table PSNR {cfg['psnr_db']:.2f} dB, measured {psnr:.2f} dB  (k={k}, tau_k={tauk:.2f})", flush=True)
        cells = {}
        for a in atts:
            rows = []
            for emb, (iid, tx) in embeds:
                v = r512(attack_pil_any(a, r512(emb), dev=dev)); rows.append(read_views(comp, order, v, iid, tx))
            m1, a1 = stats(rows, tau1); mk, ak = stats(rows, tauk)
            ok1 = m1 >= tau1 and a1 >= 0.9; okk = mk >= tauk and ak >= 0.9
            cells[a] = {"mean_tau1": m1, "rate_tau1": a1, "mean_tauk": mk, "rate_tauk": ak, "ok_tau1": ok1, "ok_tauk": okk, "n": len(rows)}
            print(f"     {a:12s} at tau1 {tau1:.2f}: mean {m1:.3f} rate {a1:.2f} {'ok' if ok1 else 'FAIL'}   | at tau_k {tauk:.2f}: mean {mk:.3f} rate {ak:.2f} {'ok' if okk else 'FAIL'}   [{time.time()-t0:.0f}s]", flush=True)
        res["configs"][lab] = {"desc": desc, "k": k, "tau1": tau1, "tauk": tauk, "psnr_table": cfg["psnr_db"], "psnr_measured": psnr, "cells": cells,
                               "feasible_tau1": all(c["ok_tau1"] for c in cells.values()), "feasible_tauk": all(c["ok_tauk"] for c in cells.values())}
        print(f"     -> {lab}: feasible at tau1 {res['configs'][lab]['feasible_tau1']}, at tau_k {res['configs'][lab]['feasible_tauk']}", flush=True)
    out.append(res)
    json.dump(out, open(f"{SC}/measure_ablation_live.json", "w"), indent=1)
print("ABLATION_LIVE_DONE", flush=True)
