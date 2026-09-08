"""Re-measure the geometric front-end curves under the CANONICAL attack definitions.

The front-end cells are stored as an effect: the accuracy with the front-end on, beside a control taken
in the same run with it off. That subtraction is only meaningful if both sides, and the base curve the
effect is later added to, refer to the same attack. They had drifted: the in-process campaign measured
resync against its own locally-defined rotation (default fill, black corners) while the base rotation
curve now comes from the reported matrix's operator (reflect-pad, no corner loss). This overlay
re-measures both sides with the matrix's operators imported directly, so the effect and the base it
modifies finally describe one attack.

It also calls the DEPLOYED cascade rather than re-implementing part of it. The front-end is not
SyncSeal alone: the deployed decoder rectifies, retries every fragment, sweeps small residual angles,
searches the nested ring over scale, and then sweeps the full +/-180 degree range coarsely before
refining -- every step gated by the keyed verification. An overlay that measured only the rectify step
would credit the front-end with a fraction of what it does, and would do so precisely on the attack
where rectification is weakest: the reported rotation reflect-pads and crops back, so the image corners
SyncSeal predicts are no longer inside the frame, and the blind angle sweep is what actually recovers
the payload there.

Emits: base_resync_{F}|rot9, raw_synced_noresync_{F}|rot9 (the same-run control),
       base_nested_VINE|{crop75,crop50}, nested_VINE_no_scale_search|{crop75,crop50}, nested_penalty.
Front-end cells record the DETECTION rate of the deployed cascade; the paired control records the
detection rate of the same composite decoded without it, so the stored effect is the cascade's own
contribution rather than a mixture of the cascade and the fragment.

Usage: python make_frontend_overlay.py [N=50]
"""
import sys, os, json, glob, time, subprocess
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense"):
    sys.path.insert(0, p)
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.shortened_bch import ShortenedBCH
from src.image_pool import sample as _pool_sample, composition_of
from src.syncseal_frontend import load_sync, sync_embed, sync_rectify, DEFAULT_JIT
from composite_external_eval import nested_vine_embed
from eval_matrix import GEO, OursComposite                   # the SAME operators AND the SAME decoder

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
sb = ShortenedBCH(); NB = sb.n; KEY = b"v5_key_encoder_master"; dev = "cuda"
GRID = {f: [round(float(v), 4) for v in np.linspace(lo, hi, 9)]
        for f, (lo, hi) in {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6),
                            "VideoSeal": (0.5, 1.5)}.items()}
FRAGS = ["VINE", "TrustMark", "VideoSeal"]
VINE_SCALE_GRID = np.arange(0.34, 1.0001, 0.005)             # decode-side search the deployment uses
print(f"N={N}  rot9/crop from eval_matrix.GEO", flush=True)

FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}
_CACHE = {}
def _composite(cfg):
    """The deployed composite, configured as the solver would configure it. Cached per (subset,
    front-ends) so the fragments and the SyncSeal model are loaded once, not once per strength."""
    key = (tuple(cfg["order"]), cfg["resync"], cfg["nested"])
    if key not in _CACHE:
        _CACHE[key] = OursComposite(dev, tm_variant="B", vine_variant="R", config=cfg)
    c = _CACHE[key]
    c.strength = dict(c.DEFAULT_STRENGTH); c.strength.update(cfg["strengths"])
    return c

FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
      "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
      "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}
sync = load_sync(DEFAULT_JIT, dev)

def tb(i): return np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
def r512(im): return im if im.size == (512, 512) else im.resize((512, 512))
def hard(fn, pil):
    f = FR[fn]; v = f.raw_probs(pil) if fn == "VINE" else f.raw_logits(pil)
    v = np.asarray(v).ravel()[:NB]
    return (v > (0.5 if fn == "VINE" else 0.0)).astype(np.uint8)
def mse(a, b):
    x = np.asarray(a, np.float64); y = np.asarray(b, np.float64); return float(np.mean((x - y) ** 2))

files = _pool_sample(N, offset=0)   # across all five sources, in the evaluation set's proportions
assert len(files) == N, f"pool sample short: {len(files)}"
print('sources:', composition_of(files), flush=True)
on   = {(f, s): [] for f in FRAGS for s in GRID[f]}          # resync ON
off  = {(f, s): [] for f in FRAGS for s in GRID[f]}          # same-run control, resync OFF
nes  = {(a, s): [] for a in ("crop75", "crop50") for s in GRID["VINE"]}   # nested + scale search
nen  = {(a, s): [] for a in ("crop75", "crop50") for s in GRID["VINE"]}   # nested, plain decode
pen  = {s: [] for s in GRID["VINE"]}                          # nested MSE cost over plain VINE
t0 = time.time()
for i, fp in enumerate(files):
    cover = Image.open(fp).convert("RGB").resize((512, 512), Image.BICUBIC)
    t = tb(i)
    for f in FRAGS:
        for s in GRID[f]:
            # One fragment, one strength, carrying the deployed crypto payload so that the deployed
            # accept gate is meaningful. The control and the treatment differ ONLY in whether the
            # cascade is allowed to run.
            cfg = {"frags": [FKEY[f]], "order": [FKEY[f]], "strengths": {FKEY[f]: float(s)},
                   "resync": True, "nested": False}
            comp = _composite(cfg)
            emb, sec = comp.embed(cover, i)
            att = r512(GEO["rot9"](emb))
            _, det_off = comp.decode_no_cascade(att, sec)     # same decoder, cascade suppressed
            _, det_on = comp.decode(att, sec)                 # deployed path, cascade allowed
            off[(f, s)].append(float(det_off))
            on[(f, s)].append(float(det_on))
    m = FR["VINE"]; perm, M = m.get_perm_M("img_%05d" % i)
    from src.vine_crypto_wrapper import apply_crypto
    w = apply_crypto(t, perm, M)
    for s in GRID["VINE"]:
        nested = r512(nested_vine_embed(m, cover, w, scales=(1.0, 0.75, 0.5), strength=s))
        plain  = r512(FR["VINE"].embed_with_target(cover, t, strength=s))
        pen[s].append(mse(cover, nested) - mse(cover, plain))
        for a in ("crop75", "crop50"):
            att = r512(GEO[a](nested))
            nen[(a, s)].append(float(np.mean(hard("VINE", att) == t)))
            best = 0.0
            for k in VINE_SCALE_GRID:                          # the deployed blind scale search
                if k >= 0.999: view = att
                else:
                    q = int(round(512 * float(k))); o = (512 - q) // 2
                    view = att.crop((o, o, o + q, o + q))
                best = max(best, float(np.mean(hard("VINE", view) == t)))
            nes[(a, s)].append(best)
    if (i + 1) % 5 == 0: print(f"  ...{i+1}/{N} [{time.time()-t0:.0f}s]", flush=True)

mean = lambda L: float(np.mean(L)) if L else float("nan")
fe = {}
for f in FRAGS:
    fe[f"base_resync_{f}|rot9"] = {"xs": GRID[f], "ys": [round(mean(on[(f, s)]), 4) for s in GRID[f]]}
    fe[f"raw_synced_noresync_{f}|rot9"] = {"xs": GRID[f], "ys": [round(mean(off[(f, s)]), 4) for s in GRID[f]]}
    d = [mean(on[(f, s)]) - mean(off[(f, s)]) for s in GRID[f]]
    print(f"resync effect {f:10s}: " + ", ".join(f"{s}:{v:+.3f}" for s, v in zip(GRID[f], d)), flush=True)
for a in ("crop75", "crop50"):
    fe[f"base_nested_VINE|{a}"] = {"xs": GRID["VINE"], "ys": [round(mean(nes[(a, s)]), 4) for s in GRID["VINE"]]}
    fe[f"nested_VINE_no_scale_search|{a}"] = {"xs": GRID["VINE"], "ys": [round(mean(nen[(a, s)]), 4) for s in GRID["VINE"]]}
    print(f"nested {a}: search " + ",".join(f"{mean(nes[(a,s)]):.3f}" for s in GRID["VINE"])
          + " | plain " + ",".join(f"{mean(nen[(a,s)]):.3f}" for s in GRID["VINE"]), flush=True)
fe["nested_penalty"] = {"xs": GRID["VINE"], "ys": [round(mean(pen[s]), 4) for s in GRID["VINE"]]}
print("nested MSE penalty: " + ",".join(f"{mean(pen[s]):.2f}" for s in GRID["VINE"]), flush=True)
try:
    git_rev = subprocess.run(["git","-C",CF,"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
except Exception: git_rev = "unknown"
json.dump({"frontend": fe, "base": {}, "delta": {},
           "source": {"n": N, "pool": composition_of(files), "slice": [0, N], "git": git_rev,
                      "measured_at": os.environ.get("TS", "unknown"),
                      "attack": "rot9/crop imported from eval_matrix.GEO so the front-end effect and "
                                "the base curve it modifies describe the same attack"}},
          open(f"{SC}/surrogate_frontend_overlay.json", "w"), indent=2)
print(f"wrote {SC}/surrogate_frontend_overlay.json"); print("FRONTEND_OVERLAY_DONE")
