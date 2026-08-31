"""PHASE 2 — nested-VINE integrated into the 3-fragment composite + fusion-head gate G2.

VINE embed = nested{1,.75,.5}; VINE decode = dense centered scale search (new geo_cascade crop stage,
crypto-verify gated). TrustMark/VideoSeal = full-frame + SyncSeal rectify + rotation search.

Reports per attack:
  composite_or  : best-path (any fragment crypto-verifies, incl. VINE scale-search + TM rotation search)
  head3_fused   : primary fused-head detection on the full frame (verify or zero-bit)
  eqw_fused     : equal-weight fused detection on the full frame
GATE G2: on non-geometric attacks (clean/regen) is head3 >= equal-weight? If head3 degrades on the new
  VINE LLR distribution, it needs recalibration."""
import os, sys, glob, json, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint")
sys.path.insert(0, "scripts/defense")
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from src.fusion_head3 import load_head3
from src.syncseal_frontend import load_sync, sync_embed, sync_rectify
from _regen_util import build_regen_pipe, stable_regen
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; ALPHA = 0.70
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
head = load_head3(os.path.join("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint", "results/defense/frag3_head.pt"), dev)
sync = load_sync(dev=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def txc(iid, nm):
    p, M = F[nm].get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
def vine_nested(x, iid, Ks=(1.0, 0.75, 0.5)):
    x = to512(x)
    for K in Ks:
        if K >= 0.999: x = to512(F["vine"].embed_with_target(x, txc(iid, "vine")))
        else:
            s = int(512 * K); o = (512 - s) // 2
            m = F["vine"].embed_with_target(x.crop((o, o, o + s, o + s)), txc(iid, "vine"))
            out = x.copy(); out.paste(m.resize((s, s)), (o, o)); x = out
    return x
def embed_composite(orig, iid):
    x = vine_nested(orig, iid)                                             # 1) nested VINE
    x = scale(x, to512(F["trustmark"].embed_with_target(x, txc(iid, "trustmark"))))
    x = scale(x, to512(F["videoseal"].embed_with_target(x, txc(iid, "videoseal"))))
    return to512(sync_embed(sync, to512(x), dev))
def llr(nm, pil, iid):
    kind, getter = SPEC[nm]; p, M = F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm], getter)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP).astype(np.float32)
def cv(rl, iid): return bool(decode_and_verify(rl, iid, codec=sb)["detected"])
def rot(img, deg):
    a = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(a, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    l = (big.size[0] - 512) // 2; return big.crop((l, l, l + 512, l + 512))
VSCALES = np.round(np.arange(0.34, 1.0001, 0.03), 3).tolist()
def vine_scale_search(pil, iid, tx):
    """dense centered scale search for nested VINE; returns (verified_bool, best_llr)."""
    P = to512(pil); best_rl = None; best_ba = -1.0
    for v in VSCALES:
        view = P if v >= 0.999 else P.crop((int(512*(1-v)/2), int(512*(1-v)/2), int(512*(1-v)/2)+int(512*v), int(512*(1-v)/2)+int(512*v)))
        rl = llr("vine", view, iid); ba = float(((rl > 0).astype(np.uint8) == tx).mean())
        if cv(rl, iid) or ba >= tau: return True, rl
        if ba > best_ba: best_ba, best_rl = ba, rl
    return False, best_rl

srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:24]
pipe = build_regen_pipe()
def regen(x, j): return to512(stable_regen(pipe, x, 7000 + j, noise_step=30))
def a_crop(x, c): s = int(512*c); o = (512-s)//2; return to512(x).crop((o, o, o+s, o+s)).resize((512, 512))
ATT = {"clean": lambda x, j: x, "crop75": lambda x, j: a_crop(x, .75), "crop50": lambda x, j: a_crop(x, .50),
       "rot30": lambda x, j: rot(x, 30), "regen0.3": lambda x, j: regen(x, j), "regen0.5": lambda x, j: to512(stable_regen(pipe, x, 7000+j, noise_step=50)),
       "crop75+regen": lambda x, j: regen(a_crop(x, .75), j), "crop50+regen": lambda x, j: regen(a_crop(x, .50), j)}

def detect(att, iid, tx):
    al = {nm: llr(nm, att, iid) for nm in F}                                # full-frame per-fragment LLRs
    with torch.no_grad():
        fused_h = head(torch.tensor(al["vine"][None], device=dev), torch.tensor(al["trustmark"][None], device=dev),
                       torch.tensor(al["videoseal"][None], device=dev)).cpu().numpy()[0]
    fused_e = fuse_llrs(al, weights=None, n_codeword=n)
    h_ok = cv(fused_h, iid) or (((fused_h > 0).astype(np.uint8) == tx).mean() >= tau)
    e_ok = cv(fused_e, iid) or (((fused_e > 0).astype(np.uint8) == tx).mean() >= tau)
    best = any(cv(al[nm], iid) for nm in F)
    if not best:                                                            # geo_cascade fallbacks
        v_ok, _ = vine_scale_search(att, iid, tx)                           # VINE crop stage
        if v_ok: best = True
    if not best:
        rect, _ = sync_rectify(sync, att, dev)                              # SyncSeal + rotation for TM
        if any(cv(llr(nm, rect, iid), iid) for nm in F): best = True
        elif any(cv(llr("trustmark", rot(att, d), iid), iid) for d in (-30,-20,-10,10,20,30)): best = True
    comp = 1.0 if (h_ok or e_ok or best) else 0.0
    return comp, (1.0 if h_ok else 0.0), (1.0 if e_ok else 0.0)

res = {k: {"composite_or": [], "head3_fused": [], "eqw_fused": []} for k in ATT}
for j, fp in enumerate(srcs):
    iid = f"p2_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
    tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
    wm = embed_composite(orig, iid)
    for k, f in ATT.items():
        c, h, e = detect(f(wm, j), iid, tx)
        res[k]["composite_or"].append(c); res[k]["head3_fused"].append(h); res[k]["eqw_fused"].append(e)
out = {k: {m: round(float(np.mean(v)), 3) for m, v in d.items()} for k, d in res.items()}
print(f"{'attack':14}{'comp_or':>9}{'head3':>8}{'eqw':>8}")
for k in ATT: print(f"{k:14}{out[k]['composite_or']:>9.2f}{out[k]['head3_fused']:>8.2f}{out[k]['eqw_fused']:>8.2f}", flush=True)
json.dump(out, open("/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense/phase2_composite.json", "w"), indent=2)
print("PHASE2_COMPOSITE_DONE")
