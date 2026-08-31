"""Can we drop the OUTERMOST VINE ring (K=1.0)? Two decisive tests, local GPU.

PART A — per-ring survival (uses EXISTING attacked sets, no re-attack): for clean + CtrlRegen+ s0.5/s0.9,
decode VINE at EACH ring's canonical view (crop to central K, resize) and record bit-acc + crypto-verify per
ring. Answers: does K=1.0 uniquely carry regeneration, or do the inner rings survive it just as well?
  (search brings ring-K to the border by cropping to central fraction K; under a scale-preserving attack
   like regen the ring stays at scale K, so view f=K.)

PART B — full-composite 2-ring {0.75,0.5} vs 3-ring {1,.75,.5}: PSNR + detection on clean/rot90/crop75/crop50
through the deployed cascade. Answers: does dropping K=1.0 change robustness or quality on non-regen axes?"""
import os, sys, glob, json, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0, "scripts/defense")
import torch
from scipy.stats import binom
from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from src.syncseal_frontend import load_sync, sync_embed, sync_rectify
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; ALPHA = 0.70
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SC = "/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks"
OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense"
print(f"[gpu] free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
FR = ["vine", "trustmark", "videoseal"]; sync = load_sync(dev=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scl(c, w):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + ALPHA * (W - C), 0, 255).astype(np.uint8))
def txc(iid, nm):
    p, M = F[nm].get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
def vine_nested(x, iid, scales):
    x = to512(x)
    for K in scales:
        if K >= 0.999:
            x = to512(F["vine"].embed_with_target(x, txc(iid, "vine")))
        else:
            s = int(512 * K); o = (512 - s) // 2
            m = F["vine"].embed_with_target(x.crop((o, o, o + s, o + s)), txc(iid, "vine"))
            out = x.copy(); out.paste(m.resize((s, s)), (o, o)); x = out
    return x
def embed(orig, iid, scales):
    x = to512(orig)
    for nm in FR:
        x = vine_nested(x, iid, scales) if nm == "vine" else scl(x, to512(F[nm].embed_with_target(x, txc(iid, nm))))
    return to512(sync_embed(sync, to512(x), dev))
def llr(nm, pil, iid):
    kind, g = SPEC[nm]; p, M = F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm], g)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP).astype(np.float32)
def cv(rl, iid): return bool(decode_and_verify(rl, iid, codec=sb)["detected"])
def view_at(pil, f):
    P = to512(pil)
    if f >= 0.999: return P
    s = int(round(512 * float(f))); o = (512 - s) // 2; return P.crop((o, o, o + s, o + s))
def rot(img, deg):
    ar = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(ar, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    l = (big.size[0] - 512) // 2; return big.crop((l, l, l + 512, l + 512))
VS = np.arange(0.34, 1.0001, 0.005)
def vine_scale_search(pil, iid, tx):
    for f in VS:
        rl = llr("vine", view_at(pil, float(f)), iid)
        if cv(rl, iid) or ((rl > 0).astype(np.uint8) == tx).mean() >= tau: return True
    return False
def detect(att, iid, tx):
    al = {nm: llr(nm, att, iid) for nm in FR}
    fused = fuse_llrs(al, weights=None, n_codeword=n)
    if cv(fused, iid) or ((fused > 0).astype(np.uint8) == tx).mean() >= tau: return 1.0
    if any(cv(al[nm], iid) for nm in FR): return 1.0
    if vine_scale_search(att, iid, tx): return 1.0
    rect, _ = sync_rectify(sync, att, dev)
    if any(cv(llr(nm, rect, iid), iid) for nm in FR): return 1.0
    best_d, best_ba = 0.0, -1.0
    for d in np.arange(-180, 180.01, 10.0):
        rl = llr("trustmark", rot(att, float(d)), iid)
        if cv(rl, iid): return 1.0
        ba = float(((rl > 0).astype(np.uint8) == tx).mean())
        if ba > best_ba: best_ba, best_d = ba, d
    if best_ba < 0.60: return 0.0
    for d in np.arange(best_d - 10, best_d + 10.01, 3.0):
        r = rot(att, float(d))
        for nm in ("trustmark", "videoseal"):
            if cv(llr(nm, r, iid), iid): return 1.0
    return 0.0

report = {}
# ---------- PART A: per-ring survival ----------
print("=== PART A: per-ring VINE survival (which ring carries each condition?) ===", flush=True)
metaD = json.load(open(f"{SC}/p3embed_D/meta.json"))["items"]
SETS = {"clean": (f"{SC}/p3embed_D", metaD), "ctrlregen_s05": (f"{SC}/p3cr_D_s05", metaD),
        "ctrlregen_s09": (f"{SC}/p3cr_D_s09", metaD)}
report["per_ring"] = {}
for sname, (d, meta) in SETS.items():
    per = {K: {"ba": [], "verify": []} for K in (1.0, 0.75, 0.5)}
    for m in meta:
        p = f"{d}/{m['fname']}"
        if not os.path.exists(p): continue
        iid = m["iid"]; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        im = to512(Image.open(p).convert("RGB"))
        for K in (1.0, 0.75, 0.5):
            rl = llr("vine", view_at(im, K), iid)
            per[K]["ba"].append(float(((rl > 0).astype(np.uint8) == tx).mean()))
            per[K]["verify"].append(1.0 if cv(rl, iid) else 0.0)
    report["per_ring"][sname] = {f"K{K}": {"ba": round(float(np.mean(v["ba"])), 3), "verify": round(float(np.mean(v["verify"])), 3),
                                            "n": len(v["ba"])} for K, v in per.items()}
    r = report["per_ring"][sname]
    print(f"  {sname:14}: K1.0 ba={r['K1.0']['ba']:.3f} vf={r['K1.0']['verify']:.2f} | "
          f"K0.75 ba={r['K0.75']['ba']:.3f} vf={r['K0.75']['verify']:.2f} | "
          f"K0.5 ba={r['K0.5']['ba']:.3f} vf={r['K0.5']['verify']:.2f}", flush=True)

# ---------- PART B: 2-ring vs 3-ring full composite ----------
print("=== PART B: 2-ring {0.75,0.5} vs 3-ring {1,.75,.5} — PSNR + geometry ===", flush=True)
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:15]
VARIANTS = {"3ring_{1,.75,.5}": (1.0, 0.75, 0.5), "2ring_{.75,.5}": (0.75, 0.5)}
report["variant"] = {}
for vname, scales in VARIANTS.items():
    wms = []; ps = []; ss = []
    for j, fp in enumerate(srcs):
        iid = f"dor_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        wm = embed(orig, iid, scales); wms.append((iid, wm, tx, orig))
        ps.append(psnr_fn(np.asarray(orig, np.float64), np.asarray(wm, np.float64), data_range=255))
        ss.append(ssim_fn(np.asarray(orig, np.float64), np.asarray(wm, np.float64), channel_axis=2, data_range=255))
    def crop(x, c):
        s = int(512 * c); o = (512 - s) // 2; return to512(x).crop((o, o, o + s, o + s)).resize((512, 512))
    conds = {"clean": lambda w: w, "rot90": lambda w: rot(w, 90), "crop75": lambda w: crop(w, 0.75), "crop50": lambda w: crop(w, 0.5)}
    det = {}
    for cn, fn in conds.items():
        det[cn] = round(float(np.mean([detect(fn(wm), iid, tx) for iid, wm, tx, _ in wms])), 3)
    report["variant"][vname] = {"psnr": round(float(np.mean(ps)), 2), "ssim": round(float(np.mean(ss)), 4), "detect": det, "n": len(wms)}
    print(f"  {vname:18}: PSNR={report['variant'][vname]['psnr']:.2f} SSIM={report['variant'][vname]['ssim']:.4f} | " +
          "  ".join(f"{k}={v:.2f}" for k, v in det.items()), flush=True)
json.dump(report, open(f"{OUT}/drop_outer_ring.json", "w"), indent=2)
print("DROP_OUTER_RING_DONE", flush=True)
