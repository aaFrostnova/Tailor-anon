"""Minimal-embedding sweep: which fragment/ring combination maximizes PSNR while keeping the defense?
Each config embeds a subset (VINE rings + TM@strength + VS@strength + SyncSeal) and is decoded with a
config-aware version of the deployed cascade. Measures PSNR/SSIM + detection on clean/jpeg/blur/noise/
rot30/rot90/crop75/crop50 (all local, cheap). Regeneration + UnMarker are cross-env and confirmed
separately for the Pareto winner. Prior ablation: VINE+TM ≈ 3-frag (VS marginal); drop-outer-ring: +2.4dB."""
import os, sys, glob, json, io, numpy as np
from PIL import Image, ImageFilter
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
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense"
print(f"[gpu] free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
sync = load_sync(dev=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scl(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
def txc(iid, nm):
    p, M = F[nm].get_perm_M(iid); return apply_crypto(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)), p, M)
def vine_nested(x, iid, scales):
    x = to512(x)
    for K in scales:
        if K >= 0.999: x = to512(F["vine"].embed_with_target(x, txc(iid, "vine")))
        else:
            s = int(512 * K); o = (512 - s) // 2
            m = F["vine"].embed_with_target(x.crop((o, o, o + s, o + s)), txc(iid, "vine"))
            out = x.copy(); out.paste(m.resize((s, s)), (o, o)); x = out
    return x
def llr(nm, pil, iid):
    kind, g = SPEC[nm]; p, M = F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm], g)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP).astype(np.float32)
def cv(rl, iid): return bool(decode_and_verify(rl, iid, codec=sb)["detected"])
def view_at(P, f):
    if f >= 0.999: return P
    s = int(round(512 * float(f))); o = (512 - s) // 2; return P.crop((o, o, o + s, o + s))
def rot(img, deg):
    ar = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(ar, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    l = (big.size[0] - 512) // 2; return big.crop((l, l, l + 512, l + 512))
VS = np.arange(0.34, 1.0001, 0.005)

def embed(orig, iid, cfg):
    x = to512(orig)
    if cfg["vine"]: x = vine_nested(x, iid, cfg["vine"])
    if cfg["tm"] > 0: x = scl(x, to512(F["trustmark"].embed_with_target(x, txc(iid, "trustmark"))), cfg["tm"])
    if cfg["vs"] > 0: x = scl(x, to512(F["videoseal"].embed_with_target(x, txc(iid, "videoseal"))), cfg["vs"])
    return to512(sync_embed(sync, to512(x), dev))
def detect(att, iid, tx, frags):
    P = to512(att)
    al = {nm: llr(nm, P, iid) for nm in frags}
    if len(al) > 1:
        fused = fuse_llrs(al, weights=None, n_codeword=n)
        if cv(fused, iid) or ((fused > 0).astype(np.uint8) == tx).mean() >= tau: return 1.0
    if any(cv(al[nm], iid) for nm in frags): return 1.0
    if "vine" in frags:
        for f in VS:                                    # VERIFY-ONLY (FPR-safe; no zerobit in the repeated search)
            if cv(llr("vine", view_at(P, float(f)), iid), iid): return 1.0
    rect, _ = sync_rectify(sync, P, dev)
    if any(cv(llr(nm, rect, iid), iid) for nm in frags): return 1.0
    if "trustmark" in frags:
        best_d, best_ba = 0.0, -1.0
        for d in np.arange(-180, 180.01, 10.0):
            rl = llr("trustmark", rot(P, float(d)), iid)
            if cv(rl, iid): return 1.0
            ba = float(((rl > 0).astype(np.uint8) == tx).mean())
            if ba > best_ba: best_ba, best_d = ba, d
        if best_ba >= 0.60:
            for d in np.arange(best_d - 10, best_d + 10.01, 3.0):
                for nm in ("trustmark", "videoseal"):
                    if nm in frags and cv(llr(nm, rot(P, float(d)), iid), iid): return 1.0
    return 0.0

# candidate configs (fragments present -> which are decoded)
CONFIGS = {
    "C0_3ring+TM.7+VS.7": {"vine": (1.0, 0.75, 0.5), "tm": 0.70, "vs": 0.70},
    "C1_3ring+TM.7":      {"vine": (1.0, 0.75, 0.5), "tm": 0.70, "vs": 0.0},
    "C2_2ring+TM.7":      {"vine": (0.75, 0.5),      "tm": 0.70, "vs": 0.0},
    "C3_1ring+TM.7":      {"vine": (0.5,),           "tm": 0.70, "vs": 0.0},
    "C4_2ring+TM.5":      {"vine": (0.75, 0.5),      "tm": 0.50, "vs": 0.0},
    "C5_2ring+VS.7":      {"vine": (0.75, 0.5),      "tm": 0.0,  "vs": 0.70},
}
def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def crop(im, c):
    s = int(512 * c); o = (512 - s) // 2; return im.crop((o, o, o + s, o + s)).resize((512, 512))
def noise(im, sd):
    a = np.asarray(im, np.float32) + np.random.RandomState(0).randn(512, 512, 3) * sd * 255
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
AX = {"clean": lambda w: w, "jpeg40": lambda w: jpeg(w, 40), "blur2": lambda w: w.filter(ImageFilter.GaussianBlur(2.0)),
      "noise.05": lambda w: noise(w, 0.05), "rot30": lambda w: rot(w, 30), "rot90": lambda w: rot(w, 90),
      "crop75": lambda w: crop(w, 0.75), "crop50": lambda w: crop(w, 0.50)}
srcs = sorted(glob.glob("/work/pi_shiqingma_umass_edu/mingzheli/W-Bench/DISTORTION_1K/image/*.png"))[:15]
report = {"tau": tau}
for cname, cfg in CONFIGS.items():
    frags = (["vine"] if cfg["vine"] else []) + (["trustmark"] if cfg["tm"] > 0 else []) + (["videoseal"] if cfg["vs"] > 0 else [])
    nlayers = len(cfg["vine"]) + (1 if cfg["tm"] > 0 else 0) + (1 if cfg["vs"] > 0 else 0)
    wms, ps, ss = [], [], []
    for j, fp in enumerate(srcs):
        iid = f"me_{j:04d}"; orig = to512(Image.open(fp).convert("RGB"))   # unique iid per image -> unique crypto target
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        wm = embed(orig, iid, cfg); wms.append((iid, wm, tx, orig))
        ps.append(psnr_fn(np.asarray(orig, np.float64), np.asarray(wm, np.float64), data_range=255))
        ss.append(ssim_fn(np.asarray(orig, np.float64), np.asarray(wm, np.float64), channel_axis=2, data_range=255))
    det = {}
    for ax, fn in AX.items():
        det[ax] = round(float(np.mean([detect(fn(wm), iid, tx, frags) for iid, wm, tx, _ in wms])), 3)
    report[cname] = {"n_embed_layers": nlayers, "psnr": round(float(np.mean(ps)), 2), "ssim": round(float(np.mean(ss)), 4),
                     "detect": det, "min_geo_det": round(min(det["rot30"], det["rot90"], det["crop75"], det["crop50"]), 3)}
    r = report[cname]
    print(f"  {cname:20} layers={nlayers} PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} | " +
          "  ".join(f"{k}={v:.2f}" for k, v in det.items()), flush=True)
json.dump(report, open(f"{OUT}/minimal_embedding_sweep.json", "w"), indent=2)
print("MINIMAL_EMBEDDING_DONE", flush=True)
