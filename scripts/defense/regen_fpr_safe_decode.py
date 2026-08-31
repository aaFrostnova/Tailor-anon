"""Quantify the FPR-safety gap in regen/UnMarker detection. Three acceptance tiers per image:
  crypto_only : ONLY crypto-verify anywhere (primary/bestpath/scale/syncseal/rotation) — union FPR <= 2^-29.6, strict.
  deployed    : crypto_only OR the single PRIMARY fused zerobit (ba>=tau at scale 1.0) — 1 test, ~1% FPR tier.
  permissive  : deployed OR zerobit ba>=tau at ANY of 133 scales — FPR-UNSAFE (this is what the benchmark used).
Re-decodes the existing CtrlRegen+ {s03,s05,s07,s09} and UnMarker {default,strong} sets. Shows the gap so the
report's regen numbers can be corrected to the FPR-safe (deployed) rule; UnMarker should be unaffected (its rings
crypto-verify at the right scale, per oracle_ring.json)."""
import os, sys, glob, json, numpy as np
from PIL import Image
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"); sys.path.insert(0, "scripts/defense")
import torch
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs
from src.syncseal_frontend import load_sync, sync_rectify
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0
sb = ShortenedBCH(); n = sb.n; tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
SC = "/scratch/workspace/mingzhel_umass_edu-ablator/adv_attacks"
OUT = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/results/defense"
print(f"[gpu] free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB  tau={tau:.4f}", flush=True)
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits"), "videoseal": ("logit", "raw_logits")}
F = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev),
     "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev),
     "videoseal": VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev)}
FR = ["vine", "trustmark", "videoseal"]; sync = load_sync(dev=dev)
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def llr(nm, pil, iid):
    kind, g = SPEC[nm]; p, M = F[nm].get_perm_M(iid)
    return np.clip(method_soft_to_codeword_llr(getattr(F[nm], g)(pil), p, M, kind=kind, n_codeword=n), -CLAMP, CLAMP).astype(np.float32)
def cv(rl, iid): return bool(decode_and_verify(rl, iid, codec=sb)["detected"])
def zb(rl, tx): return ((rl > 0).astype(np.uint8) == tx).mean() >= tau
def view_at(P, f):
    if f >= 0.999: return P
    s = int(round(512 * float(f))); o = (512 - s) // 2; return P.crop((o, o, o + s, o + s))
def rot(img, deg):
    ar = np.asarray(img); pad = 256
    big = Image.fromarray(np.pad(ar, ((pad, pad), (pad, pad), (0, 0)), "reflect")).rotate(deg, resample=Image.BICUBIC)
    l = (big.size[0] - 512) // 2; return big.crop((l, l, l + 512, l + 512))
VS = np.arange(0.34, 1.0001, 0.005)
def tiers(att, iid, tx):
    P = to512(att)
    al = {nm: llr(nm, P, iid) for nm in FR}
    fused = fuse_llrs(al, weights=None, n_codeword=n)
    crypto = cv(fused, iid) or any(cv(al[nm], iid) for nm in FR)
    primary_zb = zb(fused, tx)
    safe_scale = perm_scale = False
    for f in VS:
        rl = llr("vine", view_at(P, float(f)), iid)
        if not safe_scale and cv(rl, iid): safe_scale = True
        if not perm_scale and zb(rl, tx): perm_scale = True
        if safe_scale and perm_scale: break
    rect, _ = sync_rectify(sync, P, dev)
    sync_hit = any(cv(llr(nm, rect, iid), iid) for nm in FR)
    rot_hit = False; best_d, best_ba = 0.0, -1.0
    if not (crypto or safe_scale or sync_hit):
        for d in np.arange(-180, 180.01, 10.0):
            rl = llr("trustmark", rot(P, float(d)), iid)
            if cv(rl, iid): rot_hit = True; break
            ba = float(((rl > 0).astype(np.uint8) == tx).mean())
            if ba > best_ba: best_ba, best_d = ba, d
        if not rot_hit and best_ba >= 0.60:
            for d in np.arange(best_d - 10, best_d + 10.01, 3.0):
                for nm in ("trustmark", "videoseal"):
                    if cv(llr(nm, rot(P, float(d)), iid), iid): rot_hit = True; break
                if rot_hit: break
    crypto_only = crypto or safe_scale or sync_hit or rot_hit
    deployed = crypto_only or primary_zb
    permissive = deployed or perm_scale
    return crypto_only, deployed, permissive

def load(embed_meta, adir, um=False):
    if um:
        return [(m["iid"], f"{adir}/{m['fname']}") for m in json.load(open(embed_meta))["items"] if os.path.exists(f"{adir}/{m['fname']}")]
    return [(m["iid"], f"{adir}/{m['fname']}") for m in json.load(open(embed_meta))["items"] if os.path.exists(f"{adir}/{m['fname']}")]
SETS = {
    "ctrlregen_s03": (f"{SC}/p3embed_D/meta.json", f"{SC}/p3cr_D_s03"),
    "ctrlregen_s05": (f"{SC}/p3embed_D/meta.json", f"{SC}/p3cr_D_s05"),
    "ctrlregen_s07": (f"{SC}/p3embed_D/meta.json", f"{SC}/p3cr_D_s07"),
    "ctrlregen_s09": (f"{SC}/p3embed_D/meta.json", f"{SC}/p3cr_D_s09"),
    "unmarker_default": (f"{SC}/p3embed_Dum/meta.json", f"{SC}/p3um_D"),
    "unmarker_strong":  (f"{SC}/p3embed_Dum/meta.json", f"{SC}/p3um_strong"),
}
report = {"tau": tau}
for sname, (mp, adir) in SETS.items():
    if not os.path.isdir(adir): print(f"  {sname}: (missing)"); continue
    items = load(mp, adir)
    co, dep, perm = [], [], []
    for iid, path in items:
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
        c, d, p = tiers(to512(Image.open(path).convert("RGB")), iid, tx)
        co.append(1.0 if c else 0.0); dep.append(1.0 if d else 0.0); perm.append(1.0 if p else 0.0)
    report[sname] = {"n": len(items), "crypto_only": round(float(np.mean(co)), 3),
                     "deployed_fpr_safe": round(float(np.mean(dep)), 3), "permissive_unsafe": round(float(np.mean(perm)), 3)}
    r = report[sname]
    print(f"  {sname:18} crypto_only={r['crypto_only']:.3f}  DEPLOYED(fpr-safe)={r['deployed_fpr_safe']:.3f}  "
          f"permissive={r['permissive_unsafe']:.3f}   (n={r['n']})", flush=True)
json.dump(report, open(f"{OUT}/regen_fpr_safe.json", "w"), indent=2)
print("REGEN_FPR_SAFE_DONE", flush=True)
