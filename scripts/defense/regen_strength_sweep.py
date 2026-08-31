"""Is the 'rot+regen compound is fatal' conclusion an artifact of ONE regen strength? Sweep the
DiffWMAttacker noise_step and measure, per strength, whether even a PERFECT geometry fix (oracle
un-rotate) lets VINE — the sole regen survivor — cross the crypto-verify floor (~0.90) under
rot30+regen. Reference = regen-ALONE aligned VINE (its regen survival without rotation).
Reuses the composite+sync embed set (only VINE is decoded; oracle un-rotate, not SyncSeal)."""
import os, sys, json
import numpy as np, torch
from PIL import Image
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO, "scripts"), os.path.join(REPO, "scripts/defense")]: sys.path.insert(0, p)
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.soft_fusion import method_soft_to_codeword_llr
from src.soft_bch import decode_and_verify
from _regen_util import build_regen_pipe, stable_regen

KEY = b"v5_key_encoder_master"; dev = "cuda"; RES = 512
SS2 = "/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_regen"
sb = ShortenedBCH(); n = sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev)
pipe = build_regen_pipe()
def rot(img, deg): return img.rotate(deg, resample=Image.BICUBIC)
def vine_ba_cv(pil, iid, tx):
    p, M = vine.get_perm_M(iid)
    llr = method_soft_to_codeword_llr(vine.raw_probs(pil), p, M, kind="prob", n_codeword=n)
    return float(np.mean((llr > 0).astype(np.uint8) == tx)), bool(decode_and_verify(llr, iid, codec=sb)["detected"])

meta = json.load(open(os.path.join(SS2, "embed", "meta.json")))
items = meta["items"][:int(sys.argv[1]) if len(sys.argv) > 1 else 16]
STRENGTHS = [30, 45, 60, 90]
print(f"regen-strength sweep, n={len(items)}, crypto floor ~0.90, zero-bit tau 0.63\n", flush=True)
res = {s: {"regen_ba": [], "regen_cv": [], "orc_ba": [], "orc_cv": []} for s in STRENGTHS}
for k, it in enumerate(items):
    iid = it["image_id"]; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
    W = Image.open(os.path.join(SS2, "embed", f"img_{it['i']:05d}.png")).convert("RGB").resize((RES, RES))
    Wr = rot(W, 30.0)
    for s in STRENGTHS:
        ra = stable_regen(pipe, W, 1234 + k, noise_step=s)                   # regen-alone (aligned)
        ba, cv = vine_ba_cv(ra, iid, tx); res[s]["regen_ba"].append(ba); res[s]["regen_cv"].append(1.0*cv)
        rr = stable_regen(pipe, Wr, 1234 + k, noise_step=s)                  # rot30 then regen
        orc = rot(rr, -30.0)                                                 # perfect geometry fix
        ba, cv = vine_ba_cv(orc, iid, tx); res[s]["orc_ba"].append(ba); res[s]["orc_cv"].append(1.0*cv)
    if (k+1) % 4 == 0: print(f"  [{k+1}/{len(items)}]", flush=True)

print(f"\n{'noise_step':10s} | {'regen-alone VINE ba/cv':22s} | {'rot30+regen + ORACLE-unrot VINE ba/cv':38s}")
print("-"*78)
for s in STRENGTHS:
    r = res[s]
    print(f"{s:<10d} | {np.mean(r['regen_ba']):.3f} / {np.mean(r['regen_cv']):.3f}            | "
          f"{np.mean(r['orc_ba']):.3f} / {np.mean(r['orc_cv']):.3f}")
print("\n(noise_step lower = weaker regen. ba=VINE bit-acc, cv=crypto-verify rate. Job-2 used noise_step=60.")
print(" If even the WEAKEST strength keeps ORACLE-unrot ba below ~0.90, the compound is fatal at ALL")
print(" tested strengths, not an artifact of one setting.)")
print("REGEN_SWEEP_DONE")
