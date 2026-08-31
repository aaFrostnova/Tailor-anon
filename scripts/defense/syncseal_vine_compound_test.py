"""Why can't SyncSeal + VINE rescue the rot30+regen COMPOUND? Isolate the two failure modes on the
already-generated rot30_regen images (composite+sync, then rot30, then regen):
  raw       : decode each fragment on the compound-attacked frame (no rectify)
  oracle    : rotate back EXACTLY -30 (grant a PERFECT geometry fix) then decode
  syncseal  : SyncSeal detect+unwarp (real learned rectify, sync mark must survive regen) then decode
Per fragment report bit-acc + crypto-verify rate. Decides:
  - if oracle+VINE bit-acc >= ~0.90 but syncseal+VINE low -> failure is SyncSeal (sync mark regen-dead).
  - if oracle+VINE bit-acc < ~0.90 too -> VINE ITSELF can't cross the crypto floor even perfectly
    rectified (rotation-recovery ceiling 0.72 + regen), i.e. the DEEPER fundamental failure.
Reference: regen-alone (aligned) VINE bit-acc = VINE's regen-survival when NOT rotated."""
import os, sys, json
import numpy as np, torch
from PIL import Image
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
for p in [REPO, os.path.join(REPO, "scripts")]: sys.path.insert(0, p)
from scipy.stats import binom
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.soft_bch import decode_and_verify
from src.syncseal_frontend import load_sync, sync_rectify

KEY = b"v5_key_encoder_master"; dev = "cuda"; RES = 512
SS2 = "/scratch/workspace/mingzhel_umass_edu-ablator/syncseal_regen"
sb = ShortenedBCH(); n = sb.n
sync = load_sync(dev=dev)
frag = {
 "vine":      (VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=n, device=dev), "raw_probs", "prob"),
 "trustmark": (TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=n, model_type="B", device=dev), "raw_logits", "logit"),
 "videoseal": (VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=n, device=dev), "raw_logits", "logit"),
}
def rot(img, deg): return img.rotate(deg, resample=Image.BICUBIC)
def frag_ba_cv(name, pil, iid, tx):
    fr, getter, kind = frag[name]; p, M = fr.get_perm_M(iid)
    llr = method_soft_to_codeword_llr(getattr(fr, getter)(pil), p, M, kind=kind, n_codeword=n)
    ba = float(np.mean((llr > 0).astype(np.uint8) == tx))
    cv = bool(decode_and_verify(llr, iid, codec=sb)["detected"])
    return ba, cv

meta = json.load(open(os.path.join(SS2, "embed", "meta.json")))
items = meta["items"]
print(f"compound rot30+regen: why SyncSeal+VINE can't rescue (n={len(items)})\n", flush=True)
acc = {v: {m: {"ba": [], "cv": []} for m in ["raw", "oracle", "syncseal"]} for v in frag}
regen_ref = {v: [] for v in frag}
for k, it in enumerate(items):
    iid = it["image_id"]; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits)).astype(np.uint8)
    comp = Image.open(os.path.join(SS2, "rot30_regen", f"img_{it['i']:05d}.png")).convert("RGB").resize((RES, RES))
    variants = {"raw": comp, "oracle": rot(comp, -30.0), "syncseal": sync_rectify(sync, comp, dev)[0]}
    for name in frag:
        for mkey, pil in variants.items():
            ba, cv = frag_ba_cv(name, pil, iid, tx)
            acc[name][mkey]["ba"].append(ba); acc[name][mkey]["cv"].append(1.0*cv)
    # regen-alone (aligned) reference
    ra = Image.open(os.path.join(SS2, "regen", f"img_{it['i']:05d}.png")).convert("RGB").resize((RES, RES))
    for name in frag: regen_ref[name].append(frag_ba_cv(name, ra, iid, tx)[0])
    if (k+1) % 8 == 0: print(f"  [{k+1}/{len(items)}]", flush=True)

print(f"\n{'fragment':10s} | regen-alone | {'raw ba/cv':13s} | {'ORACLE-unrot ba/cv':18s} | {'SyncSeal ba/cv':14s}")
print("-"*82)
for name in frag:
    ra = np.mean(regen_ref[name])
    def cell(m): return f"{np.mean(acc[name][m]['ba']):.2f}/{np.mean(acc[name][m]['cv']):.2f}"
    print(f"{name:10s} |    {ra:.2f}     | {cell('raw'):13s} | {cell('oracle'):18s} | {cell('syncseal'):14s}")
print("\n(ba=bit-acc, cv=crypto-verify rate. crypto floor ~0.90. regen-alone = VINE's regen survival when aligned.)")
print(" READ: oracle-unrot grants PERFECT geometry fix -> if VINE ba there is still <0.90, VINE itself can't")
print(" cross the floor under rot+regen; and SyncSeal col shows whether the sync mark even survives regen to rectify.")
print("COMPOUND_TEST_DONE")
