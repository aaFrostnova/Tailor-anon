"""Embed the 3-fragment composite VINE -> TrustMark -> MaskWM-ED (@alpha=0.70) on the
same 48 fresh imgs (sw_180..227, image_ids identical to the 2-frag sweep) for the
3-fragment CtrlRegen+ comparison. Writes sweep3/comp/img_%05d.png + meta.json.
"""
import os, sys, glob, json
import numpy as np
from PIL import Image
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.maskwm_wrapper import MaskWMWrapper
from src.payload import image_id_to_payload

KEY = b"v5_key_encoder_master"; ALPHA = 0.70; dev = "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
START = int(sys.argv[2]) if len(sys.argv) > 2 else 180
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(REPO, "results/defense/sweep3")
PREFIX = "sw"                                   # SAME ids as 2-frag sweep -> directly comparable
imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[START:START + N]
sb = ShortenedBCH()
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
ed = MaskWMWrapper(ckpt_path=os.path.join(REPO, "external/MaskWM/checkpoints/ED_128bits.pth"),
                   master_key=KEY, method_name="maskwm_ed", n_bits=sb.n, msg_len=128,
                   jnd_factor=1.75, device=dev, use_self_mask=False, config_name="ED_128bits")
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
os.makedirs(os.path.join(OUT, "comp"), exist_ok=True)
meta = {"alpha": ALPHA, "start": START, "fragments": ["vine", "trustmark", "maskwm_ed"], "items": []}
psnrs = []
for j, fp in enumerate(imgs):
    iid = f"{PREFIX}_{START + j:05d}"; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    orig = to512(Image.open(fp).convert("RGB"))
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid); pe, Me = ed.get_perm_M(iid)
    x = scale(orig, to512(vine.embed_with_target(orig, apply_crypto(tx, pv, Mv))), ALPHA)
    x = scale(x, to512(tm.embed_with_target(x, apply_crypto(tx, pt, Mt))), ALPHA)
    x = scale(x, to512(ed.embed_with_target(x, apply_crypto(tx, pe, Me))), ALPHA)
    x.save(os.path.join(OUT, "comp", f"img_{j:05d}.png"))
    mse = np.mean((np.asarray(orig, np.float64) - np.asarray(x, np.float64)) ** 2)
    psnrs.append(10 * np.log10(255 * 255 / mse) if mse > 1e-9 else 99.0)
    meta["items"].append({"i": j, "image_id": iid})
    if (j + 1) % 12 == 0: print(f"  embed3 [{j+1}/{len(imgs)}] mean PSNR={np.mean(psnrs):.1f}", flush=True)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
print(f"[sweep_embed3] {len(imgs)} VINE+TM+ED comp imgs @alpha={ALPHA} mean PSNR={np.mean(psnrs):.2f}dB -> {OUT}\nSWEEP_EMBED3_DONE")
