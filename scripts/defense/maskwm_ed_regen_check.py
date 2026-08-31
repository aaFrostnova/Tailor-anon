"""DECISIVE 5-min check: does MaskWM-ED survive diffusion regeneration? ED is a
pixel-space JND additive watermark (TrustMark sibling). If it dies under regen
(ba->0.5) like TrustMark, a 3rd pixel-space fragment adds NOTHING under CtrlRegen+.
Uses the cheap in-env multi-round stable_regen (faithful CtrlRegen+ proxy).

Embeds ED-only on fresh imgs, regen x1/x2/x3, decodes ED bit-acc. Also embeds
VINE-only as the known regen-survivor reference.
"""
import os, sys, glob
import numpy as np
from PIL import Image
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.maskwm_wrapper import MaskWMWrapper
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto, undo_crypto
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen

KEY = b"v5_key_encoder_master"; dev = "cuda"; ALPHA = 0.70
sb = ShortenedBCH()
ed = MaskWMWrapper(ckpt_path=os.path.join(REPO, "external/MaskWM/checkpoints/ED_128bits.pth"),
                   master_key=KEY, method_name="maskwm_ed", n_bits=sb.n, msg_len=128,
                   jnd_factor=1.75, device=dev, use_self_mask=False, config_name="ED_128bits")
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
pipe = build_regen_pipe()

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))

imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:188]
ROUNDS = [0, 1, 2, 3]
ed_ba = {k: [] for k in ROUNDS}; vine_ba = {k: [] for k in ROUNDS}
for j, fp in enumerate(imgs):
    iid = f"sw_{180 + j:05d}"
    orig = to512(Image.open(fp).convert("RGB"))
    cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    pe, Me = ed.get_perm_M(iid); pv, Mv = vine.get_perm_M(iid)
    ed_img = scale_resid(orig, to512(ed.embed_with_target(orig, apply_crypto(cw, pe, Me))), ALPHA)
    v_img = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(cw, pv, Mv))), ALPHA)
    ce, cv = ed_img, v_img
    for k in range(0, max(ROUNDS) + 1):
        if k > 0:
            ce = to512(stable_regen(pipe, ce, seed=3000 + j * 10 + k))
            cv = to512(stable_regen(pipe, cv, seed=3000 + j * 10 + k))
        if k in ROUNDS:
            ed_ba[k].append(float(np.mean(undo_crypto(ed.raw_scores(ce), pe, Me) == cw)))
            vine_ba[k].append(float(np.mean((vine.raw_probs(cv) > 0.5).astype(int) == apply_crypto(cw, pv, Mv)) if False else
                                     np.mean(undo_crypto((vine.raw_probs(cv) > 0.5).astype(np.uint8), pv, Mv) == cw)))
    print(f"  [{j+1}/{len(imgs)}]", flush=True)

print(f"\n{'regen':6s} | {'MaskWM-ED ba':>13s} | {'VINE ba':>9s}")
for k in ROUNDS:
    print(f"x{k:<5d} | {np.mean(ed_ba[k]):13.3f} | {np.mean(vine_ba[k]):9.3f}")
print("verdict: ED " + ("SURVIVES regen (worth fusing under CtrlRegen+)" if np.mean(ed_ba[1]) > 0.62 else
                         "DIES under regen like TrustMark (no CtrlRegen+ value)"))
print("EDREGEN_DONE")
