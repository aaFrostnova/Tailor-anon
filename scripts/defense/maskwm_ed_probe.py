"""GATE: verify MaskWM-ED (ED_128bits, full-image/all-ones mask) embeds+decodes our
crypto codeword correctly BEFORE fusing it as a 3rd fragment, and measure the raw-score
distribution to calibrate the score->LLR map for soft fusion.

ED uses the SAME MaskWMWrapper code path as D (encoder auto-fills ones mask when
mask_channel & mask=None; decoder with use_self_mask=False uses ones) -- only the
checkpoint/config/jnd differ. If clean bit-acc < ~0.95 the full-ones mask is OOD and we
stop + rethink the mask, rather than fuse a broken fragment.
"""
import os, sys, glob
import numpy as np
from PIL import Image
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.maskwm_wrapper import MaskWMWrapper
from src.shortened_bch import ShortenedBCH
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import undo_crypto, apply_crypto

KEY = b"v5_key_encoder_master"; dev = "cuda"
sb = ShortenedBCH()
ed = MaskWMWrapper(ckpt_path=os.path.join(REPO, "external/MaskWM/checkpoints/ED_128bits.pth"),
                   master_key=KEY, method_name="maskwm_ed", n_bits=sb.n, msg_len=128,
                   jnd_factor=1.75, device=dev, use_self_mask=False, config_name="ED_128bits")

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
def psnr(a, b):
    m = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 99.0 if m < 1e-9 else 10 * np.log10(255 * 255 / m)

imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:188]
print(f"ED_128bits gate: n={len(imgs)} imgs, jnd=1.75, full-ones mask\n", flush=True)
rows = []
s1_all, s0_all = [], []   # raw scores split by true codeword bit
for a in [1.0, 0.70]:
    bas, ps = [], []
    for j, fp in enumerate(imgs):
        iid = f"sw_{180 + j:05d}"
        orig = to512(Image.open(fp).convert("RGB"))
        cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))   # length n (ShortenedBCH, same as VINE/TM)
        perm, M = ed.get_perm_M(iid)
        tgt = apply_crypto(cw, perm, M)                                 # crypto target on the sb codeword (NOT wrapper's BCHCodec)
        wm = to512(ed.embed_with_target(orig, tgt))
        wm = scale_resid(orig, wm, a)
        scores = ed.raw_scores(wm)                                      # length n, ~near {0,1}
        rec = undo_crypto(scores, perm, M)
        ba = float(np.mean(rec == cw)); bas.append(ba); ps.append(psnr(orig, wm))
        if a == 1.0:
            tgt = np.asarray(tgt)
            s1_all += scores[tgt > 0.5].tolist(); s0_all += scores[tgt <= 0.5].tolist()
    print(f"alpha={a:.2f}: clean bit-acc = {np.mean(bas):.3f} (min {np.min(bas):.3f})  PSNR={np.mean(ps):.1f}dB", flush=True)
s1, s0 = np.array(s1_all), np.array(s0_all)
print(f"\nraw-score distribution (alpha=1.0): bit=1 mean {s1.mean():.3f} (std {s1.std():.3f}); "
      f"bit=0 mean {s0.mean():.3f} (std {s0.std():.3f})")
sep = (s1.mean() - s0.mean())
print(f"separation (s1-s0)={sep:.3f}  -> suggested LLR scale k so confident |LLR|~3: k~={3/max(abs(s1.mean()-0.5),1e-3):.1f}")
print("EDPROBE_DONE")
