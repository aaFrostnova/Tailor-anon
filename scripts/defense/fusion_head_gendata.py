"""Phase 1: generate training data for the learned fusion head.

Embed the composite (VINE+TrustMark @ alpha=0.7) on each image ONCE, then apply
a diverse attack suite (random strength per sample). For each attacked image,
decode each fragment to its ALIGNED codeword LLR (post undo-crypto) and save
(a_vine, a_tm, codeword tx, attack_label, img_idx). Encoders frozen.

Cache -> results/defense/fusion_head_data.npz  (AV, AT, TX, ATK, IMG).
The expensive VINE embed is amortized across all attacks for an image.
"""
import os, sys, glob, io
import numpy as np, torch
from PIL import Image, ImageEnhance, ImageFilter

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr
from src.payload import image_id_to_payload
from _regen_util import build_regen_pipe, stable_regen

KEY = b"v5_key_encoder_master"
ALPHA = 0.70
N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 180
dev = "cuda"
imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[:N_IMG]

sb = ShortenedBCH()
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
regen_pipe = build_regen_pipe()

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(cover, wm, a):
    c = np.asarray(to512(cover), np.float64); w = np.asarray(to512(wm), np.float64)
    return Image.fromarray(np.clip(c + a * (w - c), 0, 255).astype(np.uint8))

def att_jpeg(im, r):  # r in [0,1] -> QF 90..10
    q = int(round(90 - 80 * r)); b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
def att_blur(im, r):  return im.filter(ImageFilter.GaussianBlur(radius=0.5 + 7.5 * r))
def att_noise(im, r):
    a = np.asarray(im, np.float64) + np.random.RandomState(int(r*1e6)+1).normal(0, (0.02 + 0.08*r)*255, (512,512,3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def att_bright(im, r): return ImageEnhance.Brightness(im).enhance(0.6 + 1.2 * r)   # 0.6..1.8
def att_contrast(im, r): return ImageEnhance.Contrast(im).enhance(0.6 + 1.2 * r)
def att_crop(im, r):  # area 0.95..0.55
    area = 0.95 - 0.40 * r; s = int(round(512 * np.sqrt(area))); off = (512 - s) // 2
    return im.crop((off, off, off + s, off + s)).resize((512, 512))
def att_regen(im, r, sd): return stable_regen(regen_pipe, im, seed=sd)
def att_rinse(im, r, sd): return stable_regen(regen_pipe, stable_regen(regen_pipe, im, seed=sd), seed=sd + 1)

ATTACKS = ["clean", "jpeg", "blur", "noise", "bright", "contrast", "crop", "regen", "rinse"]

def apply_attack(name, im, r, sd):
    if name == "clean": return im
    if name == "jpeg": return att_jpeg(im, r)
    if name == "blur": return att_blur(im, r)
    if name == "noise": return att_noise(im, r)
    if name == "bright": return att_bright(im, r)
    if name == "contrast": return att_contrast(im, r)
    if name == "crop": return att_crop(im, r)
    if name == "regen": return att_regen(im, r, sd)
    if name == "rinse": return att_rinse(im, r, sd)

AV, AT, TX, ATK, IMG = [], [], [], [], []
for i, fp in enumerate(imgs):
    image_id = f"fh_{i:05d}"
    tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))
    orig = to512(Image.open(fp).convert("RGB"))
    pv, Mv = vine.get_perm_M(image_id); pt, Mt = tm.get_perm_M(image_id)
    tv = apply_crypto(tx, pv, Mv); tt = apply_crypto(tx, pt, Mt)
    v_a = scale_resid(orig, to512(vine.embed_with_target(orig, tv)), ALPHA)
    comp = scale_resid(v_a, to512(tm.embed_with_target(v_a, tt)), ALPHA)
    rng = np.random.RandomState(1000 + i)
    for k in ATTACKS:
        r = float(rng.uniform(0.3, 1.0))             # random strength (skew to non-trivial)
        att = to512(apply_attack(k, comp, r, 2000 + i))
        av = method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n)
        at = method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n)
        AV.append(av.astype(np.float32)); AT.append(at.astype(np.float32)); TX.append(tx.astype(np.uint8))
        ATK.append(k); IMG.append(i)
    if (i + 1) % 10 == 0: print(f"  [{i+1}/{len(imgs)}] samples={len(AV)}", flush=True)

out = os.path.join(REPO, "results/defense/fusion_head_data.npz")
np.savez_compressed(out, AV=np.array(AV), AT=np.array(AT), TX=np.array(TX),
                    ATK=np.array(ATK), IMG=np.array(IMG), n=sb.n, alpha=ALPHA, attacks=np.array(ATTACKS))
print(f"[saved] {len(AV)} samples ({len(imgs)} imgs x {len(ATTACKS)} attacks) -> {out}")
print("GENDATA_DONE")
