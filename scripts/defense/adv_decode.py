"""Decode one method (comp/vine/tm) on an attacked dir vs its clean watermarked dir.
Measures attack-induced SSIM/PSNR + bit-acc + detection; appends a row to a jsonl.
Usage: adv_decode.py <method> <clean_dir> <attacked_dir> <meta.json> <attack_label> <out.jsonl>"""
import os, sys, glob, json
import numpy as np
from PIL import Image
from scipy.stats import binom
from skimage.metrics import structural_similarity as ssim_fn
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload

method, clean_dir, att_dir, meta_fp, label, out_jsonl = sys.argv[1:7]
KEY = b"v5_key_encoder_master"; dev = "cuda"
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
meta = json.load(open(meta_fp))
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def llr_of(att, pv, Mv, pt, Mt):
    if method == "vine": return method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n)
    if method == "tm": return method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n)
    av = method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n)
    at = method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n)
    return fuse_llrs({"v": av, "t": at}, n_codeword=sb.n)
ss, ps, ba, det, miss = [], [], [], [], 0
for it in meta["items"]:
    i, iid = it["i"], it["image_id"]
    cfp = os.path.join(clean_dir, f"img_{i:05d}.png"); afp = os.path.join(att_dir, f"img_{i:05d}.png")
    if not os.path.exists(afp): miss += 1; continue
    clean = to512(Image.open(cfp).convert("RGB")); att = to512(Image.open(afp).convert("RGB"))
    C = np.asarray(clean, np.float64); A = np.asarray(att, np.float64); mse = np.mean((C - A) ** 2)
    ss.append(float(ssim_fn(C, A, channel_axis=2, data_range=255))); ps.append(99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse))
    tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
    llr = llr_of(att, pv, Mv, pt, Mt); b = float(np.mean(llr_to_bits(llr) == tx))
    d = 1.0 if (bool(decode_and_verify(llr, iid, codec=sb)["detected"]) or b >= TAU) else 0.0
    ba.append(b); det.append(d)
row = {"method": method, "attack": label, "ssim": float(np.mean(ss)), "psnr": float(np.mean(ps)),
       "ba": float(np.mean(ba)), "det": float(np.mean(det)), "n": len(det), "miss": miss}
open(out_jsonl, "a").write(json.dumps(row) + "\n")
print(f"[adv_decode] {method:5s} {label:16s} | SSIM {row['ssim']:.3f} PSNR {row['psnr']:.2f} | ba {row['ba']:.3f} det {row['det']:.3f} (n={row['n']})")
