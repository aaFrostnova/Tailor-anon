"""Decode the 3-fragment (VINE+TM+MaskWM-ED) composite across attacked dirs and compare
the marginal value of the ED fragment:
  VINE | TM | ED | 2frag(V+T) | 3frag(V+T+E) | oracle3
on detection (crypto-verify OR fused-ba>=tau) + mean bit-acc, plus attack SSIM/PSNR.

ED raw scores (~{0,1}, threshold 0.5) -> LLR = K_ED*(score-0.5) (kind=logit), K_ED from
the calibration probe (~5.5 -> confident |LLR|~3, comparable to VINE/TM).
"""
import os, sys, glob, json, argparse
import numpy as np, torch
from PIL import Image
from scipy.stats import binom
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.maskwm_wrapper import MaskWMWrapper
from src.soft_fusion import method_soft_to_codeword_llr
try:
    from skimage.metrics import structural_similarity as _ssim, peak_signal_noise_ratio as _psnr
    def ssim(a, b): return float(_ssim(a, b, channel_axis=2))
    def psnr(a, b): return float(_psnr(a, b, data_range=255))
except Exception:
    def psnr(a, b):
        m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2); return 99.0 if m < 1e-9 else float(10*np.log10(255*255/m))
    def ssim(a, b): return float("nan")

KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; K_ED = 5.5
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
ap = argparse.ArgumentParser()
ap.add_argument("--embed_dir", default="results/defense/sweep3")
ap.add_argument("--attack", default="ctrlregen3")
ap.add_argument("--pattern", default="results/defense/sweep3/cr_s*")
ap.add_argument("--out", default="results/defense/sweep3_results.jsonl")
args = ap.parse_args()

vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
ed = MaskWMWrapper(ckpt_path=os.path.join(REPO, "external/MaskWM/checkpoints/ED_128bits.pth"),
                   master_key=KEY, method_name="maskwm_ed", n_bits=sb.n, msg_len=128,
                   jnd_factor=1.75, device=dev, use_self_mask=False, config_name="ED_128bits")
meta = json.load(open(os.path.join(REPO, args.embed_dir, "meta.json")))
comp_dir = os.path.join(REPO, args.embed_dir, "comp")
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im

def decode_dir(d):
    av, at, ae, tx, ids, ss, ps = [], [], [], [], [], [], []
    for it in meta["items"]:
        fp = os.path.join(d, f"img_{it['i']:05d}.png"); ref = os.path.join(comp_dir, f"img_{it['i']:05d}.png")
        if not os.path.exists(fp): continue
        iid = it["image_id"]; im = to512(Image.open(fp).convert("RGB"))
        if os.path.exists(ref):
            A = np.asarray(im); B = np.asarray(to512(Image.open(ref).convert("RGB"))); ss.append(ssim(A, B)); ps.append(psnr(A, B))
        cw = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
        pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid); pe, Me = ed.get_perm_M(iid)
        av.append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(im), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP).astype(np.float32))
        at.append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(im), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP).astype(np.float32))
        ae.append(np.clip(method_soft_to_codeword_llr(K_ED * (ed.raw_scores(im) - 0.5), pe, Me, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP).astype(np.float32))
        tx.append(cw.astype(np.uint8)); ids.append(iid)
    return (np.array(av), np.array(at), np.array(ae), np.array(tx), ids,
            float(np.mean(ss)) if ss else float("nan"), float(np.mean(ps)) if ps else float("nan"))

e = 1e-6
def norm(a): return a / (a.std(1, keepdims=True) + e)
def oracle3(av, at, ae, tx):
    f = np.zeros_like(av)
    for A in (av, at, ae):
        b = ((A > 0).astype(np.uint8) == tx).mean(1)[:, None]; w = np.clip(2 * b - 1, 0, None); f += w * norm(A)
    flat = (np.abs(f).sum(1) < e); f[flat] = (norm(av) + norm(at) + norm(ae))[flat]; return f
def metrics(Fz, tx, ids):
    dets = []
    for i in range(len(Fz)):
        ver = bool(decode_and_verify(Fz[i], ids[i], codec=sb)["detected"])
        zb = ((Fz[i] > 0).astype(np.uint8) == tx[i]).mean()
        dets.append(1.0 if (ver or zb >= TAU) else 0.0)
    return float(np.mean(dets)), float(((Fz > 0).astype(np.uint8) == tx).mean())

dirs = [("clean", comp_dir)] + [(os.path.basename(d), d) for d in sorted(glob.glob(os.path.join(REPO, args.pattern))) if os.path.isdir(d)]
rows = []
hdr = f"{'point':8s}{'ssim':>6s}{'psnr':>6s} |{'VINE':>9s}{'TM':>9s}{'ED':>9s}{'2frag':>9s}{'3frag':>9s}{'oracle3':>9s}"
print(f"tau={TAU:.3f} K_ED={K_ED}\n{hdr}\n{'-'*len(hdr)}", flush=True)
for tag, d in dirs:
    av, at, ae, tx, ids, ss, ps = decode_dir(d)
    if len(av) == 0: continue
    R = {"VINE": metrics(av, tx, ids), "TM": metrics(at, tx, ids), "ED": metrics(ae, tx, ids),
         "2frag": metrics(av + at, tx, ids), "3frag": metrics(av + at + ae, tx, ids), "oracle3": metrics(oracle3(av, at, ae, tx), tx, ids)}
    def c(k): return f"{R[k][0]:.2f}/{R[k][1]:.2f}"
    print(f"{tag:8s}{ss:6.2f}{ps:6.1f} |{c('VINE'):>9s}{c('TM'):>9s}{c('ED'):>9s}{c('2frag'):>9s}{c('3frag'):>9s}{c('oracle3'):>9s}", flush=True)
    rows.append({"attack": args.attack, "point": tag, "n": int(len(av)), "ssim": ss, "psnr": ps,
                 **{k: {"det": R[k][0], "ba": R[k][1]} for k in R}})
outp = os.path.join(REPO, args.out)
existing = [json.loads(l) for l in open(outp)] if os.path.exists(outp) else []
existing = [r for r in existing if r.get("attack") != args.attack]
with open(outp, "w") as f:
    for r in existing + rows: f.write(json.dumps(r) + "\n")
print(f"[saved] {len(rows)} points -> {outp}\nSWEEP_DECODE3_DONE")
