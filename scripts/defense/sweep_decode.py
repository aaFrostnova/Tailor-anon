"""Decode a set of attacked dirs (one per advanced-attack parameter point) and
compute the COMPLETE detector comparison + attack image-quality at each point.

Per dir: decode VINE & TM aligned codeword LLRs for all images, then for each of
{VINE, TM, equal-MRC, conf-MRC, OLD-head, NEW-head, oracle} report detection
(crypto-verify OR fused-ba>=tau) and mean bit-acc; plus mean SSIM/PSNR of the
attacked image vs the embedded composite (the image-quality axis: high->low).

  python sweep_decode.py --embed_dir results/defense/sweep --attack ctrlregen \
      --pattern 'results/defense/sweep/cr_s*' --out results/defense/sweep_results.jsonl
Always also decodes the clean (unattacked) comp dir as the reference point.
"""
import os, sys, glob, json, argparse, math
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.stats import binom

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr
try:
    from skimage.metrics import structural_similarity as _ssim, peak_signal_noise_ratio as _psnr
    def ssim(a, b): return float(_ssim(a, b, channel_axis=2))
    def psnr(a, b): return float(_psnr(a, b, data_range=255))
except Exception:
    def psnr(a, b):
        m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2); return 99.0 if m < 1e-9 else float(10 * np.log10(255 * 255 / m))
    def ssim(a, b): return float("nan")

KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n


class FusionHead(nn.Module):
    def __init__(self, hid=16):
        super().__init__()
        self.cal_v = nn.Sequential(nn.Linear(1, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.cal_t = nn.Sequential(nn.Linear(1, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.gate = nn.Sequential(nn.Linear(9, hid), nn.ReLU(), nn.Linear(hid, 2))
    def context(self, av, at):
        def feats(a):
            aa = a.abs(); return torch.stack([aa.mean(1), aa.std(1), (aa < 0.5).float().mean(1), torch.tanh(aa).mean(1)], 1)
        cross = (torch.sign(av) == torch.sign(at)).float().mean(1, keepdim=True)
        return torch.cat([feats(av), feats(at), cross], 1)
    def forward(self, av, at):
        B, m = av.shape; w = F.softplus(self.gate(self.context(av, at)))
        cv = self.cal_v(av.reshape(-1, 1)).reshape(B, m); ct = self.cal_t(at.reshape(-1, 1)).reshape(B, m)
        return w[:, 0:1] * cv + w[:, 1:2] * ct, w

def load_head(p):
    h = FusionHead().to(dev); h.load_state_dict(torch.load(p, map_location=dev)); h.eval(); return h

ap = argparse.ArgumentParser()
ap.add_argument("--embed_dir", default="results/defense/sweep")
ap.add_argument("--attack", required=True)
ap.add_argument("--pattern", required=True)
ap.add_argument("--out", default="results/defense/sweep_results.jsonl")
ap.add_argument("--include_clean", type=int, default=1)
args = ap.parse_args()

vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
old_h = load_head(os.path.join(REPO, "results/defense/fusion_head.pt"))
new_h = load_head(os.path.join(REPO, "results/defense/fusion_head_adv.pt"))
meta = json.load(open(os.path.join(REPO, args.embed_dir, "meta.json")))
comp_dir = os.path.join(REPO, args.embed_dir, "comp")
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im

def decode_dir(d):
    av_l, at_l, tx_l, id_l, ss_l, ps_l = [], [], [], [], [], []
    for it in meta["items"]:
        fp = os.path.join(d, f"img_{it['i']:05d}.png")
        ref = os.path.join(comp_dir, f"img_{it['i']:05d}.png")
        if not os.path.exists(fp): continue
        iid = it["image_id"]; att = to512(Image.open(fp).convert("RGB"))
        if os.path.exists(ref):
            A = np.asarray(att); B = np.asarray(to512(Image.open(ref).convert("RGB")))
            ss_l.append(ssim(A, B)); ps_l.append(psnr(A, B))
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
        pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
        av_l.append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP).astype(np.float32))
        at_l.append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP).astype(np.float32))
        tx_l.append(tx.astype(np.uint8)); id_l.append(iid)
    return (np.array(av_l), np.array(at_l), np.array(tx_l), id_l,
            float(np.mean(ss_l)) if ss_l else float("nan"), float(np.mean(ps_l)) if ps_l else float("nan"))

e = 1e-6
def norm(a): return a / (a.std(1, keepdims=True) + e)
def oracle_w(av, at, tx):
    bav = ((av > 0).astype(np.uint8) == tx).mean(1)[:, None]; bat = ((at > 0).astype(np.uint8) == tx).mean(1)[:, None]
    wv = np.clip(2 * bav - 1, 0, None); wt = np.clip(2 * bat - 1, 0, None)
    f = wv * norm(av) + wt * norm(at); flat = (wv + wt < e)[:, 0]; f[flat] = (norm(av) + norm(at))[flat]; return f
def conf_mrc(av, at):
    wv = np.abs(av).mean(1, keepdims=True); wt = np.abs(at).mean(1, keepdims=True); return wv * norm(av) + wt * norm(at)
def hl(h, av, at):
    with torch.no_grad():
        f, _ = h(torch.tensor(av, device=dev), torch.tensor(at, device=dev)); return f.cpu().numpy()
def metrics(Fz, tx, ids):
    dets = []
    for i in range(len(Fz)):
        ver = bool(decode_and_verify(Fz[i], ids[i], codec=sb)["detected"])
        zb = ((Fz[i] > 0).astype(np.uint8) == tx[i]).mean()
        dets.append(1.0 if (ver or zb >= TAU) else 0.0)
    return float(np.mean(dets)), float(((Fz > 0).astype(np.uint8) == tx).mean())

dirs = []
if args.include_clean: dirs.append(("clean", comp_dir))
for d in sorted(glob.glob(os.path.join(REPO, args.pattern))):
    if os.path.isdir(d): dirs.append((os.path.basename(d), d))

rows = []
print(f"tau={TAU:.3f}  attack={args.attack}  dirs={len(dirs)}", flush=True)
hdr = f"{'point':16s} {'ssim':>5s} {'psnr':>5s} | {'VINE':>9s} {'TM':>9s} {'equalMRC':>9s} {'confMRC':>9s} {'OLD':>9s} {'NEW':>9s} {'oracle':>9s}"
print(hdr); print("-" * len(hdr))
for tag, d in dirs:
    av, at, tx, ids, ss, ps = decode_dir(d)
    if len(av) == 0: continue
    R = {"VINE": metrics(av, tx, ids), "TM": metrics(at, tx, ids), "equalMRC": metrics(av + at, tx, ids),
         "confMRC": metrics(conf_mrc(av, at), tx, ids), "OLD": metrics(hl(old_h, av, at), tx, ids),
         "NEW": metrics(hl(new_h, av, at), tx, ids), "oracle": metrics(oracle_w(av, at, tx), tx, ids)}
    def c(k): return f"{R[k][0]:.2f}/{R[k][1]:.2f}"
    print(f"{tag:16s} {ss:5.2f} {ps:5.1f} | {c('VINE'):>9s} {c('TM'):>9s} {c('equalMRC'):>9s} {c('confMRC'):>9s} {c('OLD'):>9s} {c('NEW'):>9s} {c('oracle'):>9s}", flush=True)
    rows.append({"attack": args.attack, "point": tag, "n": int(len(av)), "ssim": ss, "psnr": ps,
                 **{k: {"det": R[k][0], "ba": R[k][1]} for k in R}})

outp = os.path.join(REPO, args.out)
existing = [json.loads(l) for l in open(outp)] if os.path.exists(outp) else []
existing = [r for r in existing if not (r["attack"] == args.attack)]   # replace this attack's rows
with open(outp, "w") as f:
    for r in existing + rows: f.write(json.dumps(r) + "\n")
print(f"[saved] {len(rows)} points -> {outp}\nSWEEP_DECODE_DONE")
