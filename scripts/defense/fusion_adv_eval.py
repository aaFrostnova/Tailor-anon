"""Stage 1: measure HEADROOM for a learned fusion head under ADVANCED attacks.

Decode the real composite-embedded images that were attacked by CtrlRegen+ /
UnMarker (results/defense/adv/comp_*), recover each fragment's ALIGNED codeword
LLR (a_vine, a_tm), and compare 5 detectors per attack:
  VINE-only | TM-only | equal-MRC (Composite) | existing learned head | oracle_w
on detection rate (crypto-verify OR ba>=tau) and mean bit-acc.

This tells us (a) where equal-MRC is suboptimal (Composite<VINE => dead-TM
poisons the sum) and (b) whether the mild-attack-trained head already recovers
it. Also caches LLRs -> adv_llr_cache.npz to serve as the held-out TEST set for
the dead-fragment-aware head (stage 2).
"""
import os, sys, glob, json
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

KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
ADV = os.path.join(REPO, "results/defense/adv")


class FusionHead(nn.Module):
    def __init__(self, hid=16):
        super().__init__()
        self.cal_v = nn.Sequential(nn.Linear(1, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.cal_t = nn.Sequential(nn.Linear(1, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.gate = nn.Sequential(nn.Linear(9, hid), nn.ReLU(), nn.Linear(hid, 2))

    @staticmethod
    def context(av, at):
        def feats(a):
            aa = a.abs()
            return torch.stack([aa.mean(1), aa.std(1), (aa < 0.5).float().mean(1), torch.tanh(aa).mean(1)], 1)
        cross = (torch.sign(av) == torch.sign(at)).float().mean(1, keepdim=True)
        return torch.cat([feats(av), feats(at), cross], 1)

    def forward(self, av, at):
        B, m = av.shape
        w = F.softplus(self.gate(self.context(av, at)))
        cv = self.cal_v(av.reshape(-1, 1)).reshape(B, m)
        ct = self.cal_t(at.reshape(-1, 1)).reshape(B, m)
        return w[:, 0:1] * cv + w[:, 1:2] * ct, w


def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im

vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
meta = json.load(open(os.path.join(ADV, "meta.json")))

# adv config -> (dir, human label)
CFGS = [
    ("comp_creg03", "CtrlRegen+ s=0.3"),
    ("comp_creg05", "CtrlRegen+ s=0.5"),
    ("comp_creg07", "CtrlRegen+ s=0.7"),
    ("comp_unmark", "UnMarker (def)"),
    ("comp_um_nocrop_l", "UnMarker spec x400"),
    ("comp_um_nocrop_h", "UnMarker spec x2500"),
    ("comp_um_c95", "UnMarker crop5%"),
    ("comp_um_c90", "UnMarker crop10%"),
    ("comp_um_c85", "UnMarker crop15%"),
]

def decode_dir(d):
    av_l, at_l, tx_l, id_l = [], [], [], []
    for it in meta["items"]:
        fp = os.path.join(ADV, d, f"img_{it['i']:05d}.png")
        if not os.path.exists(fp): continue
        iid = it["image_id"]
        att = to512(Image.open(fp).convert("RGB"))
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
        pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
        av = np.clip(method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP)
        at = np.clip(method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)
        av_l.append(av.astype(np.float32)); at_l.append(at.astype(np.float32))
        tx_l.append(tx.astype(np.uint8)); id_l.append(iid)
    return np.array(av_l), np.array(at_l), np.array(tx_l), id_l

e = 1e-6
def norm(a): return a / (a.std(1, keepdims=True) + e)
def oracle_w(av, at, tx):
    bav = ((av > 0).astype(np.uint8) == tx).mean(1)[:, None]; bat = ((at > 0).astype(np.uint8) == tx).mean(1)[:, None]
    wv = np.clip(2 * bav - 1, 0, None); wt = np.clip(2 * bat - 1, 0, None)
    f = wv * norm(av) + wt * norm(at); flat = (wv + wt < e)[:, 0]; f[flat] = (norm(av) + norm(at))[flat]; return f

head = FusionHead().to(dev)
head.load_state_dict(torch.load(os.path.join(REPO, "results/defense/fusion_head.pt"), map_location=dev))
head.eval()
def head_llr(av, at):
    with torch.no_grad():
        f, w = head(torch.tensor(av, device=dev), torch.tensor(at, device=dev))
        return f.cpu().numpy(), w.cpu().numpy()

def metrics(Fz, tx, ids):
    dets = []
    for i in range(len(Fz)):
        ver = bool(decode_and_verify(Fz[i], ids[i], codec=sb)["detected"])
        zb = ((Fz[i] > 0).astype(np.uint8) == tx[i]).mean()
        dets.append(1.0 if (ver or zb >= TAU) else 0.0)
    ba = ((Fz > 0).astype(np.uint8) == tx).mean()
    return float(np.mean(dets)), float(ba)

print(f"tau={TAU:.3f}  n_per_attack varies  (detection = crypto-verify OR ba>=tau)\n")
hdr = f"{'attack':22s} | {'VINE':>11s} | {'TM':>11s} | {'equal-MRC':>11s} | {'HEAD':>11s} | {'oracle':>11s}   (det/ba)"
print(hdr); print("-" * len(hdr))
cache = {}; rows = []
for d, lab in CFGS:
    if not os.path.isdir(os.path.join(ADV, d)): continue
    av, at, tx, ids = decode_dir(d)
    if len(av) == 0: continue
    cache[d] = {"av": av, "at": at, "tx": tx}
    F_hd, W = head_llr(av, at)
    cells = {}
    cells["vine"] = metrics(av, tx, ids)
    cells["tm"] = metrics(at, tx, ids)
    cells["eq"] = metrics(av + at, tx, ids)
    cells["hd"] = metrics(F_hd, tx, ids)
    cells["or"] = metrics(oracle_w(av, at, tx), tx, ids)
    def c(k): return f"{cells[k][0]:.2f}/{cells[k][1]:.2f}"
    print(f"{lab:22s} | {c('vine'):>11s} | {c('tm'):>11s} | {c('eq'):>11s} | {c('hd'):>11s} | {c('or'):>11s}   "
          f"(n={len(av)}, head w_v/w_t={W[:,0].mean():.2f}/{W[:,1].mean():.2f})")
    rows.append({"dir": d, "label": lab, "n": int(len(av)),
                 **{f"{k}_{m}": (cells[k][0] if m == "det" else cells[k][1]) for k in cells for m in ["det", "ba"]},
                 "wv": float(W[:, 0].mean()), "wt": float(W[:, 1].mean())})

np.savez_compressed(os.path.join(REPO, "results/defense/adv_llr_cache.npz"),
                    **{f"{d}_{k}": v for d in cache for k, v in cache[d].items()},
                    dirs=np.array(list(cache.keys())))
json.dump(rows, open(os.path.join(REPO, "results/defense/fusion_adv_eval.json"), "w"), indent=2)
print("\n[saved] adv_llr_cache.npz + fusion_adv_eval.json")
print("ADVEVAL_DONE")
