"""Stage 3: statistically-powered test of the fusion head under the one-fragment-
death regime, on a FRESH disjoint image set (idx>=180, never used to train the
head which saw only fh_0..125).

Cheap faithful proxy for CtrlRegen+: multi-round stable_regen (x1/x2/x3) kills
TrustMark (pixel) while VINE (latent) partially survives -> the SAME poisoning
the real CtrlRegen+ n=12 set shows, but now at n=48 for real error bars.

Detectors: VINE-only | TM-only | equal-MRC | conf-MRC(no-train) | OLD head |
NEW head(dead-aug) | NOCROSS head(ablation: cross-agreement feature removed) |
oracle. The NOCROSS ablation isolates WHY the head beats conf-MRC: the cross-
fragment sign-agreement feature is the only observable that flags a CONFIDENTLY-
WRONG dead fragment (own |LLR| can't).
"""
import os, sys, glob, json
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.stats import binom

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr
from _regen_util import build_regen_pipe, stable_regen

torch.manual_seed(0); np.random.seed(0)
KEY = b"v5_key_encoder_master"; dev = "cuda"; CLAMP = 15.0; ALPHA = 0.70; P_DROP = 0.40
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
START = 180                          # fresh: training used fh_0..179
ROUNDS = [1, 2, 3]
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n


class FusionHead(nn.Module):
    def __init__(self, hid=16, use_cross=True):
        super().__init__()
        self.use_cross = use_cross
        self.cal_v = nn.Sequential(nn.Linear(1, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.cal_t = nn.Sequential(nn.Linear(1, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.gate = nn.Sequential(nn.Linear(9, hid), nn.ReLU(), nn.Linear(hid, 2))

    def context(self, av, at):
        def feats(a):
            aa = a.abs()
            return torch.stack([aa.mean(1), aa.std(1), (aa < 0.5).float().mean(1), torch.tanh(aa).mean(1)], 1)
        cross = (torch.sign(av) == torch.sign(at)).float().mean(1, keepdim=True)
        if not self.use_cross: cross = torch.full_like(cross, 0.5)   # ablate: constant
        return torch.cat([feats(av), feats(at), cross], 1)

    def forward(self, av, at):
        B, m = av.shape
        w = F.softplus(self.gate(self.context(av, at)))
        cv = self.cal_v(av.reshape(-1, 1)).reshape(B, m)
        ct = self.cal_t(at.reshape(-1, 1)).reshape(B, m)
        return w[:, 0:1] * cv + w[:, 1:2] * ct, w


def synth_dead(B, m, scale, gen):
    sign = torch.where(torch.rand(B, m, generator=gen, device=dev) > 0.5, 1.0, -1.0)
    mag = -scale * torch.log(torch.rand(B, m, generator=gen, device=dev).clamp_min(1e-6))
    return (sign * mag).clamp(-CLAMP, CLAMP)

def train_head(use_cross):
    d = np.load(os.path.join(REPO, "results/defense/fusion_head_data.npz"), allow_pickle=True)
    AV = np.clip(d["AV"], -CLAMP, CLAMP).astype(np.float32); AT = np.clip(d["AT"], -CLAMP, CLAMP).astype(np.float32)
    TX, IMG = d["TX"], d["IMG"]
    tr = IMG >= 12                       # exclude reserved test-pixel images
    ba_t = ((AT > 0).astype(np.uint8) == TX).mean(1); ba_v = ((AV > 0).astype(np.uint8) == TX).mean(1)
    st = float(np.median(np.abs(AT[ba_t < 0.55]))); sv = float(np.median(np.abs(AV[ba_v < 0.55])))
    h = FusionHead(use_cross=use_cross).to(dev)
    xv = torch.tensor(AV[tr], device=dev); xt = torch.tensor(AT[tr], device=dev)
    y = torch.tensor(TX[tr].astype(np.float32), device=dev); Ntr, m = xv.shape
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    opt = torch.optim.Adam(h.parameters(), lr=2e-3, weight_decay=1e-5)
    for ep in range(1200):
        opt.zero_grad(); av_b, at_b = xv.clone(), xt.clone()
        r = torch.rand(Ntr, generator=gen, device=dev)
        dt = r < P_DROP / 2; dv = (r >= P_DROP / 2) & (r < P_DROP)
        if dt.any(): at_b[dt] = synth_dead(int(dt.sum()), m, st, gen)
        if dv.any(): av_b[dv] = synth_dead(int(dv.sum()), m, sv, gen)
        f, _ = h(av_b, at_b); F.binary_cross_entropy_with_logits(f, y).backward(); opt.step()
    h.eval(); return h

print("training NEW(dead-aug, cross) + NOCROSS(ablation) heads...", flush=True)
new_h = train_head(True); nocross_h = train_head(False)
old_h = FusionHead().to(dev); old_h.load_state_dict(torch.load(os.path.join(REPO, "results/defense/fusion_head.pt"), map_location=dev)); old_h.eval()

# ---------------- build fresh test set ----------------
imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
        sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[START:START + N]
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
pipe = build_regen_pipe()
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def scale_resid(c, w, a):
    C = np.asarray(to512(c), np.float64); W = np.asarray(to512(w), np.float64)
    return Image.fromarray(np.clip(C + a * (W - C), 0, 255).astype(np.uint8))
def decode(att, iid):
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
    av = np.clip(method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP)
    at = np.clip(method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP)
    return av.astype(np.float32), at.astype(np.float32)

cache = {k: {"av": [], "at": [], "tx": [], "id": []} for k in ROUNDS}
for j, fp in enumerate(imgs):
    iid = f"rg_{START + j:05d}"; orig = to512(Image.open(fp).convert("RGB"))
    tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
    v_a = scale_resid(orig, to512(vine.embed_with_target(orig, apply_crypto(tx, pv, Mv))), ALPHA)
    comp = scale_resid(v_a, to512(tm.embed_with_target(v_a, apply_crypto(tx, pt, Mt))), ALPHA)
    cur = comp
    for k in range(1, max(ROUNDS) + 1):
        cur = to512(stable_regen(pipe, cur, seed=3000 + j * 10 + k))
        if k in ROUNDS:
            av, at = decode(cur, iid)
            cache[k]["av"].append(av); cache[k]["at"].append(at); cache[k]["tx"].append(tx.astype(np.uint8)); cache[k]["id"].append(iid)
    if (j + 1) % 8 == 0: print(f"  [{j+1}/{len(imgs)}]", flush=True)

for k in ROUNDS:
    for f in ["av", "at", "tx"]: cache[k][f] = np.array(cache[k][f])
np.savez_compressed(os.path.join(REPO, "results/defense/regen_big_cache.npz"),
                    **{f"r{k}_{f}": cache[k][f] for k in ROUNDS for f in ["av", "at", "tx"]},
                    ids=np.array([cache[ROUNDS[0]]["id"]], dtype=object), rounds=np.array(ROUNDS))

# ---------------- detectors ----------------
e = 1e-6
def norm(a): return a / (a.std(1, keepdims=True) + e)
def oracle_w(av, at, tx):
    bav = ((av > 0).astype(np.uint8) == tx).mean(1)[:, None]; bat = ((at > 0).astype(np.uint8) == tx).mean(1)[:, None]
    wv = np.clip(2 * bav - 1, 0, None); wt = np.clip(2 * bat - 1, 0, None)
    f = wv * norm(av) + wt * norm(at); flat = (wv + wt < e)[:, 0]; f[flat] = (norm(av) + norm(at))[flat]; return f
def conf_mrc(av, at):
    wv = np.abs(av).mean(1, keepdims=True); wt = np.abs(at).mean(1, keepdims=True)
    return wv * norm(av) + wt * norm(at)
def hl(h, av, at):
    with torch.no_grad():
        f, w = h(torch.tensor(av, device=dev), torch.tensor(at, device=dev)); return f.cpu().numpy(), w.cpu().numpy()
def metrics(Fz, tx, ids):
    dets = []
    for i in range(len(Fz)):
        ver = bool(decode_and_verify(Fz[i], ids[i], codec=sb)["detected"])
        zb = ((Fz[i] > 0).astype(np.uint8) == tx[i]).mean()
        dets.append(1.0 if (ver or zb >= TAU) else 0.0)
    return float(np.mean(dets)), float(((Fz > 0).astype(np.uint8) == tx).mean())

print(f"\nn={N}  tau={TAU:.3f}  fresh imgs idx[{START},{START+N})  (det/ba)")
hdr = f"{'regen':6s} | {'VINE':>9s} | {'TM':>9s} | {'equalMRC':>9s} | {'confMRC':>9s} | {'OLD':>9s} | {'NEW':>9s} | {'NOCROSS':>9s} | {'oracle':>9s}"
print(hdr); print("-" * len(hdr))
out = []
for k in ROUNDS:
    av, at, tx = cache[k]["av"], cache[k]["at"], cache[k]["tx"]; ids = cache[k]["id"]
    Fnew, Wn = hl(new_h, av, at); Fold, _ = hl(old_h, av, at); Fnc, Wnc = hl(nocross_h, av, at)
    res = {"VINE": metrics(av, tx, ids), "TM": metrics(at, tx, ids), "equalMRC": metrics(av + at, tx, ids),
           "confMRC": metrics(conf_mrc(av, at), tx, ids), "OLD": metrics(Fold, tx, ids), "NEW": metrics(Fnew, tx, ids),
           "NOCROSS": metrics(Fnc, tx, ids), "oracle": metrics(oracle_w(av, at, tx), tx, ids)}
    def c(kk): return f"{res[kk][0]:.2f}/{res[kk][1]:.2f}"
    print(f"x{k:<5d} | {c('VINE'):>9s} | {c('TM'):>9s} | {c('equalMRC'):>9s} | {c('confMRC'):>9s} | {c('OLD'):>9s} | {c('NEW'):>9s} | {c('NOCROSS'):>9s} | {c('oracle'):>9s}")
    out.append({"round": k, "n": int(len(av)), **{kk: {"det": res[kk][0], "ba": res[kk][1]} for kk in res},
                "wv_new": float(Wn[:, 0].mean()), "wt_new": float(Wn[:, 1].mean())})

json.dump(out, open(os.path.join(REPO, "results/defense/fusion_adv_bign.json"), "w"), indent=2)
print("BIGN_DONE")
