"""Phase 3+4: train the learned fusion head (frozen encoders) and evaluate.

Head = context-gated, calibrated MRC (permutation-invariant, per-position shared):
  g = global reliability/agreement context (per-fragment stats + sign-agreement)
  w = softplus(gate(g))               # per-fragment attack-adaptive weight
  f[j] = w_v * cal_v(a_v[j]) + w_t * cal_t(a_t[j])   # cal_* = learned scalar calibrators
Output f = codeword LLR -> still goes through Chase-BCH + exact 37-bit match, so the
crypto-ID FPR (2^-37) is preserved by construction. Trains on cached aligned LLRs.

Eval: positives (held-out imgs) head vs equal-MRC vs oracle_w per attack (det+ba);
negatives (UNwatermarked held-out imgs) -> FPR for head vs equal-MRC.
"""
import os, sys, glob
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from scipy.stats import binom

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload

torch.manual_seed(0)
dev = "cuda"
CLAMP = 15.0
d = np.load(os.path.join(REPO, "results/defense/fusion_head_data.npz"), allow_pickle=True)
AV = np.clip(d["AV"], -CLAMP, CLAMP); AT = np.clip(d["AT"], -CLAMP, CLAMP)
TX, ATK, IMG = d["TX"], d["ATK"], d["IMG"]
n = int(d["n"]); sb = ShortenedBCH(); tau = float(binom.ppf(0.99, n, 0.5) + 1) / n
attacks = list(d["attacks"])
cut = int(np.quantile(np.unique(IMG), 0.7)); tr = IMG < cut; te = IMG >= cut
print(f"n={n} tau={tau:.3f} | train={tr.sum()} test={te.sum()} (test imgs>= {cut})", flush=True)


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
        w = F.softplus(self.gate(self.context(av, at)))         # (B,2)
        cv = self.cal_v(av.reshape(-1, 1)).reshape(B, m)
        ct = self.cal_t(at.reshape(-1, 1)).reshape(B, m)
        return w[:, 0:1] * cv + w[:, 1:2] * ct


head = FusionHead().to(dev)
xv = torch.tensor(AV[tr], device=dev); xt = torch.tensor(AT[tr], device=dev)
y = torch.tensor(TX[tr].astype(np.float32), device=dev)
opt = torch.optim.Adam(head.parameters(), lr=2e-3, weight_decay=1e-5)
for ep in range(600):
    opt.zero_grad()
    f = head(xv, xt)
    loss = F.binary_cross_entropy_with_logits(f, y)
    loss.backward(); opt.step()
    if (ep + 1) % 150 == 0:
        ba = ((f > 0).float() == y).float().mean().item()
        print(f"  ep{ep+1} loss={loss.item():.4f} train-ba={ba:.3f}", flush=True)
torch.save(head.state_dict(), os.path.join(REPO, "results/defense/fusion_head.pt"))
head.eval()

# ---------------- positive eval (cached test split) ----------------
e = 1e-6
def norm(a): return a / (a.std(1, keepdims=True) + e)
def oracle_w(av, at, tx):
    bav = ((av > 0).astype(np.uint8) == tx).mean(1)[:, None]; bat = ((at > 0).astype(np.uint8) == tx).mean(1)[:, None]
    wv = np.clip(2 * bav - 1, 0, None); wt = np.clip(2 * bat - 1, 0, None)
    f = wv * norm(av) + wt * norm(at); flat = (wv + wt < e)[:, 0]; f[flat] = (norm(av) + norm(at))[flat]; return f
def head_llr(av, at):
    with torch.no_grad():
        return head(torch.tensor(av, device=dev), torch.tensor(at, device=dev)).cpu().numpy()

def detect_rate(F, tx, img):
    out = []
    for i in range(len(F)):
        ver = bool(decode_and_verify(F[i], f"fh_{int(img[i]):05d}", codec=sb)["detected"])
        zb = ((F[i] > 0).astype(np.uint8) == tx[i]).mean()
        out.append(1.0 if (ver or zb >= tau) else 0.0)
    return float(np.mean(out))

av, at, tx, atk, img = AV[te], AT[te], TX[te], ATK[te], IMG[te]
F_eq, F_or, F_hd = av + at, oracle_w(av, at, tx), head_llr(av, at)
print(f"\n{'attack':9s} |   equal-MRC   |    oracle_w   |  LEARNED-head   (det / ba)")
ov = {"equal": [], "oracle": [], "head": []}
for k in attacks:
    s = atk == k
    if s.sum() == 0: continue
    def cell(Fz, key):
        det = detect_rate(Fz[s], tx[s], img[s]); ba = ((Fz[s] > 0).astype(np.uint8) == tx[s]).mean()
        ov[key].append(det); return f"{det:.2f}/{ba:.2f}"
    print(f"{k:9s} |  {cell(F_eq,'equal'):>11s}  |  {cell(F_or,'oracle'):>11s}  |  {cell(F_hd,'head'):>11s}")
print(f"{'OVERALL':9s} |  {np.mean(ov['equal']):.3f}        |  {np.mean(ov['oracle']):.3f}        |  {np.mean(ov['head']):.3f}")

# ---------------- negative eval (FPR on UNwatermarked held-out imgs) ----------------
print("\n[FPR] decoding UNwatermarked held-out images...", flush=True)
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr
from PIL import Image
KEY = b"v5_key_encoder_master"
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
neg_imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
            sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:230]
NV, NT, NIMG, NTX = [], [], [], []
for j, fp in enumerate(neg_imgs):
    image_id = f"neg_{j:05d}"
    im = Image.open(fp).convert("RGB").resize((512, 512))
    pv, Mv = vine.get_perm_M(image_id); pt, Mt = tm.get_perm_M(image_id)
    NV.append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(im), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP))
    NT.append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(im), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP))
    NTX.append(sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))); NIMG.append(image_id)
NV, NT, NTX = np.array(NV, np.float32), np.array(NT, np.float32), np.array(NTX, np.uint8)
def fpr(F):
    out = []
    for i in range(len(F)):
        ver = bool(decode_and_verify(F[i], NIMG[i], codec=sb)["detected"])
        zb = ((F[i] > 0).astype(np.uint8) == NTX[i]).mean()
        out.append(1.0 if (ver or zb >= tau) else 0.0)
    return float(np.mean(out))
print(f"[FPR] equal-MRC={fpr(NV+NT):.3f}  LEARNED-head={fpr(head_llr(NV,NT)):.3f}  (n={len(NV)} unwatermarked)")
print("TRAIN_DONE")
