"""Stage 2: train a DEAD-FRAGMENT-AWARE fusion head and test it under ADVANCED
attacks (CtrlRegen+ / UnMarker), where equal-MRC is poisoned by a dead fragment.

Key change vs train_fusion_head.py: a *fragment-dropout augmentation*. With prob
P_DROP, one fragment's aligned LLRs are replaced by a synthetic DEAD sample
(random-sign, low-magnitude noise => bit-acc ~0.5, low confidence) while the
label stays the true codeword. This forces the context-gate to drive a dead
fragment's weight toward 0 and rely on the survivor -- exactly the CtrlRegen+
(TM dead) / crop (VINE dead) regime. The dead magnitude scale is estimated
empirically from the mild `regen`/`crop` samples.

Train data: mild-attack cache, but EXCLUDING the first 12 images (whose pixels
are reused by the real adv test set) -> zero pixel overlap.
Test: held-out mild split  AND  the real adv LLR cache (adv_llr_cache.npz),
attack types CtrlRegen+/UnMarker NEVER seen in training (OOD on attack axis).

Compares: VINE-only | TM-only | equal-MRC | conf-MRC (no-train) | OLD head |
NEW head | oracle, plus FPR on unwatermarked.
"""
import os, sys, glob, json
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from scipy.stats import binom

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload

torch.manual_seed(0); np.random.seed(0)
dev = "cuda"; CLAMP = 15.0; P_DROP = 0.40
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
RESERVE = 12          # first 12 images reused by adv test -> never train on them

d = np.load(os.path.join(REPO, "results/defense/fusion_head_data.npz"), allow_pickle=True)
AV = np.clip(d["AV"], -CLAMP, CLAMP).astype(np.float32); AT = np.clip(d["AT"], -CLAMP, CLAMP).astype(np.float32)
TX, ATK, IMG = d["TX"], d["ATK"], d["IMG"]; n = int(d["n"]); attacks = list(d["attacks"])
cut = int(np.quantile(np.unique(IMG), 0.7))
tr = (IMG >= RESERVE) & (IMG < cut); te = IMG >= cut
print(f"n={n} tau={TAU:.3f} | reserve<{RESERVE} | train={tr.sum()} (imgs[{RESERVE},{cut})) test={te.sum()} (imgs>={cut})", flush=True)

# ---- estimate the DEAD-fragment magnitude scale from real dead samples ----
def per_img_ba(A, TXa): return ((A > 0).astype(np.uint8) == TXa).mean(1)
ba_t = per_img_ba(AT, TX); ba_v = per_img_ba(AV, TX)
dead_t = np.abs(AT[ba_t < 0.55]); dead_v = np.abs(AV[ba_v < 0.55])
scale_t = float(np.median(dead_t)) if dead_t.size > 50 else 0.8
scale_v = float(np.median(dead_v)) if dead_v.size > 50 else 0.8
print(f"dead-mag scale: TM={scale_t:.3f} (n={ (ba_t<0.55).sum() })  VINE={scale_v:.3f} (n={ (ba_v<0.55).sum() })", flush=True)


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


def synth_dead(B, m, scale, gen):
    """random-sign exponential-magnitude LLR -> bit-acc ~0.5, low confidence."""
    sign = torch.where(torch.rand(B, m, generator=gen, device=dev) > 0.5, 1.0, -1.0)
    mag = -scale * torch.log(torch.rand(B, m, generator=gen, device=dev).clamp_min(1e-6))
    return (sign * mag).clamp(-CLAMP, CLAMP)


head = FusionHead().to(dev)
xv = torch.tensor(AV[tr], device=dev); xt = torch.tensor(AT[tr], device=dev)
y = torch.tensor(TX[tr].astype(np.float32), device=dev)
Ntr, m = xv.shape
gen = torch.Generator(device=dev); gen.manual_seed(0)
opt = torch.optim.Adam(head.parameters(), lr=2e-3, weight_decay=1e-5)
for ep in range(1200):
    opt.zero_grad()
    av_b, at_b = xv.clone(), xt.clone()
    # fragment dropout: half drop TM, half drop VINE
    r = torch.rand(Ntr, generator=gen, device=dev)
    drop_t = r < (P_DROP / 2); drop_v = (r >= P_DROP / 2) & (r < P_DROP)
    if drop_t.any(): at_b[drop_t] = synth_dead(int(drop_t.sum()), m, scale_t, gen)
    if drop_v.any(): av_b[drop_v] = synth_dead(int(drop_v.sum()), m, scale_v, gen)
    f, _ = head(av_b, at_b)
    loss = F.binary_cross_entropy_with_logits(f, y)
    loss.backward(); opt.step()
    if (ep + 1) % 300 == 0:
        with torch.no_grad():
            ba = (((f > 0).float() == y).float().mean().item())
        print(f"  ep{ep+1} loss={loss.item():.4f} train-ba(aug)={ba:.3f}", flush=True)
torch.save(head.state_dict(), os.path.join(REPO, "results/defense/fusion_head_adv.pt"))
head.eval()

# ---------------- detectors ----------------
e = 1e-6
def norm(a): return a / (a.std(1, keepdims=True) + e)
def oracle_w(av, at, tx):
    bav = ((av > 0).astype(np.uint8) == tx).mean(1)[:, None]; bat = ((at > 0).astype(np.uint8) == tx).mean(1)[:, None]
    wv = np.clip(2 * bav - 1, 0, None); wt = np.clip(2 * bat - 1, 0, None)
    f = wv * norm(av) + wt * norm(at); flat = (wv + wt < e)[:, 0]; f[flat] = (norm(av) + norm(at))[flat]; return f
def conf_mrc(av, at):                       # no-train: weight by own |LLR| confidence
    wv = np.abs(av).mean(1, keepdims=True); wt = np.abs(at).mean(1, keepdims=True)
    return wv * norm(av) + wt * norm(at)
def head_llr(h, av, at):
    with torch.no_grad():
        f, w = h(torch.tensor(av, device=dev), torch.tensor(at, device=dev)); return f.cpu().numpy(), w.cpu().numpy()

old = FusionHead().to(dev); old.load_state_dict(torch.load(os.path.join(REPO, "results/defense/fusion_head.pt"), map_location=dev)); old.eval()

def metrics(Fz, tx, ids):
    dets = []
    for i in range(len(Fz)):
        ver = bool(decode_and_verify(Fz[i], ids[i], codec=sb)["detected"])
        zb = ((Fz[i] > 0).astype(np.uint8) == tx[i]).mean()
        dets.append(1.0 if (ver or zb >= TAU) else 0.0)
    return float(np.mean(dets)), float(((Fz > 0).astype(np.uint8) == tx).mean())

def report(av, at, tx, ids, title):
    Fnew, Wn = head_llr(head, av, at); Fold, Wo = head_llr(old, av, at)
    dets = {"VINE": metrics(av, tx, ids), "TM": metrics(at, tx, ids), "equal": metrics(av + at, tx, ids),
            "conf": metrics(conf_mrc(av, at), tx, ids), "OLD": metrics(Fold, tx, ids),
            "NEW": metrics(Fnew, tx, ids), "oracle": metrics(oracle_w(av, at, tx), tx, ids)}
    def c(k): return f"{dets[k][0]:.2f}/{dets[k][1]:.2f}"
    print(f"  {title:22s} | V {c('VINE')} | T {c('TM')} | eq {c('equal')} | conf {c('conf')} "
          f"| OLD {c('OLD')} | NEW {c('NEW')} | orc {c('oracle')}  (wv/wt new={Wn[:,0].mean():.2f}/{Wn[:,1].mean():.2f})")
    return {"title": title, "n": int(len(av)), **{k: {"det": dets[k][0], "ba": dets[k][1]} for k in dets},
            "wv_new": float(Wn[:, 0].mean()), "wt_new": float(Wn[:, 1].mean())}

# ---------------- held-out MILD test split ----------------
print("\n[MILD held-out test split]  (det/ba)")
rows = []
av, at, tx, atk = AV[te], AT[te], TX[te], ATK[te]
ids_m = [f"fh_{int(i):05d}" for i in IMG[te]]
for k in attacks:
    s = atk == k
    if s.sum() == 0: continue
    rows.append(report(av[s], at[s], tx[s], [ids_m[j] for j in np.where(s)[0]], k))

# ---------------- real ADVANCED-attack test (OOD attack types) ----------------
print("\n[ADVANCED-attack test: CtrlRegen+/UnMarker (OOD, never trained)]  (det/ba)")
ADV = os.path.join(REPO, "results/defense/adv"); meta = json.load(open(os.path.join(ADV, "meta.json")))
cache = np.load(os.path.join(REPO, "results/defense/adv_llr_cache.npz"), allow_pickle=True)
LAB = {"comp_creg03": "CtrlRegen+ s=0.3", "comp_creg05": "CtrlRegen+ s=0.5", "comp_creg07": "CtrlRegen+ s=0.7",
       "comp_unmark": "UnMarker (def)", "comp_um_nocrop_l": "UnMarker spec x400", "comp_um_nocrop_h": "UnMarker spec x2500",
       "comp_um_c95": "UnMarker crop5%", "comp_um_c90": "UnMarker crop10%", "comp_um_c85": "UnMarker crop15%"}
adv_rows = []
for dd in list(cache["dirs"]):
    av, at, tx = cache[f"{dd}_av"], cache[f"{dd}_at"], cache[f"{dd}_tx"]
    ids = [meta["items"][k]["image_id"] for k in range(len(av))]   # trailing files missing -> first n items
    adv_rows.append(report(av, at, tx, ids, LAB.get(dd, dd)))

# ---------------- FPR on unwatermarked held-out ----------------
print("\n[FPR] decoding UNwatermarked held-out images...", flush=True)
from PIL import Image
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr
KEY = b"v5_key_encoder_master"
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
neg_imgs = (sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen/*.png"))) +
            sorted(glob.glob(os.path.join(REPO, "results/defense/raven_gen2/*.png"))))[180:215]
NV, NT, NTX, NID = [], [], [], []
for j, fp in enumerate(neg_imgs):
    iid = f"neg_{j:05d}"; im = Image.open(fp).convert("RGB").resize((512, 512))
    pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
    NV.append(np.clip(method_soft_to_codeword_llr(vine.raw_probs(im), pv, Mv, kind="prob", n_codeword=sb.n), -CLAMP, CLAMP))
    NT.append(np.clip(method_soft_to_codeword_llr(tm.raw_logits(im), pt, Mt, kind="logit", n_codeword=sb.n), -CLAMP, CLAMP))
    NTX.append(sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))); NID.append(iid)
NV, NT, NTX = np.array(NV, np.float32), np.array(NT, np.float32), np.array(NTX, np.uint8)
fn, _ = head_llr(head, NV, NT)
print(f"[FPR] equal-MRC={metrics(NV+NT,NTX,NID)[0]:.3f}  conf-MRC={metrics(conf_mrc(NV,NT),NTX,NID)[0]:.3f}  "
      f"NEW-head={metrics(fn,NTX,NID)[0]:.3f}  (n={len(NV)} unwatermarked)")

json.dump({"mild": rows, "adv": adv_rows}, open(os.path.join(REPO, "results/defense/fusion_adv_train_eval.json"), "w"), indent=2)
print("ADVTRAIN_DONE")
