"""Show WHY similar mean-ba gives very different detection: decode per-image bit-acc for
VINE vs Composite on the existing CtrlRegen+ attacked images, plot vs tau=0.63.
Detection (at this op point) = fraction of images with ba >= tau (crypto-ID dead at ~30 errs)."""
import os, sys, glob, json
import numpy as np
from PIL import Image
from scipy.stats import binom
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.payload import image_id_to_payload
KEY = b"v5_key_encoder_master"; dev = "cuda"
sb = ShortenedBCH(); TAU = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n
vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)
ADV = os.path.join(REPO, "results/defense/adv"); meta = json.load(open(os.path.join(ADV, "meta.json")))
def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def ba_list(method, adir):
    out = []
    for it in meta["items"]:
        fp = os.path.join(adir, f"img_{it['i']:05d}.png")
        if not os.path.exists(fp): continue
        att = to512(Image.open(fp).convert("RGB")); iid = it["image_id"]
        tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
        pv, Mv = vine.get_perm_M(iid); pt, Mt = tm.get_perm_M(iid)
        if method == "vine":
            llr = method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n)
        else:
            av = method_soft_to_codeword_llr(vine.raw_probs(att), pv, Mv, kind="prob", n_codeword=sb.n)
            at = method_soft_to_codeword_llr(tm.raw_logits(att), pt, Mt, kind="logit", n_codeword=sb.n)
            llr = fuse_llrs({"v": av, "t": at}, n_codeword=sb.n)
        out.append(float(np.mean(llr_to_bits(llr) == tx)))
    return out
data = {}
for tag, cd, vd in [("CtrlRegen+ s=0.5", "comp_creg05", "vine_creg05"), ("CtrlRegen+ s=0.7", "comp_creg07", "vine_creg07")]:
    data[tag] = {"Composite": ba_list("comp", os.path.join(ADV, cd)), "VINE": ba_list("vine", os.path.join(ADV, vd))}
    print(f"[{tag}]", flush=True)
    for m in ["VINE", "Composite"]:
        a = np.array(data[tag][m]); print(f"   {m:9s} mean ba {a.mean():.3f} | per-img {np.round(a,2).tolist()} | det(>= {TAU:.2f}) = {(a>=TAU).mean():.2f}")
json.dump({"tau": TAU, "data": data}, open(os.path.join(REPO, "results/defense/per_image_ba.json"), "w"), indent=2)
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
for k, tag in enumerate(data):
    for j, (m, col) in enumerate([("VINE", "#4c9f70"), ("Composite", "#d1495b")]):
        a = np.array(data[tag][m]); y = np.full_like(a, j) + np.linspace(-0.12, 0.12, len(a))
        ax[k].scatter(a, y, s=90, color=col, ec="#222", zorder=3, label=f"{m} (mean {a.mean():.2f}, det {(a>=TAU).mean():.2f})")
        ax[k].scatter([a.mean()], [j], marker="|", s=2000, color="black", zorder=4)
    ax[k].axvline(TAU, color="crimson", ls="--", lw=2); ax[k].text(TAU+0.005, 1.45, f"τ={TAU:.2f}", color="crimson", fontweight="bold")
    ax[k].set_yticks([0, 1]); ax[k].set_yticklabels(["VINE", "Composite"]); ax[k].set_xlabel("per-image bit-acc")
    ax[k].set_title(tag, fontweight="bold"); ax[k].set_ylim(-0.6, 1.7); ax[k].grid(axis="x", alpha=.3); ax[k].legend(loc="upper left", fontsize=8)
fig.suptitle("Detection = fraction of images with ba ≥ τ (not the mean). Dead-TM noise pushes Composite's borderline images below τ.", fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(REPO, "results/defense/per_image_ba.png")
plt.savefig(out, dpi=130); print(f"[saved] {out}\nPERIMG_DONE")
