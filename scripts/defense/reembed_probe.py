"""Re-embedding probe: does an attacked WATERMARKED image still carry enough signal that re-embedding
the SAME payload costs LESS (smaller incremental residual) than re-embedding on an attacked CLEAN image?

VINE is partially idempotent (re-embedding an already-marked image adds less). Hypothesis: an attacked
watermarked image A_w retains residual mark, so embed(A_w, right_id)-A_w is SMALL; an attacked clean image
A_c has none, so embed(A_c, right_id)-A_c is LARGE. Control: embed(A_w, WRONG_id)-A_w should be LARGE too
(residual is payload-specific, not "any watermark"). Paired (same image, same attack, same regen seed).

Per attack, report mean: ba_w (direct detect on A_w), ba_c, and the three re-embed increments, plus the
separability AUROC of the probe (incr small => watermarked) — especially where DIRECT detect has failed.
"""
import argparse, glob, io, os, sys, tempfile
import numpy as np, torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.vine_crypto_wrapper import VineCryptoWrapper
KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
RES = 512
ATTACKS = ["clean", "jpeg", "regen", "rinse2x", "rinse4x"]
SHOW = ["jpeg", "regen", "rinse4x"]


def arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def rms(x): return float(np.sqrt(np.mean(x ** 2)))
def emb(vine, pil, iid):
    W = vine.embed(pil, iid)
    return arr(W if W.size == (RES, RES) else W.resize((RES, RES)))


def auroc(pos, neg):  # pos = incr on watermarked (want small), neg = incr on clean
    pos, neg = np.asarray(pos), np.asarray(neg); c = 0
    for a in pos:
        for b in neg:
            c += (a < b) + 0.5 * (a == b)       # smaller incr => watermarked
    return c / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_stats", type=int, default=8); ap.add_argument("--heroes", nargs="+", type=int, default=[0])
    args = ap.parse_args(); dev = "cuda"
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=100, device=dev)
    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60); tmp = tempfile.mkdtemp()

    def regen1(pil, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        ip = os.path.join(tmp, "i.png"); op = os.path.join(tmp, "o.png")
        pil.save(ip); regen.attack([ip], [op]); return Image.open(op).convert("RGB").resize((RES, RES))

    def attack(pil, name, seed):
        if name == "clean": return pil
        if name == "jpeg":
            buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=25)
            return Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((RES, RES))
        k = {"regen": 1, "rinse2x": 2, "rinse4x": 4}[name]; cur = pil
        for r in range(k): cur = regen1(cur, seed + r)
        return cur

    files = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n_stats]
    D = {a: {"iwr": [], "iww": [], "icr": [], "baw": [], "bac": []} for a in ATTACKS}
    hero = {h: {} for h in args.heroes}

    for idx, fp in enumerate(files):
        rid = f"probe_{idx:05d}"; wid = f"WRONG_{idx:05d}"
        C = Image.open(fp).convert("RGB").resize((RES, RES))
        W = vine.embed(C, rid); W = W if W.size == (RES, RES) else W.resize((RES, RES))
        for a in ATTACKS:
            seed = 1234 + 100 * idx
            Aw = attack(W, a, seed); Ac = attack(C, a, seed)
            Awa, Aca = arr(Aw), arr(Ac)
            iwr = rms(emb(vine, Aw, rid) - Awa); iww = rms(emb(vine, Aw, wid) - Awa)
            icr = rms(emb(vine, Ac, rid) - Aca)
            baw = vine.detect(Aw, rid)["bit_accuracy"]; bac = vine.detect(Ac, rid)["bit_accuracy"]
            D[a]["iwr"].append(iwr); D[a]["iww"].append(iww); D[a]["icr"].append(icr)
            D[a]["baw"].append(baw); D[a]["bac"].append(bac)
            if idx in args.heroes and a in SHOW:
                hero[idx][a] = {"Aw": Awa, "Ac": Aca,
                                "rw": emb(vine, Aw, rid) - Awa, "rc": emb(vine, Ac, rid) - Aca,
                                "iwr": iwr, "icr": icr}
        print(f"  img{idx:05d} done", flush=True)

    print(f"\n=== re-embedding probe (VINE, n={args.n_stats}) ===")
    print(f"{'attack':<9}{'direct ba_w':>12}{'ba_c':>7}{'incr_w_RIGHT':>14}{'incr_w_wrong':>14}{'incr_c_right':>14}{'probe_AUROC':>13}")
    for a in ATTACKS:
        d = D[a]; au = auroc(d["iwr"], d["icr"])
        print(f"{a:<9}{np.mean(d['baw']):>12.3f}{np.mean(d['bac']):>7.3f}"
              f"{np.mean(d['iwr']):>14.2f}{np.mean(d['iww']):>14.2f}{np.mean(d['icr']):>14.2f}{au:>13.3f}")
    print("\nHypothesis holds if incr_w_RIGHT < incr_c_right ~= incr_w_wrong, and probe_AUROC > direct detection"
          " where ba_w ~ 0.5.")

    # ---- bar chart ----
    x = np.arange(len(ATTACKS)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, [np.mean(D[a]["iwr"]) for a in ATTACKS], w, label="incr_w_RIGHT (mark+aligned)", color="#1f77b4")
    ax.bar(x, [np.mean(D[a]["iww"]) for a in ATTACKS], w, label="incr_w_wrong (mark, wrong id)", color="#ff7f0e")
    ax.bar(x + w, [np.mean(D[a]["icr"]) for a in ATTACKS], w, label="incr_c_right (no mark)", color="#888888")
    ax.set_xticks(x); ax.set_xticklabels(ATTACKS); ax.set_ylabel("re-embed incremental residual RMS")
    ax.set_title("Re-embed probe: marked+correct-id re-embedding costs LESS (residual mark survives the attack)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(REPO, "results/figures/fig_reembed_probe_bars.png"), dpi=130, bbox_inches="tight")
    print("[fig] fig_reembed_probe_bars.png")

    # ---- residual visualization for hero ----
    h = args.heroes[0]; AMP = 22; cols = SHOW
    fig, axes = plt.subplots(4, len(cols), figsize=(3.2 * len(cols), 12))
    rows = ["A_w (attacked watermarked)", "re-embed incr (RIGHT id) x22", "A_c (attacked clean)", "re-embed incr (no mark) x22"]
    for c, a in enumerate(cols):
        hd = hero[h][a]
        axes[0, c].imshow(np.clip(hd["Aw"], 0, 255).astype(np.uint8))
        axes[1, c].imshow(np.clip(hd["rw"] * AMP + 128, 0, 255).astype(np.uint8))
        axes[2, c].imshow(np.clip(hd["Ac"], 0, 255).astype(np.uint8))
        axes[3, c].imshow(np.clip(hd["rc"] * AMP + 128, 0, 255).astype(np.uint8))
        axes[0, c].set_title(f"{a}", fontsize=11)
        axes[1, c].set_title(f"incr_w={hd['iwr']:.2f}", fontsize=9)
        axes[3, c].set_title(f"incr_c={hd['icr']:.2f}", fontsize=9)
    for r in range(4):
        axes[r, 0].set_ylabel(rows[r], fontsize=8)
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"img{h:05d}: re-embed increment is FAINTER on attacked-watermarked (row2) than attacked-clean (row4) "
                 f"=> residual mark survives", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(os.path.join(REPO, "results/figures/fig_reembed_probe_resid.png"), dpi=125, bbox_inches="tight")
    print("[fig] fig_reembed_probe_resid.png\nREEMBED_PROBE_DONE")


if __name__ == "__main__":
    main()
