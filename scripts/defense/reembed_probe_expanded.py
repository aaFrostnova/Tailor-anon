"""Expanded re-embed probe (STABLE regen) — many attacks, multi-round re-embedding, larger n, find regularities.

For each image and attack a, paired (same image, same regen seed):
  A_w = attack(W),  A_c = attack(C)
  direct detect: ba_w, ba_c
  multi-round re-embed (K rounds, accumulating): on A_w with RIGHT id -> incr_w_right[1..K];
     on A_c with RIGHT id -> incr_c_right[1..K]; on A_w with WRONG id -> incr_w_wrong[1]
  probe AUROC = separability of incr_w_right[1] (small => watermarked) vs incr_c_right[1]
  divergence metric (regen-family only): surviving-residual RMS = ||regen(W)-regen(C)|| (small=converge, large=diverge)
Outputs JSON + bar (per-attack probe) + multi-round trend figure.
"""
import argparse, glob, io, json, os, sys
import numpy as np, torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "defense"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.vine_crypto_wrapper import VineCryptoWrapper
from _regen_util import build_regen_pipe, stable_regen
KEY = b"v5_key_encoder_master"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
RES = 512
ATTACKS = ["clean", "jpeg", "noise", "blur", "regen", "rinse2x", "vae_c", "crop75", "rot9", "hflip"]
REGEN_FAM = {"regen", "rinse2x", "vae_c"}


def arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def rms(x): return float(np.sqrt(np.mean(x ** 2)))
def emb(vine, pil, iid):
    W = vine.embed(pil, iid); return W if W.size == (RES, RES) else W.resize((RES, RES))
def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg); c = 0.0
    for a in pos:
        for b in neg: c += (a < b) + 0.5 * (a == b)
    return c / (len(pos) * len(neg) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default="results/defense/reembed_probe_expanded.json")
    args = ap.parse_args(); dev = "cuda"; K = args.rounds
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=100, device=dev)
    pipe = build_regen_pipe()
    from wmattacker import VAEWMAttacker, GaussianBlurAttacker, GaussianNoiseAttacker, JPEGAttacker
    import tempfile
    tmp = tempfile.mkdtemp()
    sig = {"jpeg": JPEGAttacker(quality=25), "blur": GaussianBlurAttacker(5, 1), "noise": GaussianNoiseAttacker(std=0.05)}
    try: vae_c = VAEWMAttacker("cheng2020-anchor", quality=3, metric="mse", device=dev)
    except Exception as e: vae_c = None; print(f"[vae_c] skip: {e}", flush=True)

    def sig_apply(att, pil):
        ip = os.path.join(tmp, "i.png"); op = os.path.join(tmp, "o.png")
        pil.save(ip); att.attack([ip], [op]); return Image.open(op).convert("RGB").resize((RES, RES))
    def geo(name, pil):
        if name == "hflip": return pil.transpose(Image.FLIP_LEFT_RIGHT)
        if name == "crop75":
            W, H = pil.size; f = 0.75; cw, ch = int(W * f), int(H * f); l, t = (W - cw) // 2, (H - ch) // 2
            return pil.crop((l, t, l + cw, t + ch)).resize((W, H), Image.BICUBIC)
        if name == "rot9":
            a = np.array(pil); pad = max(pil.size) // 2
            big = Image.fromarray(np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")).rotate(9, resample=Image.BICUBIC)
            bw, bh = big.size; return big.crop(((bw - 512) // 2, (bh - 512) // 2, (bw - 512) // 2 + 512, (bh - 512) // 2 + 512))

    def attack(name, W, C, seed):
        if name == "clean": return W, C
        if name in sig: return sig_apply(sig[name], W), sig_apply(sig[name], C)
        if name == "vae_c":
            return (sig_apply(vae_c, W), sig_apply(vae_c, C)) if vae_c else (W, C)
        if name in ("regen", "rinse2x"):
            kk = 1 if name == "regen" else 2
            aw, ac = W, C
            for r in range(kk):
                aw = stable_regen(pipe, aw, seed + r, denoise_steps=args.steps)
                ac = stable_regen(pipe, ac, seed + r, denoise_steps=args.steps)
            return aw, ac
        return geo(name, W), geo(name, C)

    def multi_reembed(pil, iid, k):
        cur = arr(pil); out = []
        for _ in range(k):
            nxt = arr(emb(vine, Image.fromarray(cur.astype(np.uint8)), iid)); out.append(rms(nxt - cur)); cur = nxt
        return out

    files = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n]
    D = {a: {"ba_w": [], "ba_c": [], "iwr": [], "iww": [], "icr": [], "surv": []} for a in ATTACKS}
    for idx, fp in enumerate(files):
        rid = f"exp_{idx:05d}"; wid = f"WR_{idx:05d}"
        C = Image.open(fp).convert("RGB").resize((RES, RES))
        W = emb(vine, C, rid)
        for a in ATTACKS:
            seed = 1234 + 100 * idx
            Aw, Ac = attack(a, W, C, seed)
            D[a]["ba_w"].append(vine.detect(Aw, rid)["bit_accuracy"]); D[a]["ba_c"].append(vine.detect(Ac, rid)["bit_accuracy"])
            D[a]["iwr"].append(multi_reembed(Aw, rid, K)); D[a]["icr"].append(multi_reembed(Ac, rid, K))
            D[a]["iww"].append(multi_reembed(Aw, wid, 1)[0])
            D[a]["surv"].append(rms(arr(Aw) - arr(Ac)) if a in REGEN_FAM else float("nan"))
        print(f"  [{idx+1}/{len(files)}] done", flush=True)

    # summarize
    S = {}
    for a in ATTACKS:
        d = D[a]; iwr1 = [v[0] for v in d["iwr"]]; icr1 = [v[0] for v in d["icr"]]
        S[a] = {"ba_w": float(np.mean(d["ba_w"])), "ba_c": float(np.mean(d["ba_c"])),
                "iwr_rounds": [float(np.mean([v[r] for v in d["iwr"]])) for r in range(K)],
                "icr_rounds": [float(np.mean([v[r] for v in d["icr"]])) for r in range(K)],
                "iww1": float(np.mean(d["iww"])), "iwr1": float(np.mean(iwr1)), "icr1": float(np.mean(icr1)),
                "probe_auroc": auroc(iwr1, icr1),
                "surv_rms": float(np.nanmean(d["surv"])) if a in REGEN_FAM else None}
    json.dump({"n": len(files), "rounds": K, "steps": args.steps, "attacks": ATTACKS, "summary": S, "raw": D},
              open(os.path.join(REPO, args.out), "w"), indent=2)

    print(f"\n=== expanded re-embed probe (STABLE {args.steps}-step, n={len(files)}, rounds={K}) ===")
    print(f"{'attack':<9}{'ba_w':>7}{'ba_c':>7}{'iwr1':>7}{'iww1':>7}{'icr1':>7}{'AUROC':>8}{'surv_rms':>10}")
    for a in ATTACKS:
        s = S[a]; sv = f"{s['surv_rms']:.1f}" if s["surv_rms"] is not None else "-"
        print(f"{a:<9}{s['ba_w']:>7.3f}{s['ba_c']:>7.3f}{s['iwr1']:>7.2f}{s['iww1']:>7.2f}{s['icr1']:>7.2f}{s['probe_auroc']:>8.3f}{sv:>10}")

    # bar fig
    x = np.arange(len(ATTACKS)); w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, [S[a]["iwr1"] for a in ATTACKS], w, label="incr_w_RIGHT (mark+aligned)", color="#1f77b4")
    ax.bar(x, [S[a]["iww1"] for a in ATTACKS], w, label="incr_w_wrong", color="#ff7f0e")
    ax.bar(x + w, [S[a]["icr1"] for a in ATTACKS], w, label="incr_c_right (no mark)", color="#888")
    for i, a in enumerate(ATTACKS): ax.text(i, max(S[a]["icr1"], S[a]["iww1"]) + 0.05, f"{S[a]['probe_auroc']:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(ATTACKS, rotation=20); ax.set_ylabel("re-embed incr RMS (round1)")
    ax.set_title("Re-embed probe across attacks (number=probe AUROC). Gap incr_c>incr_w => residual mark survives")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(REPO, "results/figures/fig_reembed_expanded_bars.png"), dpi=130, bbox_inches="tight")

    # multi-round trend
    fig, ax = plt.subplots(figsize=(9, 5)); rr = np.arange(1, K + 1)
    for a in ["clean", "jpeg", "regen", "vae_c", "crop75"]:
        ax.plot(rr, S[a]["iwr_rounds"], "o-", label=f"{a} W(right)")
        ax.plot(rr, S[a]["icr_rounds"], "s--", alpha=0.6, label=f"{a} C(right)")
    ax.set_xlabel("re-embed round"); ax.set_ylabel("incremental RMS"); ax.set_xticks(rr)
    ax.set_title("Multi-round re-embed increment: watermarked (solid) starts lower & converges (idempotency)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(REPO, "results/figures/fig_reembed_expanded_rounds.png"), dpi=130, bbox_inches="tight")
    print("[figs] fig_reembed_expanded_bars.png + fig_reembed_expanded_rounds.png\nEXPANDED_PROBE_DONE")


if __name__ == "__main__":
    main()
