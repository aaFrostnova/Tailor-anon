"""Test reactive projection onto the regen-preserved subspace.

Two arms:
  ARM 1 (positive control, flexible carrier): a self-contained spread-spectrum (SS) watermark whose carrier
    we control. Compare WHITE carriers (full band) vs REACTIVELY-PROJECTED carriers (regen-kept band only),
    at MATCHED PSNR. If projection works, regen bit-acc and survival-ratio go UP for the projected version.
  ARM 2 (negative control, band-locked carrier): TrustMark's own residual projected to the kept band.
    Its signal lives in the regen-KILLED band, so projection should DESTROY decodability -- you cannot move a
    band-locked watermark into the kept band by projection.

Metrics per arm: clean bit-acc, regen bit-acc, survival-ratio (decode-free), PSNR vs original.
"""
import argparse, glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image
from scipy import ndimage

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.regen_ortho_projection import measure_survival_spectrum, build_kept_mask, project, survival_ratio
from src.trustmark_fragment import TrustMarkFragment
KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
RES = 512


def arr(pil): return np.asarray(pil.convert("RGB").resize((RES, RES)), np.float32)
def to_pil(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def psnr(a, b):
    m = np.mean((a - b) ** 2); return float(99.0 if m < 1e-9 else 10 * np.log10(255.0 ** 2 / m))


def make_carriers(key, n_bits, H, W):
    cs = []
    base = int.from_bytes(key[:4], "little")
    for i in range(n_bits):
        rs = np.random.RandomState((base + 1000 * i) % (2 ** 31))
        c = rs.randn(H, W).astype(np.float64)
        c -= c.mean(); c /= (np.linalg.norm(c) + 1e-9)
        cs.append(c)
    return cs


def embed_ss(x, eff_carriers, bits, target_psnr):
    H, W, _ = x.shape
    d1 = np.zeros((H, W))
    for c, b in zip(eff_carriers, bits):
        d1 += (2 * int(b) - 1) * c
    d3 = np.repeat(d1[:, :, None], 3, axis=2)
    rmse_t = 255.0 / (10 ** (target_psnr / 20.0))
    d3 *= rmse_t / (np.sqrt(np.mean(d3 ** 2)) + 1e-12)
    xw = np.clip(x + d3, 0, 255)
    return xw, d3


def decode_ss(att, eff_carriers, hp_sigma=12.0):
    g = att.mean(axis=2)
    if hp_sigma: g = g - ndimage.gaussian_filter(g, hp_sigma)   # drop very-low-freq image content
    return np.array([1 if np.sum(g * c) > 0 else 0 for c in eff_carriers], dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=8); ap.add_argument("--n_bits", type=int, default=64)
    ap.add_argument("--psnr", type=float, default=38.0); ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args(); dev = "cuda"
    tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=100, model_type="B", device=dev)
    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    tmp = tempfile.mkdtemp()

    def regen_seeded(a, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        ip = os.path.join(tmp, "i.png"); op = os.path.join(tmp, "o.png")
        to_pil(a).save(ip); regen.attack([ip], [op])
        return arr(Image.open(op).convert("RGB"))

    carriers = make_carriers(KEY, args.n_bits, RES, RES)
    files = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n_images]
    rng = np.random.RandomState(0)
    S_attack = 777
    band = {"low": [], "mid": [], "high": []}; kept_frac = []
    R = {k: {"clean": [], "regen": [], "sr": [], "psnr": []}
         for k in ["SS_white", "SS_reactive", "TM_base", "TM_proj"]}

    for i, fp in enumerate(files):
        x = arr(Image.open(fp).convert("RGB"))
        bits = rng.randint(0, 2, args.n_bits).astype(np.uint8)
        bits100 = rng.randint(0, 2, 100).astype(np.uint8)

        rho, S = measure_survival_spectrum(x, regen_seeded, probe_seed=1234, n_probes=8, probe_std=3.0)
        mask = build_kept_mask(S, RES, RES, mode="hard", thresh=args.thresh)
        kept_frac.append(float(mask.mean()))
        band["low"].append(float(S[rho < 0.10].mean()))
        band["mid"].append(float(S[(rho >= 0.10) & (rho < 0.40)].mean()))
        band["high"].append(float(S[rho >= 0.40].mean()))

        eff_white = carriers
        eff_proj = [project(np.repeat(c[:, :, None], 3, axis=2), mask)[:, :, 0] for c in carriers]
        # renorm each projected carrier to unit norm for fair correlation decode
        eff_proj = [c / (np.linalg.norm(c) + 1e-9) for c in eff_proj]

        base_att = regen_seeded(x, S_attack)                 # regen(x) for SR baseline

        # ARM 1: SS white vs reactive
        for k, eff in [("SS_white", eff_white), ("SS_reactive", eff_proj)]:
            xw, d3 = embed_ss(x, eff, bits, args.psnr)
            att = regen_seeded(xw, S_attack)
            R[k]["clean"].append(float(np.mean(decode_ss(xw, eff) == bits)))
            R[k]["regen"].append(float(np.mean(decode_ss(att, eff) == bits)))
            R[k]["sr"].append(survival_ratio(d3, base_att, att))
            R[k]["psnr"].append(psnr(xw, x))

        # ARM 2: TrustMark base vs projected
        xw_tm = arr(tm.embed_with_target(to_pil(x), bits100).resize((RES, RES)))
        d_tm = xw_tm - x
        xw_tm_proj = np.clip(x + project(d_tm, mask, renorm=True), 0, 255)
        for k, xw in [("TM_base", xw_tm), ("TM_proj", xw_tm_proj)]:
            att = regen_seeded(xw, S_attack)
            R[k]["clean"].append(float(np.mean((tm.raw_logits(to_pil(xw)) > 0).astype(np.uint8) == bits100)))
            R[k]["regen"].append(float(np.mean((tm.raw_logits(to_pil(att)) > 0).astype(np.uint8) == bits100)))
            R[k]["sr"].append(survival_ratio(xw - x, base_att, att))
            R[k]["psnr"].append(psnr(xw, x))

        print(f"  [{i+1}/{len(files)}] kept={kept_frac[-1]:.3f}  "
              f"SSwhite r={R['SS_white']['regen'][-1]:.2f} | SSreact r={R['SS_reactive']['regen'][-1]:.2f} | "
              f"TMbase r={R['TM_base']['regen'][-1]:.2f} | TMproj r={R['TM_proj']['regen'][-1]:.2f}", flush=True)

    print(f"\n=== reactive regen-orthogonal projection (n={len(files)}, n_bits SS={args.n_bits}, "
          f"PSNR target={args.psnr}, kept thresh S>={args.thresh}) ===")
    print(f"regen survival spectrum: low(<0.1)={np.mean(band['low']):.2f}  "
          f"mid(0.1-0.4)={np.mean(band['mid']):.2f}  high(>0.4)={np.mean(band['high']):.2f}   "
          f"kept-frequency fraction={np.mean(kept_frac):.3f}")
    print(f"\n{'condition':<14}{'clean_ba':>10}{'regen_ba':>10}{'surv_ratio':>12}{'PSNR':>9}")
    for k in ["SS_white", "SS_reactive", "TM_base", "TM_proj"]:
        print(f"{k:<14}{np.mean(R[k]['clean']):>10.3f}{np.mean(R[k]['regen']):>10.3f}"
              f"{np.mean(R[k]['sr']):>12.3f}{np.mean(R[k]['psnr']):>9.2f}")
    print("\nARM1 positive control: SS_white vs SS_reactive (same carrier, projected to regen-kept band)")
    print("ARM2 negative control: TM_base vs TM_proj (band-locked high-freq mark cannot move to kept band)")
    print("REGEN_ORTHO_DONE")


if __name__ == "__main__":
    main()
