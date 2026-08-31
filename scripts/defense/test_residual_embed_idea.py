"""Test the user's idea: embed the watermark conditioned on the REGEN RESIDUAL, add to original.

Does shaping the watermark by what regen changes (r = regen(x) - x) make it more regen-robust?
We compare, on the SAME bits, bit-acc CLEAN and AFTER one regen, plus publish PSNR vs original:

  C1 TM-baseline        : publish = TM.embed(x, bits)
  C2 TM-into-residual   : r=regen(x)-x; r_img=normalize(r); delta=TM.embed(r_img,bits)-r_img; publish=x+delta   (the user's idea)
  C3 TM-on-regenfixpt   : publish = TM.embed(regen(x), bits)   (inverse: watermark a near regen-fixed-point image)
  C4 VINE-baseline      : publish = VINE.embed(x, bits)        (latent-domain anchor: already survives regen)

TrustMark is pixel/high-freq (dies on regen); VINE is latent/low-mid-freq (survives). If a pixel-placement
trick could fix regen-robustness, C2/C3 would approach C4. Prediction: they don't -- the fix is DOMAIN, not placement.
"""
import argparse, glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.trustmark_fragment import TrustMarkFragment
from src.vine_crypto_wrapper import VineCryptoWrapper
KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
RES = 512


def arr(pil):
    return np.asarray(pil.convert("RGB").resize((RES, RES)), np.float32)


def pil(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def psnr(a, b):
    m = np.mean((a - b) ** 2)
    return float(99.0 if m < 1e-9 else 10 * np.log10(255.0 ** 2 / m))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_images", type=int, default=12); args = ap.parse_args()
    dev = "cuda"
    tm = TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=100, model_type="B", device=dev)
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=100, device=dev)
    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    tmp = tempfile.mkdtemp()

    def do_regen(p):
        ip = os.path.join(tmp, "i.png"); op = os.path.join(tmp, "o.png")
        p.save(ip); regen.attack([ip], [op])
        return Image.open(op).convert("RGB").resize((RES, RES))

    def tm_ba(att_pil, bits):
        return float(np.mean((tm.raw_logits(att_pil) > 0).astype(np.uint8) == bits))

    def vine_ba(att_pil, bits):
        return float(np.mean((vine.raw_probs(att_pil) > 0.5).astype(np.uint8) == bits))

    files = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n_images]
    rng = np.random.RandomState(0)
    R = {k: {"clean": [], "regen": [], "psnr": []} for k in ["C1_tm_base", "C2_tm_resid", "C3_tm_fixpt", "C4_vine"]}

    for i, fp in enumerate(files):
        bits = rng.randint(0, 2, 100).astype(np.uint8)
        x = Image.open(fp).convert("RGB").resize((RES, RES)); xa = arr(x)
        rg = do_regen(x); rga = arr(rg)                      # one regen of the clean image

        # C1 baseline
        p1 = tm.embed_with_target(x, bits).resize((RES, RES))
        # C2 embed into normalized regen-residual, add delta to original
        r = rga - xa; rn = (r - r.min()) / (r.max() - r.min() + 1e-9) * 255.0
        r_img = pil(rn); rw = tm.embed_with_target(r_img, bits).resize((RES, RES))
        delta = arr(rw) - arr(r_img)
        p2 = pil(xa + delta)
        # C3 watermark the regen fixed-point
        p3 = tm.embed_with_target(rg, bits).resize((RES, RES))
        # C4 VINE baseline
        p4 = vine.embed_with_target(x, bits).resize((RES, RES))

        for k, p, ba_fn in [("C1_tm_base", p1, tm_ba), ("C2_tm_resid", p2, tm_ba),
                            ("C3_tm_fixpt", p3, tm_ba), ("C4_vine", p4, vine_ba)]:
            R[k]["clean"].append(ba_fn(p, bits))
            R[k]["regen"].append(ba_fn(do_regen(p), bits))
            R[k]["psnr"].append(psnr(arr(p), xa))
        print(f"  [{i+1}/{len(files)}] "
              f"C1 {R['C1_tm_base']['regen'][-1]:.2f} | C2 {R['C2_tm_resid']['regen'][-1]:.2f} | "
              f"C3 {R['C3_tm_fixpt']['regen'][-1]:.2f} | C4 {R['C4_vine']['regen'][-1]:.2f}  (regen bit-acc)", flush=True)

    print(f"\n=== residual-embed idea test (n={len(files)}, one regen) ===")
    print(f"{'condition':<16}{'clean_ba':>10}{'regen_ba':>10}{'PSNR_vs_orig':>14}")
    for k in ["C1_tm_base", "C2_tm_resid", "C3_tm_fixpt", "C4_vine"]:
        print(f"{k:<16}{np.mean(R[k]['clean']):>10.3f}{np.mean(R[k]['regen']):>10.3f}{np.mean(R[k]['psnr']):>14.2f}")
    print("\nC1=TM baseline  C2=TM into regen-residual (the idea)  C3=TM on regen-fixed-point  C4=VINE latent anchor")
    print("RESIDUAL_EMBED_DONE")


if __name__ == "__main__":
    main()
