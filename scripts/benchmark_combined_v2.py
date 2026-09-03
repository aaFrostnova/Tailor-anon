"""Full combined system v2, with our OWN reference-free geometric tier.

Tiers (OR-combined):
  - Tier-1 fused: VINE + QIM (shared BCH codeword, adaptive LLR + Chase) -- distortions.
                  DFT-Kred is REMOVED; its DFT-magnitude domain is taken by the geo tier.
  - Tier-2 Gaussian Shading (latent, generation time) -- regeneration / VAE / rinse.
  - Tier-3 scale-aware geometric fragment (our DFT carrier + pilot sync) -- resize / rotation
            / crop, REFERENCE-FREE (replaces the external TrustMark for PURE geometry).
  - Tier-4 TrustMark (deep post-hoc) -- kept as the compound geometry+compression backstop.

Reports each tier's per-attack detection + the OR, and clean self-acc (interference check).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
GS_DIR = REPO / "external/Gaussian-Shading"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(GS_DIR))

from inverse_stable_diffusion import InversableStableDiffusionPipeline  # noqa: E402
from diffusers import DPMSolverMultistepScheduler  # noqa: E402
from watermark import Gaussian_Shading_chacha  # noqa: E402
from image_utils import transform_img, set_random_seed  # noqa: E402
from PIL import Image  # noqa: E402
from src.fused_detector import FusedDetector  # noqa: E402
from src.scaleaware_method import ScaleAwareGeoFragment  # noqa: E402
from src.soft_fusion import fuse_llrs  # noqa: E402
from src.soft_bch import decode_and_verify  # noqa: E402
from benchmark_fused import apply_attack, psnr, adaptive_weights  # noqa: E402
from benchmark_combined import PROMPTS  # noqa: E402
from wbench.methods import build_methods  # noqa: E402

DEVICE = "cuda"


def rot(pil, deg):
    return pil.rotate(deg, resample=Image.BILINEAR, expand=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=30)
    p.add_argument("--gs_model", default="/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1-base")
    p.add_argument("--geo_ckpt", default="results/logpolar/scaleaware_C.pt")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--det_thresh", type=float, default=0.75)
    p.add_argument("--tm_thresh", type=float, default=0.95)
    p.add_argument("--attacks", nargs="+", default=[
        "clean", "jpeg_50", "blur_2.5", "noise_50", "crop_90", "crop_70",
        "resize_110", "rot_5", "vae", "regen_10_sd15", "regen_20_sd15", "regen_30_sd15", "rinse_2"])
    p.add_argument("--output", default="results/fused/combined_v2.json")
    args = p.parse_args()

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.gs_model, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.gs_model, scheduler=scheduler, torch_dtype=torch.float16)
    pipe.safety_checker = None; pipe = pipe.to(DEVICE); pipe.set_progress_bar_config(disable=True)
    text_emb = pipe.get_text_embedding("")
    fd = FusedDetector(device=DEVICE, use_dft=False)               # Tier-1 = VINE + QIM
    geo = ScaleAwareGeoFragment(str(REPO / args.geo_ckpt), device=DEVICE,
                                angles_deg=(-6, -3, 0, 3, 6))       # Tier-3 ref-free geometric
    tm = build_methods(["trustmark"], DEVICE)["trustmark"]         # Tier-4 backstop

    tiers = ["tier1", "gs", "geo", "trustmark"]
    agg = {a: {t: [] for t in tiers} | {"or": []} for a in args.attacks}
    self_acc = {"gs": [], "geo": [], "tm": [], "tier1": []}
    psnrs = []

    @torch.no_grad()
    def gs_detect(att_pil, gs_obj):
        t = transform_img(att_pil).unsqueeze(0).to(text_emb.dtype).to(DEVICE)
        rev = pipe.forward_diffusion(latents=pipe.get_image_latents(t, sample=False),
                                     text_embeddings=text_emb, guidance_scale=1,
                                     num_inference_steps=args.steps)
        return gs_obj.eval_watermark(rev)

    for i in range(args.n_images):
        image_id = f"cv2_{i:05d}"
        set_random_seed(i)
        gs = Gaussian_Shading_chacha(1, 8, 1e-6, 1_000_000)
        w = gs.create_watermark_and_return_w()
        base = pipe(PROMPTS[i % len(PROMPTS)], num_images_per_prompt=1, guidance_scale=7.5,
                    num_inference_steps=args.steps, height=512, width=512, latents=w).images[0]
        stacked = fd.embed(base, image_id)                          # VINE + QIM
        tm_bits = np.random.RandomState(2000 + i).randint(0, 2, tm.n_bits).astype(np.uint8)
        stacked = tm.embed(stacked, tm_bits)                        # TrustMark
        stacked = geo.embed(stacked, image_id)                      # DFT geometric LAST (carriers on top)
        psnrs.append(psnr(base, stacked))

        # clean self-acc
        self_acc["gs"].append(gs_detect(stacked, gs))
        self_acc["geo"].append(geo.detect(stacked, image_id)["payload_bit_acc"])
        rec0 = tm.decode(stacked); n0 = min(len(rec0), len(tm_bits))
        self_acc["tm"].append(float(np.mean(rec0[:n0] == tm_bits[:n0])))
        al0, _ = fd.aligned_llrs(stacked, image_id)
        from src.soft_fusion import llr_to_bits
        self_acc["tier1"].append(float(np.mean(llr_to_bits(fuse_llrs(al0, n_codeword=fd.sb.n))
                                               == fd.codeword(image_id))))

        for atk in args.attacks:
            att = rot(stacked, 5) if atk == "rot_5" else apply_attack(atk, stacked)
            al, _ = fd.aligned_llrs(att, image_id)
            fused = fuse_llrs(al, weights=adaptive_weights(al), n_codeword=fd.sb.n)
            t1 = decode_and_verify(fused, image_id, codec=fd.sb, p=8)["detected"]
            g = gs_detect(att, gs) >= args.det_thresh
            ge = geo.detect(att, image_id)["detected"]
            rec = tm.decode(att); n = min(len(rec), len(tm_bits))
            tmd = float(np.mean(rec[:n] == tm_bits[:n])) >= args.tm_thresh
            agg[atk]["tier1"].append(float(t1)); agg[atk]["gs"].append(float(g))
            agg[atk]["geo"].append(float(ge)); agg[atk]["trustmark"].append(float(tmd))
            agg[atk]["or"].append(float(t1 or g or ge or tmd))
        print(f"  [{i+1}/{args.n_images}] psnr={np.mean(psnrs):.1f} self gs={np.mean(self_acc['gs']):.2f} "
              f"geo={np.mean(self_acc['geo']):.2f} tm={np.mean(self_acc['tm']):.2f} "
              f"t1={np.mean(self_acc['tier1']):.2f}", flush=True)

    summary = {"n_images": args.n_images, "psnr_stack": float(np.mean(psnrs)),
               "self_acc": {k: float(np.mean(v)) for k, v in self_acc.items()}, "attacks": {}}
    for atk in args.attacks:
        d = agg[atk]
        summary["attacks"][atk] = {t: float(np.mean(d[t])) for t in tiers} | {"OR": float(np.mean(d["or"]))}

    Path(REPO / args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(REPO / args.output, "w"), indent=2)
    print(f"\n{'='*82}")
    print(f"COMBINED v2 (ref-free geometric tier)  PSNR={summary['psnr_stack']:.1f}dB  "
          f"self: gs={summary['self_acc']['gs']:.2f} geo={summary['self_acc']['geo']:.2f} "
          f"tm={summary['self_acc']['tm']:.2f} t1={summary['self_acc']['tier1']:.2f}")
    print(f"{'attack':<16}{'Tier1':>7}{'GS':>6}{'GEO':>6}{'TrustM':>8}{'OR':>7}")
    print("-" * 82)
    for atk in args.attacks:
        s = summary["attacks"][atk]
        print(f"{atk:<16}{s['tier1']:>7.2f}{s['gs']:>6.2f}{s['geo']:>6.2f}{s['trustmark']:>8.2f}{s['OR']:>7.2f}")
    print("=" * 82)
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
