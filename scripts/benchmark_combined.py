"""Full complementary system: GS (latent) + VINE+DFT+QIM (fused) + TrustMark (crop).

Stacks all tiers in ONE image and evaluates the OR-combination across the full
attack suite, to test whether the RAVEN-inspired complementary set gives union
coverage:
  - Tier-1  fused VINE+DFT+QIM (shared BCH codeword, adaptive LLR + Chase): distortions;
  - Tier-2  Gaussian Shading (generation-time latent): regeneration / VAE / rinse;
  - Tier-3  TrustMark (deep post-hoc): geometric (crop / resize).

Reports each tier's per-attack detection, the OR-combination, and clean self-acc
in the stack (to expose any stacking interference). RivaGAN is an alternative
Tier-3 (omitted from the stacked image to avoid deep-deep overwrite with VINE/TrustMark).
"""
import argparse
import json
import sys
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
from src.fused_detector import FusedDetector  # noqa: E402
from src.soft_fusion import fuse_llrs  # noqa: E402
from src.soft_bch import decode_and_verify  # noqa: E402
from benchmark_fused import apply_attack, psnr, adaptive_weights  # noqa: E402
from wbench.methods import build_methods  # noqa: E402

DEVICE = "cuda"
PROMPTS = [
    "a photograph of an astronaut riding a horse", "a bowl of fresh fruit on a table",
    "a serene mountain lake at sunrise", "a busy city street at night with neon lights",
    "a close-up portrait of a brown dog", "an old wooden sailboat on calm water",
    "a plate of spaghetti with tomato sauce", "a field of sunflowers under blue sky",
    "a cozy living room with a fireplace", "a red sports car on a coastal road",
    "a snowy forest in winter", "a cup of coffee next to a book",
    "a butterfly on a purple flower", "a lighthouse on a rocky cliff",
    "a slice of chocolate cake on a plate", "a hot air balloon over green hills",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=30)
    p.add_argument("--gs_model", default="/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--det_thresh", type=float, default=0.75)
    p.add_argument("--tm_thresh", type=float, default=0.95)
    p.add_argument("--attacks", nargs="+", default=[
        "clean", "jpeg_50", "blur_2.5", "noise_50", "crop_90", "crop_70",
        "resize_110", "vae", "regen_10_sd15", "regen_20_sd15", "regen_30_sd15", "rinse_2",
    ])
    p.add_argument("--output", default="results/fused/combined.json")
    args = p.parse_args()

    # Tier-2 GS pipeline
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.gs_model, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.gs_model, scheduler=scheduler, torch_dtype=torch.float16)
    pipe.safety_checker = None
    pipe = pipe.to(DEVICE); pipe.set_progress_bar_config(disable=True)
    text_emb = pipe.get_text_embedding("")
    # Tier-1 fused + Tier-3 TrustMark
    fd = FusedDetector(device=DEVICE)
    tm = build_methods(["trustmark"], DEVICE)["trustmark"]

    tiers = ["tier1_fused", "gs", "trustmark"]
    agg = {a: {t: [] for t in tiers} for a in args.attacks}
    for a in args.attacks:
        agg[a]["or"] = []
    self_acc = {"gs": [], "trustmark": [], "tier1_bitacc": []}
    psnrs = []

    for i in range(args.n_images):
        image_id = f"comb_{i:05d}"
        set_random_seed(i)
        gs = Gaussian_Shading_chacha(1, 8, 1e-6, 1_000_000)
        w = gs.create_watermark_and_return_w()
        base = pipe(PROMPTS[i % len(PROMPTS)], num_images_per_prompt=1, guidance_scale=7.5,
                    num_inference_steps=args.steps, height=512, width=512, latents=w).images[0]
        # Tier-1 stack (VINE+DFT+QIM, shared codeword)
        stacked = fd.embed(base, image_id)
        # Tier-3 TrustMark on top
        tm_bits = np.random.RandomState(1000 + i).randint(0, 2, tm.n_bits).astype(np.uint8)
        stacked = tm.embed(stacked, tm_bits)
        psnrs.append(psnr(base, stacked))

        # clean self-acc in the stack
        rev0 = pipe.forward_diffusion(
            latents=pipe.get_image_latents(
                transform_img(stacked).unsqueeze(0).to(text_emb.dtype).to(DEVICE), sample=False),
            text_embeddings=text_emb, guidance_scale=1, num_inference_steps=args.steps)
        self_acc["gs"].append(gs.eval_watermark(rev0))
        rec0 = tm.decode(stacked); n0 = min(len(rec0), len(tm_bits))
        self_acc["trustmark"].append(float(np.mean(rec0[:n0] == tm_bits[:n0])))
        al0, _ = fd.aligned_llrs(stacked, image_id)
        tx = fd.codeword(image_id)
        from src.soft_fusion import llr_to_bits
        self_acc["tier1_bitacc"].append(
            float(np.mean(llr_to_bits(fuse_llrs(al0, n_codeword=fd.sb.n)) == tx)))

        for atk in args.attacks:
            att = apply_attack(atk, stacked)
            # Tier-1: adaptive fused + Chase identity
            al, _ = fd.aligned_llrs(att, image_id)
            fused = fuse_llrs(al, weights=adaptive_weights(al), n_codeword=fd.sb.n)
            t1 = decode_and_verify(fused, image_id, codec=fd.sb, p=8)["detected"]
            # Tier-2: GS invert
            rev = pipe.forward_diffusion(
                latents=pipe.get_image_latents(
                    transform_img(att).unsqueeze(0).to(text_emb.dtype).to(DEVICE), sample=False),
                text_embeddings=text_emb, guidance_scale=1, num_inference_steps=args.steps)
            g = gs.eval_watermark(rev) >= args.det_thresh
            # Tier-3: TrustMark
            rec = tm.decode(att); n = min(len(rec), len(tm_bits))
            tmd = float(np.mean(rec[:n] == tm_bits[:n])) >= args.tm_thresh
            agg[atk]["tier1_fused"].append(float(t1))
            agg[atk]["gs"].append(float(g))
            agg[atk]["trustmark"].append(float(tmd))
            agg[atk]["or"].append(float(t1 or g or tmd))
        print(f"  [{i+1}/{args.n_images}] psnr={np.mean(psnrs):.1f} "
              f"self gs={np.mean(self_acc['gs']):.2f} tm={np.mean(self_acc['trustmark']):.2f} "
              f"t1={np.mean(self_acc['tier1_bitacc']):.2f}", flush=True)

    summary = {"n_images": args.n_images, "psnr_stack": float(np.mean(psnrs)),
               "self_acc": {k: float(np.mean(v)) for k, v in self_acc.items()},
               "attacks": {}}
    for atk in args.attacks:
        d = agg[atk]
        summary["attacks"][atk] = {t: float(np.mean(d[t])) for t in tiers}
        summary["attacks"][atk]["OR"] = float(np.mean(d["or"]))

    Path(REPO / args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(REPO / args.output, "w"), indent=2)

    print(f"\n{'='*80}")
    print(f"FULL COMBINED SYSTEM  PSNR(stack)={summary['psnr_stack']:.1f}dB  "
          f"clean self-acc: gs={summary['self_acc']['gs']:.2f} "
          f"tm={summary['self_acc']['trustmark']:.2f} tier1={summary['self_acc']['tier1_bitacc']:.2f}")
    print(f"{'attack':<16}{'Tier1':>8}{'GS':>7}{'TrustM':>8}{'OR(all)':>9}")
    print("-" * 80)
    for atk in args.attacks:
        s = summary["attacks"][atk]
        print(f"{atk:<16}{s['tier1_fused']:>8.2f}{s['gs']:>7.2f}{s['trustmark']:>8.2f}{s['OR']:>9.2f}")
    print("=" * 80)
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
