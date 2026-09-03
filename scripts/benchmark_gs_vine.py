"""Complementary combination: Gaussian Shading (latent) + VINE (post-hoc).

Motivated by RAVEN (arXiv:2601.08832) Table 2: latent methods (Gaussian Shading)
are near-perfect under regeneration / VAE / purification (0.99-1.0) where post-hoc
methods degrade, while deep post-hoc methods (VINE) are the most resistant to
geometric and semantic-view attacks. We embed BOTH in one image (GS at generation
time, VINE post-hoc on top) and show the OR-combination covers the union:

  - GS carries regeneration (where VINE only partially survives);
  - VINE carries the rest;
  - combined detection = GS_detect OR VINE_detect.

GS detection inverts the (attacked) image back to the initial latent via the
repo's InversableStableDiffusionPipeline and votes the recovered bits.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
GS_DIR = REPO / "external/Gaussian-Shading"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(GS_DIR))

from inverse_stable_diffusion import InversableStableDiffusionPipeline  # noqa: E402
from diffusers import DPMSolverMultistepScheduler  # noqa: E402
from watermark import Gaussian_Shading_chacha  # noqa: E402
from image_utils import transform_img, set_random_seed  # noqa: E402
from src.vine_crypto_wrapper import VineCryptoWrapper  # noqa: E402
from benchmark_fused import apply_attack, psnr  # noqa: E402

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
    p.add_argument("--n_images", type=int, default=12)
    # SD-1.5 (local): eps-prediction, 512px, 4x64x64 latent -> compatible with GS.
    p.add_argument("--model", default="/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-v1-5")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--det_thresh", type=float, default=0.75)
    p.add_argument("--attacks", nargs="+", default=[
        "clean", "jpeg_50", "blur_2.5", "noise_50", "crop_70",
        "regen_10_sd15", "regen_20_sd15", "regen_30_sd15",
    ])
    p.add_argument("--output", default="results/fused/gs_vine.json")
    args = p.parse_args()

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model, subfolder="scheduler")
    try:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model, scheduler=scheduler, torch_dtype=torch.float16, variant="fp16")
    except Exception:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model, scheduler=scheduler, torch_dtype=torch.float16)
    pipe.safety_checker = None
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    vine = VineCryptoWrapper(master_key=b"v5_key_encoder_master", method_name="vine",
                             n_bits=100, device=DEVICE)
    text_emb = pipe.get_text_embedding("")

    agg = {a: {"gs_acc": [], "vine_acc": [], "gs_det": [], "vine_det": [], "or_det": []}
           for a in args.attacks}
    psnrs = []

    for i in range(args.n_images):
        seed = i
        set_random_seed(seed)
        wm = Gaussian_Shading_chacha(ch_factor=1, hw_factor=8, fpr=1e-6, user_number=1_000_000)
        w = wm.create_watermark_and_return_w()
        prompt = PROMPTS[i % len(PROMPTS)]
        img = pipe(prompt, num_images_per_prompt=1, guidance_scale=args.guidance,
                   num_inference_steps=args.steps, height=512, width=512, latents=w).images[0]
        image_id = f"gsv_{i:05d}"
        img_v = vine.embed(img, image_id)            # post-hoc VINE on top of GS image
        psnrs.append(psnr(img, img_v))

        for atk in args.attacks:
            att = apply_attack(atk, img_v)
            # GS detect: DDIM inversion -> recovered latent -> vote
            t = transform_img(att).unsqueeze(0).to(text_emb.dtype).to(DEVICE)
            lat = pipe.get_image_latents(t, sample=False)
            rev = pipe.forward_diffusion(latents=lat, text_embeddings=text_emb,
                                         guidance_scale=1, num_inference_steps=args.steps)
            acc_gs = wm.eval_watermark(rev)
            # VINE detect
            acc_vine = vine.detect(att, image_id)["bit_accuracy"]
            dg = acc_gs >= args.det_thresh
            dv = acc_vine >= args.det_thresh
            agg[atk]["gs_acc"].append(acc_gs); agg[atk]["vine_acc"].append(acc_vine)
            agg[atk]["gs_det"].append(float(dg)); agg[atk]["vine_det"].append(float(dv))
            agg[atk]["or_det"].append(float(dg or dv))
        print(f"  [{i+1}/{args.n_images}] psnr(VINE-on-GS)={np.mean(psnrs):.1f}", flush=True)

    summary = {"n_images": args.n_images, "psnr_vine_on_gs": float(np.mean(psnrs)),
               "det_thresh": args.det_thresh, "attacks": {}}
    for atk in args.attacks:
        d = agg[atk]
        summary["attacks"][atk] = {k: float(np.mean(v)) for k, v in d.items()}

    Path(REPO / args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(REPO / args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*78}")
    print(f"GS+VINE complementary combination  (PSNR VINE-on-GS={summary['psnr_vine_on_gs']:.1f}dB)")
    print(f"{'attack':<16}{'GS_acc':>8}{'VINE_acc':>9}{'GS_det':>8}{'VINE_det':>9}{'OR_det':>8}")
    print("-" * 78)
    for atk in args.attacks:
        s = summary["attacks"][atk]
        print(f"{atk:<16}{s['gs_acc']:>8.3f}{s['vine_acc']:>9.3f}"
              f"{s['gs_det']:>8.2f}{s['vine_det']:>9.2f}{s['or_det']:>8.2f}")
    print("=" * 78)
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
