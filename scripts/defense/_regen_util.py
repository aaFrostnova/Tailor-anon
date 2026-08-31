"""Stable regen utility — same attack strength as DiffWMAttacker (noise_step=60) but MORE denoise steps
so the DPMSolver sampler doesn't occasionally collapse to color noise (the 3-step instability bug).

stable_regen(pipe, pil, seed, noise_step=60, denoise_steps=8) replicates DiffWMAttacker.attack's path
(vae.encode->sample(gen1024) -> add_noise(t=noise_step) -> pipe denoise) but with head_start_step =
50 - denoise_steps (default 8 steps instead of 3). Verified: 8 steps removes the img0/seed1234 noise.

Self-test (python _regen_util.py): runs imgs 0..7 x seeds at 3 vs 8 steps; flags any corr<0.5 (=noise).
"""
import os, sys, glob
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
RES = 512


def build_regen_pipe(dtype=torch.float16, device="cuda"):
    from regen_pipe import ReSDPipeline
    from diffusers import DPMSolverMultistepScheduler
    p = ReSDPipeline.from_pretrained(SD21, torch_dtype=dtype)
    p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
    p.set_progress_bar_config(disable=True)
    return p.to(device)


def stable_regen(pipe, pil, seed, noise_step=60, denoise_steps=8, device="cuda"):
    """Same as DiffWMAttacker but head_start_step = 50 - denoise_steps (more steps => stable)."""
    with torch.no_grad():
        gen = torch.Generator(device).manual_seed(1024)
        torch.manual_seed(seed); np.random.seed(seed)
        ts = torch.tensor([noise_step], dtype=torch.long, device=device)
        im = (np.asarray(pil.convert("RGB").resize((RES, RES)), np.float32) / 255 - 0.5) * 2
        dt = pipe.dtype
        x = torch.tensor(im, dtype=dt, device=device).permute(2, 0, 1)[None]
        sf = pipe.vae.config.scaling_factor
        lat = pipe.vae.encode(x).latent_dist.sample(gen) * sf
        noise = torch.randn([1, 4, RES // 8, RES // 8], device=device).to(lat.dtype)
        lat = pipe.scheduler.add_noise(lat, noise, ts).to(dt)
        imgs = pipe([""], head_start_latents=lat, head_start_step=50 - denoise_steps,
                    guidance_scale=7.5, generator=gen)
        return imgs[0][0].resize((RES, RES))


def _arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def _corr(a, b): return float(np.corrcoef(_arr(a).ravel(), _arr(b).ravel())[0, 1])


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    from src.vine_crypto_wrapper import VineCryptoWrapper
    COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
    vine = VineCryptoWrapper(master_key=b"v5_key_encoder_master", method_name="vine", n_bits=100, device="cuda")
    pipe = build_regen_pipe()
    files = sorted(glob.glob(os.path.join(COCO, "*.jpg")))
    print("self-test: corr(regen(W), clean) — noise if <0.5. cols: 3-step vs 8-step")
    bad3 = bad8 = 0
    for i in range(8):
        C = Image.open(files[4000 + i]).convert("RGB").resize((RES, RES))
        W = vine.embed(C, f"probe_{i:05d}")
        W = W if W.size == (RES, RES) else W.resize((RES, RES))
        for seed in [1234 + 100 * i, 42]:
            c3 = _corr(stable_regen(pipe, W, seed, denoise_steps=3), C)
            c8 = _corr(stable_regen(pipe, W, seed, denoise_steps=8), C)
            bad3 += c3 < 0.5; bad8 += c8 < 0.5
            flag = "  <-- 3step NOISE" if c3 < 0.5 else ""
            print(f"  img{i:05d} seed{seed:<6}: 3step={c3:.3f}  8step={c8:.3f}{flag}")
    print(f"\nnoise cases: 3-step={bad3}/16   8-step={bad8}/16")
    print("STABLE_REGEN_SELFTEST_DONE")
