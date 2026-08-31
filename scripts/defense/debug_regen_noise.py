"""DEBUG: why does regen(W) for img0 come out as color noise? Hypothesis: fp16 VAE overflow -> NaN -> noise.

(1) Reproduce: img0 vs img5, regen(W)/regen(C) at the actual seed used; report PIL-level stats (noise => huge std).
(2) Root-cause probe: run the VAE roundtrip (encode->decode) on img0's W in fp16 vs fp32; count NaN/Inf and range.
"""
import glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.vine_crypto_wrapper import VineCryptoWrapper
KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
RES = 512


def arr(p): return np.asarray(p.convert("RGB").resize((RES, RES)), np.float32)
def stats(name, pil):
    a = arr(pil)
    print(f"  {name:<26} min={a.min():6.1f} max={a.max():6.1f} mean={a.mean():6.1f} std={a.std():6.1f}", flush=True)


def main():
    dev = "cuda"
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=100, device=dev)
    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60); tmp = tempfile.mkdtemp()

    def regen_seeded(pil, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        ip = os.path.join(tmp, "i.png"); op = os.path.join(tmp, "o.png")
        pil.save(ip); regen.attack([ip], [op]); return Image.open(op).convert("RGB").resize((RES, RES))

    files = sorted(glob.glob(os.path.join(COCO, "*.jpg")))
    print("=== (1) reproduce: PIL-level stats (uniform color noise => std ~70-75) ===")
    for idx in [0, 5]:
        C = Image.open(files[4000 + idx]).convert("RGB").resize((RES, RES))
        W = vine.embed(C, f"probe_{idx:05d}")
        if W.size != (RES, RES): W = W.resize((RES, RES))
        seed = 1234 + 100 * idx                       # the exact seed reembed_probe used for this idx
        print(f"img{idx:05d} (seed={seed}):")
        stats("W (embedded)", W)
        stats("regen(W)", regen_seeded(W, seed))
        stats("regen(C)", regen_seeded(C, seed))

    print("\n=== (2) root-cause probe: VAE roundtrip on img0 W, fp16 vs fp32 ===")
    C0 = Image.open(files[4000]).convert("RGB").resize((RES, RES))
    W0 = vine.embed(C0, "probe_00000")
    if W0.size != (RES, RES): W0 = W0.resize((RES, RES))
    x = torch.from_numpy(arr(W0) / 127.5 - 1).permute(2, 0, 1)[None].to(dev)
    vae = pipe.vae; sf = vae.config.scaling_factor
    for tag, dt in [("fp16", torch.float16), ("fp32", torch.float32)]:
        vae.to(dt); xx = x.to(dt)
        with torch.no_grad():
            z = vae.encode(xx).latent_dist.mean * sf
            dec = vae.decode(z / sf).sample
        zn = torch.isnan(z).sum().item() + torch.isinf(z).sum().item()
        dn = torch.isnan(dec).sum().item() + torch.isinf(dec).sum().item()
        zf = z.float()
        print(f"  {tag}: latent z range=[{zf.min():.1f},{zf.max():.1f}] nan/inf(z)={zn}  "
              f"decode nan/inf={dn} range=[{dec.float().min():.2f},{dec.float().max():.2f}]")
    vae.to(torch.float16)
    print("DEBUG_DONE")


if __name__ == "__main__":
    main()
