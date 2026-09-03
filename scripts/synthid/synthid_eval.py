"""Step 3+4 — apply the SAME attack suite as our composite eval to the SynthID images,
detect each attacked image, and report per-attack detection rate. Local attacks are free;
only detection costs (~$0.0005/img via Gemini-QA). Mirrors the attack params used in
scripts/defense/wbench_eval.py so results are directly comparable to our pipeline.

  python synthid_eval.py --wm results/synthid/wm --n 64 --out results/synthid/synthid_robustness.json
  # add --regen to include the (slow, needs diffusers+SD) regeneration axis
"""
import os, io, glob, json, argparse
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import synthid_detect as D          # reuse detect() + make_client()

def to512(im): return im.resize((512, 512)) if im.size != (512, 512) else im
def a_jpeg(x, q): b = io.BytesIO(); x.save(b, "JPEG", quality=int(q)); b.seek(0); return Image.open(b).convert("RGB")
def a_blur(x, s): return x.filter(ImageFilter.GaussianBlur(float(s)))
def a_noise(x, s):
    arr = np.asarray(x, np.float64) + np.random.RandomState(0).normal(0, float(s) * 255, (512, 512, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
def a_bright(x, f): return ImageEnhance.Brightness(x).enhance(float(f))
def a_contrast(x, f): return ImageEnhance.Contrast(x).enhance(float(f))
def a_crop(x, r): s = int(512 * r); o = (512 - s) // 2; return x.crop((o, o, o + s, o + s)).resize((512, 512))
def a_rot(x, a): return x.rotate(float(a), resample=Image.BILINEAR)

# same attack set/params as wbench_eval.py DISTORT (extend as needed)
ATTACKS = {
    "clean":     lambda x: x,
    "jpeg25":    lambda x: a_jpeg(x, 25),
    "jpeg10":    lambda x: a_jpeg(x, 10),
    "blur3":     lambda x: a_blur(x, 3),
    "noise0.05": lambda x: a_noise(x, 0.05),
    "bright1.5": lambda x: a_bright(x, 1.5),
    "contrast1.5": lambda x: a_contrast(x, 1.5),
    "crop75":    lambda x: a_crop(x, 0.75),
    "crop50":    lambda x: a_crop(x, 0.5),
    "rot9":      lambda x: a_rot(x, 9),
    "rot30":     lambda x: a_rot(x, 30),
    "rot90":     lambda x: a_rot(x, 90),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True, help="dir of SynthID images from synthid_gen.py")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--regen", action="store_true", help="add SD img2img regen axis (needs diffusers+SD)")
    ap.add_argument("--regen_strengths", default="0.2,0.4,0.6")
    ap.add_argument("--out", default="results/synthid/synthid_robustness.json")
    a = ap.parse_args()
    client = D.make_client()
    imgs = sorted(glob.glob(os.path.join(a.wm, "*.png")))[:a.n]
    print(f"SynthID robustness eval: n={len(imgs)}, {len(ATTACKS)} attacks", flush=True)

    attacks = dict(ATTACKS)
    if a.regen:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        SD = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD, torch_dtype=torch.float16, safety_checker=None, local_files_only=True).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True)
        for s in [float(x) for x in a.regen_strengths.split(",")]:
            attacks[f"regen{s}"] = (lambda x, s=s: to512(pipe(prompt="", image=x, strength=s, num_inference_steps=50, guidance_scale=1.0).images[0]))

    res = {}
    for name, fn in attacks.items():
        dets = []
        for fp in imgs:
            wm = to512(Image.open(fp).convert("RGB"))
            att = to512(fn(wm))
            d = D.detect(client, att)
            if d >= 0: dets.append(d)
        res[name] = {"detection": float(np.mean(dets)) if dets else float("nan"), "n": len(dets)}
        print(f"  {name:12s} det={res[name]['detection']:.3f} (n={res[name]['n']})", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\n=== SynthID robustness (detection rate per attack) ===")
    for k, v in res.items(): print(f"  {k:12s} {v['detection']:.3f}")
    print(f"[done] -> {a.out}\nSYNTHID_EVAL_DONE")

if __name__ == "__main__":
    main()
