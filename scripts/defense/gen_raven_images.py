"""Generate SD-v2-1 images from prompts to match RAVEN's eval distribution (DiffusionDB/SD-Prompts).
512x512, CFG 7.5, 50 DDIM steps. Saves to results/defense/raven_gen/."""
import os, sys, json, torch
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
SD21="/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
OUT=os.path.join(REPO,"results/defense/raven_gen"); os.makedirs(OUT,exist_ok=True)
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
# --- prompts ---
prompts=None
try:
    from datasets import load_dataset
    ds=load_dataset("Gustavosta/Stable-Diffusion-Prompts",split="train")
    prompts=[ds[i]["Prompt"] for i in range(N)]; print(f"[prompts] SD-Prompts (Gustavosta): {len(prompts)}",flush=True)
except Exception as e:
    print(f"[prompts] dataset load failed ({str(e)[:60]}); using fallback prompts",flush=True)
    base=["a photorealistic portrait of a woman, detailed","a fantasy landscape with mountains and a river at sunset",
          "a cyberpunk city street at night, neon lights","a cute corgi puppy in a garden, bokeh","an astronaut riding a horse on mars",
          "a bowl of fresh fruit on a wooden table, studio light","a medieval castle on a cliff, dramatic sky","a steaming cup of coffee on a cafe table"]
    prompts=[base[i%len(base)]+f", v{i}" for i in range(N)]
from diffusers import StableDiffusionPipeline, DDIMScheduler
pipe=StableDiffusionPipeline.from_pretrained(SD21,torch_dtype=torch.float16,safety_checker=None,requires_safety_checker=False)
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True); pipe=pipe.to("cuda")
meta=[]
for i,pr in enumerate(prompts):
    g=torch.Generator("cuda").manual_seed(1000+i)
    img=pipe(pr,num_inference_steps=50,guidance_scale=7.5,height=512,width=512,generator=g).images[0]
    img.save(os.path.join(OUT,f"img_{i:05d}.png")); meta.append({"id":i,"prompt":pr})
    if (i+1)%10==0: print(f"  generated [{i+1}/{N}]",flush=True)
json.dump(meta,open(os.path.join(OUT,"prompts.json"),"w"),indent=2)
print(f"[done] {N} SD-2.1 images -> {OUT}\nGEN_DONE")
