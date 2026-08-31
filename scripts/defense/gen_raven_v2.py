"""Generate SD-v2-1 images from prompts (RAVEN distribution). 512, CFG7.5, 50 DDIM.
Args: N dataset outdir seed0 ; dataset in {gustavosta, diffusiondb}."""
import os, sys, json, torch
REPO="/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
SD21="/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
DATASET=sys.argv[2] if len(sys.argv)>2 else "gustavosta"
OUT=os.path.join(REPO,sys.argv[3]) if len(sys.argv)>3 else os.path.join(REPO,"results/defense/raven_gen")
SEED0=int(sys.argv[4]) if len(sys.argv)>4 else 1000
os.makedirs(OUT,exist_ok=True)
prompts=None
try:
    from datasets import load_dataset
    if DATASET=="diffusiondb":
        ds=load_dataset("poloclub/diffusiondb","2m_random_1k",split="train",trust_remote_code=True)
        prompts=[ds[i]["prompt"] for i in range(min(N,len(ds)))]
    else:
        ds=load_dataset("Gustavosta/Stable-Diffusion-Prompts",split="train")
        prompts=[ds[(SEED0-1000+i)%len(ds)]["Prompt"] for i in range(N)]
    print(f"[prompts] {DATASET}: {len(prompts)}",flush=True)
except Exception as e:
    print(f"[prompts] load failed ({str(e)[:80]}); fallback",flush=True)
    base=["a photorealistic portrait of a woman","a fantasy landscape at sunset","a cyberpunk city at night",
          "a corgi puppy in a garden","an astronaut on mars","a bowl of fruit","a castle on a cliff","a coffee cup"]
    prompts=[base[i%len(base)]+f", detailed, v{SEED0+i}" for i in range(N)]
from diffusers import StableDiffusionPipeline, DDIMScheduler
pipe=StableDiffusionPipeline.from_pretrained(SD21,torch_dtype=torch.float16,safety_checker=None,requires_safety_checker=False)
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True); pipe=pipe.to("cuda")
meta=[]
for i,pr in enumerate(prompts):
    g=torch.Generator("cuda").manual_seed(SEED0+i)
    img=pipe(pr,num_inference_steps=50,guidance_scale=7.5,height=512,width=512,generator=g).images[0]
    img.save(os.path.join(OUT,f"img_{i:05d}.png")); meta.append({"id":i,"prompt":pr})
    if (i+1)%20==0: print(f"  generated [{i+1}/{N}]",flush=True)
json.dump(meta,open(os.path.join(OUT,"prompts.json"),"w"),indent=2)
print(f"[done] {len(prompts)} imgs ({DATASET}) -> {OUT}\nGEN_DONE")
