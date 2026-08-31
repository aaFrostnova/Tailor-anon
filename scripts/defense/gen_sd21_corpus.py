"""Generate an on-manifold SD-2.1 image corpus (the PhaseMark paper's domain).

PhaseMark's >=35 dB is reported on SD-v2-1-base GENERATED 512x512 images, where the VAE
round-trip ceiling is ~36 dB. Our composite/PhaseMark were measured on COCO natural photos
(VAE ceiling ~24 dB). This generates the proper domain so we can (a) reproduce the paper's
quality regime and (b) run the regen-robust large-dataset eval on the domain that actually
matters for an AI-image watermark.
"""
import argparse, os, sys
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"

PROMPTS = [
    "a photograph of a mountain lake at sunrise, mist over the water",
    "a busy city street at night with neon signs and rain",
    "a close-up portrait of an elderly fisherman, weathered face",
    "a bowl of fresh fruit on a wooden table, soft window light",
    "an astronaut riding a horse on the moon, photorealistic",
    "a cozy living room with a fireplace and bookshelves",
    "a red sports car parked on a coastal road, golden hour",
    "a field of sunflowers under a dramatic cloudy sky",
    "a steaming cup of coffee on a cafe table, morning light",
    "a snow-covered cabin in a pine forest at dusk",
    "a tropical beach with turquoise water and palm trees",
    "a vintage bicycle leaning against a brick wall with ivy",
    "a plate of sushi beautifully arranged, studio lighting",
    "a golden retriever puppy playing in autumn leaves",
    "a futuristic city skyline with flying cars at sunset",
    "a still life of old books and a candle, chiaroscuro",
    "a hot air balloon over a patchwork of green fields",
    "a waterfall in a lush rainforest, long exposure",
    "a market stall full of colorful spices, vibrant",
    "a lighthouse on a rocky cliff during a storm",
    "a macro shot of a dewy spider web at dawn",
    "a chef plating a gourmet dish in a restaurant kitchen",
    "a desert landscape with towering sand dunes at noon",
    "a cat sleeping on a windowsill in the sun",
    "a violin resting on sheet music, warm light",
    "a crowded train platform with motion blur",
    "an old stone bridge over a calm river in fog",
    "a bouquet of roses in a glass vase on a table",
    "a skier carving down a powder slope, spray of snow",
    "a neon-lit diner interior, retro americana",
    "a child flying a kite on a windy hill",
    "a glass of red wine beside a cheese board",
    "a vintage typewriter on a writer's cluttered desk",
    "a coral reef teeming with tropical fish, underwater",
    "a foggy forest path with tall ancient trees",
    "a plate of pancakes with syrup and berries",
    "a sailboat on a calm sea under a pink sky",
    "a blacksmith working at a glowing forge",
    "a row of colorful houses along a canal in europe",
    "a telescope pointed at a starry night sky",
    "a freshly baked loaf of bread on a cutting board",
    "a deer standing in a misty meadow at dawn",
    "a classic motorcycle on an empty desert highway",
    "a potter shaping clay on a spinning wheel",
    "a snowy mountain peak reflected in an alpine lake",
    "a street musician playing guitar under a lamppost",
    "a tray of macarons in pastel colors, patisserie",
    "a vintage camera and film rolls on a map",
    "a peacock displaying its feathers in a garden",
    "a quiet library with tall shelves and a reading lamp",
    "a surfer riding a large wave, spray backlit",
    "a campfire under a sky full of stars in the wilderness",
    "a bustling fish market in the early morning",
    "a single tree on a hill silhouetted against sunset",
    "a plate of spaghetti with tomato sauce and basil",
    "a snowman in a backyard during gentle snowfall",
    "a hummingbird feeding from a bright red flower",
    "a cobblestone alley with hanging lanterns at night",
    "a bowl of ramen with egg and scallions, steam rising",
    "an autumn forest with golden and red foliage",
    "a grand piano in an empty concert hall",
    "a kayaker paddling through a narrow canyon river",
    "a vintage globe and stacked books on a shelf",
    "a misty tea plantation on terraced hills",
    "a chess board mid-game with carved wooden pieces",
    "a flock of birds over a wheat field at sunset",
    "a cozy bookstore cafe with warm lighting",
    "a frozen waterfall in a winter landscape",
    "a colorful parrot perched on a tropical branch",
    "a rustic farmhouse kitchen with copper pots",
    "a dramatic thunderstorm over the open ocean",
    "a bowl of fresh strawberries with cream",
    "a mountain biker on a rugged trail, dust trailing",
    "a serene zen garden with raked sand and stones",
    "a vintage record player with vinyl records",
    "a city park in full bloom during spring",
    "a fishing boat returning to harbor at dusk",
    "a close-up of a butterfly on a lavender flower",
    "a snowy village square with a lit christmas tree",
    "a hot spring surrounded by snow-covered rocks",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out_dir", default="results/sd21_corpus")
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(SD21, torch_dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)

    n = min(args.n, len(PROMPTS))
    for i in range(n):
        g = torch.Generator(device=dev).manual_seed(1000 + i)
        img = pipe(PROMPTS[i], num_inference_steps=args.steps, guidance_scale=7.5,
                   height=512, width=512, generator=g).images[0]
        img.save(os.path.join(args.out_dir, f"sd_{i:04d}.png"))
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] generated", flush=True)
    print(f"[done] {n} SD-2.1 images -> {args.out_dir}\nSD_CORPUS_DONE")


if __name__ == "__main__":
    main()
