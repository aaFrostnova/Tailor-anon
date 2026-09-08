"""Two OUT-OF-DOMAIN image sets, for the transfer half of the certification.

In-domain = the pool the offline table was fitted on (A COCO, B DiffusionDB SD1.5, C UltraEdit,
D DALL-E 3, E DIV2K+LSDIR), held-out images. Out-of-domain = sources the pool never contained:
  ood_content   ImageNet-1k validation: real photographs, a content and resolution shift
  ood_generator open-image-preferences-v1: FLUX / SD3-era generations, a generator shift (the pool's
                synthetic half is SD1.5 and DALL-E 3)
Usage: python build_ood_sets.py <content|generator> [n=100]
"""
import os, sys, io, itertools
from PIL import Image
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")), "hf_cache"))
from datasets import load_dataset
WHICH = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SPEC = {"content": ("ILSVRC/imagenet-1k", None, "validation", "ood_content"),
        "generator": ("data-is-better-together/open-image-preferences-v1", None, "cleaned", "ood_generator")}
name, cfg, split, out = SPEC[WHICH]
d = f"{SC}/{out}"; os.makedirs(d, exist_ok=True)
ds = load_dataset(name, cfg, split=split, streaming=True) if cfg else load_dataset(name, split=split, streaming=True)
def first_image(ex):
    for v in ex.values():
        if isinstance(v, Image.Image): return v
        if isinstance(v, dict) and v.get("bytes"):
            try: return Image.open(io.BytesIO(v["bytes"]))
            except Exception: pass
    return None
n = 0
for ex in ds:
    im = first_image(ex)
    if im is None: continue
    im = im.convert("RGB")
    if min(im.size) < 256: continue                      # too small to resize to 512 without inventing detail
    im.resize((512, 512), Image.BICUBIC).save(f"{d}/i{n:05d}.png")
    n += 1
    if n >= N: break
print(f"wrote {n} images to {d} from {name}:{split}  OOD_BUILD_DONE", flush=True)
