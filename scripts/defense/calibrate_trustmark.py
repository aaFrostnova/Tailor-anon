"""Find the TrustMark variant matching RAVEN's Regen robustness (0.999).
We currently use model_type='Q' (quality, low-robustness) -> dies under Regen (0.495).
RAVEN reports TrustMark Regen=0.999, so it uses a robust variant. Test Q/P/B/C."""
import glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from regen_pipe import ReSDPipeline
from wmattacker import DiffWMAttacker
from diffusers import DPMSolverMultistepScheduler
from trustmark import TrustMark

SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
n = 6
imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))[4000:4000+n]
pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True); pipe = pipe.to("cuda")
attacker = DiffWMAttacker(pipe, batch_size=n, noise_step=60)
tmp = tempfile.mkdtemp()
print(f"{'variant':<10}{'clean':>8}{'Regen':>8}{'vae_rt':>8}  (RAVEN Regen=0.999)", flush=True)
for mt in ["Q", "P", "B", "C"]:
    try:
        tm = TrustMark(verbose=False, model_type=mt)
    except Exception as e:
        print(f"{mt:<10} load FAIL: {type(e).__name__}: {str(e)[:50]}", flush=True); continue
    cap = tm.schemaCapacity()
    ins, outs, secrets = [], [], []
    for i, p in enumerate(imgs):
        cover = Image.open(p).convert("RGB").resize((512, 512))
        s = "".join(str(x) for x in np.random.RandomState(i).randint(0, 2, cap))
        wm = tm.encode(cover, s, MODE="binary")
        if wm.size != (512, 512): wm = wm.resize((512, 512))
        ip = os.path.join(tmp, f"{mt}_{i}_i.png"); op = os.path.join(tmp, f"{mt}_{i}_o.png")
        wm.save(ip); ins.append(ip); outs.append(op); secrets.append(s)
    def acc(paths):
        a = []
        for i, pth in enumerate(paths):
            out = tm.decode(Image.open(pth).convert("RGB").resize((512,512)), MODE="binary")
            d = out[0] if isinstance(out,(tuple,list)) else out
            d = (d or "")[:len(secrets[i])]
            if not d: a.append(0.0); continue
            a.append(np.mean([c1==c2 for c1,c2 in zip(d, secrets[i])]))
        return float(np.mean(a))
    clean = acc(ins)
    attacker.attack(ins, outs)
    regen = acc(outs)
    print(f"{mt:<10}{clean:>8.3f}{regen:>8.3f}{'-':>8}  cap={cap}", flush=True)
