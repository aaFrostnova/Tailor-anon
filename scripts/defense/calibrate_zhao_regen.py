"""Calibrate the faithful Zhao Regen attack (RAVEN's 'Regen') and check it matches RAVEN.

RAVEN's Regen = DiffWMAttacker(noise_step=60) from XuandongZhao/WatermarkAttacker (NeurIPS'24):
VAE-encode -> add noise to timestep 60/1000 -> denoise last ~3 steps. Gentle round-trip.
RAVEN MS-COCO reference under Regen: TrustMark 0.999, VINE 0.881, DwtDct 0.519, DwtDctSvd 0.644.
If our reimpl reproduces those LEVELS, we are aligned with the paper.
"""
import glob
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))

from regen_pipe import ReSDPipeline                      # noqa: E402
from wmattacker import DiffWMAttacker                     # noqa: E402
from wbench.methods import build_methods                  # noqa: E402

SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
RAVEN_REF = {"DwtDct": 0.519, "DwtDctSvd": 0.644, "RivaGAN": 0.608,
             "TrustMark-Q": 0.999, "VINE-R": 0.881}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    noise_step = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    device = "cuda"
    print(f"[calibrate] loading ReSDPipeline (SD-2.1), noise_step={noise_step}", flush=True)
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    try:
        pipe.safety_checker = None
    except Exception:
        pass
    attacker = DiffWMAttacker(pipe, batch_size=n, noise_step=noise_step)

    methods = build_methods(["dwtDct", "dwtDctSvd", "rivaGan", "trustmark", "vine_r"], device)
    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))[4000:4000 + n]

    tmp = tempfile.mkdtemp()
    print(f"{'defense':<14}{'our Regen':>11}{'RAVEN':>9}{'clean':>8}", flush=True)
    for name, m in methods.items():
        bits_list, ins, outs = [], [], []
        for i, p in enumerate(imgs):
            cover = Image.open(p).convert("RGB").resize((512, 512))
            b = np.random.RandomState(i).randint(0, 2, m.n_bits).astype(np.uint8)
            wm = m.embed(cover, b)
            if wm.size != (512, 512):
                wm = wm.resize((512, 512))
            ip = os.path.join(tmp, f"{name}_{i}_in.png"); op = os.path.join(tmp, f"{name}_{i}_out.png")
            wm.save(ip); bits_list.append(b); ins.append(ip); outs.append(op)
        clean = float(np.mean([np.mean(m.decode(Image.open(ins[i]))[:m.n_bits] == bits_list[i]) for i in range(len(ins))]))
        attacker.attack(ins, outs)
        accs = []
        for i in range(len(outs)):
            att = Image.open(outs[i]).convert("RGB")
            if att.size != (512, 512):
                att = att.resize((512, 512))
            rec = m.decode(att); nb = min(len(rec), len(bits_list[i]))
            accs.append(np.mean(rec[:nb] == bits_list[i][:nb]))
        ours = float(np.mean(accs))
        ref = RAVEN_REF.get(m.name, float("nan"))
        print(f"{m.name:<14}{ours:>11.3f}{ref:>9.3f}{clean:>8.3f}", flush=True)


if __name__ == "__main__":
    main()
