"""Fingerprint-env bridge: regen/rinse the StegaStamp wm PNGs saved by measure_stega_aiwm.py,
so aiwm can then decode them. Keeps the SD regen in the env that has it."""
import os,sys,glob
sys.path.insert(0,"/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
from PIL import Image
from _regen_util import build_regen_pipe, stable_regen
WORK="/scratch/workspace/mingzhel_umass_edu-ablator/stega_work"
N=int(sys.argv[1]) if len(sys.argv)>1 else 50
pipe=build_regen_pipe()
for i in range(N):
    p=WORK+f"/wm_{i:04d}.png"
    if not os.path.exists(p): continue
    im=Image.open(p).convert("RGB")
    r=stable_regen(pipe,im,1234); r.save(WORK+f"/regen_{i:04d}.png")
    r2=stable_regen(pipe,r,1235);  r2.save(WORK+f"/rinse_{i:04d}.png")
    if (i+1)%10==0: print(f"  regen [{i+1}/{N}]",flush=True)
print("REGEN_PNGS_DONE")
