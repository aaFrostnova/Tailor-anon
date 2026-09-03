"""Decisive eval of the geometry-finetuned VINE vs the PRETRAINED VINE-R. Did geometric adversarial
finetuning (a) lift rot9/rot30/crop bit-acc above the 0.50 floor, (b) keep clean ~1.0, (c) forget
regen? Loads both models, embeds a random 100-bit secret (exact wrapper convention), attacks, decodes."""
import os, sys, glob, io, argparse
import numpy as np, torch
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
import kornia.geometry.transform as KT
VINE_REPO = "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
sys.path.insert(0, VINE_REPO); sys.path.insert(0, os.path.join(VINE_REPO, "vine", "src"))
sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
dev = "cuda"

def load_vine(ckpt=None):
    enc = VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc").to(dev).eval()
    dec = CustomConvNeXt.from_pretrained("Shilin-LU/VINE-R-Dec").to(dev).eval()
    if ckpt:
        enc.unet.load_state_dict(torch.load(os.path.join(ckpt, "UNet2DConditionModel.pth"), map_location="cpu"))
        enc.vae_a2b.load_state_dict(torch.load(os.path.join(ckpt, "vae.pth"), map_location="cpu"))
        enc.sec_encoder.load_state_dict(torch.load(os.path.join(ckpt, "ConditionAdaptor.pth"), map_location="cpu"))
        dec.load_state_dict(torch.load(os.path.join(ckpt, "CustomConvNeXt.pth"), map_location="cpu"))
        enc.to(dev).eval(); dec.to(dev).eval()
    return enc, dec

t256 = transforms.Compose([transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])
def embed(enc, img01, sec):
    with torch.no_grad(): wm = enc(img01 * 2 - 1, sec)
    return ((wm + 1) / 2).clamp(0, 1)
def dec_ba(dec, x01, sec):
    with torch.no_grad(): p = dec(x01)
    return float(((p > 0.5).float() == sec).float().mean())

def a_rot(x, d): return KT.rotate(x, torch.tensor([float(d)], device=x.device), mode="bilinear", padding_mode="reflection")
def a_crop(x, r):
    H = x.shape[-1]; c = int(H * r); return KT.resize(KT.center_crop(x, (c, c)), (H, H))
def a_jpeg(x, q):
    im = Image.fromarray((x[0].permute(1, 2, 0) * 255).byte().cpu().numpy()); b = io.BytesIO()
    im.save(b, "JPEG", quality=q); b.seek(0)
    return transforms.ToTensor()(Image.open(b).convert("RGB")).unsqueeze(0).to(x.device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/workspace/mingzhel_umass_edu-ablator/vine_geo_ft/checkpoint-6000")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob("results/defense/ext_vtv100/*.png"))[:args.n]
    imgs = [t256(Image.open(f).convert("RGB")).unsqueeze(0).to(dev) for f in files]
    torch.manual_seed(0)
    secs = [torch.randint(0, 2, (1, 100)).float().to(dev) for _ in files]
    ATT = {"clean": lambda x: x, "rot9": lambda x: a_rot(x, 9), "rot30": lambda x: a_rot(x, 30),
           "crop90": lambda x: a_crop(x, 0.9), "crop75": lambda x: a_crop(x, 0.75), "jpeg50": lambda x: a_jpeg(x, 50)}
    pipe = None
    if args.regen:
        from _regen_util import build_regen_pipe, stable_regen
        pipe = build_regen_pipe()
    models = {"pretrained": load_vine(None), "finetuned": load_vine(args.ckpt)}
    print(f"VINE geo-finetune eval, n={len(files)}, ckpt={os.path.basename(args.ckpt)}\n", flush=True)
    print(f"{'attack':8s} | {'pretrained':10s} | {'finetuned':10s} | delta")
    print("-" * 46)
    res = {m: {a: [] for a in list(ATT) + (["regen"] if args.regen else [])} for m in models}
    for k, (img, sec) in enumerate(zip(imgs, secs)):
        for mname, (enc, dec) in models.items():
            wm01 = embed(enc, img, sec)
            for a, fn in ATT.items():
                res[mname][a].append(dec_ba(dec, fn(wm01).clamp(0, 1), sec))
            if args.regen:
                pil = Image.fromarray((wm01[0].permute(1, 2, 0) * 255).byte().cpu().numpy())
                rg = stable_regen(pipe, pil, 1234 + k).resize((256, 256))
                rt = transforms.ToTensor()(rg).unsqueeze(0).to(dev)
                res[mname]["regen"].append(dec_ba(dec, rt, sec))
    for a in list(ATT) + (["regen"] if args.regen else []):
        pa = np.mean(res["pretrained"][a]); fa = np.mean(res["finetuned"][a])
        print(f"{a:8s} | {pa:10.3f} | {fa:10.3f} | {fa-pa:+.3f}")
    print("\n(bit-acc; crypto-verify needs ~0.90. Did finetune lift rot/crop ABOVE pretrained without losing clean/regen?)")
    print("EVAL_DONE")

if __name__ == "__main__":
    main()
