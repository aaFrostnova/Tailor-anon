"""Finetune the PUBLIC VINE-R encoder+decoder (the exact weights our fragment uses) with GEOMETRIC
adversarial augmentation ADDED to VINE's ORIGINAL distortion categories (jpeg/blur/noise/contrast/
brightness/cropout) so the model does NOT forget its existing robustness -- plus a frozen SD-VAE
regen surrogate (regen was the main casualty of the geometry-only v1 and is not an original category).

Conventions match src/vine_crypto_wrapper.py exactly: encoder eats [-1,1] 256px + a 100-bit secret and
returns [-1,1]; decoder eats [0,1] 256px -> 100 sigmoid probs. Full finetune (unet+vae+sec_encoder+dec).
Multi-GPU via accelerate. Run `--smoke` first to validate conventions + a few steps."""
import os, sys, io, math, argparse, glob, random
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
import kornia.geometry.transform as KT
import kornia.filters as KF

VINE_REPO = "/project/pi_shiqingma_umass_edu/mingzheli/watermark/vine/repo"
sys.path.insert(0, VINE_REPO); sys.path.insert(0, os.path.join(VINE_REPO, "vine", "src"))
from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"

# ---------------- distortion: geometry + ORIGINAL VINE categories + regen surrogate (all in [0,1]) ----------------
def _st_jpeg(x01, q):  # real jpeg, straight-through gradient
    out = []
    for i in range(x01.shape[0]):
        im = Image.fromarray((x01[i].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
        b = io.BytesIO(); im.save(b, "JPEG", quality=int(q)); b.seek(0)
        out.append(transforms.ToTensor()(Image.open(b).convert("RGB")))
    j = torch.stack(out).to(x01.device, x01.dtype)
    return x01 + (j - x01).detach()

def distort(x01, cover01, step, args, vae_regen):
    """x01 in [0,1] B3HW. Geometry ramps in; original photometric/compression categories + a regen
    surrogate stay at full strength throughout (the 'do not forget' part). Each applied independently
    so a sample gets a realistic SUBSET (composed), some samples stay clean."""
    B, _, H, W = x01.shape; dev = x01.device
    def p(pr): return float(torch.rand(1, device=dev)[0]) < pr
    # --- geometry (ramped 0 -> full) ---
    g = min(max((step - args.geometric_step) / max(1, args.geometric_ramp), 0.0), 1.0)
    if step >= args.geometric_step and p(args.geometric_prob):
        r = float(torch.rand(1, device=dev)[0])
        if r < 0.6:
            ang = (torch.rand(B, device=dev) * 2 - 1) * (args.max_rot_deg * g)
            x01 = KT.rotate(x01, ang, mode="bilinear", padding_mode="reflection")
        elif r < 0.8:
            keep = 1.0 - (1.0 - args.min_keep) * g * float(torch.rand(1, device=dev)[0])
            ch = max(16, int(H * keep)); cw = max(16, int(W * keep))
            x01 = KT.resize(KT.center_crop(x01, (ch, cw)), (H, W))
        else:
            ds = 0.10 * g
            pts = torch.tensor([[0., 0.], [W-1, 0.], [W-1, H-1], [0., H-1]], device=dev).unsqueeze(0).repeat(B, 1, 1)
            jit = (torch.rand(B, 4, 2, device=dev) * 2 - 1) * ds * torch.tensor([W, H], device=dev)
            M = KT.get_perspective_transform(pts, (pts + jit).clamp(min=0))
            x01 = KT.warp_perspective(x01, M, (H, W), mode="bilinear")
    # --- ORIGINAL VINE categories (full strength, keep the pretrained robustness) ---
    if p(0.3):                                                     # gaussian blur
        s = float(torch.empty(1).uniform_(0.5, 2.0)); k = 7
        x01 = KF.gaussian_blur2d(x01, (k, k), (s, s))
    if p(0.3):                                                     # gaussian noise
        x01 = (x01 + torch.randn_like(x01) * float(torch.empty(1).uniform_(0.01, 0.05)))
    if p(0.3):                                                     # contrast + brightness
        c = float(torch.empty(1).uniform_(0.7, 1.3)); brt = float(torch.empty(1).uniform_(-0.1, 0.1))
        m = x01.mean(dim=(1, 2, 3), keepdim=True); x01 = (m + (x01 - m) * c + brt)
    if p(0.3):                                                     # jpeg (straight-through)
        x01 = _st_jpeg(x01.clamp(0, 1), float(torch.empty(1).uniform_(40, 85)))
    if p(0.2):                                                     # cropout: paste a random patch of the COVER (kills mark locally)
        ph, pw = int(H * 0.3), int(W * 0.3)
        ty, tx = random.randint(0, H - ph), random.randint(0, W - pw)
        x01 = x01.clone(); x01[:, :, ty:ty+ph, tx:tx+pw] = cover01[:, :, ty:ty+ph, tx:tx+pw]
    # --- regen surrogate: frozen SD-VAE roundtrip (the main thing v1 forgot) ---
    if vae_regen is not None and p(args.regen_prob):
        x11 = (x01.clamp(0, 1) * 2 - 1)
        z = vae_regen.encode(x11).latent_dist.mode()
        x01 = ((vae_regen.decode(z).sample.clamp(-1, 1)) + 1) / 2
    return x01.clamp(0, 1)

def load_models(dev):
    enc = VINE_Turbo.from_pretrained("Shilin-LU/VINE-R-Enc").to(dev)
    dec = CustomConvNeXt.from_pretrained("Shilin-LU/VINE-R-Dec").to(dev)
    for m in (enc.unet, enc.vae_a2b, enc.sec_encoder): m.requires_grad_(True); m.train()
    dec.requires_grad_(True); dec.train()
    return enc, dec

def embed(enc, img01, secret):
    wm = enc(img01 * 2 - 1, secret)
    return wm
def decode(dec, x01): return dec(x01)

class ImgSecretData(torch.utils.data.Dataset):
    def __init__(self, folder, n_bits=100, size=256, limit=None):
        self.files = []
        for e in ("*.jpg", "*.jpeg", "*.png"): self.files += glob.glob(os.path.join(folder, e))
        self.files = sorted(self.files)[:limit] if limit else sorted(self.files)
        self.n_bits = n_bits
        self.tf = transforms.Compose([transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
                                      transforms.ToTensor()])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB")), torch.randint(0, 2, (self.n_bits,)).float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_folder", default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    ap.add_argument("--output_dir", default="/scratch/workspace/mingzhel_umass_edu-ablator/vine_geo_ft2")
    ap.add_argument("--train_batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--max_train_steps", type=int, default=20000)
    ap.add_argument("--checkpointing_steps", type=int, default=2000)
    ap.add_argument("--dataloader_num_workers", type=int, default=6)
    ap.add_argument("--mixed_precision", default="bf16")
    ap.add_argument("--secret_size", type=int, default=100)
    ap.add_argument("--secret_loss_scale", type=float, default=1.5)
    ap.add_argument("--image_loss_scale", type=float, default=2.0)
    ap.add_argument("--lpips_loss_scale", type=float, default=1.0)
    ap.add_argument("--no_im_loss_steps", type=int, default=200)
    ap.add_argument("--geometric_step", type=int, default=0)
    ap.add_argument("--geometric_ramp", type=int, default=3000)
    ap.add_argument("--geometric_prob", type=float, default=0.5)
    ap.add_argument("--max_rot_deg", type=float, default=30.0)
    ap.add_argument("--min_keep", type=float, default=0.75)
    ap.add_argument("--regen_prob", type=float, default=0.3)
    ap.add_argument("--no_regen_surrogate", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    acc = Accelerator(mixed_precision=args.mixed_precision)
    if args.seed is not None: set_seed(args.seed)
    dev = acc.device
    enc, dec = load_models(dev)
    B = args.train_batch_size
    if hasattr(enc, "timesteps"): enc.timesteps = enc.timesteps[:1].repeat(B)
    if hasattr(enc, "fixed_a2b_emb_base") and enc.fixed_a2b_emb_base is not None:
        e = enc.fixed_a2b_emb_base; enc.fixed_a2b_emb_base = e[:1].repeat(B, *([1] * (e.dim() - 1)))
    acc.print(f"[batch-fix] timesteps={tuple(enc.timesteps.shape)} emb={tuple(enc.fixed_a2b_emb_base.shape)} B={B}")

    vae_regen = None
    if not args.no_regen_surrogate:
        from diffusers import AutoencoderKL
        vae_regen = AutoencoderKL.from_pretrained(SD21, subfolder="vae").to(dev).eval()
        vae_regen.requires_grad_(False)
        acc.print("[regen-surrogate] loaded frozen SD-2.1 VAE")
    import lpips as lpips_mod
    net_lpips = lpips_mod.LPIPS(net="vgg").to(dev); net_lpips.requires_grad_(False); net_lpips.eval()

    opt = torch.optim.Adam(list(enc.unet.parameters()) + list(enc.vae_a2b.parameters())
                           + list(enc.sec_encoder.parameters()) + list(dec.parameters()), lr=args.learning_rate)
    ds = ImgSecretData(args.dataset_folder, n_bits=args.secret_size, limit=2000 if args.smoke else None)
    dl = torch.utils.data.DataLoader(ds, batch_size=B, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True)
    (enc.unet, enc.vae_enc, enc.vae_dec, enc.sec_encoder, dec, opt, dl) = acc.prepare(
        enc.unet, enc.vae_enc, enc.vae_dec, enc.sec_encoder, dec, opt, dl)

    if args.smoke:
        enc.eval(); dec.eval()
        img, sec = next(iter(dl)); img, sec = img.to(dev), sec.to(dev)
        acc.print(f"[SMOKE] img={tuple(img.shape)} mp={args.mixed_precision}")
        with torch.no_grad():
            wm01 = ((embed(enc, img, sec) + 1) / 2).clamp(0, 1)
            ba_clean = ((decode(dec, wm01) > 0.5).float() == sec).float().mean().item()
            ba_dist = ((decode(dec, distort(wm01.clone(), img, 10**9, args, vae_regen)) > 0.5).float() == sec).float().mean().item()
            psnr = 10 * math.log10(1.0 / max(1e-9, F.mse_loss(wm01, img).item()))
        acc.print(f"[SMOKE] CLEAN bit-acc={ba_clean:.3f} (want ~1.0)  full-distort bit-acc={ba_dist:.3f}  PSNR={psnr:.1f}dB")
        enc.train(); dec.train()
        for s in range(3):
            img, sec = next(iter(dl)); img, sec = img.to(dev), sec.to(dev)
            wm = embed(enc, img, sec); wm01 = ((wm + 1) / 2).clamp(0, 1)
            probs = decode(dec, distort(wm01, img, s, args, vae_regen))
            loss = args.secret_loss_scale * F.binary_cross_entropy(probs.clamp(1e-6, 1-1e-6), sec) + args.image_loss_scale * F.mse_loss(wm, img*2-1)
            opt.zero_grad(); acc.backward(loss); opt.step()
            acc.print(f"[SMOKE step {s}] loss={loss.item():.3f}")
        acc.print("SMOKE_OK"); return

    os.makedirs(args.output_dir, exist_ok=True)
    step = 0; it = iter(dl)
    while step < args.max_train_steps:
        try: img, sec = next(it)
        except StopIteration: it = iter(dl); img, sec = next(it)
        img, sec = img.to(dev), sec.to(dev)
        wm = embed(enc, img, sec); wm01 = ((wm + 1) / 2).clamp(0, 1)
        probs = decode(dec, distort(wm01, img, step, args, vae_regen))
        secret_loss = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), sec)
        if step < args.no_im_loss_steps:
            loss = secret_loss
        else:
            loss = (args.secret_loss_scale * secret_loss + args.image_loss_scale * F.mse_loss(wm, img * 2 - 1)
                    + args.lpips_loss_scale * net_lpips(wm, img * 2 - 1).mean())
        opt.zero_grad(); acc.backward(loss); opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                ba = ((probs > 0.5).float() == sec).float().mean().item()
                ba_clean = ((decode(dec, wm01) > 0.5).float() == sec).float().mean().item()
            acc.print(f"[{step}/{args.max_train_steps}] loss={loss.item():.3f} secret={secret_loss.item():.3f} ba_dist={ba:.3f} ba_clean={ba_clean:.3f}")
        step += 1
        if step % args.checkpointing_steps == 0 and acc.is_main_process:
            d = os.path.join(args.output_dir, f"checkpoint-{step}"); os.makedirs(d, exist_ok=True)
            torch.save(acc.unwrap_model(enc.unet).state_dict(), os.path.join(d, "UNet2DConditionModel.pth"))
            torch.save(enc.vae_a2b.state_dict(), os.path.join(d, "vae.pth"))
            torch.save(acc.unwrap_model(enc.sec_encoder).state_dict(), os.path.join(d, "ConditionAdaptor.pth"))
            torch.save(acc.unwrap_model(dec).state_dict(), os.path.join(d, "CustomConvNeXt.pth"))
            acc.print(f"[ckpt] saved {d}")
    acc.print("TRAIN_DONE")

if __name__ == "__main__":
    main()
