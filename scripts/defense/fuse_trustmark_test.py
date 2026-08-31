"""Empirical test: does FUSING TrustMark (vs keeping it as an independent OR) help or hurt?

TrustMark supports 100 raw bits with use_ECC=False (decoder net outputs secret_len=100;
soft per-bit signal = self.decoder.decoder(stego), thresholded at >0). So we can make it a
fusible fragment carrying the SAME shortened-BCH codeword under its own crypto key, expose
raw_logits, and fuse it with VINE/DFT/QIM.

Compare fused bit-acc with 3 fragments {VINE,DFT,QIM} vs 4 {…,+TrustMark}, per attack.
Hypothesis: under regen TrustMark is at chance but CONFIDENT -> MRC fusion drags down.
"""
import argparse, glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image
from torchvision import transforms

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto, derive_method_keyed_constants
from src.learned_fragment_methods import DFTKredMethod, QuantQIMMethod
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.payload import image_id_to_payload
KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"


class TrustMarkFused:
    """TrustMark as a fusible fragment: 100 raw bits (no ECC), crypto-keyed, soft logits."""
    def __init__(self, master_key, method_name, n_bits=100, model_type="B", device="cuda"):
        from trustmark import TrustMark
        self.tm = TrustMark(verbose=False, model_type=model_type, use_ECC=False, secret_len=n_bits)
        self.n_bits = n_bits; self.master_key = master_key; self.method_name = method_name
        self.device = device

    def get_perm_M(self, image_id):
        return derive_method_keyed_constants(self.master_key, image_id, self.method_name, self.n_bits)

    def embed_with_target(self, pil, target_bits):
        s = "".join(str(int(b)) for b in np.asarray(target_bits, np.uint8)[:self.n_bits])
        return self.tm.encode(pil.convert("RGB"), s, MODE="binary")

    def raw_logits(self, pil):
        r = self.tm.model_resolution_dec
        img = pil.convert("RGB").resize((r, r), Image.BILINEAR)
        x = transforms.ToTensor()(img).unsqueeze(0).to(self.tm.decoder.device) * 2.0 - 1.0
        with torch.no_grad():
            logits = self.tm.decoder.decoder(x).cpu().numpy().reshape(-1)[:self.n_bits]
        return logits.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=15)
    ap.add_argument("--attacks", nargs="+", default=["clean", "jpeg", "regen", "rinse2x", "vae_c", "hflip", "crop75", "rs256"])
    args = ap.parse_args()
    dev = "cuda"
    sb = ShortenedBCH()
    vine = VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev)
    dft = DFTKredMethod(os.path.join(REPO, "results/dft_fftaware_baseline/ckpt.pt"), KEY, "dft_kred", dev)
    qim = QuantQIMMethod(os.path.join(REPO, "results/quant_qim_frozen_d006/ckpt.pt"), KEY, "quant_qim", dev)
    tmf = TrustMarkFused(KEY, "trustmark_fused", n_bits=sb.n, model_type="B", device=dev)

    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker, VAEWMAttacker, JPEGAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    jp = JPEGAttacker(quality=25); vae_c = VAEWMAttacker("cheng2020-anchor", quality=3, metric="mse", device=dev)
    tmp = tempfile.mkdtemp()

    def attack(a, pil):
        ip = os.path.join(tmp, "i.png"); pil.save(ip); op = os.path.join(tmp, "o.png")
        if a == "clean": return pil
        if a == "jpeg": jp.attack([ip], [op])
        elif a == "vae_c": vae_c.attack([ip], [op])
        elif a == "regen": regen.attack([ip], [op])
        elif a == "rinse2x":
            cur = ip
            for r in range(2):
                nxt = os.path.join(tmp, f"r{r}.png"); regen.attack([cur], [nxt]); cur = nxt
            return Image.open(cur).convert("RGB").resize((512, 512))
        elif a == "hflip":
            return pil.transpose(Image.FLIP_LEFT_RIGHT)
        elif a == "crop75":
            W, H = pil.size; f = 0.75; cw, ch = int(W * f), int(H * f)
            l, t = (W - cw) // 2, (H - ch) // 2
            return pil.crop((l, t, l + cw, t + ch)).resize((W, H), Image.BICUBIC)
        elif a == "rs256":
            return pil.resize((256, 256), Image.BICUBIC).resize((512, 512), Image.BICUBIC)
        return Image.open(op).convert("RGB").resize((512, 512))

    imgs = sorted(glob.glob("/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017/*.jpg"))[4000:4000 + args.n_images]
    # accumulators
    R = {a: {"fa3": [], "fa4": [], "tm": [], "vine": []} for a in args.attacks}
    # verify TrustMark-100bit-noECC clean recovery once
    for i, fp in enumerate(imgs):
        image_id = f"ft_{i:05d}"; tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))
        img = Image.open(fp).convert("RGB").resize((512, 512))
        # embed all 4 carrying the SAME codeword (VINE->DFT->QIM->TrustMark)
        pv, Mv = vine.get_perm_M(image_id); img = vine.embed_with_target(img, apply_crypto(tx, pv, Mv)).resize((512, 512))
        img = dft.embed(img, image_id, tx).resize((512, 512))
        img = qim.embed(img, image_id, tx).resize((512, 512))
        pt, Mt = tmf.get_perm_M(image_id); img = tmf.embed_with_target(img, apply_crypto(tx, pt, Mt)).resize((512, 512))
        for a in args.attacks:
            att = attack(a, img)
            al = {}
            for name, m, kind, getter, idkey in [
                ("vine", vine, "prob", "raw_probs", None), ("dft", dft, "logit", "raw_logits", None),
                ("qim", qim, "logit", "raw_logits", None), ("tm", tmf, "logit", "raw_logits", None)]:
                perm, M = m.get_perm_M(image_id)
                soft = getattr(m, getter)(att)
                al[name] = method_soft_to_codeword_llr(soft, perm, M, kind=kind, n_codeword=sb.n)
            f3 = fuse_llrs({k: al[k] for k in ("vine", "dft", "qim")}, weights=None, n_codeword=sb.n)
            f4 = fuse_llrs(al, weights=None, n_codeword=sb.n)
            R[a]["fa3"].append(float(np.mean(llr_to_bits(f3) == tx)))
            R[a]["fa4"].append(float(np.mean(llr_to_bits(f4) == tx)))
            R[a]["tm"].append(float(np.mean(llr_to_bits(al["tm"]) == tx)))
            R[a]["vine"].append(float(np.mean(llr_to_bits(al["vine"]) == tx)))
        print(f"  [{i+1}/{len(imgs)}]", flush=True)

    print(f"\n=== FUSE TrustMark? bit-acc, n={len(imgs)} (TrustMark-B, 100bit no-ECC) ===")
    print(f"{'attack':<10}{'fuse3':>9}{'fuse4(+TM)':>12}{'delta':>8}{'TM_alone':>10}{'VINE_alone':>12}")
    for a in args.attacks:
        fa3 = np.mean(R[a]["fa3"]); fa4 = np.mean(R[a]["fa4"])
        print(f"{a:<10}{fa3:>9.3f}{fa4:>12.3f}{fa4-fa3:>+8.3f}{np.mean(R[a]['tm']):>10.3f}{np.mean(R[a]['vine']):>12.3f}")
    print("\nfuse3={VINE,DFT,QIM}  fuse4=+TrustMark.  delta<0 => fusing TrustMark HURTS.\nFUSE_TM_DONE")


if __name__ == "__main__":
    main()
