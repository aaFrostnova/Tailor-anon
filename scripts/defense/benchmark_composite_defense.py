"""OUR multi-fragment COMBINED defense vs the faithful RAVEN attack suite.

Embeds the full post-hoc composite (one image, cross-subspace, correct order):
    PhaseMark (VAE-latent phase) -> VINE (SDXL latent) -> DFT-Kred (FFT-mag QIM)
    -> Quant-QIM (block QIM) -> TrustMark (deep pixel, last so PhaseMark's VAE
       round-trip doesn't overwrite it)
The 4 codeword fragments (PhaseMark/VINE/DFT/QIM) carry the SAME shortened-BCH(100,37)
codeword under per-method crypto keys and are soft-LLR fused + Chase-BCH decoded; TrustMark
is an independent OR corroborator.

Attacks = RAVEN's exact suite (Zhao WatermarkAttacker): clean, jpeg, blur, noise, bright,
contrast, bm3d, regen(noise_step=60), rinse2x, rinse4x, vae_b(bmshj q3), vae_c(cheng q3).

Reports, per attack, the COMPOSITE detection: fused-BCH-verify (exact id), fused zero-bit
(fused bit-acc >= 1%-FPR tau), and the OR with TrustMark -- plus the fused bit-acc and the
per-fragment bit-acc for insight. Persistent JSON + log.
"""
import argparse, glob, json, os, sys, tempfile
import numpy as np, torch
from PIL import Image
from scipy.stats import binom

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))

from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.phasemark import PhaseMarkWrapper
from src.trustmark_fragment import TrustMarkFragment
from src.videoseal_fragment import VideoSealFragment
from src.learned_fragment_methods import DFTKredMethod, QuantQIMMethod
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload

SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
KEY = b"v5_key_encoder_master"
ATTACKS = ["clean","jpeg","blur","noise","bright","contrast","bm3d","regen","rinse2x","rinse4x","vae_b","vae_c"]
GEO_ATTACKS = ["crop75","crop50","rot9","rs256","hflip","crop_jpeg"]   # geometric (RAVEN/WAVES style)


def _center_crop_resize(img, frac):
    W, H = img.size; cw, ch = int(round(W * frac)), int(round(H * frac))
    l, t = (W - cw) // 2, (H - ch) // 2
    return img.crop((l, t, l + cw, t + ch)).resize((W, H), Image.BICUBIC)


def _rotate_reflect(img, deg):
    import numpy as _np
    W, H = img.size; arr = _np.array(img); pad = max(W, H) // 2
    refl = _np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    big = Image.fromarray(refl).rotate(deg, resample=Image.BICUBIC, expand=False)
    bw, bh = big.size; l, t = (bw - W) // 2, (bh - H) // 2
    return big.crop((l, t, l + W, t + H))


def _resize_down_up(img, mid):
    W, H = img.size
    return img.resize((mid, mid), Image.BICUBIC).resize((W, H), Image.BICUBIC)


GEO = {
    "crop75": lambda im: _center_crop_resize(im, 0.75),
    "crop50": lambda im: _center_crop_resize(im, 0.50),
    "rot9":   lambda im: _rotate_reflect(im, 9.0),
    "rs256":  lambda im: _resize_down_up(im, 256),
    "hflip":  lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
}


def _crop_then_jpeg(ip, op, frac=0.75, quality=25):
    import io
    img = _center_crop_resize(Image.open(ip).convert("RGB"), frac)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality)
    Image.open(io.BytesIO(buf.getvalue())).convert("RGB").save(op)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=10)
    p.add_argument("--start_idx", type=int, default=4000)
    p.add_argument("--attacks", nargs="+", default=ATTACKS)
    p.add_argument("--image_dir", default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017",
                   help="cover image dir (COCO photos default; pass the SD corpus for on-manifold eval)")
    p.add_argument("--image_glob", default="*.jpg")
    p.add_argument("--output", default="results/defense/composite_defense.json")
    p.add_argument("--fragments", nargs="+", default=["phasemark", "vine", "dft", "qim"],
                   help="codeword fragments to stack+fuse, in embed order")
    p.add_argument("--pm_gamma", type=float, default=1.0, help="PhaseMark soft-steer (1.0=hard APM)")
    p.add_argument("--pm_strength", type=float, default=1.0, help="PhaseMark residual strength")
    p.add_argument("--pm_perceptual", action="store_true", help="PhaseMark perceptual residual mask")
    p.add_argument("--tm_variant", default="B", help="TrustMark model variant (B strongest; Q P B C)")
    args = p.parse_args()
    dev = "cuda"
    sb = ShortenedBCH(); tau = float(binom.ppf(0.99, sb.n, 0.5) + 1) / sb.n

    # --- fragments (all carry the shortened-BCH codeword) + TrustMark (independent) ---
    _builders = {
        "phasemark": lambda: PhaseMarkWrapper(master_key=KEY, method_name="phasemark", n_bits=sb.n, vae_key="sd21", device=dev,
                                              gamma=args.pm_gamma, strength=args.pm_strength, perceptual=args.pm_perceptual),
        "vine":      lambda: VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev),
        "dft":       lambda: DFTKredMethod(os.path.join(REPO, "results/dft_fftaware_baseline/ckpt.pt"), KEY, "dft_kred", dev),
        "qim":       lambda: QuantQIMMethod(os.path.join(REPO, "results/quant_qim_frozen_d006/ckpt.pt"), KEY, "quant_qim", dev),
        "trustmark": lambda: TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type=args.tm_variant, device=dev),
        "videoseal": lambda: VideoSealFragment(master_key=KEY, method_name="videoseal", n_bits=sb.n, device=dev),
    }
    SPEC = {"phasemark": ("logit", "raw_scores", "target"), "vine": ("prob", "raw_probs", "target"),
            "dft": ("logit", "raw_logits", "id_tx"), "qim": ("logit", "raw_logits", "id_tx"),
            "trustmark": ("logit", "raw_logits", "target"),
            "videoseal": ("logit", "raw_logits", "target")}    # geometry fragment (replaces trustmark)
    frag = {n: _builders[n]() for n in args.fragments}
    # Separate OR-TrustMark corroborator ONLY when no pixel geometry fragment is already fused.
    use_or_tm = "trustmark" not in args.fragments and "videoseal" not in args.fragments
    if use_or_tm:
        from wbench.methods import TrustMarkMethod
        tm = TrustMarkMethod(args.tm_variant)

    # --- faithful RAVEN attackers ---
    from regen_pipe import ReSDPipeline
    from wmattacker import (DiffWMAttacker, VAEWMAttacker, GaussianBlurAttacker,
                            GaussianNoiseAttacker, JPEGAttacker, BrightnessAttacker, ContrastAttacker, BM3DAttacker)
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    sig = {"jpeg": JPEGAttacker(quality=25), "blur": GaussianBlurAttacker(5, 1),
           "noise": GaussianNoiseAttacker(std=0.05), "bright": BrightnessAttacker(0.2),
           "contrast": ContrastAttacker(0.2), "bm3d": BM3DAttacker()}
    vae = {}
    for k, mn in [("vae_b", "bmshj2018-hyperprior"), ("vae_c", "cheng2020-anchor")]:
        try: vae[k] = VAEWMAttacker(mn, quality=3, metric="mse", device=dev)
        except Exception as e: print(f"[vae] skip {k}: {e}", flush=True)
    tmp = tempfile.mkdtemp()

    def attack_one(a, ip, op):
        if a == "clean": Image.open(ip).save(op); return
        if a in sig: sig[a].attack([ip], [op]); return
        if a in vae: vae[a].attack([ip], [op]); return
        if a == "regen": regen.attack([ip], [op]); return
        if a in ("rinse2x", "rinse4x"):
            k = 2 if a == "rinse2x" else 4; cur = ip
            for r in range(k):
                nxt = op.replace(".png", f"_r{r}.png"); regen.attack([cur], [nxt]); cur = nxt
            Image.open(cur).save(op); return
        if a in GEO:
            out = GEO[a](Image.open(ip).convert("RGB"))
            if out.size != (512, 512): out = out.resize((512, 512))
            out.save(op); return
        if a == "crop_jpeg":
            _crop_then_jpeg(ip, op); return
        raise ValueError(a)

    imgs = sorted(glob.glob(os.path.join(args.image_dir, args.image_glob)))[args.start_idx:args.start_idx+args.n_images]
    print(f"[composite-defense] {len(imgs)} imgs; fragments=PhaseMark+VINE+DFT+QIM (fused) + TrustMark(OR); tau={tau:.2f}", flush=True)

    R = {a: {"fused_verify": [], "fused_zerobit": [], "composite_or": [], "fused_ba": [],
             "tm": [], **{n: [] for n in args.fragments}} for a in args.attacks}
    coexist = []

    def embed_composite(cover, image_id, tx):
        img = cover
        for name in args.fragments:
            m = frag[name]
            if SPEC[name][2] == "id_tx":
                img = m.embed(img, image_id, tx)
            else:
                perm, M = m.get_perm_M(image_id); img = m.embed_with_target(img, apply_crypto(tx, perm, M))
            if img.size != (512, 512): img = img.resize((512, 512))
        tmbits = None
        if use_or_tm:
            tmbits = np.random.RandomState(hash(image_id) % (2**31)).randint(0, 2, tm.n_bits).astype(np.uint8)
            img = tm.embed(img, tmbits)
            if img.size != (512, 512): img = img.resize((512, 512))
        return img, tmbits

    def decode_fragments(att, image_id, tx):
        aligned, per = {}, {}
        for name in args.fragments:
            kind, getter, _ = SPEC[name]
            m = frag[name]; perm, M = m.get_perm_M(image_id)
            soft = getattr(m, getter)(att)
            a = method_soft_to_codeword_llr(soft, perm, M, kind=kind, n_codeword=sb.n)
            aligned[name] = a; per[name] = float(np.mean(llr_to_bits(a) == tx))
        return aligned, per

    for i, fp in enumerate(imgs):
        image_id = f"def_{i:05d}"; tx = sb.encode(image_id_to_payload(image_id, n_bits=sb.data_bits))
        cover = Image.open(fp).convert("RGB").resize((512, 512))
        comp, tmbits = embed_composite(cover, image_id, tx)
        al, per = decode_fragments(comp, image_id, tx)
        coexist.append({k: per[k] for k in per})
        ip = os.path.join(tmp, f"comp_{i}.png"); comp.save(ip)
        for a in args.attacks:
            op = os.path.join(tmp, f"comp_{i}_{a}.png")
            try:
                attack_one(a, ip, op)
                att = Image.open(op).convert("RGB")
                if att.size != (512, 512): att = att.resize((512, 512))
                al, per = decode_fragments(att, image_id, tx)
                fused = fuse_llrs(al, weights=None, n_codeword=sb.n)
                fba = float(np.mean(llr_to_bits(fused) == tx))
                fver = float(decode_and_verify(fused, image_id, codec=sb)["detected"])
                fzb = 1.0 if fba >= tau else 0.0
                if use_or_tm:
                    tmrec = tm.decode(att); ntm = min(len(tmrec), len(tmbits))
                    tmba = float(np.mean(tmrec[:ntm] == tmbits[:ntm])); tmdet = 1.0 if tmba >= 0.75 else 0.0
                else:
                    tmba = float("nan"); tmdet = 0.0
                R[a]["fused_verify"].append(fver); R[a]["fused_zerobit"].append(fzb)
                R[a]["composite_or"].append(1.0 if (fver or fzb or tmdet) else 0.0)
                R[a]["fused_ba"].append(fba); R[a]["tm"].append(tmba)
                for k in args.fragments: R[a][k].append(per[k])
            except Exception as e:
                print(f"  [{a}] img{i} FAIL: {type(e).__name__}: {str(e)[:60]}", flush=True)
        print(f"  [{i+1}/{len(imgs)}] embedded+attacked", flush=True)

    def mean(x): return float(np.mean(x)) if x else None
    out = {"n_images": len(imgs), "tau": tau,
           "clean_coexist": {k: mean([c[k] for c in coexist]) for k in coexist[0]} if coexist else {},
           "attacks": {a: {k: mean(R[a][k]) for k in R[a]} for a in args.attacks}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)

    print(f"\nclean coexistence (per-fragment bit-acc): {out['clean_coexist']}")
    print(f"\n=== COMPOSITE DEFENSE vs RAVEN attack suite (n={len(imgs)}) ===")
    fcols = "".join(f"{n[:5]:>7}" for n in args.fragments)
    print(f"{'attack':<10}{'COMPOSITE':>10}{'fused_ver':>10}{'fused_0bit':>11}{'fused_ba':>9}{'TM':>6}{fcols}")
    for a in args.attacks:
        c = out["attacks"][a]
        fv = "".join(f"{(c[n] if c[n] is not None else float('nan')):>7.2f}" for n in args.fragments)
        print(f"{a:<10}{c['composite_or']:>10.2f}{c['fused_verify']:>10.2f}{c['fused_zerobit']:>11.2f}"
              f"{c['fused_ba']:>9.2f}{c['tm']:>6.2f}{fv}")
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
