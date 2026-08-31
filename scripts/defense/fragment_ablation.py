"""Leave-one-out ablation: marginal contribution of each fragment to the FUSED bit-acc.

For each image+attack, decode all 4 fragments -> aligned LLRs, then compute fused bit-acc for the
full set and for each leave-one-out subset. Delta_f = fba(all) - fba(drop f) = how much fragment f
adds to the fusion (positive = helps, ~0 = redundant, negative = hurts). Answers 'is DFT/QIM
droppable' and 'which fragments are essential'.
"""
import argparse, glob, os, sys, tempfile
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "external", "WatermarkAttacker"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.learned_fragment_methods import DFTKredMethod, QuantQIMMethod
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.payload import image_id_to_payload
from scripts.defense.benchmark_composite_defense import GEO, _crop_then_jpeg
KEY = b"v5_key_encoder_master"
SD21 = "/project/pi_shiqingma_umass_edu/mingzheli/model/stable-diffusion-2-1"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
SPEC = {"vine": ("prob", "raw_probs", "target"), "dft": ("logit", "raw_logits", "id_tx"),
        "qim": ("logit", "raw_logits", "id_tx"), "trustmark": ("logit", "raw_logits", "target")}
FRAGS = ["vine", "dft", "qim", "trustmark"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=30)
    ap.add_argument("--attacks", nargs="+",
                    default=["clean", "jpeg", "bright", "bm3d", "regen", "rinse2x", "rinse4x", "vae_c", "hflip", "crop75"])
    ap.add_argument("--output", default="results/defense/fragment_ablation.json")
    args = ap.parse_args()
    dev = "cuda"; sb = ShortenedBCH()
    frag = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev),
            "dft": DFTKredMethod(os.path.join(REPO, "results/dft_fftaware_baseline/ckpt.pt"), KEY, "dft_kred", dev),
            "qim": QuantQIMMethod(os.path.join(REPO, "results/quant_qim_frozen_d006/ckpt.pt"), KEY, "quant_qim", dev),
            "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)}
    from regen_pipe import ReSDPipeline
    from wmattacker import DiffWMAttacker, VAEWMAttacker, JPEGAttacker, BrightnessAttacker, BM3DAttacker
    from diffusers import DPMSolverMultistepScheduler
    pipe = ReSDPipeline.from_pretrained(SD21, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True); pipe = pipe.to(dev)
    regen = DiffWMAttacker(pipe, batch_size=1, noise_step=60)
    sig = {"jpeg": JPEGAttacker(quality=25), "bright": BrightnessAttacker(0.2), "bm3d": BM3DAttacker()}
    vae = {"vae_c": VAEWMAttacker("cheng2020-anchor", quality=3, metric="mse", device=dev)}
    tmp = tempfile.mkdtemp()

    def attack(a, pil):
        if a == "clean": return pil
        ip = os.path.join(tmp, "i.png"); pil.save(ip); op = os.path.join(tmp, "o.png")
        if a in sig: sig[a].attack([ip], [op])
        elif a in vae: vae[a].attack([ip], [op])
        elif a == "regen": regen.attack([ip], [op])
        elif a == "rinse2x":
            cur = ip
            for r in range(2):
                nx = os.path.join(tmp, f"r{r}.png"); regen.attack([cur], [nx]); cur = nx
            return Image.open(cur).convert("RGB").resize((512, 512))
        elif a == "rinse4x":
            cur = ip
            for r in range(4):
                nx = os.path.join(tmp, f"s{r}.png"); regen.attack([cur], [nx]); cur = nx
            return Image.open(cur).convert("RGB").resize((512, 512))
        elif a in GEO:
            o = GEO[a](pil); return o if o.size == (512, 512) else o.resize((512, 512))
        elif a == "crop_jpeg": _crop_then_jpeg(ip, op)
        return Image.open(op).convert("RGB").resize((512, 512))

    imgs = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n_images]
    acc = {a: {"all": [], **{f"drop_{f}": [] for f in FRAGS}, **{f"only_{f}": [] for f in FRAGS}} for a in args.attacks}

    for i, fp in enumerate(imgs):
        iid = f"ab_{i:05d}"; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
        img = Image.open(fp).convert("RGB").resize((512, 512))
        pv, Mv = frag["vine"].get_perm_M(iid); img = frag["vine"].embed_with_target(img, apply_crypto(tx, pv, Mv)).resize((512, 512))
        img = frag["dft"].embed(img, iid, tx).resize((512, 512))
        img = frag["qim"].embed(img, iid, tx).resize((512, 512))
        pt, Mt = frag["trustmark"].get_perm_M(iid); img = frag["trustmark"].embed_with_target(img, apply_crypto(tx, pt, Mt)).resize((512, 512))
        for a in args.attacks:
            att = attack(a, img)
            al = {}
            for f in FRAGS:
                kind, getter, _ = SPEC[f]; m = frag[f]; perm, M = m.get_perm_M(iid)
                al[f] = method_soft_to_codeword_llr(getattr(m, getter)(att), perm, M, kind=kind, n_codeword=sb.n)
            def fba(keys): return float(np.mean(llr_to_bits(fuse_llrs({k: al[k] for k in keys}, n_codeword=sb.n)) == tx))
            acc[a]["all"].append(fba(FRAGS))
            for f in FRAGS:
                acc[a][f"drop_{f}"].append(fba([k for k in FRAGS if k != f]))
                acc[a][f"only_{f}"].append(fba([f]))
        print(f"  [{i+1}/{len(imgs)}]", flush=True)

    import json
    out = {a: {k: float(np.mean(v)) for k, v in acc[a].items()} for a in args.attacks}
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\n=== leave-one-out ablation: fused bit-acc (n={len(imgs)}) ===")
    print(f"{'attack':<10}{'ALL':>6}{'  Δ(drop) each fragment = contribution':<10}")
    print(f"{'':<10}{'all':>6}{'-VINE':>8}{'-DFT':>8}{'-QIM':>8}{'-TM':>8}")
    for a in args.attacks:
        o = out[a]
        print(f"{a:<10}{o['all']:>6.2f}{o['all']-o['drop_vine']:>+8.2f}{o['all']-o['drop_dft']:>+8.2f}"
              f"{o['all']-o['drop_qim']:>+8.2f}{o['all']-o['drop_trustmark']:>+8.2f}")
    print("\n(Δ = fba(all) − fba(drop f); + = fragment helps fusion, ~0 = redundant, − = hurts)")
    print("ABLATION_DONE")


if __name__ == "__main__":
    main()
