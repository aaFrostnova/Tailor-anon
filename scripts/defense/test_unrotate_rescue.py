"""Does image-level warp-back rescue rot9 for the FINAL config (VINE+TrustMark-B)?

Embed VINE+TM composite -> rotate 9° (the attack) -> rotate −9° (oracle un-rotate, i.e. assume
the angle is known/estimated) -> decode fused. If fused bit-acc recovers, then rot9 is rescuable
by angle-estimation + warp-back (unlike the sparse-carrier DFT QIM, which dies even at the oracle).
"""
import argparse, glob, os, sys
import numpy as np, torch
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.shortened_bch import ShortenedBCH
from src.vine_crypto_wrapper import VineCryptoWrapper, apply_crypto
from src.trustmark_fragment import TrustMarkFragment
from src.soft_fusion import method_soft_to_codeword_llr, fuse_llrs, llr_to_bits
from src.soft_bch import decode_and_verify
from src.payload import image_id_to_payload
KEY = b"v5_key_encoder_master"
COCO = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"
SPEC = {"vine": ("prob", "raw_probs"), "trustmark": ("logit", "raw_logits")}


def rot(img, deg):
    W, H = img.size; arr = np.array(img); pad = max(W, H) // 2
    refl = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    big = Image.fromarray(refl).rotate(deg, resample=Image.BICUBIC, expand=False)
    bw, bh = big.size; l, t = (bw - W) // 2, (bh - H) // 2
    return big.crop((l, t, l + W, t + H))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_images", type=int, default=15); ap.add_argument("--deg", type=float, default=9.0)
    args = ap.parse_args(); dev = "cuda"; sb = ShortenedBCH()
    tau = 0.63
    frag = {"vine": VineCryptoWrapper(master_key=KEY, method_name="vine", n_bits=sb.n, device=dev),
            "trustmark": TrustMarkFragment(master_key=KEY, method_name="trustmark", n_bits=sb.n, model_type="B", device=dev)}
    imgs = sorted(glob.glob(os.path.join(COCO, "*.jpg")))[4000:4000 + args.n_images]

    def decode(att, iid, tx):
        al = {}
        for name in ["vine", "trustmark"]:
            kind, getter = SPEC[name]; m = frag[name]; perm, M = m.get_perm_M(iid)
            al[name] = method_soft_to_codeword_llr(getattr(m, getter)(att), perm, M, kind=kind, n_codeword=sb.n)
        fused = fuse_llrs(al, weights=None, n_codeword=sb.n)
        ba = float(np.mean(llr_to_bits(fused) == tx))
        det = (ba >= tau) or bool(decode_and_verify(fused, iid, codec=sb)["detected"])
        return ba, det

    A = {"clean": [], "rot9": [], "rot9_unrot": []}; D = {"clean": [], "rot9": [], "rot9_unrot": []}
    for i, fp in enumerate(imgs):
        iid = f"ur_{i:05d}"; tx = sb.encode(image_id_to_payload(iid, n_bits=sb.data_bits))
        o = Image.open(fp).convert("RGB").resize((512, 512))
        pv, Mv = frag["vine"].get_perm_M(iid); img = frag["vine"].embed_with_target(o, apply_crypto(tx, pv, Mv)).resize((512, 512))
        pt, Mt = frag["trustmark"].get_perm_M(iid); img = frag["trustmark"].embed_with_target(img, apply_crypto(tx, pt, Mt)).resize((512, 512))
        for tag, att in [("clean", img), ("rot9", rot(img, args.deg)), ("rot9_unrot", rot(rot(img, args.deg), -args.deg))]:
            ba, det = decode(att, iid, tx); A[tag].append(ba); D[tag].append(det)
        print(f"  [{i+1}/{len(imgs)}] clean={A['clean'][-1]:.2f} rot9={A['rot9'][-1]:.2f} unrot={A['rot9_unrot'][-1]:.2f}", flush=True)

    print(f"\n=== warp-back rot{args.deg:.0f}° rescue (VINE+TrustMark-B, n={len(imgs)}) ===")
    for tag in ["clean", "rot9", "rot9_unrot"]:
        print(f"{tag:<12} fused_ba={np.mean(A[tag]):.3f}  detect={np.mean(D[tag]):.3f}")
    print("UNROT_DONE")


if __name__ == "__main__":
    main()
