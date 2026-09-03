"""Test whether the reference-needed log-polar alignment + correlation path
(MultiMethodFingerprint) detects crop_70, which the reference-free bit tiers cannot
(half the carriers are physically cropped away, so payload recovery is impossible;
correlation presence over the surviving region can still fire)."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from src.multi_method_fingerprint import MultiMethodFingerprint
from benchmark_fused import apply_attack

DEVICE = "cuda"
IMG_DIR = "/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017"


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--n_images", type=int, default=20); a = p.parse_args()
    mf = MultiMethodFingerprint(device=DEVICE, use_vine=True)
    files = sorted([f for f in Path(IMG_DIR).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])[4000:4000 + a.n_images]
    conds = {"clean": "clean", "crop_90": "crop_90", "crop_70": "crop_70", "resize_110": "resize_110"}
    agg = {c: {"ref_free": [], "with_ref": []} for c in conds}
    for i, fp in enumerate(files):
        pil = Image.open(fp).convert("RGB").resize((512, 512), Image.LANCZOS)
        image_id = f"crop_{i:05d}"
        wm = mf.embed(pil, image_id)
        for c, atk in conds.items():
            att = apply_attack(atk, wm)
            # reference-free (no original): only VINE neural decode
            rf = mf.detect(att, image_id, pil_original=None)
            # reference-needed (verifier holds the clean original): adds alignment+correlation
            wr = mf.detect(att, image_id, pil_original=pil)
            agg[c]["ref_free"].append(float(rf["detected"]))
            agg[c]["with_ref"].append(float(wr["detected"]))
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)
    print(f"\n{'condition':<14}{'ref-free':>10}{'with-ref (align+corr)':>24}")
    for c in conds:
        print(f"{c:<14}{np.mean(agg[c]['ref_free']):>10.2f}{np.mean(agg[c]['with_ref']):>24.2f}", flush=True)


if __name__ == "__main__":
    main()
