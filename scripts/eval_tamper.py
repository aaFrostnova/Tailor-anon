"""Component C eval: tamper detection (clean vs attacked) + localization IoU.

- detection: tamper_score / is_tampered should separate clean from attacked,
  including the regime where the payload still decodes;
- localization: for a local edit (center box), the QIM block region flagged
  should overlap the true edited region (IoU).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.fused_detector import FusedDetector
from src import tamper_detect as TD
from wbench.editing import local_edit
from benchmark_fused import apply_attack


def iou(b1, b2):
    ya0, xa0, ya1, xa1 = b1
    yb0, xb0, yb1, xb1 = b2
    iy0, ix0 = max(ya0, yb0), max(xa0, xb0)
    iy1, ix1 = min(ya1, yb1), min(xa1, xb1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    a1 = (ya1 - ya0) * (xa1 - xa0); a2 = (yb1 - yb0) * (xb1 - xb0)
    return inter / (a1 + a2 - inter + 1e-9)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=10)
    p.add_argument("--image_dir",
                   default="/project/pi_shiqingma_umass_edu/mingzheli/KCMP/EXP_data/train2017")
    p.add_argument("--start_idx", type=int, default=4100)
    p.add_argument("--output", default="results/fused/tamper_eval.json")
    args = p.parse_args()

    calib = json.load(open(REPO / "results/attack_char/erasure_calib.json"))
    fd = FusedDetector(device="cuda")
    files = sorted([f for f in Path(args.image_dir).iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[args.start_idx:args.start_idx + args.n_images]

    conds = ["clean", "blur_2.5", "regen_20_sd15", "local_edit"]
    scores = {c: [] for c in conds}
    decoded_ok = {c: [] for c in conds}   # did the payload still decode (Tier-1 fused)?
    ious = []
    # true center box in 256-space (QIM analysis resolution), frac=0.5
    true_box = (64, 64, 192, 192)

    for i, fp in enumerate(files):
        image_id = f"tamp_{i:05d}"
        wm = fd.embed(Image.open(fp).convert("RGB"), image_id)  # 512
        for c in conds:
            if c == "clean":
                att = wm
            elif c == "local_edit":
                att = local_edit(wm, device="cuda", frac=0.5, model_key="sd15", strength=0.8)
            else:
                att = apply_attack(c, wm)
            res = TD.analyze(fd, att, image_id, calib=calib)
            scores[c].append(res["tamper_score"])
            decoded_ok[c].append(float(fd.detect(att, image_id)["detected"]))
            if c == "local_edit" and res.get("qim_tamper_bbox_px"):
                ious.append(iou(tuple(res["qim_tamper_bbox_px"]), true_box))
        print(f"  [{i+1}/{len(files)}]", flush=True)

    def auc(pos, neg):  # P(pos > neg), Mann-Whitney
        pos, neg = np.asarray(pos), np.asarray(neg)
        return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))

    clean = np.asarray(scores["clean"])
    thr = float(np.percentile(clean, 90))  # calibrate to ~10% clean FPR
    attacked_conds = [c for c in conds if c != "clean"]
    summary = {
        "n_images": len(files),
        "threshold_clean_fpr10": thr,
        "tamper_score_mean": {c: float(np.mean(scores[c])) for c in conds},
        "clean_fpr": float(np.mean(clean > thr)),
        "tpr_at_clean_fpr10": {c: float(np.mean(np.asarray(scores[c]) > thr)) for c in attacked_conds},
        "auc_vs_clean": {c: auc(scores[c], scores["clean"]) for c in attacked_conds},
        "payload_still_decodes": {c: float(np.mean(decoded_ok[c])) for c in conds},
        "local_edit_localization_iou_mean": float(np.mean(ious)) if ious else None,
    }
    Path(REPO / args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(REPO / args.output, "w"), indent=2)

    print(f"\n{'='*72}")
    print(f"Tamper detection (threshold@clean-FPR10%={thr:.3f}, clean FPR={summary['clean_fpr']:.2f})")
    print(f"{'condition':<16}{'score':>9}{'TPR':>7}{'AUC':>7}{'payload_decodes':>17}")
    print("-" * 72)
    for c in conds:
        tpr = summary["tpr_at_clean_fpr10"].get(c, float('nan'))
        a = summary["auc_vs_clean"].get(c, float('nan'))
        print(f"{c:<16}{summary['tamper_score_mean'][c]:>9.3f}"
              f"{tpr:>7.2f}{a:>7.2f}{summary['payload_still_decodes'][c]:>17.2f}")
    print(f"\nlocal-edit localization IoU (vs center box): "
          f"{summary['local_edit_localization_iou_mean']}")
    print("=" * 72)
    print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
