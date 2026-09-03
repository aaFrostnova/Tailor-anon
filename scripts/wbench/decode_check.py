"""Decode the original watermark from images AFTER you applied your own watermark on top.

Usage:
  1. Generate with generate_watermarked.py (writes manifest.json + *_wm.png).
  2. Apply YOUR post-processing watermark to each *_wm.png, saving the result with
     suffix "_post.png" next to it (e.g. img0_wm.png -> img0_wm_post.png), OR pass
     --post_dir pointing to a directory with the same filenames.
  3. Run this: it decodes the ORIGINAL watermark from each post-processed image and
     reports how much survived (bit accuracy vs the saved payload), per method.

If a post image is missing, that entry is skipped (reported).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from wbench.methods import build_methods


def _post_path(wm_png, post_dir, suffix="wm_post"):
    wm = Path(wm_png)
    name = wm.name.replace("_wm.png", f"_{suffix}.png")
    if post_dir:
        return Path(post_dir) / name
    return wm.with_name(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",
                   default="results/wbench_baselines/generated/manifest.json")
    p.add_argument("--post_dir", default=None,
                   help="dir with post-processed images named <orig>_<suffix>.png; "
                        "default: alongside each wm png")
    p.add_argument("--post_suffix", default="wm_post",
                   help="suffix of the post-processed file (e.g. 'GG' for imgN_GG.png)")
    p.add_argument("--resize_to", type=int, default=None,
                   help="resize the post image to this square size before decoding "
                        "(use the embedding resolution, e.g. 512, to undo any rescaling)")
    p.add_argument("--output", default="results/wbench_baselines/generated/decode_report.json")
    args = p.parse_args()

    man = json.load(open(REPO / args.manifest if not Path(args.manifest).is_absolute() else args.manifest))
    method_names = sorted({e["method"] for e in man["entries"]})
    methods = build_methods(method_names, "cuda")

    per_method = defaultdict(lambda: {"survived": [], "self": [], "missing": 0})
    rows = []
    for e in man["entries"]:
        m = e["method"]
        if m not in methods:
            continue
        bits = np.load(e["bits_npy"])
        post = _post_path(e["wm_png"], args.post_dir, args.post_suffix)
        if not post.exists():
            per_method[m]["missing"] += 1
            rows.append({**{k: e[k] for k in ("method", "display_name", "img_index")},
                         "post_png": str(post), "status": "MISSING"})
            continue
        img = Image.open(post).convert("RGB")
        if args.resize_to:
            img = img.resize((args.resize_to, args.resize_to), Image.LANCZOS)
        rec = methods[m].decode(img)
        n = min(len(rec), len(bits))
        acc = float(np.mean(rec[:n] == bits[:n]))
        per_method[m]["survived"].append(acc)
        per_method[m]["self"].append(e["self_bit_acc"])
        rows.append({"method": m, "display_name": e["display_name"],
                     "img_index": e["img_index"], "self_bit_acc": e["self_bit_acc"],
                     "survived_bit_acc": round(acc, 4), "status": "ok"})
        print(f"  {e['display_name']:<12} img{e['img_index']}: "
              f"self={e['self_bit_acc']:.3f} -> after_your_wm={acc:.3f}", flush=True)

    summary = {"per_method": {}, "rows": rows}
    print("\n" + "=" * 70)
    print(f"{'method':<14}{'self':>8}{'after_post':>12}{'delta':>9}{'imgs':>6}{'missing':>9}")
    print("-" * 70)
    for m in method_names:
        d = per_method[m]
        if d["survived"]:
            s = float(np.mean(d["self"])); a = float(np.mean(d["survived"]))
            summary["per_method"][m] = {"self_mean": round(s, 4),
                                        "after_post_mean": round(a, 4),
                                        "delta": round(a - s, 4),
                                        "n": len(d["survived"]), "missing": d["missing"]}
            disp = next(e["display_name"] for e in man["entries"] if e["method"] == m)
            print(f"{disp:<14}{s:>8.3f}{a:>12.3f}{a - s:>9.3f}{len(d['survived']):>6}{d['missing']:>9}")
        else:
            summary["per_method"][m] = {"missing": d["missing"], "n": 0}
            print(f"{m:<14}{'-':>8}{'-':>12}{'-':>9}{0:>6}{d['missing']:>9}")
    print("=" * 70)
    print("delta < 0 means your post-processing watermark degraded the original watermark.")

    with open(REPO / args.output if not Path(args.output).is_absolute() else args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
