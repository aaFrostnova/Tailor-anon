"""Merge stage JSON summaries into one markdown comparison table (bit accuracy)."""
from __future__ import annotations

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="stage JSON files")
    p.add_argument("--output_md", required=True)
    args = p.parse_args()

    # Merge: method -> {display_name, n_bits, psnr, {attack: acc}}
    merged = {}
    attack_order = []
    for path in args.inputs:
        s = json.load(open(path))
        for atk in s["attacks"]:
            if atk != "clean" and atk not in attack_order:
                attack_order.append(atk)
        for m, d in s["methods"].items():
            e = merged.setdefault(m, {"display": d["display_name"], "n_bits": d["n_bits"],
                                      "psnr": d.get("psnr_mean"), "acc": {}})
            for atk, v in d["bit_acc"].items():
                if v is not None:
                    e["acc"][atk] = v

    cols = ["clean"] + attack_order
    method_order = ["dwtDct", "dwtDctSvd", "rivaGan", "trustmark", "vine_b", "vine_r"]
    method_order = [m for m in method_order if m in merged] + [m for m in merged if m not in method_order]

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    # Markdown
    hdr = "| Method | bits | PSNR | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (3 + len(cols))
    lines = [hdr, sep]
    for m in method_order:
        e = merged[m]
        row = [e["display"], str(e["n_bits"]), f"{e['psnr']:.1f}" if e["psnr"] else "—"]
        row += [fmt(e["acc"].get(c)) for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines)

    with open(args.output_md, "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[done] -> {args.output_md}")


if __name__ == "__main__":
    main()
