"""Cross-env advanced-attack eval (CtrlRegen+, UnMarker): embed in fingerprint env ->
external batch attack in its OWN conda env (dir->dir) -> decode back in fingerprint env.
Secrets are deterministic in the image index, so decode regenerates them without state.

  --mode embed  : watermark N images with --method, save e_{i:05d}.png into --embed_dir
  --mode decode : for each e_*.png in --attacked_dir, regen secret by index, decode, and
                  MERGE {--attack_name: {bit_acc,tpr}} into --out (per-method json).
"""
import argparse, glob, os, sys, json
import numpy as np
from PIL import Image

REPO = "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "defense"))
from eval_matrix import build_method

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["embed", "decode"], required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--image_dir"); ap.add_argument("--image_glob", default="*.png")
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--embed_dir"); ap.add_argument("--attacked_dir")
    ap.add_argument("--attack_name"); ap.add_argument("--out")
    ap.add_argument("--tm_variant", default="B")
    ap.add_argument("--geo", action="store_true")
    a = ap.parse_args()
    method = build_method(a.method, "cuda", a.tm_variant, geo=a.geo)

    if a.mode == "embed":
        os.makedirs(a.embed_dir, exist_ok=True)
        imgs = sorted(glob.glob(os.path.join(a.image_dir, a.image_glob)))[a.start_idx:a.start_idx + a.n]
        for i, fp in enumerate(imgs):
            cover = Image.open(fp).convert("RGB").resize((512, 512))
            emb, _ = method.embed(cover, i)
            if emb.size != (512, 512): emb = emb.resize((512, 512))
            emb.save(os.path.join(a.embed_dir, f"e_{i:05d}.png"))
        print(f"[embed] {a.method}: {len(imgs)} -> {a.embed_dir}", flush=True)
    else:
        files = sorted(glob.glob(os.path.join(a.attacked_dir, "e_*.png")))
        bas, dets = [], []
        for fp in files:
            i = int(os.path.basename(fp).split("_")[1].split(".")[0])
            att = Image.open(fp).convert("RGB")
            if att.size != (512, 512): att = att.resize((512, 512))
            ba, det = method.decode(att, method.secret_for(i))
            bas.append(ba); dets.append(float(det))
        m = lambda x: float(np.mean(x)) if x else None
        out = json.load(open(a.out)) if (a.out and os.path.exists(a.out)) else {}
        out.setdefault("method", method.name); out.setdefault("n_bits", method.n_bits)
        out.setdefault("attacks", {})
        out["attacks"][a.attack_name] = {"bit_acc": m(bas), "tpr": m(dets), "n": len(files)}
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"[decode] {a.attack_name} {a.method}: ba={m(bas):.3f} tpr={m(dets):.3f} n={len(files)} -> {a.out}", flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()
