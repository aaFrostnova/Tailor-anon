"""Cross-environment live measurement for the CEGAR loop.

W.live_required_at refuses to let the offline table settle UnMarker (adversarial, per-image) or any
column whose table value sits within k image-to-image standard deviations of the threshold. UnMarker and
CtrlRegen+ cannot run inside the fingerprint environment, so without an executor those columns come back
as `pending_live` and the request is never live-certified. This is that executor: it embeds the user's
images with the configuration the solve returned, hands them to the attack's own conda environment, and
reads the result back with the deployed best-path decoder -- the same measurement certify_classes_xenv.py
makes offline, driven per request instead of per campaign.

    measure = xenv_measurer(covers, workdir="/scratch/.../live_xenv", n_unmarker=8)
    solve_with_live(scen, in_process_measure, sg, LIVE_OK, xenv_measure=measure,
                    xenv_ok=("unmarker", "ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07"))

UnMarker costs about 3.5 GPU-minutes per image, so `n_unmarker` caps that column separately; the
CtrlRegen+ steps run on all the covers.
"""
import os, glob, shutil, subprocess, math
import numpy as np
from PIL import Image
import watermark_smt_v2 as W

CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
CRPY = os.environ.get("WM_PY_CTRLREGEN", "python")
UMPY = os.environ.get("WM_PY_UNMARKER", "python")
FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}


def _run(cmd, cwd=CF):
    env = dict(os.environ)
    env.setdefault("HF_HOME", os.path.join(os.path.dirname(SC), "hf_cache"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(env["HF_HOME"], "hub"))
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed rc={p.returncode}: {p.stderr[-600:]}")
    return p


def xenv_measurer(covers, dev="cuda", workdir=None, n_unmarker=8, keep=False):
    """`(cfg, attack) -> {fragment: (mean, se, n)}` for the columns that need their own environment."""
    from eval_matrix import OursComposite
    workdir = workdir or os.path.join(SC, "live_xenv")
    os.makedirs(workdir, exist_ok=True)
    cache = {}

    def composite(cfg):
        key = (tuple(cfg["order"]), tuple(sorted(cfg["fe"].items())), tuple(sorted(cfg["s"].items())))
        if key not in cache:
            cache.clear()
            c = OursComposite(dev, tm_variant="B", vine_variant="R",
                              config={"frags": [FKEY[f] for f in cfg["order"]],
                                      "order": [FKEY[f] for f in cfg["order"]],
                                      "strengths": {}, **W.frontend_config(cfg["fe"])})
            c.strength = dict(c.DEFAULT_STRENGTH)
            c.strength.update({FKEY[f]: float(v) for f, v in cfg["s"].items()})
            cache[key] = c
        return cache[key]

    def measure(cfg, attack):
        comp = composite(cfg)
        n = min(n_unmarker, len(covers)) if attack == "unmarker" else len(covers)
        qdir = os.path.join(workdir, f"q_{abs(hash((tuple(cfg['order']), tuple(sorted(cfg['s'].items())), attack))) % 10**8:08d}")
        emb = os.path.join(qdir, "embed"); att = os.path.join(qdir, f"att_{attack}")
        os.makedirs(emb, exist_ok=True)
        secrets = {}
        for i in range(n):
            v, sec = comp.embed(covers[i], i); secrets[i] = sec
            v.save(os.path.join(emb, f"i{i:05d}.png"))
        if attack == "unmarker":
            _run([UMPY, "scripts/attack/unmarker_batch.py", "--in_dir", emb, "--out_dir", att,
                  "--config", "attack_configs/Vine.yaml", "--n", str(n), "--start_idx", "0", "--batch", "1"])
        elif attack.startswith("ctrlregen_s0"):
            # --step (not --step_list): with a step list the script appends _s<tag> to out_dir
            step = f"0.{attack[-1]}"
            _run([CRPY, "scripts/attack/ctrlregen_batch.py", "--in_dir", emb, "--out_dir", att,
                  "--step", step])
        else:
            raise ValueError(f"xenv_measurer cannot run {attack}")
        files = sorted(glob.glob(os.path.join(att, "i*.png")))
        if not files:
            raise RuntimeError(f"{attack}: the executor produced no images in {att}")
        per = {f: [] for f in cfg["order"]}
        thr = cfg.get("threshold", 0.0)
        for fp in files:
            idx = int(os.path.basename(fp)[1:6]); iid, tx = secrets[idx]
            img = Image.open(fp).convert("RGB")
            if img.size != (512, 512): img = img.resize((512, 512))
            acc = {f: float(np.mean((comp._frag_llr(FKEY[f], img, iid) > 0).astype(np.uint8) == tx)) for f in cfg["order"]}
            if comp.geo and max(acc.values()) < thr:
                ok, vw = comp.geo_cascade(img, iid, tx, return_view=True)
                if ok and vw is not None:
                    acc_c = {f: float(np.mean((comp._frag_llr(FKEY[f], vw, iid) > 0).astype(np.uint8) == tx)) for f in cfg["order"]}
                    if max(acc_c.values()) > max(acc.values()): acc = acc_c
            for f in cfg["order"]: per[f].append(acc[f])
        if not keep: shutil.rmtree(qdir, ignore_errors=True)
        return {f: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0, len(v))
                for f, v in per.items()}

    return measure


XENV_OK = ("unmarker", "ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07")
