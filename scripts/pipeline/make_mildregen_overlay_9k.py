"""Mild diffusion regeneration ("regen") and its rinse repeats at NINE strength knots, sharded.

The measurement is the one make_mildregen_overlay_n100.py made -- DiffWMAttacker(ReSDPipeline(SD-2-1),
noise_step=60), the regeneration eval_matrix.py calls "regen", applied 1x/2x/4x for regen/rinse2x/
rinse4x, over the three solo fragment grids plus all six ordered pairs at the mid strengths -- with two
changes:

  * nine knots per fragment instead of five, matching every other column of the table. Thinning the
    nine-knot columns back to five costs a median 0.0066 and up to 0.0488 bit accuracy at the dropped
    knots, and the diffusion curves carry 1.09x the cheap family's curvature at the same knot count
    (diag_diffusion_sampling.py), so the five-knot penalty applied here too.
  * the image loop is sharded, because nine knots make ~231 regeneration passes per image. Each shard
    writes its per-image raw accuracies after every image (so a preempted shard resumes), and `merge`
    refuses to write the overlay unless every shard is present and complete -- a partial campaign
    cannot become a table.

Usage:
  python make_mildregen_overlay_9k.py measure N NSHARD SHARD   -> {WORK}/shardKKofNN.json
  python make_mildregen_overlay_9k.py merge   N NSHARD         -> $SC/surrogate_mildregen_overlay_n100.json
Env: OUT_NAME overrides the overlay filename (a smoke run must not overwrite the canonical input).
"""
import sys, os, json, time, subprocess
import numpy as np
from PIL import Image
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
for p in (CF, f"{CF}/scripts", f"{CF}/scripts/defense", f"{CF}/external/WatermarkAttacker"):
    sys.path.insert(0, p)
from src.image_pool import sample as _pool_sample, composition_of

STAGE = sys.argv[1] if len(sys.argv) > 1 else "measure"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
NSHARD = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SHARD = int(sys.argv[4]) if len(sys.argv) > 4 else 0
assert STAGE in ("measure", "merge"), f"unknown stage {STAGE!r}"
assert 0 <= SHARD < NSHARD, f"shard {SHARD} outside [0,{NSHARD})"
SLICE_START = 100          # per-source offset: disjoint from the in-process fit slice (offset 0, n=100)
WORK = f"{SC}/mildregen9_n{N}"; os.makedirs(WORK, exist_ok=True)
OUT_NAME = os.environ.get("OUT_NAME", "surrogate_mildregen_overlay_n100.json")

GRID = {f: [round(float(v), 4) for v in np.linspace(lo, hi, 9)]
        for f, (lo, hi) in {"VINE": (0.2, 1.0), "TrustMark": (0.4, 1.6), "VideoSeal": (0.5, 1.5)}.items()}
RANGES = {f: (GRID[f][0], GRID[f][-1]) for f in GRID}
MID = {f: GRID[f][len(GRID[f]) // 2] for f in GRID}     # the middle knot, as make_surrogate_ext9.py takes it
assert MID == {"VINE": 0.6, "TrustMark": 1.0, "VideoSeal": 1.0}, MID
FRAGS = ["VINE", "TrustMark", "VideoSeal"]
ORDERED_PAIRS = [(a, b) for a in FRAGS for b in FRAGS if a != b]
PASSES = {"regen": 1, "rinse2x": 2, "rinse4x": 4}
CELLS = sum(len(GRID[f]) for f in FRAGS) + len(ORDERED_PAIRS)          # 27 + 6 = 33

files = _pool_sample(N, offset=SLICE_START)
assert len(files) == N, f"pool sample short: {len(files)}"
lo, hi = (N * SHARD) // NSHARD, (N * (SHARD + 1)) // NSHARD
def shard_path(k): return f"{WORK}/shard{k:02d}of{NSHARD:02d}.json"
def skey(a, f, s): return f"{a}|{f}|{s}"
def pkey(a, f1, f2): return f"{a}|{f1}->{f2}"
mean = lambda L: float(np.mean(L)) if L else float("nan")

if STAGE == "measure":
    import torch
    from src.vine_crypto_wrapper import VineCryptoWrapper
    from src.trustmark_fragment import TrustMarkFragment
    from src.videoseal_fragment import VideoSealFragment
    from src.shortened_bch import ShortenedBCH
    print(f"[measure] N={N} shard {SHARD}/{NSHARD} images [{lo},{hi}) slice_start={SLICE_START} "
          f"cells/img={CELLS} passes/img={CELLS * sum(PASSES.values())} sources={composition_of(files)}", flush=True)
    sb = ShortenedBCH(); NB = sb.n; KEY = b"v5_key_encoder_master"; dev = "cuda"
    print("loading fragments...", flush=True)
    FR = {"VINE": VineCryptoWrapper(KEY, "vine", NB, dev, variant="R"),
          "TrustMark": TrustMarkFragment(KEY, "trustmark", NB, model_type="B", device=dev),
          "VideoSeal": VideoSealFragment(KEY, "videoseal", NB, device=dev)}
    print("loading the mild regeneration pipeline (SD-2-1)...", flush=True)
    from src.attacks import regen_attacker, RINSE_PASSES      # single definition of the regeneration
    assert RINSE_PASSES == PASSES, RINSE_PASSES
    REGEN = regen_attacker(dev)

    def target_bits(i): return np.random.RandomState(i).randint(0, 2, NB).astype(np.uint8)
    def r512(im): return im if im.size == (512, 512) else im.resize((512, 512))
    def hard(fn, pil):
        f = FR[fn]; v = f.raw_probs(pil) if fn == "VINE" else f.raw_logits(pil)
        v = np.asarray(v).ravel()[:NB]
        return (v > (0.5 if fn == "VINE" else 0.0)).astype(np.uint8)
    _tmp = f"{WORK}/_t{SHARD:02d}"; os.makedirs(_tmp, exist_ok=True)
    def regen_k(img, k, tag):
        """Apply the regeneration k times in sequence -- the repo's rinse definition."""
        cur = f"{_tmp}/{tag}_in.png"; img.save(cur)
        for r in range(k):
            nxt = f"{_tmp}/{tag}_r{r}.png"; REGEN.attack([cur], [nxt]); cur = nxt
        return Image.open(cur).convert("RGB")

    # resume: a shard file left by a preempted run holds the images already measured
    rec = {"N": N, "nshard": NSHARD, "shard": SHARD, "range": [lo, hi], "grid": GRID, "mid": MID,
           "slice_start": SLICE_START, "images": [],
           "solo": {skey(a, f, s): [] for a in PASSES for f in FRAGS for s in GRID[f]},
           "pair": {pkey(a, f1, f2): [] for a in PASSES for (f1, f2) in ORDERED_PAIRS}}
    if os.path.exists(shard_path(SHARD)):
        old = json.load(open(shard_path(SHARD)))
        if old.get("grid") == GRID and old.get("range") == [lo, hi] and old.get("N") == N:
            rec = old; print(f"[measure] resuming: {len(rec['images'])} images already done", flush=True)
    t0 = time.time()
    for i in range(lo, hi):
        if i in rec["images"]: continue
        cover = Image.open(files[i]).convert("RGB").resize((512, 512), Image.BICUBIC)
        t = target_bits(SLICE_START + i)
        emb = {f: {s: r512(FR[f].embed_with_target(cover, t, strength=s)) for s in GRID[f]} for f in FRAGS}
        comp = {(f1, f2): r512(FR[f2].embed_with_target(emb[f1][MID[f1]], t, strength=MID[f2]))
                for (f1, f2) in ORDERED_PAIRS}
        for a, k in PASSES.items():
            for f in FRAGS:
                for s in GRID[f]:
                    rec["solo"][skey(a, f, s)].append(float(np.mean(hard(f, regen_k(emb[f][s], k, f"s{f}{s}")) == t)))
            for (f1, f2) in ORDERED_PAIRS:
                rec["pair"][pkey(a, f1, f2)].append(float(np.mean(hard(f1, regen_k(comp[(f1, f2)], k, f"p{f1}{f2}")) == t)))
        rec["images"].append(i)
        rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = shard_path(SHARD) + ".tmp"
        json.dump(rec, open(tmp, "w")); os.replace(tmp, shard_path(SHARD))      # atomic checkpoint
        print(f"  ...{len(rec['images'])}/{hi - lo} (image {i}) [{time.time() - t0:.0f}s]", flush=True)
    assert rec["images"] == list(range(lo, hi)), "shard finished with images missing"
    print(f"[measure] shard {SHARD}/{NSHARD} complete -> {shard_path(SHARD)}", flush=True)
    print("MILDREGEN9_SHARD_DONE", flush=True)

else:   # merge
    shards = []
    for k in range(NSHARD):
        assert os.path.exists(shard_path(k)), f"shard {k}/{NSHARD} missing: {shard_path(k)}"
        d = json.load(open(shard_path(k)))
        elo, ehi = (N * k) // NSHARD, (N * (k + 1)) // NSHARD
        assert d["images"] == list(range(elo, ehi)), f"shard {k} incomplete: {len(d['images'])}/{ehi - elo} images"
        assert d["grid"] == GRID and d["N"] == N and d["nshard"] == NSHARD and d["mid"] == MID, f"shard {k} was measured under a different design"
        shards.append(d)
    solo = {key: sum((d["solo"][key] for d in shards), []) for key in shards[0]["solo"]}
    pair = {key: sum((d["pair"][key] for d in shards), []) for key in shards[0]["pair"]}
    assert all(len(v) == N for v in solo.values()) and all(len(v) == N for v in pair.values()), "merged cell counts != N"
    base, delta, sd = {}, {}, {}
    for a in PASSES:
        for f in FRAGS:
            ys = [mean(solo[skey(a, f, s)]) for s in GRID[f]]
            base[f"{f}|{a}"] = {"xs": list(GRID[f]), "ys": [round(y, 4) for y in ys]}
            sd[f"{f}|{a}"] = [round(float(np.std(solo[skey(a, f, s)], ddof=1)), 4) for s in GRID[f]]
            print(f"[{a}] base_{f} = " + ", ".join(f"{s}:{y:.3f}" for s, y in zip(GRID[f], ys)), flush=True)
        for (f1, f2) in ORDERED_PAIRS:          # composite f1->f2 measures delta_{f2->f1}
            dv = max(0.0, mean(solo[skey(a, f1, MID[f1])]) - mean(pair[pkey(a, f1, f2)]))
            rlo, rhi = RANGES[f2]
            delta[f"{f2}|{f1}|{a}"] = {"xs": [rlo, rhi], "ys": [round(dv, 4), round(dv, 4)]}
        print(f"[{a}] deltas: " + ", ".join(f"{f2}->{f1}:{delta[f'{f2}|{f1}|{a}']['ys'][0]:.4f}"
                                            for (f1, f2) in ORDERED_PAIRS), flush=True)
    try:
        git_rev = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=True).stdout.strip()
    except Exception:
        git_rev = "unknown"
    out = {"attacks_added": list(PASSES), "base": base, "delta": delta, "sd": sd,
           "source": {"n": N, "pool": composition_of(files), "slice": [SLICE_START, SLICE_START + N],
                      "knots": 9, "shards": NSHARD, "per_image_raw": f"{WORK}/shard*of{NSHARD:02d}.json",
                      "measured_at": max(d.get("finished_at", "") for d in shards), "git": git_rev,
                      "script": os.path.basename(__file__),
                      "attack": "DiffWMAttacker(ReSDPipeline(stable-diffusion-2-1), noise_step=60) -- the "
                                "mild diffusion regeneration eval_matrix.py calls 'regen'; rinse2x/4x apply "
                                "it 2 and 4 times in sequence. Distinct from CtrlRegen (advanced attack).",
                      "grids": GRID, "mid": MID}}
    json.dump(out, open(f"{SC}/{OUT_NAME}", "w"), indent=2)
    json.dump(out, open(f"{WORK}/{OUT_NAME}", "w"), indent=2)
    print(f"wrote {SC}/{OUT_NAME}  (N={N}, {NSHARD} shards, 9 knots)", flush=True)
    print("MILDREGEN9_MERGE_DONE", flush=True)
