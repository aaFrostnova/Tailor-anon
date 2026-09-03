"""Live calibration: the solver's answer, certified on the user's own images.

The table is a search heuristic; the answer is backed by measurement. The loop
  1. solves to the certified optimum on the table (a copy carrying this request's patches),
  2. live-measures, per selected fragment, every in-scope attack the executor can apply to THAT
     configuration on the user's images,
  3. where the measured feasibility verdict contradicts the table's, patches the curve the solve read
     (the front-end replacement curve when a stage is on) by a gated, shrunk, asymmetric offset, and
  4. solves again -- until no contradiction remains, the offsets are all within noise, or the round
     budget is spent. Every constraint comes back with its provenance (table or live).
This is the entry point both the evaluation scripts and a deployment call; the measurement is
injected (`measure`) so the loop can be exercised without a GPU.
"""
import math, time
import numpy as np
import watermark_smt_v2 as W

FKEY = {"VINE": "vine", "TrustMark": "trustmark", "VideoSeal": "videoseal"}


def read_config(built, m):
    o, u, rs, ns, al, nf, ps, tm = built
    S = [f for f in W.FR if str(m.eval(u[f])) == "True"]
    pv = getattr(o, "_pvars", {})
    def before(x, y):
        for k, v in pv.items():
            if k == (x, y): return str(m.eval(v)) == "True"
            if k == (y, x): return str(m.eval(v)) != "True"
        return None
    order = sorted(S, key=lambda f: sum(1 for g in S if g != f and before(g, f) is True))
    fe = W.frontend_decisions(o, m, rs, ns)
    return {"S": S, "order": order, "s": {f: float(m.eval(o._svars[f]).as_fraction()) for f in S},
            "fe": fe, "fe_on": [k for k, v in fe.items() if v],
            "psnr_proxy": float(m.eval(ps).as_fraction()), "ms": float(m.eval(tm).as_fraction())}


def table_value(sg, cfg, f, a):
    """What the solve read for (f, a): the enabled stage's replacement curve when it has one."""
    for k in cfg["fe_on"]:
        c = sg.frontend(sg.fe_key(k, f, a))
        if c is not None:
            return min(1.0, float(c.eval(cfg["s"][f]))), k
    return float(sg.base(f, a).eval(cfg["s"][f])), None


def solve_with_live(scen, measure, sg0, live_ok, max_rounds=4, k=2.0, prior_sd=None, asymmetric=True, log=None):
    """Returns a dict with verdict, cfg, rounds, patched cells and per-attack provenance.

    `measure(cfg, attack) -> {fragment: (mean, se, n)}` is the live measurement of THIS configuration
    on the user's images (cfg carries `threshold`, the value coverage is judged against)."""
    offsets, patched, t0 = {}, [], time.time()
    say = log or (lambda *a, **kw: None)
    for rnd in range(1, max_rounds + 1):
        sg = sg0.with_live(offsets)
        built, m, _r, cert = W.solve_exact_model(scen, enable_order=True, continuous_strength=True, surrogate=sg)
        if built is None:
            return {"verdict": "UNSAT", "cfg": None, "rounds": rnd, "patched": patched,
                    "provenance": {a: "table" for a in scen["attacks"]}, "sec": time.time() - t0}
        cfg = read_config(built, m)
        thr = min(1.0, W.presence_threshold(scen["min_ba"], len(cfg["order"])) + float(scen.get("margin", 0.0)))
        cfg["threshold"] = thr
        contradictions, prov, moved = [], {}, False
        for a in scen["attacks"]:
            if a not in live_ok:
                prov[a] = "table"; continue
            tab = max(table_value(sg, cfg, f, a)[0] for f in cfg["order"] if (f, a) in sg._base)
            live = measure(cfg, a)
            live_best = max(v[0] for v in live.values())
            prov[a] = "live"
            if (live_best >= thr - 1e-9) == (tab >= thr - 1e-9):
                continue
            contradictions.append(a)
            for f, (mean, se, n) in live.items():
                if (f, a) not in sg._base: continue
                stage = table_value(sg0, cfg, f, a)[1]
                d = sg0.live_offset(f, a, cfg["s"][f], mean, se_live=se, prior_sd=prior_sd, k=k, asymmetric=asymmetric, fe=stage)
                key = ("fe", stage, f, a) if stage else (f, a)
                tv = table_value(sg0, cfg, f, a)[0]
                patched.append({"round": rnd, "frag": f, "attack": a, "stage": stage, "table": round(tv, 4),
                                "live": round(float(mean), 4), "se": round(float(se), 4), "n": n, "offset": round(float(d), 4)})
                if abs(d) > 1e-12 and abs(offsets.get(key, 0.0) - d) > 1e-12:
                    offsets[key] = d; moved = True
        say(f"  round {rnd}: {'+'.join(cfg['order'])} s={ {f: round(v, 3) for f, v in cfg['s'].items()} } "
            f"fe={cfg['fe_on'] or ['-']} thr={thr:.3f} -> {len(contradictions)} contradiction(s)" + (f" {contradictions}" if contradictions else ""))
        if not contradictions:
            return {"verdict": "SAT (live-certified)", "cfg": cfg, "rounds": rnd, "patched": patched, "provenance": prov, "sec": time.time() - t0}
        if not moved:
            # every disagreement is inside the live measurement's own noise: nothing to patch, so
            # re-solving would only re-propose the same configuration
            return {"verdict": "SAT (live-inconclusive: disagreement within noise)", "cfg": cfg, "rounds": rnd,
                    "patched": patched, "provenance": prov, "sec": time.time() - t0}
    return {"verdict": "SAT (uncertified: round budget)", "cfg": cfg, "rounds": max_rounds, "patched": patched,
            "provenance": prov, "sec": time.time() - t0}


def composite_measurer(covers, dev="cuda"):
    """The GPU measurement: per-fragment best-path bit accuracy of a configuration under one attack, on
    the given cover images, with the cascade's accepted view used only when the plain view misses the
    threshold (the deployed decoder fires the cascade on a primary miss only)."""
    import gc, torch
    from eval_matrix import OursComposite
    from src.attacks import attack_pil_any
    live = {}
    def composite(cfg):
        key = (tuple(cfg["order"]), tuple(sorted(cfg["fe"].items())))
        if key not in live:
            for kk in list(live): del live[kk]
            gc.collect(); torch.cuda.empty_cache()
            live[key] = OursComposite(dev, tm_variant="B", vine_variant="R",
                                      config={"frags": [FKEY[f] for f in cfg["order"]], "order": [FKEY[f] for f in cfg["order"]],
                                              "strengths": {}, **W.frontend_config(cfg["fe"])})
        c = live[key]
        c.strength = dict(c.DEFAULT_STRENGTH); c.strength.update({FKEY[f]: float(v) for f, v in cfg["s"].items()})
        return c
    def measure(cfg, a):
        comp = composite(cfg); acc = {f: [] for f in cfg["order"]}; thr = cfg.get("threshold", 0.0)
        for i, cov in enumerate(covers):
            emb, sec = comp.embed(cov, i)
            v = attack_pil_any(a, emb, dev=dev)
            if v.size != (512, 512): v = v.resize((512, 512))
            iid, tx = sec
            per = {f: float(np.mean((comp._frag_llr(FKEY[f], v, iid) > 0).astype(np.uint8) == tx)) for f in cfg["order"]}
            if comp.geo and max(per.values()) < thr:
                ok, vw = comp.geo_cascade(v, iid, tx, return_view=True)
                if ok and vw is not None:
                    per_c = {f: float(np.mean((comp._frag_llr(FKEY[f], vw, iid) > 0).astype(np.uint8) == tx)) for f in cfg["order"]}
                    if max(per_c.values()) > max(per.values()): per = per_c
            for f in cfg["order"]: acc[f].append(per[f])
        return {f: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0, len(v)) for f, v in acc.items()}
    return measure
