"""Fold the decoded fe_xenv cells into the overlays the canonical builder reads.

Input: perimage_ext/stage_<stage>_xenv/<frag>_<strength>.json (fe_xenv_campaign.py decode). Output:
  surrogate_fe_xenv_<stage>_<frag>.json          replacement curves (base_fe / ctrl_fe / det_fe / detctrl_fe)
                                                 on the CtrlRegen+ and UnMarker columns, full knot sets
  surrogate_fe_xenv_latency.json                 latency_ms_<stage>|<column>: decode with the stage on minus
                                                 decode with the cascade suppressed, pooled over fragments,
                                                 knots and images (the solver clamps a negative value at 0)
  surrogate_fe_components_ext_ring2__0of1.json   the scale/VINE knots 0.14 and 0.17 on those columns: they
                                                 EXTEND the ring curves measured earlier, so they go through
                                                 merge_range_ext.py, which splices knots into an existing
                                                 curve instead of replacing it
The per-image values themselves reach the table through merge_perimage.py (stage_ directory branch).
"""
import json, glob, os, subprocess
import numpy as np
CF = os.environ.get("WM_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SC = os.environ.get("WM_SCRATCH", "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k")
COLS = ("ctrlregen_s03", "ctrlregen_s05", "ctrlregen_s07", "unmarker")
try: git = subprocess.run(["git", "-C", CF, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
except Exception: git = "unknown"
TS = os.environ.get("TS", "unknown")
lat = {}
written = []
for stage in ("resync", "tile", "scale"):
    for frag in ("VINE", "TrustMark", "VideoSeal"):
        files = sorted(glob.glob(f"{SC}/perimage_ext/stage_{stage}_xenv/{frag}_*.json"))
        if not files: continue
        cells = {}
        for p in files:
            d = json.load(open(p)); cells[float(d["strength"])] = d
        fe, n_by_col = {}, {}
        for col in COLS:
            xs = sorted(s for s, d in cells.items() if col in d.get("fe", {}))
            if not xs: continue
            def col_mean(key): return [round(float(np.mean(cells[s]["fe"][col][key])), 4) for s in xs]
            fe[f"base_fe_{stage}_{frag}|{col}"] = {"xs": xs, "ys": col_mean("on")}
            fe[f"ctrl_fe_{stage}_{frag}|{col}"] = {"xs": xs, "ys": col_mean("off")}
            fe[f"det_fe_{stage}_{frag}|{col}"] = {"xs": xs, "ys": col_mean("don")}
            fe[f"detctrl_fe_{stage}_{frag}|{col}"] = {"xs": xs, "ys": col_mean("doff")}
            n_by_col[col] = int(min(cells[s]["fe"][col]["n"] for s in xs))
            for s in xs:
                c = cells[s]["fe"][col]
                lat.setdefault((stage, col), []).extend((np.asarray(c["lat_on_ms"]) - np.asarray(c["lat_off_ms"])).tolist())
        if not fe: continue
        src = {"n": min(n_by_col.values()), "n_by_column": n_by_col, "knots": sorted(cells), "git": git, "measured_at": TS,
               "note": "solo composite with exactly this stage on; CtrlRegen+ and UnMarker in their own environments; "
                       "base_fe = decode with the stage on, ctrl_fe = the same decode with the cascade suppressed; "
                       "per-image values in perimage_ext/stage_<stage>_xenv"}
        if (stage, frag) == ("scale", "VINE"):
            out = f"{SC}/surrogate_fe_components_ext_ring2__0of1.json"
            json.dump({"frontend": fe, "base": {}, "delta": {}, "source": {**src, "new_knots": {"VINE": sorted(cells)}}}, open(out, "w"), indent=1)
        else:
            out = f"{SC}/surrogate_fe_xenv_{stage}_{frag}.json"
            json.dump({"frontend": fe, "latency": {}, "base": {}, "delta": {}, "source": src}, open(out, "w"), indent=1)
        written.append(out)
        print(f"{os.path.basename(out)}: {len(fe)} curves, knots {sorted(cells)}, n {n_by_col}", flush=True)
        for k, v in fe.items():
            if k.startswith("base_fe_"): print(f"    {k:40s} " + " ".join(f"{x:g}:{y:.3f}" for x, y in zip(v['xs'], v['ys'])), flush=True)
LAT = {f"latency_ms_{stage}|{col}": round(float(np.mean(v)), 2) for (stage, col), v in sorted(lat.items())}
if LAT:
    out = f"{SC}/surrogate_fe_xenv_latency.json"
    json.dump({"frontend": {}, "latency": LAT, "base": {}, "delta": {},
               "source": {"n": 30, "git": git, "measured_at": TS,
                          "note": "stage latency on the cross-environment columns: decode with the stage on minus decode "
                                  "with the cascade suppressed, pooled over fragments, knots and images"}}, open(out, "w"), indent=1)
    written.append(out); print("latency:", LAT, flush=True)
print(f"FEX_MERGE_DONE {len(written)} files", flush=True)
