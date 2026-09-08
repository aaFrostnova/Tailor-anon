"""The class-level live numbers carry a bootstrap interval over the held-out images.

All configurations are measured on the same test set, so the class-level live-feasible rate has the
uncertainty of that image set; the interval comes from resampling images with replacement and re-judging
every request, keeping the pairing across requests and attacks.
"""
import os, sys
import numpy as np
SC = "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k"
sys.path.insert(0, SC); sys.path.insert(0, "/work/pi_shiqingma_umass_edu/mingzheli/cryptographic_fingerprint/scripts/defense")
import pytest
pytest.importorskip("certify_full_verdicts")
import certify_full_verdicts as V


def _data(spread):
    rng = np.random.default_rng(1); n = 100
    def cell(mean):
        ba = np.clip(rng.normal(mean, spread, n), 0, 1)
        return {"ba": {"VINE": ba.tolist()}, "ver": {"VINE": [False] * n}, "det": [1.0] * n}
    g0 = {"gidx": 0, "order": ["VINE"], "fe": [], "s": {"VINE": 0.5}, "attacks": {"jpeg25": cell(0.70), "regen": cell(0.66)}}
    g1 = {"gidx": 1, "order": ["VINE"], "fe": [], "s": {"VINE": 0.9}, "attacks": {"jpeg25": cell(0.95), "regen": cell(0.90)}}
    groups = {V.key_of({"order": g["order"], "fe": {}, "s": g["s"]}): g for g in (g0, g1)}
    recs = []
    for i in range(40):                       # 20 requests on each configuration, budgets 1e-1 / 1e-2 alternating
        g = g0 if i % 2 == 0 else g1
        recs.append({"i": i, "fpr": 0.1 if i % 4 < 2 else 0.01, "min_ba": 0.57 if i % 4 < 2 else 0.63, "min_bits": 0, "stress": i % 10 == 0,
                     "attacks": ["jpeg25", "regen"], "solver": {"order": g["order"], "fe": {}, "s": g["s"]}})
    return recs, groups


def test_interval_brackets_the_point_estimate_and_narrows_with_less_spread():
    recs, groups = _data(spread=0.08)
    rows, S = V.verdicts(recs, groups, det_min=0.9, B=400)
    b = S["bootstrap"]["all"]
    assert b["mean_ok"][0] <= S["mean_ok"] <= b["mean_ok"][1] and b["rate_ok"][0] <= S["rate_ok"] <= b["rate_ok"][1]
    assert 0 <= b["mean_ok"][0] and b["mean_ok"][1] <= 1
    recs2, groups2 = _data(spread=0.005)
    _, S2 = V.verdicts(recs2, groups2, det_min=0.9, B=400)
    w1 = b["mean_ok"][1] - b["mean_ok"][0]; w2 = S2["bootstrap"]["all"]["mean_ok"][1] - S2["bootstrap"]["all"]["mean_ok"][0]
    assert w2 <= w1


def test_stress_and_within_ceiling_are_reported_apart():
    recs, groups = _data(spread=0.05)
    _, S = V.verdicts(recs, groups, det_min=0.9, B=200)
    assert S["bootstrap"]["within_ceiling"]["n"] == 36 and S["bootstrap"]["stress"]["n"] == 4


def test_pending_requests_are_left_out_of_the_interval():
    recs, groups = _data(spread=0.05)
    for g in groups.values(): g["attacks"].pop("regen")          # the cross-environment cell has not landed
    rows, S = V.verdicts(recs, groups, det_min=0.9, B=100)
    assert S["n_done"] == 0 and "bootstrap" not in S and all(r["pending"] == ["regen"] for r in rows)
