"""Unit tests for the array-level tamper-detection helpers (Component C.2/C.3)."""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.tamper_detect import (  # noqa: E402
    carrier_eff_decode, degradation_score, radial_nonuniformity,
    localize_dft_band, qim_low_region,
)


def test_carrier_eff_decode():
    delta = 10.0
    # on bit-0 lattice -> 0 ; near delta/2 -> 1
    mag = np.array([0.0, 10.0, 20.0, 5.0, 15.0])
    out = carrier_eff_decode(mag, delta)
    assert out.tolist() == [0, 0, 0, 1, 1]


def test_degradation_score():
    assert degradation_score(5.0, 5.0) == 0.0          # intact
    assert degradation_score(0.0, 5.0) == 1.0          # fully collapsed
    assert 0.4 < degradation_score(2.5, 5.0) < 0.6     # half


def test_radial_nonuniformity():
    assert radial_nonuniformity([1, 1, 1, 1]) == 0.0          # uniform
    assert radial_nonuniformity([1, 1, 0.2, 0.1]) > 0.8       # concentrated damage


def test_localize_dft_band():
    surv = [1.0, 0.9, 0.2, 0.95]
    bins = np.array([0, 10, 20, 30, 40], dtype=float)
    out = localize_dft_band(surv, bins)
    assert out["band"] == 2
    assert out["r_lo"] == 20.0 and out["r_hi"] == 30.0


def test_qim_low_region_contiguous():
    # a contiguous 2x2 low-survival region -> clustering 1.0, tight bbox
    m = np.ones((8, 8))
    m[2:4, 5:7] = 0.0
    clustering, bbox = qim_low_region(m, thresh=0.6)
    assert clustering == 1.0
    assert bbox == (2, 5, 3, 6)


def test_qim_low_region_scattered():
    # scattered single low blocks -> clustering < 1
    m = np.ones((8, 8))
    m[0, 0] = 0.0
    m[7, 7] = 0.0
    m[3, 4] = 0.0
    clustering, bbox = qim_low_region(m, thresh=0.6)
    assert clustering < 0.5


def test_qim_low_region_none():
    m = np.ones((8, 8))
    clustering, bbox = qim_low_region(m, thresh=0.6)
    assert clustering == 0.0 and bbox is None
