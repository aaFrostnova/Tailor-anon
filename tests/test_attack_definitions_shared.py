"""The evaluation harness and the measurement campaigns must apply the SAME attacks.

They did not. Each program restated the operators, and the restatements diverged: crop read its
fraction as an area rather than a side (so "crop75" kept 75% of the area in one and 56% in the other),
rotation filled corners with black in one and reflected in the other, and brightness/contrast/noise
were materially milder in the campaign. The surrogate the solver reasons over therefore described a
different threat than the matrix it is scored against, and neither program could detect it.

These tests pin the invariant structurally: there is one definition, both entry points reach it, and
applying an attack through either path gives the identical image.
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
sys.path.insert(0, os.path.join(REPO, "external/WatermarkAttacker"))
import numpy as np
import pytest
from PIL import Image
from src import attacks as A


def _img(seed=0, n=128):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, (n, n, 3), dtype=np.uint8))


def test_eval_matrix_imports_rather_than_restates():
    """The harness must reach the shared GEO object itself, not a copy that can drift from it."""
    import eval_matrix as EM
    assert EM.GEO is A.GEO, "eval_matrix holds its own GEO; it must import the shared one"


@pytest.mark.parametrize("name", sorted(A.GEO))
def test_geometry_identical_through_both_entry_points(name):
    """GEO[...] and attack_pil(...) are the same operator, so they must agree bit for bit."""
    im = _img()
    direct = A.GEO[name](im.convert("RGB"))
    if direct.size != im.size:
        direct = direct.resize(im.size, Image.BICUBIC)
    viaapi = A.attack_pil(name, im)
    assert np.array_equal(np.asarray(direct), np.asarray(viaapi)), f"{name} differs between paths"


def test_crop_fraction_is_an_area_not_a_side():
    """crop75 keeps 75% of the AREA, so the side is sqrt(0.75). The alternative reading -- 0.75 per
    side, i.e. 56% of the area -- is a materially harsher attack, and the two conventions coexisting
    unnoticed is what made the surrogate and the matrix disagree. The convention is asserted here so a
    future change has to be deliberate."""
    im = _img(n=100)
    W, H = im.size
    kept = A.center_crop_area(im, 0.75)
    assert kept.size == (W, H)                       # canvas restored
    assert int(W * 0.75 ** 0.5) == 86                # side is 86 of 100, not 75
    assert A.GEO["crop75"] is not None
    assert np.array_equal(np.asarray(A.GEO["crop75"](im)), np.asarray(kept))


def test_rotation_uses_the_default_fill():
    """GEO['rot9'] rotates in place and leaves the corners black. That is lost content, but it is also
    a localisation cue a corner-predicting front-end can exploit, so numbers measured under it are the
    easier of the two conventions and must not be silently swapped for the reflection-padded variant."""
    im = Image.fromarray(np.full((128, 128, 3), 200, dtype=np.uint8))
    black = (np.asarray(A.GEO["rot9"](im), np.float32).sum(2) < 6).mean()
    assert black > 0.01, "the adopted rotation must leave black corners"
    refl = (np.asarray(A.rotate_reflect(im, 9.0), np.float32).sum(2) < 6).mean()
    assert refl == 0.0, "the reflection-padded variant is retained for the convention ablation"


def test_signal_parameters_are_declared_once():
    """The harness must build its attackers from the shared parameter table."""
    import eval_matrix as EM
    assert EM.SIGNAL_PARAMS is A.SIGNAL_PARAMS
    assert A.SIGNAL_PARAMS["noise"]["std"] == 0.05
    assert A.SIGNAL_PARAMS["bright"]["factor"] == 0.2
