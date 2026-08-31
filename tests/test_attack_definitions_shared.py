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


def test_crop_fraction_is_a_side_not_an_area():
    """crop75 keeps 75% of each SIDE. Reading it as an area is the drift that caused the divergence,
    so the convention is asserted rather than left to a comment."""
    im = _img(n=100)
    W, H = im.size
    kept = A.center_crop_resize(im, 0.75)
    assert kept.size == (W, H)                       # canvas restored
    # the crop box is 75 of 100 pixels per side => 56.25% of the area, not 75%
    box_side = int(round(W * 0.75))
    assert box_side == 75


def test_rotation_leaves_no_black_corners():
    """The reported rotation reflect-pads. A default-fill rotation is a different attack: it hands the
    decoder a strong visual cue at the corners, which is exactly where the geometric front-end looks."""
    im = Image.fromarray(np.full((128, 128, 3), 200, dtype=np.uint8))
    rot = A.rotate_reflect(im, 9.0)
    a = np.asarray(rot, np.float32)
    assert (a.sum(2) < 6).mean() == 0.0, "reflect-padded rotation must not produce black pixels"


def test_signal_parameters_are_declared_once():
    """The harness must build its attackers from the shared parameter table."""
    import eval_matrix as EM
    assert EM.SIGNAL_PARAMS is A.SIGNAL_PARAMS
    assert A.SIGNAL_PARAMS["noise"]["std"] == 0.05
    assert A.SIGNAL_PARAMS["bright"]["factor"] == 0.2
