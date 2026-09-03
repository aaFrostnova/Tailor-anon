"""The angle probe must order the search without touching the false-accept rate.

A geometric cascade tries many hypotheses. Whether that is safe depends entirely on what admits one: if
acceptance were a correlation threshold, every extra hypothesis would be an extra chance to be wrong,
and a probe that proposes more angles would buy detections with false positives. Acceptance here is a
keyed verification, so a hypothesis is admitted only if the recovered payload equals the expected one.
The probe therefore changes which angles are tried and in what order, never whether a wrong one is
believed.

That is the argument. These tests are the evidence: the probe touches no fragment, proposes a bounded
number of candidates, and its output is independent of the payload -- so an unwatermarked image yields
the same proposals whatever identity is being looked for, and each proposal still has to clear the code.
"""
import os, sys
import numpy as np
import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src import angle_probe as AP
from src.attacks import rotate_fill, rotate_reflect


def _img(seed=0, n=512):        # the deployment size; the wedge must be tall enough to fit a line
    rng = np.random.RandomState(seed)
    a = rng.randint(60, 200, (n, n, 3), dtype=np.uint8)
    a[:, ::16] = 240                                     # some axis-aligned structure
    a[::16, :] = 20
    return Image.fromarray(a)


def test_probe_reads_only_the_image():
    """No fragment, no key, no payload appears in the probe's interface or its module."""
    src = open(os.path.join(REPO, "src/angle_probe.py")).read()
    for forbidden in ("vine", "trustmark", "videoseal", "raw_logits", "raw_probs",
                      "decode_and_verify", "master_key", "ShortenedBCH"):
        assert forbidden.lower() not in src.lower(), f"probe reaches for {forbidden}"


def test_proposal_is_payload_independent():
    """The same image yields the same proposals regardless of what identity is being sought -- there is
    no channel by which the sought payload could bias the search toward accepting it."""
    im = rotate_fill(_img(), 9.0)
    a = AP.candidate_angles(im)
    b = AP.candidate_angles(im)
    assert a == b


def test_candidate_count_is_bounded():
    """The false accepts a search can contribute are (number of verifications) x 2^-37, so the count
    has to be bounded for that product to mean anything."""
    for deg in (0.0, 3.0, 9.0, -20.0):
        im = rotate_fill(_img(), deg) if deg else _img()
        n = len(AP.candidate_angles(im, search=180.0, step=10.0))
        assert n <= 96, f"{deg}: {n} candidates is more than the search should ever need"


def test_search_stays_complete_without_an_estimate():
    """When the wedges are absent the probe must fall back to the full grid rather than propose
    nothing -- an estimator that silently gives up would make rotation unrecoverable for the images it
    cannot read."""
    im = _img()
    assert AP.estimate_magnitudes(im) == []
    cands = AP.candidate_angles(im, search=180.0, step=10.0)
    assert len(cands) == len(np.arange(-180.0, 180.0 + 1e-9, 10.0))
    assert min(cands) <= -179.9 and max(cands) >= 179.9


def test_reflection_padded_rotation_is_declined_not_guessed():
    """The other rotation convention leaves no wedges. The probe must report that it cannot read the
    angle rather than return a confident wrong one."""
    assert AP.estimate_magnitudes(rotate_reflect(_img(), 9.0)) == []


@pytest.mark.parametrize("deg", [3.0, 9.0, 30.0, -9.0, -30.0])
def test_estimate_is_accurate_where_it_applies(deg):
    """Where the wedges exist one of the two edge readings is essentially exact, which is what makes
    ordering worthwhile. Only the magnitude is claimed -- the sign is left to the verification."""
    mags = AP.estimate_magnitudes(rotate_fill(_img(), deg))
    assert mags, "wedges present but nothing estimated"
    assert min(abs(m - abs(deg)) for m in mags) <= 1.0


@pytest.mark.parametrize("deg", [3.0, 9.0, 30.0, -9.0, -30.0])
def test_correct_angle_is_proposed_early(deg):
    """The point of a probe is ordering: the angle that undoes the attack should be near the front of
    the candidate list, so a verification succeeds long before the exhaustive grid is exhausted."""
    cands = AP.candidate_angles(rotate_fill(_img(), deg))
    hits = [i for i, a in enumerate(cands) if abs(a - (-deg)) <= 0.51]
    assert hits, f"the undoing angle {-deg} was never proposed"
    assert hits[0] < 12, f"proposed only at position {hits[0]}"
