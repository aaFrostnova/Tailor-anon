"""Capacity is derived from bit accuracy, so a live bit-accuracy measurement re-certifies it.

The solver used to read measured soft-information capacity curves (42 cells) and fall back to a
strength-flat constant elsewhere (18 cells). Neither can be checked live: the live loop measures bit
accuracy. The binary-symmetric-channel bound n(1 - H(1 - ba)) turns that measurement into reliable bits,
sits below the measured curves at every requested payload size, exists on every cell, and moves with
the with_live patch like everything else the solver reads.
"""
import os, sys, json
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import pytest
import numpy as np
from surrogate_model import PWL, synthetic_surrogate
import watermark_smt_v2 as W

SC = "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k"
CANON = os.path.join(SC, "surrogate_canonical.json")
ATT = "regen"


def test_the_bound_and_its_inverse_agree():
    for b in (1, 10, 20, 37, 50, 80, 99):
        assert abs(W.ba_to_bits(W.bits_to_ba(b)) - b) < 1e-6
    assert W.bits_to_ba(0) == 0.5 and W.bits_to_ba(100) == 1.0
    assert W.ba_to_bits(0.5) == 0.0 and W.ba_to_bits(0.3) == 0.0 and W.ba_to_bits(1.0) == 100.0
    assert abs(W.bits_to_ba(37) - 0.8418) < 5e-4            # the deployed ID needs 0.84 mean bit accuracy
    assert W.ba_to_bits(0.90) > 50                          # the identity path's 0.90 sits well above it


@pytest.mark.skipif(not os.path.exists(CANON), reason="canonical table not present")
def test_the_bound_never_grants_a_payload_the_measurement_refuses():
    """At every knot of every measured capacity curve: wherever the measurement is below a payload size
    B, so is the bound, for every B up to 77 bits (requests ask for at most 50). Where the bound does
    exceed the measurement it is by under 4 bits, at saturation of the soft estimator."""
    table = json.load(open(CANON))
    pts = []
    for key, c in table["cap"].items():
        b = table["base"][key]
        for x, bits in zip(c["xs"], c["ys"]):
            pts.append((key, x, float(np.interp(x, b["xs"], b["ys"])), bits))
    assert len(pts) > 200
    for B in (10, 20, 37, 50, 77):
        for key, x, ba, bits in pts:
            assert not (bits < B and W.ba_to_bits(ba) >= B), (B, key, x, ba, bits, W.ba_to_bits(ba))
    assert max(W.ba_to_bits(ba) - bits for _, _, ba, bits in pts) < 4.0


def _sg(level):
    sg = synthetic_surrogate(attacks=(ATT,))
    for f in W.FR:
        lo, hi = sg.range(f)
        sg._base[(f, ATT)] = PWL([lo, hi], [level[f], level[f]])
    return sg


def _sat(sg, bits):
    scen = dict(min_psnr=0.0, max_ms=1e9, attacks=[ATT], min_ba=0.63, allow_resync=False, allow_nested=False,
                min_bits=bits, resolution=512, margin=0.0)
    built, *_ = W.solve_exact_model(scen, enable_order=True, continuous_strength=True, surrogate=sg)
    return built is not None


def test_the_payload_binds_through_bit_accuracy():
    sg = _sg({"VINE": 0.86, "TrustMark": 0.60, "VideoSeal": 0.60})
    assert _sat(sg, 37)                 # 0.86 carries 37 bits (needs 0.844)
    assert not _sat(sg, 50)             # 50 bits need 0.887


def test_a_live_patch_moves_the_capacity_with_it():
    sg = _sg({"VINE": 0.86, "TrustMark": 0.60, "VideoSeal": 0.60})
    assert _sat(sg, 37)
    live = sg.with_live({("VINE", ATT): -0.03})       # measured 0.83 on the user's images
    assert not _sat(live, 37), "the payload must follow the live bit accuracy"
    assert _sat(live, 30)
    assert sg.base("VINE", ATT).eval(0.5) == 0.86, "the shared table must not be written"


@pytest.mark.skipif(not os.path.exists(CANON), reason="canonical table not present")
def test_every_cell_of_the_table_now_has_a_capacity():
    """The 18 cells without a capacity curve (the diffusion and adversarial columns) used to take a
    strength-flat constant; now the capacity is strength-resolved wherever bit accuracy is."""
    from surrogate_model import Surrogate
    sg = Surrogate.from_dict(json.load(open(CANON)))
    for a in sg.attacks:
        for f in W.FR:
            assert sg.base(f, a) is not None, (f, a)
    # the regen ladder binds where it should: VINE reaches 0.949 (70 bits) at full strength
    def sat(attack, bits):
        scen = dict(min_psnr=0.0, max_ms=1e9, attacks=[attack], min_ba=0.63, allow_resync=True,
                    allow_nested=True, min_bits=bits, resolution=512)
        built, *_ = W.solve_exact_model(scen, enable_order=True, continuous_strength=True, surrogate=sg)
        return built is not None
    assert sat("regen", 37) and not sat("regen", 90)
    assert sat("rinse2x", 37) and not sat("rinse2x", 90)
    assert not sat("ctrlregen_s07", 37), "0.640 bit accuracy carries 6 bits, not a 37-bit identity"
