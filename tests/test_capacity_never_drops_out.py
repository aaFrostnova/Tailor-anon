"""An identity payload is never granted on a column whose capacity nobody measured.

The 9-knot diffusion campaign renamed three columns (rinse -> rinse2x/rinse4x, ctrlregen ->
ctrlregen_s05_x2) that the capacity measurements still carry under the old names. A capacity cell that
matches nothing left the clause empty, and an empty clause was simply not asserted, so the payload
constraint dropped out: a 90-bit identity under rinse2x came back SAT while the same request under
regen (capacity 80.9 bit, measured) is correctly UNSAT. The measured constants still back
achievable_bits and the discrete path; the strength-resolved solver derives capacity from bit accuracy.
"""
import os, sys, json
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts/defense"))
import pytest
import watermark_smt_v2 as W

SC = "/scratch/workspace/mingzhel_umass_edu-ablator/wm_dataset10k"
CANON = os.path.join(SC, "surrogate_canonical.json")


def test_the_alias_recovers_the_measurement_taken_under_the_old_name():
    assert W.cap_constant("VINE", "rinse2x") == W.CAP["VINE"]["rinse"]
    assert W.cap_constant("TrustMark", "rinse2x") == W.CAP["TrustMark"]["rinse"]


def test_a_column_harder_than_anything_measured_carries_no_bits():
    """rinse4x and ctrlregen_s05_x2 are out of the request scope now (the canonical build drops them),
    but the rule that named them stays: a column strictly harder than anything measured credits no bits,
    so re-admitting one cannot silently grant a payload."""
    for a in ("rinse4x", "ctrlregen_s05_x2"):
        for f in W.FR:
            assert W.cap_constant(f, a) == 0.0, (f, a)


@pytest.mark.skipif(not os.path.exists(CANON), reason="canonical table not present")
def test_the_dropped_columns_are_not_in_the_request_scope():
    table = json.load(open(CANON))
    for a in ("rinse4x", "ctrlregen_s05_x2"):
        assert a not in table["attacks"], f"{a} is back in the table but has no measured capacity"


def test_an_unknown_column_is_reported_as_unknown_not_as_zero():
    assert W.cap_constant("VINE", "a_column_nobody_measured") is None


@pytest.mark.skipif(not os.path.exists(CANON), reason="canonical table not present")
def test_the_payload_clause_cannot_come_out_empty():
    """Capacity is now derived from the bit-accuracy curve of every cell (W.ba_to_bits), so the clause
    exists wherever coverage does and the ladder binds on the columns that once had no curve."""
    from surrogate_model import Surrogate
    sg = Surrogate.from_dict(json.load(open(CANON)))

    def sat(attack, bits, beta=0.63):
        scen = dict(min_psnr=0.0, max_ms=1e9, attacks=[attack], min_ba=beta, allow_resync=True,
                    allow_nested=True, min_bits=bits, resolution=512)
        built, *_ = W.solve_exact_model(scen, enable_order=True, continuous_strength=True, surrogate=sg)
        return built is not None

    assert sat("regen", 37) and not sat("regen", 90)
    assert sat("rinse2x", 37) and not sat("rinse2x", 90), "the payload was unbounded on rinse2x"
    for a in sg.attacks:
        assert all(sg.base(f, a) is not None for f in W.FR), a
