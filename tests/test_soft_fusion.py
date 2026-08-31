"""Unit tests for soft LLR fusion (A.1) and Chase soft-BCH decode (A.3 / C.1)."""

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.soft_fusion import (  # noqa: E402
    prob_to_llr,
    logits_passthrough,
    qim_margin_to_llr,
    sum_redundant_carriers,
    align_llr_to_codeword,
    fuse_llrs,
    llr_to_bits,
    erasure_positions,
)
from src.soft_bch import (  # noqa: E402
    chase_decode,
    decode_with_erasures,
    decode_and_verify,
    hard_from_llr,
)
from src.vine_crypto_wrapper import (  # noqa: E402
    derive_method_keyed_constants,
    apply_crypto,
    undo_crypto,
)
from src.payload import BCHCodec, image_id_to_payload  # noqa: E402

KEY = b"unit_test_master_key"


# --------------------------------------------------------------- soft -> LLR

def test_prob_to_llr_monotone_and_sign():
    assert prob_to_llr(0.9) > 0
    assert prob_to_llr(0.1) < 0
    assert abs(prob_to_llr(0.5)) < 1e-9


def test_qim_margin_sign():
    delta = 0.06
    # exactly on bit-0 lattice -> evidence for 0 (LLR < 0)
    assert qim_margin_to_llr(np.array([0.0]), delta)[0] < 0
    assert qim_margin_to_llr(np.array([delta]), delta)[0] < 0
    # on bit-1 lattice -> evidence for 1 (LLR > 0)
    assert qim_margin_to_llr(np.array([delta / 2]), delta)[0] > 0


def test_sum_redundant_carriers():
    # 2 bits, 3 carriers each
    per = np.array([1.0, 1.0, 1.0, -2.0, 1.0, 0.0])
    out = sum_redundant_carriers(per, carriers_per_bit=3)
    assert np.allclose(out, [3.0, -1.0])


# --------------------------------------------------------------- align == undo_crypto

@pytest.mark.parametrize("n_bits", [100, 127])
def test_align_matches_undo_crypto(n_bits):
    rng = np.random.RandomState(0)
    for trial in range(20):
        image_id = f"img_{trial}"
        perm, M = derive_method_keyed_constants(KEY, image_id, "vine", n_bits)
        # random decoder probabilities in (0,1), avoid exactly 0.5
        decoded = rng.uniform(0.001, 0.999, size=n_bits)
        decoded[np.abs(decoded - 0.5) < 1e-3] = 0.501

        recovered_hard = undo_crypto(decoded, perm, M)

        llr_target = prob_to_llr(decoded)
        aligned = align_llr_to_codeword(llr_target, perm, M, n_codeword=n_bits)
        recovered_soft = llr_to_bits(aligned)

        assert np.array_equal(recovered_hard, recovered_soft)


def test_align_full_pipeline_roundtrip():
    # encode a real codeword, scramble per-method, decode perfectly -> recover cw
    codec = BCHCodec()
    n = codec.n
    payload = image_id_to_payload("roundtrip", n_bits=codec.data_bits)
    cw = codec.encode(payload)
    perm, M = derive_method_keyed_constants(KEY, "roundtrip", "dft_kred", n)
    target = apply_crypto(cw, perm, M)  # encoder targets in {0,1}
    # perfect decoder reports the target bit as a confident probability
    decoded = np.where(target > 0.5, 0.999, 0.001)
    aligned = align_llr_to_codeword(prob_to_llr(decoded), perm, M, n_codeword=n)
    assert np.array_equal(llr_to_bits(aligned), cw)


# --------------------------------------------------------------- fusion improves

def test_fusion_beats_single_method():
    rng = np.random.RandomState(42)
    codec = BCHCodec()
    n = codec.n
    payload = image_id_to_payload("fuse", n_bits=codec.data_bits)
    cw = codec.encode(payload)

    methods = ["vine", "dft_kred", "quant_qim"]
    signal = 0.7  # low per-method SNR
    aligned = {}
    single_errs = []
    for m in methods:
        perm, M = derive_method_keyed_constants(KEY, "fuse", m, n)
        target = apply_crypto(cw, perm, M)
        latent = (2 * target - 1) * signal + rng.randn(n) * 1.0
        prob = 1.0 / (1.0 + np.exp(-latent))
        a = align_llr_to_codeword(prob_to_llr(prob), perm, M, n_codeword=n)
        aligned[m] = a
        single_errs.append(np.mean(llr_to_bits(a) != cw))

    fused = fuse_llrs(aligned, n_codeword=n)
    fused_err = np.mean(llr_to_bits(fused) != cw)
    assert fused_err < min(single_errs)


# --------------------------------------------------------------- Chase decode

def test_chase_decodes_clean():
    codec = BCHCodec()
    payload = image_id_to_payload("clean_img", n_bits=codec.data_bits)
    cw = codec.encode(payload)
    llr = (2.0 * cw - 1.0) * 10.0
    res = decode_and_verify(llr, "clean_img", codec=codec)
    assert res["detected"] is True
    assert np.array_equal(res["payload_bits"], payload)


def test_chase_beats_raw_decode_beyond_t():
    codec = BCHCodec()
    n = codec.n
    payload = image_id_to_payload("hard_img", n_bits=codec.data_bits)
    cw = codec.encode(payload)
    llr = (2.0 * cw - 1.0) * 10.0

    rng = np.random.RandomState(7)
    pos = rng.permutation(n)
    low_conf_wrong = pos[:5]    # 5 errors on low-confidence bits (chase can flip)
    conf_wrong = pos[5:13]      # 8 errors on confident bits (<= t, BCH corrects)

    for i in low_conf_wrong:
        llr[i] = -np.sign(llr[i]) * 0.4   # wrong sign, small magnitude
    for i in conf_wrong:
        llr[i] = -np.sign(llr[i]) * 10.0  # wrong sign, large magnitude

    hard = hard_from_llr(llr)
    # raw single decode sees 13 errors > t=10 -> fails
    _, n_err_raw = codec.decode(hard)
    assert n_err_raw < 0

    res = decode_and_verify(llr, "hard_img", codec=codec, p=8)
    assert res["detected"] is True
    assert np.array_equal(res["payload_bits"], payload)


def test_erasure_decode():
    codec = BCHCodec()
    n = codec.n
    payload = image_id_to_payload("erase_img", n_bits=codec.data_bits)
    cw = codec.encode(payload)
    llr = (2.0 * cw - 1.0) * 10.0

    rng = np.random.RandomState(11)
    pos = rng.permutation(n)
    erasures = pos[:6]      # damaged carriers; bits set wrong
    conf_wrong = pos[6:15]  # 9 confident errors (<= t)

    for i in erasures:
        llr[i] = -np.sign(llr[i]) * 10.0   # confidently WRONG; only erasure flag saves it
    for i in conf_wrong:
        llr[i] = -np.sign(llr[i]) * 10.0

    hard = hard_from_llr(llr)
    _, n_err_raw = codec.decode(hard)
    assert n_err_raw < 0  # 15 errors, uncorrectable raw

    res = decode_and_verify(llr, "erase_img", codec=codec, p=8,
                            erasure_idx=list(erasures))
    assert res["detected"] is True
    assert np.array_equal(res["payload_bits"], payload)


def test_erasure_positions_helper():
    llr = np.array([5.0, 0.1, -4.0, 0.0, -0.05])
    idx = erasure_positions(llr, abs_thresh=0.2)
    assert set(idx.tolist()) == {1, 3, 4}
