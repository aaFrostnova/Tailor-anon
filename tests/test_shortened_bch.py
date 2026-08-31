"""Tests for the shortened BCH(100,37,t=10) layer + generic Chase decode on it."""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.shortened_bch import ShortenedBCH  # noqa: E402
from src.payload import image_id_to_payload  # noqa: E402
from src.soft_bch import chase_decode, decode_and_verify, hard_from_llr  # noqa: E402


def test_dimensions():
    sb = ShortenedBCH(n_tx=100)
    assert sb.n == 100
    assert sb.data_bits == 37
    assert sb.ecc_bits == 63


def test_roundtrip_clean():
    sb = ShortenedBCH()
    payload = image_id_to_payload("img_a", n_bits=sb.data_bits)
    tx = sb.encode(payload)
    assert len(tx) == 100
    data, n_err = sb.decode(tx)
    assert n_err == 0
    assert np.array_equal(data, payload)


def test_corrects_up_to_t():
    sb = ShortenedBCH()
    payload = image_id_to_payload("img_b", n_bits=sb.data_bits)
    tx = sb.encode(payload).copy()
    rng = np.random.RandomState(3)
    flip = rng.choice(100, size=10, replace=False)
    tx[flip] ^= 1
    data, n_err = sb.decode(tx)
    assert data is not None and n_err >= 0
    assert np.array_equal(data, payload)


def test_fails_beyond_t_raw():
    sb = ShortenedBCH()
    payload = image_id_to_payload("img_c", n_bits=sb.data_bits)
    tx = sb.encode(payload).copy()
    rng = np.random.RandomState(5)
    flip = rng.choice(100, size=16, replace=False)
    tx[flip] ^= 1
    data, n_err = sb.decode(tx)
    assert n_err < 0  # 16 > t=10 uncorrectable


def test_chase_on_shortened_beyond_t():
    sb = ShortenedBCH()
    image_id = "img_d"
    payload = image_id_to_payload(image_id, n_bits=sb.data_bits)
    tx = sb.encode(payload)
    llr = (2.0 * tx - 1.0) * 10.0

    rng = np.random.RandomState(9)
    pos = rng.permutation(100)
    low = pos[:5]    # low-confidence errors -> chase flips
    high = pos[5:13]  # 8 confident errors (<= t)
    for i in low:
        llr[i] = -np.sign(llr[i]) * 0.4
    for i in high:
        llr[i] = -np.sign(llr[i]) * 10.0

    # raw decode of hard bits fails (13 errors)
    _, n_err_raw = sb.decode(hard_from_llr(llr))
    assert n_err_raw < 0

    res = decode_and_verify(llr, image_id, codec=sb, p=8)
    assert res["detected"] is True
    assert np.array_equal(res["payload_bits"], payload)
