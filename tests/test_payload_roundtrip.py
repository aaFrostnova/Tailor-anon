"""Unit tests for v2 payload round-trip and BCH ECC tolerance.

Covers:
  - BCH(127, 64, t=10) encode/decode round-trip
  - Tolerance to up to t bit flips
  - Detection of >t flips as uncorrectable
  - image_id → payload determinism + collision distinctness
  - Sign-mapping inverse
  - embed_multi_domain_signed → verify_multi_domain_v2 end-to-end
    on a randomly textured 128x128 image (no attack)
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.bit_assignment import assign_bits_to_regions  # noqa: E402
from src.embed import embed_multi_domain_signed  # noqa: E402
from src.fragment import generate_fragment  # noqa: E402
from src.fragment_config import get_v2_configs  # noqa: E402
from src.jnd import compute_jnd_mask  # noqa: E402
from src.keygen import derive_all_subkeys  # noqa: E402
from src.payload import (  # noqa: E402
    BCHCodec,
    bits_to_signs,
    image_id_to_payload,
    payload_to_hex,
    signs_to_bits,
)
from src.verify import verify_multi_domain_v2  # noqa: E402


# ----------------------------------------------------------------- BCH tests

@pytest.fixture(scope="module")
def codec() -> BCHCodec:
    return BCHCodec()


def test_bch_dimensions(codec: BCHCodec) -> None:
    assert codec.n == 127
    assert codec.data_bits == 64
    assert codec.t == 10


def test_bch_clean_roundtrip(codec: BCHCodec) -> None:
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, codec.data_bits, dtype=np.uint8)
    cw = codec.encode(payload)
    rec, n_err = codec.decode(cw)
    assert rec is not None
    assert np.array_equal(rec, payload)
    assert n_err == 0


@pytest.mark.parametrize("n_flips", [1, 3, 5, 8, 10])
def test_bch_correctable_flips(codec: BCHCodec, n_flips: int) -> None:
    rng = np.random.default_rng(n_flips)
    payload = rng.integers(0, 2, codec.data_bits, dtype=np.uint8)
    cw = codec.encode(payload).copy()
    flip_idx = rng.choice(codec.n, n_flips, replace=False)
    cw[flip_idx] = 1 - cw[flip_idx]
    rec, n_err = codec.decode(cw)
    assert rec is not None, f"BCH failed at {n_flips} flips"
    assert np.array_equal(rec, payload)
    assert n_err == n_flips


@pytest.mark.parametrize("n_flips", [15, 20, 30])
def test_bch_uncorrectable(codec: BCHCodec, n_flips: int) -> None:
    rng = np.random.default_rng(100 + n_flips)
    payload = rng.integers(0, 2, codec.data_bits, dtype=np.uint8)
    cw = codec.encode(payload).copy()
    flip_idx = rng.choice(codec.n, n_flips, replace=False)
    cw[flip_idx] = 1 - cw[flip_idx]
    rec, n_err = codec.decode(cw)
    # Either uncorrectable (None) or silently miscorrected to a wrong codeword
    assert rec is None or not np.array_equal(rec, payload)


# ------------------------------------------------------------ image_id mapping

def test_image_id_determinism() -> None:
    p1 = image_id_to_payload("cat42")
    p2 = image_id_to_payload("cat42")
    assert np.array_equal(p1, p2)


def test_image_id_distinct() -> None:
    p1 = image_id_to_payload("cat42")
    p2 = image_id_to_payload("dog99")
    assert not np.array_equal(p1, p2)


def test_payload_to_hex_length() -> None:
    p = image_id_to_payload("abc")
    h = payload_to_hex(p)
    assert len(h) == 16  # 64 bits / 4 bits per hex char


# --------------------------------------------------------------- sign mapping

def test_sign_inverse() -> None:
    bits = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
    s = bits_to_signs(bits)
    b = signs_to_bits(s)
    assert np.array_equal(b, bits)


# ---------------------------------------------------- end-to-end embed/verify

def _make_textured_image(h: int = 128, w: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, (3, h, w)).astype(np.float32)
    # Add some structure so DCT has signal
    y, x = np.mgrid[0:h, 0:w].astype(np.float32) / max(h, w)
    base[0] += 0.3 * np.sin(8 * np.pi * x)
    base[1] += 0.3 * np.cos(8 * np.pi * y)
    return np.clip(base, 0, 1).astype(np.float32)


def test_embed_verify_no_attack_roundtrip() -> None:
    img = _make_textured_image()
    configs = get_v2_configs()
    K = len(configs)
    master_key = b"\x42" * 32
    image_id = "test_image_x"
    epsilon = 16.0 / 255.0

    codec = BCHCodec()
    payload = image_id_to_payload(image_id, codec.data_bits)
    codeword = codec.encode(payload)
    subkeys = derive_all_subkeys(master_key, image_id, K)
    fragments = [
        generate_fragment(sk, img.shape, epsilon, fragment_index=k)
        for k, sk in enumerate(subkeys)
    ]
    sign_masks, bit_idx = assign_bits_to_regions(
        configs, img.shape, codec.n, master_key, image_id,
    )
    jnd = compute_jnd_mask(img)
    fp = embed_multi_domain_signed(
        img, fragments, sign_masks, bit_idx, codeword,
        configs, weights=None, jnd_mask=jnd,
    )

    mse = float(np.mean((fp - img) ** 2))
    assert mse > 0, "embedding produced identical image"
    psnr = 10 * np.log10(1.0 / mse)
    assert psnr > 28, f"PSNR={psnr:.2f} dB too low"

    res = verify_multi_domain_v2(
        img, fp, master_key, image_id, configs,
        epsilon=epsilon, n_bits=codec.n,
    )
    assert res["image_id_match"], (
        f"payload mismatch: got {res['recovered_image_id_hex']!r}, "
        f"expected {res['expected_image_id_hex']!r}"
    )
    assert res["detected"], f"detected=False on no-attack round-trip"
    assert res["n_err"] <= codec.t, f"too many bit errors: {res['n_err']}"


def test_verify_with_wrong_key_misses() -> None:
    img = _make_textured_image(seed=1)
    configs = get_v2_configs()
    K = len(configs)
    correct_key = b"\x42" * 32
    wrong_key = b"\x37" * 32
    image_id = "test_image_y"
    epsilon = 16.0 / 255.0

    codec = BCHCodec()
    payload = image_id_to_payload(image_id, codec.data_bits)
    codeword = codec.encode(payload)
    subkeys = derive_all_subkeys(correct_key, image_id, K)
    fragments = [
        generate_fragment(sk, img.shape, epsilon, fragment_index=k)
        for k, sk in enumerate(subkeys)
    ]
    sign_masks, bit_idx = assign_bits_to_regions(
        configs, img.shape, codec.n, correct_key, image_id,
    )
    fp = embed_multi_domain_signed(
        img, fragments, sign_masks, bit_idx, codeword, configs,
    )

    # Verifier uses WRONG key → should NOT detect
    res = verify_multi_domain_v2(
        img, fp, wrong_key, image_id, configs,
        epsilon=epsilon, n_bits=codec.n,
    )
    # With a wrong key, payload must not match (false positive rate ≈ 1/2^64)
    assert not res["image_id_match"], (
        "wrong-key verifier should NOT recover the correct payload"
    )
    assert not res["detected"]
