"""Per-fragment cryptographic key derivation.

Each fragment k gets its own (σ_k, M_k) derived from:
    master_key + image_id + fragment_id → HKDF → (permutation σ_k, mask M_k)

σ_k permutes ONLY within fragment k's bit range [k*31, (k+1)*31).
M_k is a ±1 mask of length 31 for fragment k.

The composed global (σ, M) of length 127 is block-diagonal:
    σ[k*31 + i] = k*31 + σ_k[i]
    M[k*31 + i] = M_k[i]

Security: compromising one fragment's (σ_k, M_k) reveals nothing about others.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.sign_envelope import _hkdf_uint64_stream
from src.fragment_payload import FragmentedCodec
from src.payload import image_id_to_payload


def derive_fragment_keyed_constants(
    master_key: bytes,
    image_id: str,
    K: int = 4,
    frag_n: int = 31,
    total_n: int = 127,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive per-fragment (σ_k, M_k), compose into global (σ, M).

    Returns (perm, M) arrays of length total_n=127.
    """
    global_perm = np.arange(total_n, dtype=np.int64)
    global_M = np.ones(total_n, dtype=np.int8)

    for k in range(K):
        salt = b"v5_frag_envelope/" + image_id.encode("utf-8") + f"/frag_{k}".encode()
        offset = k * frag_n

        # σ_k: permutation within [offset, offset+frag_n)
        perm_stream = _hkdf_uint64_stream(master_key, salt + b"/perm", frag_n)
        local_perm = np.arange(frag_n, dtype=np.int64)
        for i in range(frag_n - 1, 0, -1):
            j = int(perm_stream[i] % (i + 1))
            local_perm[i], local_perm[j] = local_perm[j], local_perm[i]
        global_perm[offset:offset + frag_n] = offset + local_perm

        # M_k: ±1 mask for fragment k
        mask_stream = _hkdf_uint64_stream(master_key, salt + b"/mask", frag_n)
        local_M = np.where(mask_stream % 2 == 0, 1, -1).astype(np.int8)
        global_M[offset:offset + frag_n] = local_M

    # Padding positions [K*frag_n, total_n): identity perm, M=+1 (already set)
    return global_perm, global_M


def build_frag_id_pool(
    master_key: bytes,
    pool_size: int,
    codec: FragmentedCodec,
) -> np.ndarray:
    """Pre-compute (pool_size, 127) sign arrays using per-fragment crypto.

    Same output shape as build_id_pool in train_T_lat.py.
    """
    signs = np.zeros((pool_size, codec.n), dtype=np.float32)

    for i in range(pool_size):
        image_id = f"trainpool_{i:06d}"
        payload = image_id_to_payload(image_id, n_bits=codec.data_bits)
        codeword = codec.encode(payload)
        perm, M = derive_fragment_keyed_constants(
            master_key, image_id, K=codec.K, frag_n=codec.frag_n, total_n=codec.n,
        )

        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
        bit_at_region = codeword[inv_perm]
        sign_per_region = M.astype(np.float32) * (1.0 - 2.0 * bit_at_region.astype(np.float32))
        signs[i] = sign_per_region

    return signs


def verify_fragments(
    decoded_bits: np.ndarray,
    master_key: bytes,
    image_id: str,
    codec: FragmentedCodec,
) -> dict:
    """Verify decoded 127 bits against expected payload using per-fragment crypto.

    Returns dict with per-fragment and aggregate results.
    """
    perm, M = derive_fragment_keyed_constants(
        master_key, image_id, K=codec.K, frag_n=codec.frag_n, total_n=codec.n,
    )
    expected_payload = image_id_to_payload(image_id, n_bits=codec.data_bits)

    # Undo crypto: recover codeword bits from decoded region signs
    recovered_cw = np.zeros(codec.n, dtype=np.uint8)
    for j in range(codec.n):
        r = int(perm[j])
        decoded_sign = 1 if decoded_bits[r] == 0 else -1
        original_sign = decoded_sign * int(M[r])
        recovered_cw[j] = 0 if original_sign > 0 else 1

    # Decode each fragment independently
    full_payload, frag_results = codec.decode(recovered_cw)

    # Per-fragment verification
    frag_details = []
    for k in range(codec.K):
        frag_payload, n_err = frag_results[k]
        expected_frag = expected_payload[k * codec.frag_data_bits:(k + 1) * codec.frag_data_bits]
        frag_ok = frag_payload is not None and np.array_equal(frag_payload, expected_frag)
        s, e = codec.frag_range(k)
        raw_acc = float(np.mean(recovered_cw[s:e] == codec.encode(expected_payload)[s:e]))
        frag_details.append({
            "fragment": k,
            "detected": bool(frag_ok),
            "n_errors": int(n_err),
            "bit_accuracy": raw_acc,
        })

    n_detected = sum(f["detected"] for f in frag_details)
    image_detected = full_payload is not None and np.array_equal(full_payload, expected_payload)

    return {
        "image_detected": bool(image_detected),
        "fragments_detected": n_detected,
        "fragments_total": codec.K,
        "fragment_details": frag_details,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    codec = FragmentedCodec()
    master_key = b"test_master_key"

    # Test: different fragments have different (σ, M)
    perm, M = derive_fragment_keyed_constants(master_key, "img_001")
    print(f"Global perm[:5]={perm[:5]}, perm[31:36]={perm[31:36]}")
    print(f"Global M[:5]={M[:5]}, M[31:36]={M[31:36]}")

    # Test: two images have different (σ, M)
    p1, m1 = derive_fragment_keyed_constants(master_key, "img_001")
    p2, m2 = derive_fragment_keyed_constants(master_key, "img_002")
    print(f"Same image? perm equal={np.array_equal(p1,p2)}, M equal={np.array_equal(m1,m2)}")

    # Test: build pool
    pool = build_frag_id_pool(master_key, 4, codec)
    print(f"Pool shape: {pool.shape}, range: [{pool.min()}, {pool.max()}]")

    # Test: verify roundtrip
    payload = image_id_to_payload("trainpool_000000", n_bits=codec.data_bits)
    codeword = codec.encode(payload)
    result = verify_fragments(codeword, master_key, "trainpool_000000", codec)
    print(f"Verify: image_detected={result['image_detected']}, "
          f"fragments={result['fragments_detected']}/{result['fragments_total']}")
