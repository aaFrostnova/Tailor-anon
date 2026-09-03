"""Map codeword bits to spatial sub-regions of each fragment.

Each signed fragment carries B_k codeword bits, with each bit modulating
the sign of one disjoint spatial sub-region of the fragment. The partition
into sub-regions follows a deterministic grid derived from the fragment's
overall region mask plus an HKDF-derived permutation salt.

Why per-region (not per-coefficient) sign modulation
----------------------------------------------------
Embedding via existing strategies (`embed_dct`, `embed_dwt_dct`, `embed_pixel`)
operates block-locally on 8x8 cells. A pixel-domain sign change in a
contiguous spatial region therefore translates to a sign change in the
corresponding DCT/DWT blocks without spectral leakage between bit groups.
The decoder later sums per-region inner products to recover each bit's sign.

Bit budgets per fragment come from the config dict's `bit_budget` field
(see fragment_config.py). Fragments without a `bit_budget` (or with
`bit_budget == 0`) contribute redundant detection energy only.
"""

from typing import Dict, List, Tuple

import hmac
import hashlib

import numpy as np

from .fragment_config import get_spatial_mask


# ---------------------------------------------------------------- partitioning

def _hkdf_int_stream(key: bytes, salt: bytes, n_uint64: int) -> np.ndarray:
    """Generate a stream of n_uint64 deterministic uint64 values via HMAC-SHA512.

    Equivalent crypto strength to derive_fragment_subkey but cheaper for the
    short streams we need here (permutation indices).
    """
    out = bytearray()
    counter = 0
    while len(out) < n_uint64 * 8:
        chunk = hmac.new(key, salt + counter.to_bytes(4, "big"), hashlib.sha512).digest()
        out.extend(chunk)
        counter += 1
    arr = np.frombuffer(bytes(out[: n_uint64 * 8]), dtype=np.uint64).copy()
    return arr


def _key_perm(key: bytes, salt: bytes, n: int) -> np.ndarray:
    """Deterministic permutation of [0, n) seeded from key+salt."""
    stream = _hkdf_int_stream(key, salt, n)
    # Fisher-Yates using the deterministic stream
    perm = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = int(stream[i] % (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _grid_partition(mask: np.ndarray, n_groups: int) -> List[np.ndarray]:
    """Partition the True pixels of `mask` (shape C,H,W) into n_groups roughly
    equal-area contiguous sub-regions using a grid layout over the bounding box.

    Returns a list of n_groups boolean arrays of mask.shape.
    """
    if n_groups <= 0:
        return []
    C, H, W = mask.shape
    # Find bounding box of the mask in (H, W)
    ys, xs = np.where(mask[0] > 0)
    if len(ys) == 0:
        return [np.zeros_like(mask) for _ in range(n_groups)]
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    # Choose grid dims close to square for n_groups
    nrows = int(np.ceil(np.sqrt(n_groups)))
    ncols = int(np.ceil(n_groups / nrows))

    cell_h = (y1 - y0) / nrows
    cell_w = (x1 - x0) / ncols

    groups: List[np.ndarray] = []
    cell_idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if cell_idx >= n_groups:
                break
            sub = np.zeros_like(mask)
            yy0 = int(y0 + r * cell_h)
            yy1 = int(y0 + (r + 1) * cell_h) if r < nrows - 1 else y1
            xx0 = int(x0 + c * cell_w)
            xx1 = int(x0 + (c + 1) * cell_w) if c < ncols - 1 else x1
            sub[:, yy0:yy1, xx0:xx1] = mask[:, yy0:yy1, xx0:xx1]
            groups.append(sub)
            cell_idx += 1
    while len(groups) < n_groups:
        groups.append(np.zeros_like(mask))
    return groups[:n_groups]


# -------------------------------------------------------------- public API

def assign_bits_to_regions(
    configs: List[Dict],
    image_shape: Tuple[int, int, int],
    n_bits: int,
    master_key: bytes,
    image_id: str,
) -> Tuple[List[List[np.ndarray]], List[List[int]]]:
    """Distribute n_bits across configs[*]'s spatial sub-regions.

    For each fragment k:
      - read B_k = configs[k].get("bit_budget", 0)
      - partition fragment's spatial region into B_k contiguous sub-regions
      - assign a deterministic permutation of bit indices to these sub-regions

    The global bit index allocation across all fragments must sum to >= n_bits
    (a ValueError is raised otherwise). Extra capacity beyond n_bits is unused.

    Args:
        configs: list of K fragment config dicts (must include "region",
            optional "bit_budget").
        image_shape: (C, H, W).
        n_bits: total codeword length (e.g., 127).
        master_key: master secret key.
        image_id: image identifier (used in salt so different images get
            different bit-region permutations).

    Returns:
        region_masks: list of length K. region_masks[k] is a list of B_k
            boolean masks of image_shape.
        bit_indices: list of length K. bit_indices[k] is a list of B_k codeword
            indices, the i-th sub-region carries codeword bit bit_indices[k][i].
    """
    if len(image_shape) != 3:
        raise ValueError(f"image_shape must be (C, H, W), got {image_shape}")

    K = len(configs)
    budgets = [int(configs[k].get("bit_budget", 0)) for k in range(K)]
    total = sum(budgets)
    if total < n_bits:
        raise ValueError(
            f"sum of bit_budget across fragments ({total}) < n_bits ({n_bits}). "
            f"Per-fragment budgets: {budgets}"
        )

    # Build a deterministic bit-index allocation: pick the FIRST n_bits slots
    # from a key-permuted ordering of all (fragment_k, slot_in_k) pairs.
    salt = b"bit_assignment/" + image_id.encode("utf-8")
    slot_pairs: List[Tuple[int, int]] = []
    for k, b in enumerate(budgets):
        for i in range(b):
            slot_pairs.append((k, i))
    perm = _key_perm(master_key, salt + b"/perm", len(slot_pairs))

    # Apply: codeword bit `j` is carried by slot (k=slot_pairs[perm[j]][0],
    # i=slot_pairs[perm[j]][1]).
    bit_indices: List[List[int]] = [[-1] * b for b in budgets]
    for j in range(n_bits):
        k, i = slot_pairs[int(perm[j])]
        bit_indices[k][i] = j

    # Build per-fragment region masks
    region_masks: List[List[np.ndarray]] = []
    for k in range(K):
        region = configs[k].get("region", "full")
        b = budgets[k]
        if b == 0:
            region_masks.append([])
            continue
        full_mask = get_spatial_mask(image_shape, region)
        groups = _grid_partition(full_mask, b)
        region_masks.append(groups)

    return region_masks, bit_indices


# ------------------------------------------------------------- apply / decode

def apply_bit_signs(
    fragment: np.ndarray,
    region_masks: List[np.ndarray],
    bit_index_map: List[int],
    codeword_bits: np.ndarray,
) -> np.ndarray:
    """Apply per-sub-region sign modulation to a single fragment.

    Args:
        fragment: (C, H, W) Gaussian fragment from generate_fragment.
        region_masks: B_k boolean masks of fragment.shape (output of
            assign_bits_to_regions for this fragment).
        bit_index_map: B_k codeword indices (output of assign_bits_to_regions).
        codeword_bits: shape (n_bits,) {0,1} codeword.

    Returns:
        signed_fragment of same shape as `fragment`, with per-region signs
        applied. Unsigned regions remain unchanged.
    """
    signed = fragment.copy()
    for mask, bit_idx in zip(region_masks, bit_index_map):
        if bit_idx < 0:
            continue
        # bit==0 → +1 sign, bit==1 → -1 sign (matches payload.bits_to_signs)
        sign = 1.0 if int(codeword_bits[bit_idx]) == 0 else -1.0
        if sign < 0:
            # Multiply only within the masked region. Sign in the rest of
            # the fragment stays as the original Gaussian.
            signed = np.where(mask > 0, -signed, signed)
    return signed.astype(fragment.dtype)


def recover_bits_from_residual(
    residual: np.ndarray,
    expected_fragment: np.ndarray,
    region_masks: List[np.ndarray],
    bit_index_map: List[int],
    n_bits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode per-region sign of (residual · expected_fragment) into bits.

    Args:
        residual: (C, H, W) residual = extract_single_fragment(orig, fp, config).
            For non-pixel strategies this is already brought back into pixel
            space by extract_*.
        expected_fragment: the ORIGINAL Gaussian fragment (no sign applied)
            from generate_fragment, same shape.
        region_masks: per-bit sub-region masks for this fragment.
        bit_index_map: per-bit codeword indices for this fragment.
        n_bits: codeword length.

    Returns:
        (sparse_bits, sparse_scores) — only the indices owned by this fragment
        are set; caller merges across fragments.
        sparse_bits[j] ∈ {0, 1} for j in bit_index_map; -1 otherwise.
        sparse_scores[j] = signed inner product magnitude (for soft-decision).
    """
    out_bits = -np.ones(n_bits, dtype=np.int8)
    out_scores = np.zeros(n_bits, dtype=np.float32)
    for mask, bit_idx in zip(region_masks, bit_index_map):
        if bit_idx < 0:
            continue
        ip = float(np.sum(residual * expected_fragment * (mask > 0)))
        out_scores[bit_idx] = ip
        out_bits[bit_idx] = 0 if ip >= 0 else 1
    return out_bits, out_scores
