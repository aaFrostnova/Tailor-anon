"""
Fingerprint fragment generation using cryptographic PRNG.

Each fragment is a Gaussian-distributed perturbation pattern derived
deterministically from a sub-key via AES-256-CTR CSPRNG + Box-Muller transform.
"""

import struct
import math
from typing import Tuple

import numpy as np

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def _aes_ctr_stream(key: bytes, nonce: bytes, num_bytes: int) -> bytes:
    """Generate a pseudorandom byte stream using AES-256-CTR.

    Args:
        key: 32-byte AES-256 key.
        nonce: 16-byte nonce/IV for CTR mode.
        num_bytes: Number of bytes to generate.

    Returns:
        Pseudorandom byte stream.
    """
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce))
    encryptor = cipher.encryptor()
    # Encrypt zeros to get the keystream
    plaintext = b"\x00" * num_bytes
    return encryptor.update(plaintext) + encryptor.finalize()


def _bytes_to_uniform(stream: bytes, count: int) -> np.ndarray:
    """Convert raw bytes to uniform [0, 1) float64 values.

    Uses 8 bytes per float for high precision.
    """
    values = np.frombuffer(stream[: count * 8], dtype=np.uint64)
    return values.astype(np.float64) / np.float64(2**64)


def _box_muller(uniform: np.ndarray) -> np.ndarray:
    """Box-Muller transform: convert pairs of uniform samples to Gaussian.

    Args:
        uniform: Array of uniform [0, 1) samples. Length must be even.

    Returns:
        Array of standard normal samples (half the input length).
    """
    n = len(uniform) // 2
    u1 = uniform[:n]
    u2 = uniform[n : 2 * n]

    # Clamp to avoid log(0)
    u1 = np.clip(u1, 1e-15, 1.0 - 1e-15)

    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z0 = r * np.cos(theta)
    return z0


def generate_fragment(
    subkey: bytes,
    shape: Tuple[int, ...],
    epsilon: float = 4.0 / 255.0,
    fragment_index: int = 0,
) -> np.ndarray:
    """Generate a single fingerprint fragment as a Gaussian perturbation.

    Pipeline: sub-key → AES-CTR CSPRNG → uniform → Box-Muller → Gaussian → scale

    Args:
        subkey: 32-byte sub-key for this fragment.
        shape: Output shape (C, H, W) or (H, W) for the perturbation.
        epsilon: Perturbation budget (standard deviation of the Gaussian).
        fragment_index: Used to construct a unique nonce.

    Returns:
        Numpy array of shape `shape` with Gaussian perturbation scaled to epsilon.
    """
    total_elements = 1
    for s in shape:
        total_elements *= s

    # We need 2x elements for Box-Muller (pairs), and 8 bytes per uniform sample
    num_uniform = total_elements * 2
    num_bytes = num_uniform * 8

    # Construct nonce from fragment index (16 bytes for AES-CTR IV)
    nonce = struct.pack(">QQ", 0, fragment_index)

    # Generate CSPRNG stream
    stream = _aes_ctr_stream(subkey, nonce, num_bytes)

    # Convert to uniform then Gaussian
    uniform = _bytes_to_uniform(stream, num_uniform)
    gaussian = _box_muller(uniform)

    # Take exactly the elements we need and reshape
    fragment = gaussian[:total_elements].reshape(shape)

    # Normalize to unit variance, then scale by epsilon
    std = fragment.std()
    if std > 0:
        fragment = fragment / std

    fragment = fragment * epsilon

    return fragment.astype(np.float32)


def generate_all_fragments(
    subkeys: list,
    shape: Tuple[int, ...],
    epsilon: float = 4.0 / 255.0,
    weights: list = None,
) -> Tuple[list, np.ndarray]:
    """Generate all K fragments and compute the aggregated perturbation.

    Args:
        subkeys: List of K sub-keys.
        shape: Image shape (C, H, W).
        epsilon: Per-fragment perturbation budget.
        weights: Per-fragment scaling weights. Uniform (1/K) if None.

    Returns:
        Tuple of (list of individual fragments, aggregated perturbation).
    """
    K = len(subkeys)
    if weights is None:
        weights = [1.0 / K] * K

    fragments = []
    aggregated = np.zeros(shape, dtype=np.float32)

    for k, subkey in enumerate(subkeys):
        delta = generate_fragment(subkey, shape, epsilon, fragment_index=k)
        fragments.append(delta)
        aggregated += weights[k] * delta

    return fragments, aggregated
