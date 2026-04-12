"""
Cryptographic key generation and derivation for fingerprint fragments.

Uses HKDF-SHA256 to derive per-fragment sub-keys from a master secret key
and image identifier, ensuring per-image uniqueness and fragment independence.
"""

import os
import hashlib
from typing import List, Optional

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def generate_master_key(length: int = 32) -> bytes:
    """Generate a cryptographically secure master key.

    Args:
        length: Key length in bytes. Default 32 (AES-256).

    Returns:
        Random bytes of specified length.
    """
    return os.urandom(length)


def save_master_key(key: bytes, path: str) -> None:
    """Save master key to file (hex-encoded)."""
    with open(path, "w") as f:
        f.write(key.hex())


def load_master_key(path: str) -> bytes:
    """Load master key from file (hex-encoded)."""
    with open(path, "r") as f:
        return bytes.fromhex(f.read().strip())


def derive_fragment_subkey(
    master_key: bytes,
    image_id: str,
    fragment_index: int,
    key_length: int = 32,
) -> bytes:
    """Derive a unique sub-key for a specific fragment of a specific image.

    Uses HKDF-SHA256 with info = image_id || fragment_index to ensure:
    - Per-image uniqueness (different image_id → different sub-keys)
    - Fragment independence (different fragment_index → different sub-keys)
    - Cryptographic unlinkability without the master key

    Args:
        master_key: The 256-bit master secret key.
        image_id: Unique identifier for the image.
        fragment_index: Index of the fragment (0 to K-1).
        key_length: Output sub-key length in bytes.

    Returns:
        Derived sub-key bytes.
    """
    info = f"fingerprint:v1:image={image_id}:fragment={fragment_index}".encode("utf-8")
    salt = hashlib.sha256(b"cryptographic-fingerprint-salt-v1").digest()

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=key_length,
        salt=salt,
        info=info,
    )
    return hkdf.derive(master_key)


def derive_all_subkeys(
    master_key: bytes,
    image_id: str,
    num_fragments: int,
    key_length: int = 32,
) -> List[bytes]:
    """Derive sub-keys for all fragments of an image.

    Args:
        master_key: The master secret key.
        image_id: Unique image identifier.
        num_fragments: Number of fragments K.
        key_length: Sub-key length in bytes.

    Returns:
        List of K sub-keys.
    """
    return [
        derive_fragment_subkey(master_key, image_id, k, key_length)
        for k in range(num_fragments)
    ]


def image_id_from_path(image_path: str) -> str:
    """Derive a stable image ID from file path using SHA-256 of the basename."""
    basename = os.path.basename(image_path)
    return hashlib.sha256(basename.encode("utf-8")).hexdigest()[:16]
