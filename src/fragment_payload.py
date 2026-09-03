"""Fragmented BCH codec: K independent fragments with per-fragment error correction.

Layout for K=4, total_n=127:
  Fragment 0: bits [0, 31)   → BCH(31, 16, t=3)
  Fragment 1: bits [31, 62)  → BCH(31, 16, t=3)
  Fragment 2: bits [62, 93)  → BCH(31, 16, t=3)
  Fragment 3: bits [93, 124) → BCH(31, 16, t=3)
  Padding:    bits [124, 127) = zeros

Each fragment independently corrects up to 3 bit errors.
Total data capacity: 4 × 16 = 64 bits (same as single BCH(127,64,t=10)).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.payload import BCHCodec, image_id_to_payload


class FragmentedCodec:

    def __init__(self, K: int = 4, frag_m: int = 5, frag_t: int = 3):
        self.K = K
        self.frag_m = frag_m
        self.frag_t = frag_t
        self.codecs = []
        for _ in range(K):
            c = BCHCodec(m=frag_m, t=frag_t)
            assert c.n == 31, f"Expected BCH n=31 for m=5, got {c.n}"
            self.codecs.append(c)
        self.frag_n = self.codecs[0].n  # 31
        self.frag_data_bits = self.codecs[0].data_bits  # 16
        self._total_n = 127
        self._pad = self._total_n - K * self.frag_n  # 3

    @property
    def n(self) -> int:
        return self._total_n

    @property
    def data_bits(self) -> int:
        return self.K * self.frag_data_bits

    @property
    def t(self) -> int:
        return self.frag_t

    def frag_range(self, k: int) -> Tuple[int, int]:
        return k * self.frag_n, (k + 1) * self.frag_n

    def encode(self, payload_bits: np.ndarray) -> np.ndarray:
        """Encode 64-bit payload into 127-bit fragmented codeword."""
        if len(payload_bits) != self.data_bits:
            raise ValueError(
                f"payload_bits must have length {self.data_bits}, got {len(payload_bits)}"
            )
        fragments = []
        for k in range(self.K):
            start = k * self.frag_data_bits
            end = start + self.frag_data_bits
            frag_cw = self.codecs[k].encode(payload_bits[start:end])
            fragments.append(frag_cw)
        out = np.zeros(self._total_n, dtype=np.uint8)
        for k, frag in enumerate(fragments):
            s, e = self.frag_range(k)
            out[s:e] = frag
        return out

    def decode(
        self, codeword_bits: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], List[Tuple[Optional[np.ndarray], int]]]:
        """Decode 127-bit codeword, each fragment independently.

        Returns:
            (full_payload_or_None, per_fragment_results)
            full_payload is the concatenated 64-bit payload if ALL fragments decode,
            otherwise None.
            per_fragment_results[k] = (16-bit payload or None, n_errors)
        """
        if len(codeword_bits) != self._total_n:
            raise ValueError(
                f"codeword must have length {self._total_n}, got {len(codeword_bits)}"
            )
        frag_results = []
        all_ok = True
        payloads = []
        for k in range(self.K):
            s, e = self.frag_range(k)
            frag_cw = codeword_bits[s:e]
            decoded, n_err = self.codecs[k].decode(frag_cw)
            frag_results.append((decoded, n_err))
            if decoded is None:
                all_ok = False
                payloads.append(np.zeros(self.frag_data_bits, dtype=np.uint8))
            else:
                payloads.append(decoded)

        if all_ok:
            full_payload = np.concatenate(payloads)
        else:
            full_payload = None
        return full_payload, frag_results

    def count_surviving_fragments(
        self, codeword_bits: np.ndarray,
    ) -> Tuple[int, List[bool]]:
        """Count how many fragments successfully BCH-decode."""
        _, frag_results = self.decode(codeword_bits)
        survived = [r[0] is not None for r in frag_results]
        return sum(survived), survived


if __name__ == "__main__":
    codec = FragmentedCodec()
    print(f"FragmentedCodec: K={codec.K}, frag_n={codec.frag_n}, "
          f"frag_data={codec.frag_data_bits}, total_n={codec.n}, data_bits={codec.data_bits}")

    payload = image_id_to_payload("test_image_001", n_bits=codec.data_bits)
    print(f"payload ({len(payload)} bits): {payload[:16]}...")

    codeword = codec.encode(payload)
    print(f"codeword ({len(codeword)} bits)")

    # Perfect decode
    full, frags = codec.decode(codeword)
    print(f"Perfect decode: full={'OK' if full is not None else 'FAIL'}, "
          f"fragments: {[f[1] for f in frags]}")

    # Flip 2 bits in fragment 0 (should survive, t=3)
    noisy = codeword.copy()
    noisy[0] ^= 1
    noisy[3] ^= 1
    full, frags = codec.decode(noisy)
    n_surv, survived = codec.count_surviving_fragments(noisy)
    print(f"2 flips in frag 0: full={'OK' if full is not None else 'FAIL'}, "
          f"surviving={n_surv}/4, per-frag errors={[f[1] for f in frags]}")

    # Flip 5 bits in fragment 1 (should fail, t=3)
    noisy2 = codeword.copy()
    for i in range(31, 36):
        noisy2[i] ^= 1
    full, frags = codec.decode(noisy2)
    n_surv, survived = codec.count_surviving_fragments(noisy2)
    print(f"5 flips in frag 1: full={'OK' if full is not None else 'FAIL'}, "
          f"surviving={n_surv}/4, survived={survived}")
