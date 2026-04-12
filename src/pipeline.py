"""
End-to-end fingerprint generation pipeline.

Supports two modes:
- Single-domain: all fragments embedded in one strategy (legacy)
- Multi-domain: each fragment in its own domain/region (recommended)
"""

import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from PIL import Image

from .keygen import (
    generate_master_key,
    save_master_key,
    load_master_key,
    derive_all_subkeys,
    image_id_from_path,
)
from .fragment import generate_fragment, generate_all_fragments
from .embed import embed, extract, embed_multi_domain, extract_multi_domain
from .verify import verify_fingerprint, verify_multi_domain
from .metrics import compute_all_metrics
from .fragment_config import get_default_configs


def load_config(config_path: str) -> Dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_image(path: str) -> np.ndarray:
    """Load an image as float32 numpy array in (C, H, W) format, range [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)


def save_image(arr: np.ndarray, path: str) -> None:
    """Save a (C, H, W) float32 [0,1] array as an image file."""
    hwc = arr.transpose(1, 2, 0)
    hwc = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(hwc).save(path)


class FingerprintPipeline:
    """End-to-end pipeline for fingerprint generation and verification.

    Args:
        master_key: Secret key (generated if None).
        config: Config dict.
        config_path: Path to YAML config.
        multi_domain: If True (default), embed each fragment in its own domain.
        fragment_configs: Custom per-fragment configs (auto-generated if None).
    """

    def __init__(
        self,
        master_key: Optional[bytes] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        multi_domain: bool = True,
        fragment_configs: Optional[List[Dict]] = None,
    ):
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = load_config(
                os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
            )

        if master_key:
            self.master_key = master_key
        else:
            key_len = self.config.get("key", {}).get("length", 32)
            self.master_key = generate_master_key(key_len)

        self.num_fragments = self.config["fragments"]["num_fragments"]
        self.epsilon = self.config["fragments"]["epsilon"]
        self.weights = self.config["fragments"].get("weights")
        self.strategy = self.config["embedding"]["strategy"]
        self.multi_domain = multi_domain

        # Per-fragment configs for multi-domain mode
        if fragment_configs:
            self.fragment_configs = fragment_configs
        else:
            self.fragment_configs = get_default_configs(self.num_fragments)

    def _get_embed_kwargs(self) -> Dict:
        """Get strategy-specific kwargs from config (single-domain mode)."""
        kwargs = {}
        if self.strategy == "dct":
            dct_cfg = self.config["embedding"].get("dct", {})
            kwargs["block_size"] = dct_cfg.get("block_size", 8)
            kwargs["freq_range"] = tuple(dct_cfg.get("freq_range", [10, 40]))
        elif self.strategy == "dwt-dct":
            dwt_cfg = self.config["embedding"].get("dwt", {})
            dct_cfg = self.config["embedding"].get("dct", {})
            kwargs["wavelet"] = dwt_cfg.get("wavelet", "haar")
            kwargs["level"] = dwt_cfg.get("level", 2)
            kwargs["block_size"] = dct_cfg.get("block_size", 8)
            kwargs["freq_range"] = tuple(dct_cfg.get("freq_range", [10, 40]))
        return kwargs

    def fingerprint_image(
        self,
        image: np.ndarray,
        image_id: str,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply fingerprint to a single image.

        Args:
            image: Float32 image in [0, 1], shape (C, H, W).
            image_id: Unique identifier for this image.

        Returns:
            Tuple of (fingerprinted image, metadata dict).
        """
        subkeys = derive_all_subkeys(self.master_key, image_id, self.num_fragments)

        # Generate individual fragments
        fragments = [
            generate_fragment(subkeys[k], image.shape, self.epsilon, fragment_index=k)
            for k in range(self.num_fragments)
        ]

        if self.multi_domain:
            # Each fragment embedded in its own domain/region
            K = self.num_fragments
            weights = self.weights or [1.0 / K] * K
            fingerprinted = embed_multi_domain(
                image, fragments, self.fragment_configs, weights
            )
            mode = "multi-domain"
        else:
            # All fragments aggregated and embedded in one strategy
            K = self.num_fragments
            weights = self.weights or [1.0 / K] * K
            aggregated = sum(w * f for w, f in zip(weights, fragments))
            embed_kwargs = self._get_embed_kwargs()
            fingerprinted = embed(image, aggregated, strategy=self.strategy, **embed_kwargs)
            mode = f"single-domain ({self.strategy})"

        metadata = {
            "image_id": image_id,
            "num_fragments": self.num_fragments,
            "epsilon": self.epsilon,
            "mode": mode,
            "perturbation_norm_l2": float(np.linalg.norm(fingerprinted - image)),
            "perturbation_norm_linf": float(np.max(np.abs(fingerprinted - image))),
        }

        if self.multi_domain:
            metadata["fragment_configs"] = [
                {"strategy": c["strategy"], "region": c.get("region", "full"),
                 "freq_range": c.get("freq_range"), "level": c.get("level")}
                for c in self.fragment_configs
            ]

        return fingerprinted, metadata

    def fingerprint_file(
        self,
        input_path: str,
        output_path: str,
        image_id: Optional[str] = None,
    ) -> Dict:
        """Fingerprint an image file and save the result."""
        image = load_image(input_path)

        if image_id is None:
            image_id = image_id_from_path(input_path)

        fingerprinted, metadata = self.fingerprint_image(image, image_id)
        save_image(fingerprinted, output_path)

        metrics = compute_all_metrics(image, fingerprinted, include_lpips=False)
        metadata["metrics"] = metrics

        return metadata

    def verify_image(
        self,
        original: np.ndarray,
        fingerprinted: np.ndarray,
        image_id: str,
        alpha: float = 0.001,
    ) -> Dict:
        """Verify fingerprint on image arrays.

        Args:
            original: Original image, (C, H, W).
            fingerprinted: Fingerprinted (possibly transformed) image.
            image_id: Image identifier.
            alpha: Significance level.

        Returns:
            Verification result dict.
        """
        if self.multi_domain:
            return verify_multi_domain(
                original, fingerprinted,
                self.master_key, image_id,
                configs=self.fragment_configs,
                epsilon=self.epsilon,
                alpha=alpha,
            )
        else:
            embed_kwargs = self._get_embed_kwargs()
            residual = extract(original, fingerprinted,
                             strategy=self.strategy, **embed_kwargs)
            return verify_fingerprint(
                residual, self.master_key, image_id,
                num_fragments=self.num_fragments,
                epsilon=self.epsilon, alpha=alpha,
            )

    def verify_file(
        self,
        original_path: str,
        fingerprinted_path: str,
        image_id: Optional[str] = None,
        alpha: float = 0.001,
    ) -> Dict:
        """Verify fingerprint by comparing original and fingerprinted files."""
        original = load_image(original_path)
        fingerprinted = load_image(fingerprinted_path)

        if image_id is None:
            image_id = image_id_from_path(original_path)

        return self.verify_image(original, fingerprinted, image_id, alpha)

    def save_key(self, path: str) -> None:
        """Save the master key to a file."""
        save_master_key(self.master_key, path)

    @classmethod
    def from_key_file(cls, key_path: str, **kwargs) -> "FingerprintPipeline":
        """Create pipeline from a saved key file."""
        key = load_master_key(key_path)
        return cls(master_key=key, **kwargs)
