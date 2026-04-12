#!/usr/bin/env python3
"""Tests for the fingerprint generation pipeline."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.keygen import (
    generate_master_key,
    derive_fragment_subkey,
    derive_all_subkeys,
    save_master_key,
    load_master_key,
)
from src.fragment import generate_fragment, generate_all_fragments
from src.embed import embed, extract
from src.verify import verify_fingerprint, pearson_correlation, fisher_combine_pvalues
from src.metrics import psnr, ssim
from src.pipeline import FingerprintPipeline, save_image, load_image


class TestKeyGen(unittest.TestCase):
    def test_master_key_length(self):
        key = generate_master_key(32)
        self.assertEqual(len(key), 32)

    def test_master_key_randomness(self):
        k1 = generate_master_key()
        k2 = generate_master_key()
        self.assertNotEqual(k1, k2)

    def test_subkey_determinism(self):
        key = generate_master_key()
        s1 = derive_fragment_subkey(key, "img001", 0)
        s2 = derive_fragment_subkey(key, "img001", 0)
        self.assertEqual(s1, s2)

    def test_subkey_uniqueness_per_image(self):
        key = generate_master_key()
        s1 = derive_fragment_subkey(key, "img001", 0)
        s2 = derive_fragment_subkey(key, "img002", 0)
        self.assertNotEqual(s1, s2)

    def test_subkey_uniqueness_per_fragment(self):
        key = generate_master_key()
        s1 = derive_fragment_subkey(key, "img001", 0)
        s2 = derive_fragment_subkey(key, "img001", 1)
        self.assertNotEqual(s1, s2)

    def test_save_load_key(self):
        key = generate_master_key()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".key", delete=False) as f:
            path = f.name
        try:
            save_master_key(key, path)
            loaded = load_master_key(path)
            self.assertEqual(key, loaded)
        finally:
            os.unlink(path)


class TestFragmentGeneration(unittest.TestCase):
    def setUp(self):
        self.key = generate_master_key()
        self.subkey = derive_fragment_subkey(self.key, "test", 0)
        self.shape = (3, 64, 64)

    def test_fragment_shape(self):
        fragment = generate_fragment(self.subkey, self.shape)
        self.assertEqual(fragment.shape, self.shape)

    def test_fragment_determinism(self):
        f1 = generate_fragment(self.subkey, self.shape)
        f2 = generate_fragment(self.subkey, self.shape)
        np.testing.assert_array_equal(f1, f2)

    def test_fragment_gaussian_distribution(self):
        fragment = generate_fragment(self.subkey, (3, 256, 256), epsilon=1.0)
        # Should be approximately unit variance after normalization * epsilon=1.0
        self.assertAlmostEqual(float(np.std(fragment)), 1.0, places=1)

    def test_fragment_epsilon_scaling(self):
        eps = 4.0 / 255.0
        fragment = generate_fragment(self.subkey, self.shape, epsilon=eps)
        self.assertAlmostEqual(float(np.std(fragment)), eps, places=3)

    def test_all_fragments(self):
        subkeys = derive_all_subkeys(self.key, "test", 4)
        fragments, aggregated = generate_all_fragments(subkeys, self.shape)
        self.assertEqual(len(fragments), 4)
        self.assertEqual(aggregated.shape, self.shape)

    def test_fragments_independence(self):
        subkeys = derive_all_subkeys(self.key, "test", 4)
        fragments, _ = generate_all_fragments(subkeys, self.shape)
        # Fragments should have low correlation with each other
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                r, _ = pearson_correlation(fragments[i], fragments[j])
                self.assertLess(abs(r), 0.1, f"Fragments {i} and {j} correlated: r={r}")


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.key = generate_master_key()
        self.image = np.random.rand(3, 64, 64).astype(np.float32)
        subkeys = derive_all_subkeys(self.key, "test", 4)
        _, self.perturbation = generate_all_fragments(subkeys, self.image.shape)

    def test_pixel_embed_shape(self):
        fp = embed(self.image, self.perturbation, strategy="pixel")
        self.assertEqual(fp.shape, self.image.shape)

    def test_pixel_embed_range(self):
        fp = embed(self.image, self.perturbation, strategy="pixel")
        self.assertTrue(np.all(fp >= 0) and np.all(fp <= 1))

    def test_pixel_roundtrip(self):
        fp = embed(self.image, self.perturbation, strategy="pixel")
        residual = extract(self.image, fp, strategy="pixel")
        # Should recover perturbation (with clipping artifacts)
        r, _ = pearson_correlation(residual, self.perturbation)
        self.assertGreater(r, 0.9)

    def test_dct_embed(self):
        # Need image divisible by block_size
        image = np.random.rand(3, 64, 64).astype(np.float32)
        subkeys = derive_all_subkeys(self.key, "test", 4)
        _, pert = generate_all_fragments(subkeys, image.shape)
        fp = embed(image, pert, strategy="dct")
        self.assertEqual(fp.shape, image.shape)

    def test_dwt_dct_embed(self):
        image = np.random.rand(3, 64, 64).astype(np.float32)
        subkeys = derive_all_subkeys(self.key, "test", 4)
        _, pert = generate_all_fragments(subkeys, image.shape)
        fp = embed(image, pert, strategy="dwt-dct")
        self.assertEqual(fp.shape, image.shape)


class TestVerification(unittest.TestCase):
    def test_correct_key_detects(self):
        key = generate_master_key()
        image = np.random.rand(3, 64, 64).astype(np.float32)
        image_id = "test123"

        subkeys = derive_all_subkeys(key, image_id, 8)
        _, pert = generate_all_fragments(subkeys, image.shape)
        fp = embed(image, pert, strategy="pixel")
        residual = extract(image, fp, strategy="pixel")

        result = verify_fingerprint(residual, key, image_id, num_fragments=8)
        self.assertTrue(result["detected"])
        self.assertLess(result["combined_p_value"], 0.001)

    def test_wrong_key_fails(self):
        key1 = generate_master_key()
        key2 = generate_master_key()
        image = np.random.rand(3, 64, 64).astype(np.float32)
        image_id = "test123"

        subkeys = derive_all_subkeys(key1, image_id, 8)
        _, pert = generate_all_fragments(subkeys, image.shape)
        fp = embed(image, pert, strategy="pixel")
        residual = extract(image, fp, strategy="pixel")

        result = verify_fingerprint(residual, key2, image_id, num_fragments=8)
        self.assertFalse(result["detected"])

    def test_wrong_image_id_fails(self):
        key = generate_master_key()
        image = np.random.rand(3, 64, 64).astype(np.float32)

        subkeys = derive_all_subkeys(key, "correct_id", 8)
        _, pert = generate_all_fragments(subkeys, image.shape)
        fp = embed(image, pert, strategy="pixel")
        residual = extract(image, fp, strategy="pixel")

        result = verify_fingerprint(residual, key, "wrong_id", num_fragments=8)
        self.assertFalse(result["detected"])

    def test_fisher_combine(self):
        # Very small p-values should combine to even smaller
        pvals = [0.01] * 8
        chi2, combined_p = fisher_combine_pvalues(pvals)
        self.assertLess(combined_p, 0.001)


class TestMetrics(unittest.TestCase):
    def test_identical_images(self):
        img = np.random.rand(3, 64, 64).astype(np.float32)
        self.assertEqual(psnr(img, img), float("inf"))
        self.assertAlmostEqual(ssim(img, img), 1.0, places=3)

    def test_quality_above_threshold(self):
        key = generate_master_key()
        image = np.random.rand(3, 64, 64).astype(np.float32)
        subkeys = derive_all_subkeys(key, "test", 8)
        _, pert = generate_all_fragments(subkeys, image.shape, epsilon=4/255)
        fp = embed(image, pert, strategy="pixel")

        p = psnr(image, fp)
        s = ssim(image, fp)
        self.assertGreater(p, 35)  # relaxed for small random images
        self.assertGreater(s, 0.9)


class TestPipeline(unittest.TestCase):
    def test_end_to_end(self):
        pipeline = FingerprintPipeline()
        image = np.random.rand(3, 64, 64).astype(np.float32)

        fp, metadata = pipeline.fingerprint_image(image, "test_e2e")
        self.assertEqual(fp.shape, image.shape)
        self.assertIn("image_id", metadata)

    def test_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image
            image = np.random.rand(3, 64, 64).astype(np.float32)
            orig_path = os.path.join(tmpdir, "original.png")
            fp_path = os.path.join(tmpdir, "fingerprinted.png")
            save_image(image, orig_path)

            pipeline = FingerprintPipeline()
            metadata = pipeline.fingerprint_file(orig_path, fp_path)

            self.assertTrue(os.path.exists(fp_path))
            self.assertIn("metrics", metadata)
            self.assertGreater(metadata["metrics"]["psnr_db"], 15)

            # Verify
            result = pipeline.verify_file(orig_path, fp_path)
            self.assertTrue(result["detected"])


class TestMultiDomain(unittest.TestCase):
    """Tests for multi-domain fragment embedding."""

    def test_multi_domain_fingerprint_and_verify(self):
        """Multi-domain embed + verify should detect with correct key."""
        pipeline = FingerprintPipeline(multi_domain=True)
        image = np.random.rand(3, 64, 64).astype(np.float32)

        fp, metadata = pipeline.fingerprint_image(image, "test_md")
        self.assertEqual(fp.shape, image.shape)
        self.assertEqual(metadata["mode"], "multi-domain")
        self.assertIn("fragment_configs", metadata)

        result = pipeline.verify_image(image, fp, "test_md")
        self.assertTrue(result["detected"])

    def test_multi_domain_wrong_key_fails(self):
        """Multi-domain verify with wrong key should fail."""
        key1 = generate_master_key()
        key2 = generate_master_key()
        image = np.random.rand(3, 64, 64).astype(np.float32)

        p1 = FingerprintPipeline(master_key=key1, multi_domain=True)
        fp, _ = p1.fingerprint_image(image, "test_wk")

        p2 = FingerprintPipeline(master_key=key2, multi_domain=True)
        result = p2.verify_image(image, fp, "test_wk")
        self.assertFalse(result["detected"])

    def test_multi_domain_per_fragment_details(self):
        """Verify returns per-fragment strategy/region info."""
        pipeline = FingerprintPipeline(multi_domain=True)
        image = np.random.rand(3, 64, 64).astype(np.float32)

        fp, _ = pipeline.fingerprint_image(image, "test_det")
        result = pipeline.verify_image(image, fp, "test_det")

        details = result["per_fragment"]["details"]
        self.assertEqual(len(details), pipeline.num_fragments)
        # Check that multiple strategies are used
        strategies = set(d["strategy"] for d in details)
        self.assertGreater(len(strategies), 1)

    def test_multi_domain_fragment_diversity(self):
        """Different fragments should be in different domains."""
        from src.fragment_config import get_default_configs
        configs = get_default_configs(8)

        # Should have multiple strategy types
        strategies = set(c["strategy"] for c in configs)
        self.assertGreater(len(strategies), 1)

        # Pixel fragments should have different regions
        pixel_regions = [c.get("region") for c in configs if c["strategy"] == "pixel"]
        self.assertEqual(len(pixel_regions), len(set(pixel_regions)))

    def test_single_domain_backward_compat(self):
        """Single-domain mode should still work."""
        pipeline = FingerprintPipeline(multi_domain=False)
        image = np.random.rand(3, 64, 64).astype(np.float32)

        fp, metadata = pipeline.fingerprint_image(image, "test_sd")
        self.assertIn("single-domain", metadata["mode"])

        result = pipeline.verify_image(image, fp, "test_sd")
        self.assertTrue(result["detected"])

    def test_multi_domain_file_roundtrip(self):
        """File-based fingerprint + verify in multi-domain mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image = np.random.rand(3, 64, 64).astype(np.float32)
            orig_path = os.path.join(tmpdir, "original.png")
            fp_path = os.path.join(tmpdir, "fingerprinted.png")
            save_image(image, orig_path)

            pipeline = FingerprintPipeline(multi_domain=True)
            metadata = pipeline.fingerprint_file(orig_path, fp_path)

            self.assertTrue(os.path.exists(fp_path))
            self.assertGreater(metadata["metrics"]["psnr_db"], 15)

            result = pipeline.verify_file(orig_path, fp_path)
            self.assertTrue(result["detected"])


if __name__ == "__main__":
    unittest.main()
