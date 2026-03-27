"""Test ISP (Image Signal Processing) pipeline: process and unprocess."""

import pytest
import numpy as np
import tensorflow.compat.v1 as tf

from lowlight_synthesis.core.unprocess import unprocess, get_noise_levels
from lowlight_synthesis.core.process_new import process_to_linear_rgb, apply_gamma_compression


class TestUnprocess:
    """Tests for converting sRGB to raw Bayer sensor data."""

    def test_unprocess_returns_tuple(self, small_rgb_image):
        """unprocess() should return (bayer, metadata) tuple."""
        image_tensor = tf.constant(small_rgb_image)
        result = unprocess(image_tensor)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_unprocess_metadata_has_keys(self, small_rgb_image):
        """Metadata dict should contain required ISP parameters."""
        image_tensor = tf.constant(small_rgb_image)
        _, metadata = unprocess(image_tensor)
        required_keys = {"cam2rgb", "red_gain", "blue_gain"}
        assert required_keys.issubset(metadata.keys())

    def test_unprocess_bayer_smaller_than_rgb(self, small_rgb_image):
        """Bayer output should be subsampled from RGB."""
        image_tensor = tf.constant(small_rgb_image)
        bayer, _ = unprocess(image_tensor)
        # Bayer is the RGGB pattern, subsampled
        assert bayer.ndim == 3  # height, width, channels
        # Height should be reduced (subsampling)
        assert bayer.shape[1] < small_rgb_image.shape[1]

    def test_unprocess_output_dtype(self, small_rgb_image):
        """Bayer should be float type."""
        image_tensor = tf.constant(small_rgb_image)
        bayer, _ = unprocess(image_tensor)
        assert bayer.dtype in [np.float32, np.float64, tf.float32]

    def test_unprocess_metadata_valid_gains(self, small_rgb_image):
        """Gain values should be positive."""
        image_tensor = tf.constant(small_rgb_image)
        _, metadata = unprocess(image_tensor)
        assert metadata["red_gain"] > 0
        assert metadata["blue_gain"] > 0


class TestProcessToLinearRGB:
    """Tests for converting raw Bayer to linear RGB."""

    def test_output_shape_matches_bayer_resolution(self):
        """process_to_linear_rgb() should output appropriate resolution."""
        # Create 128x128 RGGB Bayer pattern
        img_srgb = np.random.rand(128, 128, 3).astype(np.float32)
        bayer, metadata = unprocess(tf.constant(img_srgb))

        # Process back to linear RGB
        with tf.device('/CPU:0'):
            linear_rgb_batched = process_to_linear_rgb(
                tf.expand_dims(bayer, axis=0),
                tf.reshape(tf.constant(metadata["red_gain"], dtype=tf.float32), [1]),
                tf.reshape(tf.constant(metadata["blue_gain"], dtype=tf.float32), [1]),
                tf.reshape(tf.constant(metadata["cam2rgb"], dtype=tf.float32), [1, 3, 3])
            )
            linear_rgb = tf.squeeze(linear_rgb_batched, axis=0)
            linear_rgb_np = linear_rgb.numpy() if hasattr(linear_rgb, 'numpy') else linear_rgb

        # Output should be RGB (3 channels)
        assert linear_rgb_np.ndim == 3
        assert linear_rgb_np.shape[2] == 3

    def test_output_in_valid_range(self):
        """Linear RGB should be in approximately [0, 1] range."""
        img_srgb = np.random.rand(64, 64, 3).astype(np.float32)
        bayer, metadata = unprocess(tf.constant(img_srgb))

        with tf.device('/CPU:0'):
            linear_rgb_batched = process_to_linear_rgb(
                tf.expand_dims(bayer, axis=0),
                tf.reshape(tf.constant(metadata["red_gain"], dtype=tf.float32), [1]),
                tf.reshape(tf.constant(metadata["blue_gain"], dtype=tf.float32), [1]),
                tf.reshape(tf.constant(metadata["cam2rgb"], dtype=tf.float32), [1, 3, 3])
            )
            linear_rgb = tf.squeeze(linear_rgb_batched, axis=0)
            linear_rgb_np = linear_rgb.numpy() if hasattr(linear_rgb, 'numpy') else linear_rgb

        # Should be mostly in [0, 1], allowing for numerical overshoot
        assert np.percentile(linear_rgb_np, 1) >= -0.1
        assert np.percentile(linear_rgb_np, 99) <= 1.2


class TestApplyGammaCompression:
    """Tests for gamma compression (linear to sRGB)."""

    def test_gamma_compression_output_range(self):
        """apply_gamma_compression() output should be in [0, 1]."""
        linear_rgb = np.random.rand(64, 64, 3).astype(np.float32)

        with tf.device('/CPU:0'):
            srgb = apply_gamma_compression(tf.constant(linear_rgb))
            srgb_np = srgb.numpy() if hasattr(srgb, 'numpy') else srgb

        assert np.all(srgb_np >= 0) and np.all(srgb_np <= 1)

    def test_gamma_compression_brightens_linear(self):
        """Gamma compression should brighten linear RGB."""
        # Gamma = 2.2 inverse (1/2.2 ≈ 0.4545)
        # So linear value raised to 1/gamma should be brighter
        linear_rgb = np.ones((32, 32, 3), dtype=np.float32) * 0.5

        with tf.device('/CPU:0'):
            srgb = apply_gamma_compression(tf.constant(linear_rgb))
            srgb_np = srgb.numpy() if hasattr(srgb, 'numpy') else srgb

        # 0.5^(1/2.2) ≈ 0.735 > 0.5
        assert np.all(srgb_np > linear_rgb)

    def test_gamma_compression_preserves_shape(self):
        """Output shape should match input shape."""
        linear_rgb = np.random.rand(1, 64, 64, 3).astype(np.float32)

        with tf.device('/CPU:0'):
            srgb = apply_gamma_compression(tf.constant(linear_rgb))
            srgb_np = srgb.numpy() if hasattr(srgb, 'numpy') else srgb

        assert srgb_np.shape == linear_rgb.shape


class TestRoundtrip:
    """Test unprocess -> process -> unprocess cycle."""

    def test_unprocess_process_roundtrip_preserves_approximate_color(self):
        """Unprocess->Process should approximately preserve color."""
        original_srgb = np.random.rand(128, 128, 3).astype(np.float32) * 0.8 + 0.1

        # Unprocess to Bayer
        bayer, metadata = unprocess(tf.constant(original_srgb))

        # Process back to linear RGB
        with tf.device('/CPU:0'):
            linear_rgb_batched = process_to_linear_rgb(
                tf.expand_dims(bayer, axis=0),
                tf.reshape(tf.constant(metadata["red_gain"], dtype=tf.float32), [1]),
                tf.reshape(tf.constant(metadata["blue_gain"], dtype=tf.float32), [1]),
                tf.reshape(tf.constant(metadata["cam2rgb"], dtype=tf.float32), [1, 3, 3])
            )
            linear_rgb = tf.squeeze(linear_rgb_batched, axis=0)
            linear_rgb_np = linear_rgb.numpy() if hasattr(linear_rgb, 'numpy') else linear_rgb

        # Compress back to sRGB
        with tf.device('/CPU:0'):
            recovered_srgb = apply_gamma_compression(tf.constant(linear_rgb_np))
            recovered_srgb_np = recovered_srgb.numpy() if hasattr(recovered_srgb, 'numpy') else recovered_srgb

        # Should be approximately similar (ISP has some color shift)
        # But correlation should be high
        correlation = np.corrcoef(original_srgb.flatten(), recovered_srgb_np.flatten())[0, 1]
        assert correlation > 0.7  # At least 70% correlated
