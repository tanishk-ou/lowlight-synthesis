"""Test realistic sensor noise injection."""

import pytest
import numpy as np
from lowlight_synthesis.degradation.noise import add_noise, get_noise_levels


class TestGetNoiseLevels:
    """Tests for noise parameter scaling."""

    def test_noise_levels_positive(self):
        """get_noise_levels() should return positive values."""
        shot, read = get_noise_levels(noise_level=0.5)
        assert shot > 0
        assert read > 0

    def test_noise_levels_scale_with_input(self):
        """Higher noise_level should give proportionally higher noise."""
        shot_low, read_low = get_noise_levels(noise_level=0.1)
        shot_high, read_high = get_noise_levels(noise_level=1.0)

        assert shot_high > shot_low
        assert read_high > read_low

    def test_noise_levels_zero_valid(self):
        """noise_level=0 should give zero noise."""
        shot, read = get_noise_levels(noise_level=0.0)
        assert shot == 0
        assert read == 0


class TestAddNoise:
    """Tests for noise addition to images."""

    def test_noise_preserves_shape(self, small_rgb_image):
        """add_noise() should preserve image shape."""
        shot, read = get_noise_levels(0.5)
        noisy = add_noise(small_rgb_image, shot, read)
        assert noisy.shape == small_rgb_image.shape

    def test_noise_preserves_dtype(self, small_rgb_image):
        """Output should match input dtype."""
        shot, read = get_noise_levels(0.5)
        noisy = add_noise(small_rgb_image, shot, read)
        assert noisy.dtype == small_rgb_image.dtype

    def test_noise_adds_variance(self, small_rgb_image):
        """Noisy image should differ from original."""
        shot, read = get_noise_levels(0.5)
        noisy = add_noise(small_rgb_image, shot, read)
        # Should not be identical (very low probability)
        assert not np.allclose(small_rgb_image, noisy)

    def test_noise_scales_with_level(self, bright_image):
        """Higher noise_level should produce more variance."""
        shot_low, read_low = get_noise_levels(0.1)
        shot_high, read_high = get_noise_levels(0.8)
        noisy_low = add_noise(bright_image, shot_low, read_low)
        noisy_high = add_noise(bright_image, shot_high, read_high)

        # Compute variance of noise (difference from input)
        var_low = np.var(noisy_low - bright_image)
        var_high = np.var(noisy_high - bright_image)

        # Higher noise level should create more variance
        assert var_high > var_low * 2  # At least 2x more variance

    def test_noise_respects_bounds(self, bright_image):
        """Noisy output should stay in valid range."""
        shot, read = get_noise_levels(0.3)
        noisy = add_noise(bright_image, shot, read)
        # Should stay mostly in [0, 1] (may clip at edges)
        assert np.percentile(noisy, 1) >= 0
        assert np.percentile(noisy, 99) <= 1

    def test_zero_noise_unchanged(self, mid_image):
        """noise_level=0 should return unchanged image."""
        shot, read = get_noise_levels(0.0)
        noisy = add_noise(mid_image, shot, read)
        assert np.allclose(noisy, mid_image)

    def test_noise_brightness_dependent(self):
        """Bright pixels should get more shot noise than dark pixels."""
        # Create two images with same dimension
        dark_patch = np.ones((1, 64, 64, 3), dtype=np.float32) * 0.1
        bright_patch = np.ones((1, 64, 64, 3), dtype=np.float32) * 0.9

        # Add noise multiple times and check variance
        shot, read = get_noise_levels(0.5)
        dark_noised = np.stack([add_noise(dark_patch, shot, read) for _ in range(10)])
        bright_noised = np.stack([add_noise(bright_patch, shot, read) for _ in range(10)])

        dark_var = np.var(dark_noised)
        bright_var = np.var(bright_noised)

        # Bright pixels should have more variance (shot noise is brightness-dependent)
        assert bright_var > dark_var
