"""Test low-light image effects and adjustments."""

import pytest
import numpy as np
from lowlight_synthesis.degradation.illumination import (
    adjust_linear_saturation,
    apply_s_curve_contrast,
)


class TestAdjustLinearSaturation:
    """Tests for saturation adjustment while preserving luminance."""

    def test_output_shape(self, small_rgb_image):
        """Output should have same shape as input."""
        result = adjust_linear_saturation(small_rgb_image, saturation_factor=1.2)
        assert result.shape == small_rgb_image.shape

    def test_identity_at_factor_one(self, mid_image):
        """Saturation factor=1 should return unchanged image."""
        result = adjust_linear_saturation(mid_image, saturation_factor=1.0)
        # Should be nearly identical (numerical precision)
        assert np.allclose(result, mid_image, atol=1e-5)

    def test_luminance_preserved(self, small_rgb_image):
        """Luminance should be preserved at any saturation factor."""
        for factor in [0.5, 0.8, 1.0]:
            result = adjust_linear_saturation(small_rgb_image, saturation_factor=factor)

            # Compute luminance using ITU-R BT.709 weights
            lum_orig = (
                0.2126 * small_rgb_image[..., 0]
                + 0.7152 * small_rgb_image[..., 1]
                + 0.0722 * small_rgb_image[..., 2]
            )
            lum_result = (
                0.2126 * result[..., 0]
                + 0.7152 * result[..., 1]
                + 0.0722 * result[..., 2]
            )

            # Luminance should match
            assert np.allclose(lum_orig, lum_result, atol=1e-4)

    def test_increases_saturation_above_one(self, small_rgb_image):
        """factor > 1 should increase color intensity."""
        result = adjust_linear_saturation(small_rgb_image, saturation_factor=1.5)

        # Saturation increases chrominance while keeping luminance
        # Colored pixels should be more intense (distance from gray)
        orig_variance = np.var(small_rgb_image, axis=-1)
        result_variance = np.var(result, axis=-1)

        # Average variance should increase (more saturation = more color spread)
        assert np.mean(result_variance) > np.mean(orig_variance)

    def test_decreases_saturation_below_one(self, small_rgb_image):
        """factor < 1 should decrease color intensity (toward gray)."""
        result = adjust_linear_saturation(small_rgb_image, saturation_factor=0.5)

        # Desaturated image should have less color variance
        orig_variance = np.var(small_rgb_image, axis=-1)
        result_variance = np.var(result, axis=-1)

        # Average variance should decrease
        assert np.mean(result_variance) < np.mean(orig_variance)


class TestApplySCurveContrast:
    """Tests for S-curve contrast tone mapping."""

    def test_output_shape(self, small_rgb_image):
        """Output should have same shape as input."""
        result = apply_s_curve_contrast(small_rgb_image, strength=0.5)
        assert result.shape == small_rgb_image.shape

    def test_output_in_valid_range(self, small_rgb_image):
        """Output should be in [0, 1] range."""
        for strength in [0, 0.5, 1.0]:
            result = apply_s_curve_contrast(small_rgb_image, strength=strength)
            assert np.all(result >= 0) and np.all(result <= 1)

    def test_identity_at_zero_strength(self, mid_image):
        """strength=0 should return unchanged image."""
        result = apply_s_curve_contrast(mid_image, strength=0.0)
        assert np.allclose(result, mid_image)

    def test_midpoint_approximately_invariant(self, mid_image):
        """S-curve should keep x=0.5 close to 0.5."""
        result = apply_s_curve_contrast(mid_image, strength=1.0)
        # S-curve passes through (0.5, 0.5) as fixed point
        # Result should be near 0.5
        assert np.all(np.abs(result - 0.5) < 0.1)

    def test_contrast_increases_at_strength_one(self):
        """At strength=1, contrast should increase."""
        # Create a gradient from 0 to 1
        gradient = np.linspace(0, 1, 64)
        img = np.tile(gradient[np.newaxis, :, np.newaxis], (1, 1, 3))

        result = apply_s_curve_contrast(img, strength=1.0)

        # S-curve amplifies low and high values away from midpoint
        # Blacks should get darker, whites should get lighter
        assert result[0, 5, 0] < gradient[5]  # Darks darker
        assert result[0, 55, 0] > gradient[55]  # Lights lighter

    def test_strength_parameter_effect(self, small_rgb_image):
        """Stronger strength should have more contrast effect."""
        result_weak = apply_s_curve_contrast(small_rgb_image, strength=0.2)
        result_strong = apply_s_curve_contrast(small_rgb_image, strength=1.0)

        # Strong version should deviate more from identity
        diff_weak = np.mean(np.abs(result_weak - small_rgb_image))
        diff_strong = np.mean(np.abs(result_strong - small_rgb_image))

        assert diff_strong > diff_weak
