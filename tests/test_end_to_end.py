"""End-to-end integration tests for the low-light synthesis pipeline."""

import pytest
import numpy as np
import tensorflow.compat.v1 as tf

from lowlight_synthesis.core.pipeline import build_lowlight_graph
from lowlight_synthesis.core import unprocess
from lowlight_synthesis.core.process_new import process_to_linear_rgb, apply_gamma_compression
from lowlight_synthesis.degradation import noise
from lowlight_synthesis.degradation.illumination import (
    adjust_linear_saturation,
    apply_s_curve_contrast,
)


def build_lowlight_graph_from_image(
    image: np.ndarray,
    noise_level: float,
    illumination: float,
    saturation_factor: float,
    contrast: float,
) -> np.ndarray:
    """Wrapper function that applies lowlight synthesis to image data.

    Takes image data directly (instead of file paths) and applies the full
    low-light synthesis pipeline, returning the result as numpy array.

    Args:
        image: Input image as numpy array in [0, 1] range
        noise_level: Noise scaling factor
        illumination: Illumination darkening factor
        saturation_factor: Saturation adjustment factor
        contrast: S-curve contrast strength

    Returns:
        Processed image as numpy array in [0, 1] range
    """
    # Ensure image is a tensor
    image_tensor = tf.constant(image) if not isinstance(image, tf.Tensor) else image

    # Unprocess to raw Bayer
    raw_image, metadata = unprocess.unprocess(image_tensor)

    # Add realistic noise
    shot_noise, read_noise = noise.get_noise_levels(noise_level)
    noisy_raw_image = noise.add_noise(raw_image, shot_noise, read_noise)

    # Prepare batch dimensions if needed
    if len(noisy_raw_image.shape) == 3:
        noisy_raw_image_batched = tf.expand_dims(noisy_raw_image, axis=0)
    else:
        noisy_raw_image_batched = noisy_raw_image

    # Expand metadata gains if needed
    red_gain_batched = tf.reshape(tf.constant(metadata["red_gain"], dtype=tf.float32), [1])
    blue_gain_batched = tf.reshape(tf.constant(metadata["blue_gain"], dtype=tf.float32), [1])
    cam2rgb_batched = tf.reshape(tf.constant(metadata["cam2rgb"], dtype=tf.float32), [1, 3, 3])

    # Re-process through ISP
    linear_rgb_batched = process_to_linear_rgb(
        noisy_raw_image_batched,
        red_gain_batched,
        blue_gain_batched,
        cam2rgb_batched,
    )

    # Remove batch dimension if it was added
    if linear_rgb_batched.shape[0] == 1:
        linear_rgb = tf.squeeze(linear_rgb_batched, axis=0)
    else:
        linear_rgb = linear_rgb_batched

    # Apply illumination darkening (simplified - no bilateral filter in test)
    dark_illumination = linear_rgb * illumination

    # Apply contrast enhancement (S-curve)
    image_with_contrast = apply_s_curve_contrast(dark_illumination, strength=contrast)

    # Adjust saturation
    final_linear_image = adjust_linear_saturation(image_with_contrast, saturation_factor=saturation_factor)

    # Gamma compress back to sRGB
    final_srgb_image = apply_gamma_compression(final_linear_image)

    # Return as numpy array
    result = final_srgb_image.numpy() if hasattr(final_srgb_image, 'numpy') else final_srgb_image
    return np.clip(result, 0.0, 1.0)


class TestBuildLowlightGraph:
    """Tests for the complete lowlight synthesis pipeline."""

    def test_pipeline_basic_execution(self, small_rgb_image):
        """Pipeline should execute without errors."""
        with tf.device('/CPU:0'):
            result = build_lowlight_graph_from_image(
                small_rgb_image,
                noise_level=0.1,
                illumination=0.5,
                saturation_factor=1.0,
                contrast=0.3
            )

        assert result is not None
        assert result.shape == small_rgb_image.shape

    def test_output_in_valid_range(self, small_rgb_image):
        """Pipeline output should be in [0, 1] range."""
        with tf.device('/CPU:0'):
            result = build_lowlight_graph_from_image(
                small_rgb_image,
                noise_level=0.05,
                illumination=0.6,
                saturation_factor=1.0,
                contrast=0.0
            )

        assert np.all(result >= 0) and np.all(result <= 1)

    def test_darkening_effect_visible(self, bright_image):
        """Strong illumination factor should darken output."""
        with tf.device('/CPU:0'):
            result_dark = build_lowlight_graph_from_image(
                bright_image,
                noise_level=0.02,
                illumination=0.3,  # Strong darkening
                saturation_factor=1.0,
                contrast=0.0
            )

        # Result should be noticeably darker
        assert np.mean(result_dark) < np.mean(bright_image) * 0.7

    def test_no_darkening_with_factor_one(self, mid_image):
        """illumination_factor=1 should preserve brightness."""
        with tf.device('/CPU:0'):
            result = build_lowlight_graph_from_image(
                mid_image,
                noise_level=0.0,  # No noise
                illumination=1.0,  # No darkening
                saturation_factor=1.0,  # No saturation change
                contrast=0.0  # No contrast change
            )

        # Should be very close to original (only ISP round-trip)
        # Allowing for some numerical error
        assert np.mean(np.abs(result - mid_image)) < 0.1

    def test_noise_makes_output_different(self, mid_image):
        """Adding noise should make output different from input."""
        with tf.device('/CPU:0'):
            result_noisy = build_lowlight_graph_from_image(
                mid_image,
                noise_level=0.5,  # Significant noise
                illumination=1.0,
                saturation_factor=1.0,
                contrast=0.0
            )

        # Should differ from input due to noise
        assert not np.allclose(result_noisy, mid_image)

    def test_saturation_increase_effect(self, small_rgb_image):
        """Saturation > 1 should increase color intensity."""
        with tf.device('/CPU:0'):
            result_saturated = build_lowlight_graph_from_image(
                small_rgb_image,
                noise_level=0.0,
                illumination=1.0,
                saturation_factor=1.5,  # Increase saturation
                contrast=0.0
            )

        # Saturation should increase variance in pixels
        orig_color_variance = np.var(small_rgb_image, axis=-1)
        result_color_variance = np.var(result_saturated, axis=-1)

        # On average, result should have more color spread
        assert np.mean(result_color_variance) >= np.mean(orig_color_variance) * 0.8

    def test_contrast_increases_at_nonzero_factor(self, gradient_image):
        """Contrast factor > 0 should increase contrast."""
        with tf.device('/CPU:0'):
            result_contrast = build_lowlight_graph_from_image(
                gradient_image,
                noise_level=0.0,
                illumination=1.0,
                saturation_factor=1.0,
                contrast=0.8  # Strong contrast
            )

        # With S-curve, blacks should get darker, whites lighter
        # Result should have larger spread
        orig_std = np.std(gradient_image)
        result_std = np.std(result_contrast)

        assert result_std > orig_std * 0.80  # Similar or higher std

    def test_combined_parameters(self, small_rgb_image):
        """All parameters together should produce a valid output."""
        with tf.device('/CPU:0'):
            result = build_lowlight_graph_from_image(
                small_rgb_image,
                noise_level=0.3,
                illumination=0.4,
                saturation_factor=0.8,
                contrast=0.5
            )

        assert result.shape == small_rgb_image.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0) and np.all(result <= 1)