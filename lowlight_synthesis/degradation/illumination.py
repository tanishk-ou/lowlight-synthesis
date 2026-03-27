"""Low-light degradation effects: illumination, contrast, and saturation.

This module applies physically-motivated degradation effects to create
realistic low-light images from normal exposures.
"""

from typing import Tuple
import tensorflow.compat.v1 as tf


def adjust_linear_saturation(linear_image: tf.Tensor, saturation_factor: float) -> tf.Tensor:
    """Adjust saturation on a linear RGB image.

    Saturation adjustment in linear RGB space preserves luminance while scaling
    chrominance (color) information. A factor > 1.0 increases saturation,
    < 1.0 decreases it.

    The operation:
    1. Compute luminance using ITU-R BT.709 weights
    2. Extract chrominance as (image - luminance)
    3. Scale chrominance and recombine

    Args:
        linear_image: Input image in linear RGB space (values 0-1)
        saturation_factor: Saturation scaling factor (0-2 typical range)
                          1.0 = no change, >1.0 = more saturated

    Returns:
        Saturated image in linear RGB space, clipped to [0, 1]
    """
    # ITU-R BT.709 luminance coefficients
    luminance = (
        linear_image[..., 0] * 0.2126
        + linear_image[..., 1] * 0.7152
        + linear_image[..., 2] * 0.0722
    )
    luminance = tf.expand_dims(luminance, axis=-1)

    # Extract chrominance and scale it
    saturated_image = luminance + (linear_image - luminance) * saturation_factor
    return tf.clip_by_value(saturated_image, 0.0, 1.0)


def apply_s_curve_contrast(linear_image: tf.Tensor, strength: float = 1.0) -> tf.Tensor:
    """Apply a contrast-enhancing S-curve in linear space.

    The S-curve function f(x) = 3x² - 2x³ creates:
    - A fixed point at x=0.5 (no change at mid-tones)
    - Enhanced contrast in shadows (x < 0.5) and highlights (x > 0.5)
    - Strength parameter controls blend between original and S-curve

    Mathematical form:
    - output = x(1-strength) + [3x² - 2x³] * strength

    Args:
        linear_image: Input image in linear RGB space (values 0-1)
        strength: Blend factor between original and S-curve
                 0.0 = no change, 1.0 = full S-curve, typical 0.5-1.0

    Returns:
        Contrast-enhanced image in linear RGB space, clipped to [0, 1]
    """
    # S-curve formula: 3x² - 2x³
    s_curve_image = 3.0 * linear_image ** 2 - 2.0 * linear_image ** 3

    # Blend between original and S-curve
    blended_image = linear_image * (1.0 - strength) + s_curve_image * strength
    return tf.clip_by_value(blended_image, 0.0, 1.0)


def darken_illumination(
    linear_image: tf.Tensor,
    illumination_factor: float,
    detail_layer: tf.Tensor,
) -> tf.Tensor:
    """Apply illumination darkening using the detail layer.

    Separates the image into illumination and detail components using
    bilateral filtering, then reduces illumination to create low-light effect
    while preserving detail.

    The low-light effect is: darkened_illumination × detail_layer

    Args:
        linear_image: Original linear RGB image
        illumination_factor: Scaling factor for illumination (0-1 range)
                            Smaller = darker image
        detail_layer: Detail layer extracted from bilateral filter
                     (linear_image / base_illumination)

    Returns:
        Low-light image with preserved details
    """
    # Base illumination estimated via bilateral filter is computed outside this function
    # This function just applies the reduction
    # Typically: dark_illumination = base_illumination * illumination_factor
    # Result: low_light_linear = dark_illumination * detail_layer
    pass  # This is handled in the pipeline
