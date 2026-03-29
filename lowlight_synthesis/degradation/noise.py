"""Realistic sensor noise simulation."""

from typing import Tuple
import tensorflow.compat.v1 as tf


def get_noise_levels(noise_level: float) -> Tuple[float, float]:
    """Calculate shot and read noise variances scaled by a given multiplier.

    Args:
        noise_level: Scaling factor for noise (typical range 0.00005-0.005)

    Returns:
        Tuple of (shot_noise_factor, read_noise_factor)
    """
    # Base shot and read noise values (for single noiseLevel=1.0)
    shot_noise_base = 0.01
    read_noise_base = 0.0005

    return shot_noise_base * noise_level, read_noise_base * noise_level


def add_noise(
    image: tf.Tensor,
    shot_noise: float = 0.01,
    read_noise: float = 0.0005,
) -> tf.Tensor:
    """Inject signal-dependent shot noise and signal-independent read noise into an image tensor.

    Args:
        image: Input image (typically Bayer pattern in raw space)
        shot_noise: Shot noise coefficient (proportional to brightness)
        read_noise: Read noise value (independent noise floor)

    Returns:
        Noisy image with same shape as input
    """
    # Calculate variance at each pixel
    # Variance = signal-dependent component + signal-independent component
    variance = image * shot_noise + read_noise

    # Standard deviation is sqrt of variance
    stddev = tf.sqrt(variance)

    # Generate Gaussian noise with unit variance
    noise_sample = tf.random_normal(tf.shape(image))

    # Scale by per-pixel standard deviation
    return image + noise_sample * stddev
