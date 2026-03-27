# coding=utf-8
# Copyright 2025 Tanishk Gopalani.
# Based on "Unprocessing Images for Learned Raw Denoising" by Timothy Brooks.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unprocessing: converting from sRGB to realistic raw sensor data.

This module implements the reverse of the standard image signal processing (ISP)
pipeline. It converts sRGB images back to realistic raw Bayer sensor data with:
- Gamma expansion (converting from gamma-compressed to linear space)
- White balance reversal
- Color correction matrix inversion (RGB to camera space)
- Bayer mosaicing
- Realistic noise model (shot + read noise)

Reference:
    "Unprocessing Images for Learned Raw Denoising"
    Timothy Brooks, Kelvin Xu, Zhoutong Zhang
    http://timothybrooks.com/tech/unprocessing
"""

import tensorflow.compat.v1 as tf

def get_ccm():
  """Generates a neutral identity color correction matrix."""
  # An identity matrix means no color shift is applied during this step.
  rgb2cam = tf.eye(3)
  return rgb2cam

def get_gains():
  """Generates fixed gains for a consistent warm tint."""
  # A higher blue_gain in the inverse step means the blue channel is suppressed,
  # resulting in a warmer (yellowish) final image.
  # Set all to 1.0 for a perfectly neutral starting point.
  rgb_gain, red_gain, blue_gain = (1.0, 1.0, 1.0)
  return rgb_gain, red_gain, blue_gain

def inverse_smoothstep(image):
  """Approximately inverts a global tone mapping curve."""
  image = tf.clip_by_value(image, 0.0, 1.0)
  return 0.5 - tf.sin(tf.asin(1.0 - 2.0 * image) / 3.0)

def gamma_expansion(image):
  """Converts from gamma to linear space."""
  return tf.maximum(image, 1e-8) ** 2.2

def apply_ccm(image, ccm):
  """Applies a color correction matrix."""
  shape = tf.shape(image)
  image = tf.reshape(image, [-1, 3])
  image = tf.tensordot(image, ccm, axes=[[-1], [-1]])
  return tf.reshape(image, shape)

def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
  """Inverts gains while safely handling saturated pixels."""
  gains = tf.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
  gains = gains[tf.newaxis, tf.newaxis, :]
  gray = tf.reduce_mean(image, axis=-1, keepdims=True)
  inflection = 0.9
  mask = (tf.maximum(gray - inflection, 0.0) / (1.0 - inflection)) ** 2.0
  safe_gains = tf.maximum(mask + (1.0 - mask) * gains, gains)
  return image * safe_gains

def mosaic(image):
    """Extracts RGGB Bayer planes from an RGB image."""
    image.shape.assert_has_rank(3)
    image.shape.assert_is_compatible_with((None, None, 3))

    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    # Ensure even dimensions
    even_height = height - tf.math.floormod(height, 2)
    even_width = width - tf.math.floormod(width, 2)

    image = image[:even_height, :even_width, :]

    # Extract Bayer pattern planes
    red         = image[0::2, 0::2, 0]
    green_red   = image[0::2, 1::2, 1]
    green_blue  = image[1::2, 0::2, 1]
    blue        = image[1::2, 1::2, 2]

    # Stack into a 4-channel Bayer mosaic
    mosaic_image = tf.stack((red, green_red, green_blue, blue), axis=-1)
    return mosaic_image

def unprocess(image):
  """Unprocesses an image from sRGB to realistic raw data."""
  with tf.name_scope(None, 'unprocess'):
    image.shape.assert_is_compatible_with([None, None, 3])

    # Uses the new fixed functions instead of random ones
    rgb2cam = get_ccm()
    cam2rgb = tf.matrix_inverse(rgb2cam)
    rgb_gain, red_gain, blue_gain = get_gains()

    image = inverse_smoothstep(image)
    image = gamma_expansion(image)
    image = apply_ccm(image, rgb2cam)
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = mosaic(image)

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }
    return image, metadata

def get_noise_levels(noise_level):
  """Generates fixed, scalable noise levels."""
  shot_noise = 0.01
  # shot_noise = 0.001
  read_noise = 0.0005
  return shot_noise * noise_level, read_noise * noise_level

def add_noise(image, shot_noise=0.01, read_noise=0.0005):
  """Adds random shot (proportional to image) and read (independent) noise."""
  variance = image * shot_noise + read_noise
  stddev = tf.sqrt(variance)
  noise = tf.random_normal(tf.shape(image))
  return image + noise * stddev