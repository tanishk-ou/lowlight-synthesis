"""Main low-light synthesis pipeline.

Orchestrates the complete process of converting sRGB images into realistic
synthetic low-light images through sensor simulation and statistical degradation.

The pipeline:
1. Unprocess sRGB to raw Bayer with realistic noise
2. Denoise/smooth via bilateral filter
3. Decompose into illumination and detail layers
4. Apply probabilistic darkening
5. Re-process through ISP with contrast and saturation adjustments
6. Export as sRGB JPEG
"""

import os
import glob
from typing import Tuple
from datetime import datetime
import logging
import sys

import tensorflow.compat.v1 as tf
import numpy as np
import cv2
from tqdm import tqdm

# Import from local modules
from . import unprocess
from .process_new import process_to_linear_rgb, apply_gamma_compression
from lowlight_synthesis.degradation import sampling, noise
from lowlight_synthesis.degradation.illumination import (
    adjust_linear_saturation,
    apply_s_curve_contrast,
)

# Configure logging

logger = logging.getLogger(__name__)


def build_lowlight_graph(
    source_path_tensor: tf.Tensor,
    target_path_tensor: tf.Tensor,
    illumination: float,
    noise_level: float,
    contrast: float,
    saturation: float,
) -> None:
    """Build and execute the low-light synthesis TensorFlow graph.

    This function constructs a complete TensorFlow computation graph that
    transforms an sRGB image into a synthetic low-light version with
    realistic sensor degradation.

    The process:
    1. Load sRGB image and convert to [0, 1] range
    2. Unprocess to raw Bayer sensor data
    3. Add realistic shot and read noise
    4. Re-process through ISP (demosaicing, white balance, color correction)
    5. Estimate illumination via bilateral filter
    6. Decompose into illumination and detail layers
    7. Darken illumination by the specified factor
    8. Apply S-curve contrast enhancement
    9. Adjust saturation
    10. Gamma compress back to sRGB and save

    Args:
        source_path_tensor: TensorFlow path tensor to input image
        target_path_tensor: TensorFlow path tensor to output image
        illumination: Illumination darkening factor (0-1, smaller = darker)
        noise_level: Noise scaling factor (typical 0.00005-0.005)
        contrast: S-curve contrast strength (typical 0.5-1.0)
        saturation: Saturation boost factor (typical 1.0-1.5)

    Returns:
        None (writes output image to disk)
    """
    # Load and normalize sRGB image
    image_gt_bytes = tf.io.read_file(source_path_tensor)
    image_gt_tensor = tf.image.decode_image(image_gt_bytes, channels=3)
    image_gt_tensor = tf.cast(image_gt_tensor, tf.float32) / 255.0

    # Convert sRGB to raw Bayer
    raw_image, metadata = unprocess.unprocess(image_gt_tensor)

    # Add realistic noise
    shot_noise, read_noise = noise.get_noise_levels(noise_level)
    noisy_raw_image = noise.add_noise(raw_image, shot_noise, read_noise)

    # Prepare batch dimensions for ISP pipeline
    noisy_raw_image_batched = tf.expand_dims(noisy_raw_image, axis=0)
    red_gain_batched = tf.expand_dims(metadata["red_gain"], axis=0)
    blue_gain_batched = tf.expand_dims(metadata["blue_gain"], axis=0)
    cam2rgb_batched = tf.expand_dims(metadata["cam2rgb"], axis=0)

    # Re-process through ISP to get linear RGB
    linear_rgb_batched = process_to_linear_rgb(
        noisy_raw_image_batched,
        red_gain_batched,
        blue_gain_batched,
        cam2rgb_batched,
    )
    linear_rgb = tf.squeeze(linear_rgb_batched, axis=0)

    # Estimate base illumination via bilateral filter (detail preservation)
    def py_bilateral_filter(image_np):
        return cv2.bilateralFilter(
            image_np.numpy().astype(np.float32),
            d=31,
            sigmaColor=0.1,
            sigmaSpace=8.0,
        )

    base_illumination = tf.py_function(
        py_bilateral_filter, inp=[linear_rgb], Tout=tf.float32
    )
    base_illumination.set_shape(linear_rgb.shape)

    # Decompose into detail layer (high-frequency)
    detail_layer = linear_rgb / (base_illumination + 1e-8)

    # Apply illumination darkening
    dark_illumination = base_illumination * illumination

    # Recombine illumination and detail
    low_light_linear = dark_illumination * detail_layer

    # Apply contrast enhancement (S-curve)
    image_with_contrast = apply_s_curve_contrast(low_light_linear, strength=contrast)

    # Adjust saturation
    final_linear_image = adjust_linear_saturation(image_with_contrast, saturation)

    # Gamma compress back to sRGB
    final_srgb_image = apply_gamma_compression(final_linear_image)

    # Prepare for JPEG encoding
    final_image_display = tf.cast(
        tf.clip_by_value(final_srgb_image, 0.0, 1.0) * 255.0, tf.uint8
    )
    encoded_image = tf.image.encode_jpeg(final_image_display)

    # Write to disk
    return tf.io.write_file(target_path_tensor, encoded_image)


if __name__ == "__main__":
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = "generation.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(_log_file, mode="w"),
        ],
    )

    # Default paths (override with command-line arguments in production)
    SOURCE_DIR = "../Lowlightdataset/coco_original"
    TARGET_DIR = "../Lowlightdataset/coco_final_dark"

    os.makedirs(TARGET_DIR, exist_ok=True)
    logger.info(f"SOURCE_DIR: {SOURCE_DIR}")

    # Find all source images
    source_images = glob.glob(os.path.join(SOURCE_DIR, "*.jpg")) + glob.glob(
        os.path.join(SOURCE_DIR, "*.png")
    )

    logger.info(f"Found {len(source_images)} images to process in {SOURCE_DIR}.")

    # Process each image with random degradation parameters
    for source_path in tqdm(
        source_images, desc="Generating low-light images"
    ):
        # Sample random degradation parameters
        contrast = sampling.weighted_random_linear(
            min_val=0.5, max_val=1.0, main_lo=0.6, main_hi=0.8
        )
        saturation = sampling.weighted_random_linear(
            min_val=1.0, max_val=1.5, main_lo=1.1, main_hi=1.3
        )
        illumination = sampling.weighted_random_log(
            min_val=0.0001, max_val=0.001, main_hi=0.0003
        )
        noise_level = sampling.log_right_skewed_with_peak(
            min_val=0.00005, max_val=0.005, peak_val=0.0001
        )

        filename = os.path.basename(source_path)
        target_path = os.path.join(TARGET_DIR, filename)

        # Skip if already processed
        if os.path.exists(target_path):
            continue

        # Build and execute the synthesis graph
        build_lowlight_graph(
            tf.constant(source_path),
            tf.constant(target_path),
            illumination,
            noise_level,
            contrast,
            saturation,
        )

    logger.info("Data generation complete!")
