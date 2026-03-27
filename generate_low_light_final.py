import os
import glob
from tqdm import tqdm
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import logging
from datetime import datetime
import sys

# Import from lowlight_synthesis package
from lowlight_synthesis.core import unprocess
from lowlight_synthesis.core.process_new import process_to_linear_rgb, apply_gamma_compression
from lowlight_synthesis.degradation.sampling import (
    weighted_random_linear,
    weighted_random_log,
    log_right_skewed_with_peak
)
from lowlight_synthesis.degradation.illumination import (
    adjust_linear_saturation,
    apply_s_curve_contrast
)


# --- TUNABLE PARAMETERS ---
ILLUMINATION_DARKEN_FACTOR = (0.001, 0.0001)
NOISE_LEVEL = (0.005, 0.0005)
CONTRAST_STRENGTH = (0.5, 1.0)
SATURATION_BOOST = (1.0, 1.5)
# --- End Configuration ---

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = 'generation.log'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w')
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"ILLUMINATION_DARKEN_FACTOR: {ILLUMINATION_DARKEN_FACTOR}")
logger.info(f"NOISE_LEVEL: {NOISE_LEVEL}")
logger.info(f"CONTRAST_STRENGTH: {CONTRAST_STRENGTH}")
logger.info(f"SATURATION_BOOST: {SATURATION_BOOST}")

def build_lowlight_graph(source_path_tensor, target_path_tensor, illumination, noise, contrast, saturation):
    image_gt_bytes = tf.io.read_file(source_path_tensor)
    image_gt_tensor = tf.image.decode_image(image_gt_bytes, channels=3)
    image_gt_tensor = tf.cast(image_gt_tensor, tf.float32) / 255.0
    
    raw_image, metadata = unprocess.unprocess(image_gt_tensor)
    
    shot_noise, read_noise = unprocess.get_noise_levels(noise)
    noisy_raw_image = unprocess.add_noise(raw_image, shot_noise, read_noise)
    
    noisy_raw_image_batched = tf.expand_dims(noisy_raw_image, axis=0)
    red_gain_batched = tf.expand_dims(metadata['red_gain'], axis=0)
    blue_gain_batched = tf.expand_dims(metadata['blue_gain'], axis=0)
    cam2rgb_batched = tf.expand_dims(metadata['cam2rgb'], axis=0)

    linear_rgb_batched = process_to_linear_rgb(noisy_raw_image_batched,
                                               red_gain_batched,
                                               blue_gain_batched,
                                               cam2rgb_batched)
    linear_rgb = tf.squeeze(linear_rgb_batched, axis=0)

    def py_bilateral_filter(image_np):
        return cv2.bilateralFilter(image_np.numpy().astype(np.float32), d=31, sigmaColor=0.1, sigmaSpace=8.0)

    base_illumination = tf.py_function(
        py_bilateral_filter,
        inp=[linear_rgb],
        Tout=tf.float32
    )
    base_illumination.set_shape(linear_rgb.shape)

    detail_layer = linear_rgb / (base_illumination + 1e-8)

    dark_illumination = base_illumination * illumination

    low_light_linear = dark_illumination * detail_layer

    image_with_contrast = apply_s_curve_contrast(low_light_linear, strength=contrast)

    final_linear_image = adjust_linear_saturation(image_with_contrast, saturation)

    final_srgb_image = apply_gamma_compression(final_linear_image)
    
    final_image_display = tf.cast(tf.clip_by_value(final_srgb_image, 0.0, 1.0) * 255.0, tf.uint8)
    encoded_image = tf.image.encode_jpeg(final_image_display)
    
    return tf.io.write_file(target_path_tensor, encoded_image)


if __name__ == '__main__':

    SOURCE_DIR = "../Lowlightdataset/coco_original"
    TARGET_DIR = "../Lowlightdataset/coco_final_dark"

    os.makedirs(TARGET_DIR, exist_ok=True)
    logger.info(f"SOURCE_DIR: {SOURCE_DIR}")
    
    source_images = glob.glob(os.path.join(SOURCE_DIR, '*.jpg')) + \
                    glob.glob(os.path.join(SOURCE_DIR, '*.png'))
    
    logger.info(f"Found {len(source_images)} images to process in {SOURCE_DIR}.")

    for source_path in tqdm(source_images, desc=f"Generating low-light images"):
        contrast = weighted_random_linear(min_val=0.5, max_val=1.0, main_lo=0.6, main_hi=0.8) # 0.5 - 1.0
        saturation = weighted_random_linear(min_val=1.0, max_val=1.5, main_lo=1.1, main_hi=1.3) # 1.0 - 1.5
        illumination = weighted_random_log(min_val=0.0001, max_val=0.001, main_hi=0.0003) # 0.0001 - 0.001
        noise = log_right_skewed_with_peak(min_val=0.00005, max_val=0.005, peak_val=0.0001) # 0.00005 - 0.005

        filename = os.path.basename(source_path)
        target_path = os.path.join(TARGET_DIR, filename)

        if os.path.exists(target_path):
            continue
        
        build_lowlight_graph(tf.constant(source_path), tf.constant(target_path), illumination, noise, contrast, saturation)

    print(f"Data generation complete!")