import pytest
import numpy as np
import tensorflow.compat.v1 as tf

# Enable eager execution for all tests
tf.enable_eager_execution()

@pytest.fixture
def small_rgb_image():
    """Small 128x128 RGB test image in [0, 1] range."""
    np.random.seed(42)
    return np.random.rand(128, 128, 3).astype(np.float32)

@pytest.fixture
def bright_image():
    """High-brightness test image (mostly 0.8)."""
    return np.ones((64, 64, 3), dtype=np.float32) * 0.8

@pytest.fixture
def dark_image():
    """Low-brightness test image (mostly 0.2)."""
    return np.ones((64, 64, 3), dtype=np.float32) * 0.2

@pytest.fixture
def mid_image():
    """Mid-brightness test image (0.5)."""
    return np.ones((64, 64, 3), dtype=np.float32) * 0.5

@pytest.fixture
def gradient_image():
    """Gradient image for testing detail preservation."""
    h, w = 64, 64
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Horizontal gradient
    for i in range(h):
        img[i, :, :] = i / h
    return img