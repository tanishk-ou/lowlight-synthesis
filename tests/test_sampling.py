"""Test statistical sampling distributions for low-light synthesis."""

import pytest
import numpy as np
from lowlight_synthesis.degradation.sampling import (
    sample_band,
    weighted_random_linear,
    weighted_random_log,
    log_right_skewed_with_peak,
)


class TestSampleBand:
    """Tests for Beta distribution sampling."""

    def test_sample_band_range(self):
        """sample_band() should return values in [lo, hi]."""
        lo, hi = 0.1, 0.9
        for _ in range(100):
            val = sample_band(lo, hi, alpha=2, beta=5)
            assert lo <= val <= hi

    def test_sample_band_respects_bounds(self):
        """Multiple samples should fall within bounds."""
        samples = [sample_band(0.2, 0.7, alpha=3, beta=3) for _ in range(1000)]
        assert all(0.2 <= s <= 0.7 for s in samples)


class TestWeightedRandomLinear:
    """Tests for three-region weighted linear sampling."""

    def test_output_range(self):
        """Samples should be in [0, 1]."""
        for _ in range(500):
            val = weighted_random_linear(
                min_val=0.0, max_val=1.0, main_lo=0.3, main_hi=0.7
            )
            assert 0 <= val <= 1

    def test_concentration_around_peak(self):
        """Most samples should cluster in main region [0.3, 0.7]."""
        samples = [
            weighted_random_linear(
                min_val=0.0, max_val=1.0, main_lo=0.3, main_hi=0.7
            )
            for _ in range(1000)
        ]
        main_region = sum(1 for s in samples if 0.3 <= s <= 0.7)
        # ~75% should be in main region (weighted probability)
        assert main_region > 600


class TestWeightedRandomLog:
    """Tests for logarithmic scale sampling."""

    def test_output_positive(self):
        """Log-scale samples must be strictly positive."""
        for _ in range(200):
            val = weighted_random_log(
                min_val=0.0001, max_val=1.0, main_hi=0.01
            )
            assert val > 0

    def test_output_bounded(self):
        """Samples should be reasonable (not extreme)."""
        samples = [
            weighted_random_log(
                min_val=0.0001, max_val=1.0, main_hi=0.01
            )
            for _ in range(1000)
        ]
        # Should rarely exceed 1.0 (well-behaved distribution)
        extreme = sum(1 for s in samples if s > 2.0)
        assert extreme < 50  # Less than 5%


class TestLogRightSkewedWithPeak:
    """Tests for right-skewed noise distribution."""

    def test_peak_near_noise_floor(self):
        """Distribution should peak near 0.0001 (typical sensor noise)."""
        samples = [log_right_skewed_with_peak() for _ in range(2000)]
        # Most samples should be small (right-skewed)
        small_samples = sum(1 for s in samples if s < 0.001)
        assert small_samples > 1500  # 75%+

    def test_positive_values(self):
        """All noise samples must be positive."""
        samples = [log_right_skewed_with_peak() for _ in range(500)]
        assert all(s > 0 for s in samples)

    def test_distribution_shape_skewed_right(self):
        """Mean should be higher than median (right skew)."""
        samples = [log_right_skewed_with_peak() for _ in range(1000)]
        mean_val = np.mean(samples)
        median_val = np.median(samples)
        # Right-skewed: mean > median
        assert mean_val > median_val * 1.1  # 10% higher
