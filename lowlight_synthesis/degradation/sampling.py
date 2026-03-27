"""Statistical sampling functions for low-light degradation parameters.

This module provides distribution-based sampling functions that reproduce
the realistic distribution of low-light degradation found in real sensors.
The distributions are carefully tuned to match natural image statistics.
"""

import random
import math
from typing import Tuple


def sample_band(lo: float, hi: float, alpha: float, beta: float) -> float:
    """Sample from a beta distribution scaled to [lo, hi] range.

    Args:
        lo: Lower bound of the range
        hi: Upper bound of the range
        alpha: Alpha parameter for beta distribution (shape control)
        beta: Beta parameter for beta distribution (shape control)

    Returns:
        Sampled value in range [lo, hi]
    """
    x = random.betavariate(alpha, beta)
    return lo + x * (hi - lo)


def weighted_random_linear(
    min_val: float,
    max_val: float,
    main_lo: float,
    main_hi: float,
    main_weight: float = 0.75,
    low_weight: float = 0.10,
) -> float:
    """Weighted sampling with linear interpolation between ranges.

    Samples from three regions with controllable probabilities:
    - Main region [main_lo, main_hi] with high probability (concentrated)
    - Low region [min_val, main_lo] with medium probability
    - High region [main_hi, max_val] with low probability

    Uses beta distributions (5,5) for symmetric concentration within each region.

    Args:
        min_val: Absolute minimum value
        max_val: Absolute maximum value
        main_lo: Lower bound of main concentration region
        main_hi: Upper bound of main concentration region
        main_weight: Probability of sampling from main region (default: 0.75)
        low_weight: Probability of sampling from low region (default: 0.10)
                   Remaining probability (0.15) goes to high region

    Returns:
        Sampled value with realistic distribution
    """
    r = random.random()

    if r < main_weight:
        # Main region: concentrated around center (beta 5,5 is symmetric)
        return sample_band(main_lo, main_hi, 5, 5)
    elif r < main_weight + low_weight:
        # Low region: skewed towards main region (beta 2,6)
        return sample_band(min_val, main_lo, 2, 6)
    else:
        # High region: skewed towards main region (beta 6,2)
        return sample_band(main_hi, max_val, 6, 2)


def weighted_random_log(
    min_val: float,
    max_val: float,
    main_hi: float,
    main_weight: float = 0.75,
) -> float:
    """Weighted sampling with logarithmic interpolation between ranges.

    Used for parameters with logarithmic scale (e.g., noise, illumination).
    Samples in log space to match sensor behavior which is naturally logarithmic.

    - Main region: [min_val, main_hi] with high probability
    - Extended region: [min_val, max_val] with low probability

    Args:
        min_val: Absolute minimum value (log scale)
        max_val: Absolute maximum value (log scale)
        main_hi: Upper bound of main region (log scale)
        main_weight: Probability of main region (default: 0.75)

    Returns:
        Sampled value with log-linear distribution
    """
    r = random.random()

    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    log_main_hi = math.log10(main_hi)

    if r < main_weight:
        # Main region in log space: concentrated towards minimum
        log_val = sample_band(log_min, log_main_hi, 2, 8)
    else:
        # Extended region: wider distribution
        log_val = sample_band(log_min, log_max, 3, 3)

    return 10 ** log_val


def log_right_skewed_with_peak(
    min_val: float = 0.00005,
    max_val: float = 0.005,
    peak_val: float = 0.0001,
    alpha: float = 2.2,
    beta: float = 6.0,
) -> float:
    """Right-skewed log-space distribution centered at a specified peak.

    Designed for realistic sensor noise levels which follow a right-skewed
    distribution with a natural peak (mode) around typical sensor noise floor.

    The alpha and beta parameters are tuned to match observed noise behavior:
    - alpha=2.2 < beta=6.0 creates right skew (tail extends to high values)
    - Peak occurs at mode = (alpha-1)/(alpha+beta-2) ≈ 0.17 in normalized space

    Args:
        min_val: Minimum noise level
        max_val: Maximum noise level
        peak_val: Mode (peak) of the distribution
        alpha: Alpha parameter for beta distribution (left shape)
        beta: Beta parameter for beta distribution (right shape)

    Returns:
        Noise level sampled from right-skewed distribution
    """
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    log_peak = math.log10(peak_val)

    # Sample normalized value [0,1] from beta distribution
    x = random.betavariate(alpha, beta)

    # Calculate the mode (peak) of beta distribution
    mode = (alpha - 1) / (alpha + beta - 2)

    # Scale to put the mode at log_peak
    scale = (log_peak - log_min) / mode

    # Transform to log space with peak positioned correctly
    log_val = log_min + x * scale

    # Clamp to valid range
    log_val = max(log_min, min(log_val, log_max))

    return 10 ** log_val
