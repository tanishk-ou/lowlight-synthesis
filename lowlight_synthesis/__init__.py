"""Statistical Low-Light Image Synthesis with Sensor Simulation.

This package provides a statistical pipeline for generating realistic synthetic
low-light images from normal exposure photographs, using physics-based sensor
simulation and carefully tuned degradation parameters.

Key Features:
    - Realistic sensor simulation (Bayer mosaicing, white balance, noise)
    - Statistical sampling of degradation parameters
    - Detail-preserving illumination adjustment
    - Validation on extreme low-light datasets (Ellar)
    - Significantly outperforms standard low-light datasets (ARID)
"""

__version__ = "1.0.0"
__author__ = "Tanishk Gopalani"
__license__ = "Apache License 2.0"

from lowlight_synthesis.core import (
    unprocess,
    process_new as process,
)
from lowlight_synthesis.degradation import (
    sampling,
    noise,
    illumination,
)

__all__ = [
    "unprocess",
    "process",
    "sampling",
    "noise",
    "illumination",
]
