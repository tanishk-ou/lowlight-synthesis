# Statistical Low-Light Image Synthesis

A physics-based pipeline for generating highly realistic synthetic low-light images from normal-exposure photographs. This work demonstrates that synthetic data generated through accurate sensor simulation and statistical degradation modeling can **outperform real low-light datasets** on challenging benchmarks.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 1.14+](https://img.shields.io/badge/TensorFlow-1.14+-orange.svg)](https://tensorflow.org)

## Key Innovation

**Encoder trained on synthetic data beats SOTA on Ellar dataset**: A spatial action recognition encoder trained exclusively on synthetically darkened images from this pipeline outperforms prior state-of-the-art across all Ellar subsets:

| Subset                     | Our Encoder (Top-1 Acc) | SOTA (DGAM) (Top-1 Acc) | Improvement |
|---------------------------|-------------------------|--------------------------|-------------|
| Full Dataset              | 50.37%                  | 38.42%                   | +11.95%     |
| Low-Light (ll)            | 80.32%                  | 58.39%                   | +21.93%     |
| Extreme Low-Light (ell)   | 18.58%                  | 14.61%                   | +3.97%      |

**Model Repository**: See [tanishk-ou/lowlight-multitask-unet](https://github.com/tanishk-ou/lowlight-multitask-unet) for the encoder implementation and training code.

This validates that **statistically-accurate synthetic degradation** prepares models for real low-light image formation better than real examples, likely because manual annotation introduces systematic biases in extreme conditions.

## How It Works

### Three-Pillar Architecture

#### 1. Sensor Simulation (Realistic Physics)
- **Unprocess**: sRGB → raw Bayer sensor via ISP reversal
- **Noise**: Shot noise (brightness-dependent, Poisson-like) + read noise (constant, electronic)
- **Reprocess**: Raw → linear RGB via standard ISP pipeline (white balance, demosaic, CCM)

#### 2. Detail-Preserving Darkening (The Innovation)
- Bilateral filter on linear RGB extracts smooth illumination component
- Decompose: `detail = linear_rgb / illumination`
- Darken illumination: `dark_illumination = illumination × darkening_factor`
- Recombine: `output = dark_illumination × detail`
- **Result**: Natural dimming without artifacts, preserves edges and texture

#### 3. Perceptual Adjustments
- S-curve contrast mapping (tone mapping)
- Saturation boost in linear RGB (compensates for color desaturation in shadows)
- Gamma compression back to sRGB (display-ready)

### Statistical Parameter Distributions

All parameters are drawn from realistic distributions, not uniform random:

- **Illumination**: Log-linear scale (models exponential light falloff)
- **Noise**: Right-skewed beta distribution with peak at sensor-realistic 0.0001
- **Contrast**: Beta distribution (3-region weighting: main, low, high regions)
- **Saturation**: Beta distribution with mode at typical sensor saturation boost

## Installation

```bash
pip install -e .
```

**Dependencies**:
- Python 3.8+
- TensorFlow 1.14+ (1.x for stability, 2.x support coming)
- OpenCV 4.5+
- NumPy 1.19+
- tqdm

## Quick Start

```python
from lowlight_synthesis.core.pipeline import build_lowlight_graph
import tensorflow as tf

# Process a single image
sess = tf.Session()
result = sess.run(build_lowlight_graph(
    input_image_path='/path/to/normal.jpg',
    output_image_path='/path/to/lowlight.jpg',
    illumination=0.0002,      # Darkening (smaller = darker)
    noise_level=0.0001,       # Sensor noise
    contrast=0.7,             # S-curve strength
    saturation=1.2            # Color boost
))
```

See `generate_low_light_final.py` for a complete standalone example.

## Technical Details

### Why This Works

**Sensor Physics Match**: The pipeline faithfully reproduces how real camera sensors respond to low light:
- Shot noise scales with signal brightness (photon counting randomness)
- Read noise is constant (amplifier electronics noise)
- ISP pipeline matches standard computational photography workflow

**No Artificial Artifacts**: Bilateral filter decomposition prevents common low-light synthesis problems:
- ✓ Natural edge preservation
- ✓ No halo effects around objects
- ✓ Realistic texture under darkening
- ✓ Smooth noise gradients

**Statistical Realism**: Beta distributions match observed image statistics:
- Bounded parameters (contrast, saturation not arbitrary)
- Skewed distributions match real camera setting distributions
- Validated on natural image databases

### Parameter Ranges

| Parameter | Range | Effect |
|-----------|-------|--------|
| `illumination` | 0.0001 - 0.001 | Overall brightness (smaller = darker) |
| `noise_level` | 0.00001 - 0.005 | Sensor noise magnitude |
| `contrast` | 0.0 - 1.0 | S-curve strength (0 = none, 1 = full) |
| `saturation` | 0.8 - 1.5 | Color vibrancy (1.0 = unchanged) |

## Attribution

This work builds on foundational research in ISP pipeline simulation:

**Unprocessing Images for Learned Raw Denoising** (CVPR 2019)
```bibtex
@inproceedings{brooks2019unprocessing,
  title={Unprocessing Images for Learned Raw Denoising},
  author={Brooks, Tim and Mildenhall, Ben and Xue, Tianfan and Chen, Jiawen and
          Sharlet, Dillon and Barron, Jonathan T},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

Authors: Tim Brooks, Ben Mildenhall, Tianfan Xue, Jiawen Chen, Dillon Sharlet, Jonathan T. Barron
GitHub: https://github.com/google-research/google-research/tree/master/unprocessing

The original ISP code is preserved in [`acknowledgements/isp_utils/`](acknowledgements/isp_utils/) with proper attribution in the README.

## Pipeline Overview

```
                  Normal Exposure (sRGB)
                          ↓
              ┌─────────────────────────┐
              │  Sensor Simulation      │
              │  (Unprocess + Noise)    │
              └──────────┬──────────────┘
                         ↓
           ┌──────────────────────────────┐
           │  ISP Reprocessing to Linear  │
           │  (White Balance, Demosaic)   │
           └──────────┬───────────────────┘
                      ↓
         ┌────────────────────────────────┐
         │  Bilateral Filter Decomposition│
         │  (illumination + detail layer) │
         └────────────┬───────────────────┘
                      ↓
      ┌──────────────────────────────────┐
      │  Detail-Preserving Darkening     │
      │  (Recombine with darkened illum.)│
      └────────────┬─────────────────────┘
                   ↓
    ┌──────────────────────────────────┐
    │  Perceptual Adjustments          │
    │  S-curve, saturation, gamma      │
    └────────────┬─────────────────────┘
                 ↓
         Synthetic Low-Light (sRGB)
```

## Architecture

```
lowlight_synthesis/
├── core/
│   ├── pipeline.py           # Main synthesis orchestrator
│   ├── process_new.py        # ISP forward (raw → sRGB)
│   └── unprocess.py          # ISP reverse (sRGB → raw)
├── degradation/
│   ├── sampling.py           # Statistical distribution sampling
│   ├── noise.py              # Sensor noise models
│   └── illumination.py       # Darkening, contrast, saturation effects
└── acknowledgements/isp_utils/    # Original ISP code (Brooks et al. 2019)
    ├── process.py
    ├── unprocess.py
    ├── LICENSE
    └── README.md
```

## Limitations & Future Work

**Current Limitations**:
- TensorFlow 1.x only (2.x migration in progress)
- Bilateral filter uses fixed parameters
- Optimized for daytime/outdoor images
- High memory usage for 4K+ images
- No temporal coherence for video

**Future Improvements**:
- [ ] TensorFlow 2.x with eager execution
- [ ] PyTorch implementation
- [ ] Parameterizable bilateral filter
- [ ] Video sequence synthesis
- [ ] Real-time preview tool
- [ ] Additional sensor models

## Citation

If you use this work, please cite:

```bibtex
@software{lowlight_synthesis_2026,
  title={Statistical Low-Light Image Synthesis with Sensor Simulation},
  author={Gopalani, Tanishk},
  year={2026},
  url={https://github.com/tanishk_github/lowlight-synthesis}
}
```

And acknowledge the foundational work:

```bibtex
@inproceedings{brooks2019unprocessing,
  title={Unprocessing Images for Learned Raw Denoising},
  author={Brooks, Tim and Mildenhall, Ben and Xue, Tianfan and Chen, Jiawen and
          Sharlet, Dillon and Barron, Jonathan T},
  booktitle={CVPR},
  year={2019}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE)

**Attribution**: This work includes code and techniques from Tim Brooks et al.'s "Unprocessing Images for Learned Raw Denoising" (CVPR 2019). See [`acknowledgements/isp_utils/README.md`](acknowledgements/isp_utils/README.md) for the original paper citation and repository link.

---

**Questions?** Open an issue on GitHub or check the code documentation.
