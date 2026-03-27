# ISP Utilities (Third-Party Attribution)

This folder contains the original Image Signal Processing (ISP) pipeline code from the paper:

## Original Work

**Unprocessing Images for Learned Raw Denoising**

```bibtex
@inproceedings{brooks2019unprocessing,
  title={Unprocessing Images for Learned Raw Denoising},
  author={Brooks, Tim and Mildenhall, Ben and Xue, Tianfan and Chen, Jiawen and Sharlet, Dillon and Barron, Jonathan T},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019},
}
```

**Authors**: Tim Brooks, Ben Mildenhall, Tianfan Xue, Jiawen Chen, Dillon Sharlet, Jonathan T. Barron

**GitHub**: https://github.com/google-research/google-research/tree/master/unprocessing

**Paper**: https://arxiv.org/abs/1811.11721

## Files

- `process.py` - ISP forward pipeline (Raw Bayer → processed sRGB)
- `unprocess.py` - ISP reverse pipeline (sRGB → Raw Bayer)

## Note

These files are used as the foundation for the statistical low-light synthesis pipeline. The main project extends and modifies these utilities for low-light image generation, while preserving the original ISP pipeline design.
