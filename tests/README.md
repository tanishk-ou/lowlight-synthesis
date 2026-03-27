# `tests/` - Low-Light Synthesis Testing Suite

This directory contains the integration and unit tests for the `lowlight_synthesis` pipeline. The test suite validates the core image signal processing (ISP) operations, sensor noise injection, sampling distributions, and the end-to-end low-light degradation graph.

## Test Status 

Currently, the test suite is nearly completely green. **All tests pass with one deliberate exception:**

* `test_unprocess_process_roundtrip_preserves_approximate_color` (located in `tests/test_isp_pipeline.py`)

### The 70% Correlation Challenge 🎯
The failing test evaluates how well the original color is preserved when an image is sent through the complete roundtrip: `sRGB -> Raw Bayer (Unprocess) -> Linear RGB (Process) -> sRGB`. 

Because our testing inputs often include high-frequency noise (which interferes with standard bilinear demosaicing algorithms), the roundtrip currently achieves a color correlation of **40% - 45%**. 

We have intentionally set the assertion threshold back to **70%** (`> 0.7`). This is an open challenge, and **collaboration and support are highly encouraged!** If you have ideas for improving the demosaicing logic, refining the color correction matrix (CCM) application, or stabilizing the ISP roundtrip to push our correlation above 70%, we welcome your pull requests and insights.

## Running the Tests

The test suite is built using `pytest` and relies on `pytest-cov` for coverage reporting. Note that TensorFlow Eager Execution is required and is enabled globally via `tests/conftest.py`.

### 1. The Standard Test Command
To run the entire test suite, generate a coverage report for the `lowlight_synthesis` package, and save the output to a text file, run:

```bash
pytest tests/ -v --cov=lowlight_synthesis | tee test_output.txt
```

### 2. Running Specific Test Files
If you are working on a specific module and want to run only its associated tests:

```bash
# Test only the ISP pipeline roundtrip and logic
pytest tests/test_isp_pipeline.py -v

# Test only the noise injection and sampling distributions
pytest tests/test_noise.py tests/test_sampling.py -v

# Test only the end-to-end TensorFlow graph
pytest tests/test_end_to_end.py -v
```

### 3. Running the Correlation Challenge
To run *only* the failing correlation test while attempting to optimize the ISP logic:

```bash
pytest tests/test_isp_pipeline.py::TestRoundtrip::test_unprocess_process_roundtrip_preserves_approximate_color -v
```

### 4. Stop on First Failure
If you are debugging and want the test runner to halt the moment a test fails:

```bash
pytest tests/ -v -x
```

## Notes for Contributors
* **Do not add `tf.disable_v2_behavior()`:** The pipeline relies heavily on TensorFlow's Eager Execution (such as `.numpy()` conversions for OpenCV operations). Disabling v2 behavior will crash the testing suite. 
* **Shapes and Dimensions:** The core `unprocess` module strictly expects 3D tensors `[height, width, channels]`. Be mindful of adding unnecessary batch dimensions (`1`) in test fixtures.