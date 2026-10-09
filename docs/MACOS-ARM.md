# macOS / ARM (Apple Silicon)

CIL publishes no `osx-arm64` binary package, so the supported install route on
a Mac remains the Linux (x86_64) container described in the
[README](../README.md#macos). This page documents the native Apple Silicon
route that was verified on this machine: CIL built from source, and the torch
kernel operator running on the MPS backend.

## CIL from source (experimental)

CIL can be built from source on macOS/ARM using its experimental development
environment:

```bash
# Prerequisite: Apple clang (Xcode Command Line Tools) and cmake.
# The env file declares `name: cil_dev`; a custom name is fine as long as it is
# used consistently below.
conda env create -f https://raw.githubusercontent.com/TomographicImaging/CIL/master/scripts/cil_development_osx.yml
conda activate cil_dev

# Build CIL from source.
git clone https://github.com/TomographicImaging/CIL.git
pip install -e CIL/

# Install cil-krl from its own checkout (the Kernel_RL repository).
cd /path/to/Kernel_RL
pip install -e ".[dev]"
```

This uses CIL's *experimental* dev environment: some features are unavailable
on macOS/ARM.

Versions verified here: CIL `0.1.dev1+gefade2dc2` (master) built with Apple
clang on Python 3.14. The full `cil-krl` CPU test suite passes natively
against that build.

## torch and MPS

Install an arm64 build of PyTorch (`pip install torch`; verified with
torch 2.14.1). The torch kernel operator then runs on the MPS backend.

### Duplicate-OpenMP hazard

Importing torch into a process that already uses CIL's native libraries aborts
on macOS with:

```
OMP: Error #15: libomp.dylib already initialized
```

CIL's native libraries (built against conda's `llvm-openmp`) and PyTorch each
load an OpenMP runtime, and the second one refuses to initialise. The
documented workaround is:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

> **This is an unsafe workaround.** It disables OpenMP's duplicate-runtime
> safety check process-wide and can mask genuine initialisation bugs; use it
> only for the known torch + CIL coexistence case, never as a general setting.
> When the tests are enabled, `tests/test_mps_kernel_operator.py` sets it
> itself (before importing torch) so it does not need to be exported
> externally; when they are disabled the variable is left untouched.

### Backend and device selection

- `backend="auto"` deliberately resolves to numba on macOS (without importing
  torch), which avoids the duplicate-OpenMP clash. To use MPS you must request
  the torch backend explicitly: `backend="torch"`.
- With `backend="torch"`, `device="auto"` resolves cuda → mps → cpu, so on a
  Mac it selects `mps`.

### Running the MPS tests

MPS tests are opt-in like the other GPU tests:

```bash
KRL_RUN_GPU_TESTS=1 conda run -n cil_dev python -m pytest tests/test_mps_kernel_operator.py -v
```

The module skips cleanly when `KRL_RUN_GPU_TESTS` is unset or when MPS is not
available (Linux/CI).

## Verified numerics

On this Apple M5 Mac (Python 3.14, torch 2.14.1, CIL source build as above),
the MPS backend of the kernel operator was checked against the numba CPU
reference on a real CIL `ImageGeometry`, for dense/sparse x normal/hybrid
forward and adjoint:

- max absolute error ~2–3e-7 (float32) in the initial validation, ≤2e-6
  across the regression-test configurations,
- fresh adjoint-first (lazy bounded normalisation) results finite,
- adjoint dot-product identity within 1e-6 relative error (measured ~1e-7;
  `test_adjoint_dot_product_identity`).
