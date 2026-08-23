# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-23

Plugin release: the package is restructured for distribution as a CIL plugin.

### Added
- Package renamed to `cil-krl` (import as `krl`), Python >= 3.10
- `RichardsonLucy`, callbacks and all operators are now part of a clean public API (`krl.__init__`)
- Regression test ensuring `import krl` never pulls in torch (OpenMP safety)
- End-to-end integration tests running on real CIL containers
- GitHub Actions CI (Python 3.10-3.13, CPU) and opt-in GPU job; PyPI release workflow

### Fixed
- **Adjoint correctness**: the numba scatter-style adjoint kernels ran under
  `numba.prange` with unsynchronised `+=`, silently dropping contributions
  (data race). The adjoint is now exact; these kernels run serially while the
  gather-style forwards remain parallel.
  Previously masked by test stubs that replaced numba with pure Python.
- **Forward output dtype**: the kernel operator filled its result into a clone
  of the *anatomical* image, so a float32 anatomy silently truncated forward
  results to float32 and broke adjointness against float64 data (verified to
  machine precision after the fix). Output dtype now follows the input data;
  same fix applied to the torch backend.
- Removed stale module references that broke installed-package imports (`src.krl.*`)

### Changed
- **CIL is now a hard runtime requirement**; all optional-import fallbacks removed
- The torch blurring backend no longer requires CUDA: it selects cuda → mps → cpu,
  so it can run on Apple Silicon and CPU-only machines
- Tests run against real CIL (the previous conftest injected fake `cil`, `torch`
  and `numba` stubs, hiding real defects)
- Research pipelines, scripts, configs and Docker environment moved to `examples/`
  (not shipped in the wheel); dead code removed (`krl.operators.Gradient`,
  `kernel_operator_backup.py`)
- torch-touching tests are gated behind `KRL_RUN_GPU_TESTS=1`

## [0.1.0]

Initial research version: KRL/HKRL/DTV methods, numba CPU backend, PyTorch CUDA
backend with automatic backend selection, sparse masking optimisations.
