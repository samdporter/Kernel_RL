# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-08-27

### Fixed
- **HKRL hybrid activation**: `HKRLMethod` now enables `hybrid=True` when `sigma_emission > 0`; mismatch scenarios include `hybrid: true`. Previously HKRL runs were static KRL variants.
- **HKRL per-iteration emission capture**: Iterates are now recorded in emission domain using the kernel state from that iteration, not the final kernel. Pre-freeze curves are now correct.
- **Freeze timing**: `freeze_iteration=N` now freezes after N completed CIL updates (previously one update late).
- **Kernel operator validation**: `num_neighbours` must be positive odd integer; `sigma_anat`, `sigma_dist`, `sigma_emission` must be non-negative.
- **BrainWeb physical geometry**: `BrainWebDataset.voxel_mm` now reads voxel spacing from NIfTI affine; affine preserved on save/load. Runner lesion-mask truth-value error fixed.
- **SIRF activity scale**: `simulate_inputs` records `scale` in manifest; contract test validates high-count invariance.

### Changed
- Quarantined pre-fix local artifacts (`results/`, `plans/`, `data/brainweb/`, `data/patients/MK-H001/`) into `invalidated_2026-08-27/` with README. These must not be reused.
- Added `tools/quarantine_invalidated_artifacts.sh` for idempotent quarantine.

See `docs/superpowers/specs/2026-08-27-phase-0-2-3-readiness-fixes-design.md` for design.

## [0.3.0] - 2026-10-09

Fixes and cleanup from the post-release hardening pass; no new features.

### Fixed
- `backend="auto"` no longer imports torch on macOS: `get_kernel_operator` and
  `create_gaussian_blur` resolve to the numba (CPU) backend on Darwin without
  probing for torch, avoiding duplicate-OpenMP aborts when CIL's native
  libraries are already loaded.
- Callbacks are observational: `SaveIterationCallback` clones the image before
  applying its non-negative clamp, so it no longer mutates the algorithm's
  solution or an aliased operator's output. Both callbacks validate their
  `interval`/`save_first_n` arguments, and `NRMSECallback` rejects a
  non-positive or non-finite ground-truth maximum.
- `RichardsonLucy` and `MAPRL` schedule freezing, Armijo line searches and
  preconditioner updates by completed updates instead of CIL's iteration label,
  so they stay correct for any `update_objective_interval`.
- NIfTI I/O contract: `load_nifti_as_imagedata` rejects non-3-D volumes and
  singleton axes (CIL cannot represent size-1 dimensions), and
  `load_image`/`save_image` accept only `.nii`/`.nii.gz` rather than any
  `.gz` path. Saved files are documented to record voxel values and spacing
  only, not an original NIfTI header/affine.
- `LBFGSBOptimizer` keeps the template dtype in its working buffers.
- The blur operator validates its inputs: unknown backends and sigma vectors
  that are not three positive finite values raise `ValueError` at construction.

### Changed
- The torch kernel operator's `device="auto"` now resolves cuda → mps → cpu,
  consistent with the blurring backend. The kernel operator's MPS path is now
  verified end-to-end on macOS/ARM against the numba CPU reference, using CIL
  built from source (see `docs/MACOS-ARM.md`).
- Documentation consolidated to match the finished plugin: README installation
  is conda-CIL plus a source or locally built wheel (no PyPI install command or
  PyPI badge, tested Python/CIL combinations listed, macOS route via a Linux
  container), the quickstart is a complete runnable synthetic example with a
  callback, `docs/METHODS.md` documents the Python API with the actual kernel
  defaults, the historical `docs/reference/` notes are gone, the broken LICENSE
  link is gone, and `examples/` is labelled as historical research code.
- Behaviour now documented in the README: backend selection, boundary
  conditions (numba/torch zero-padding, scipy reflection, kernel mirror
  reflection), dtype vs computational precision, aligned-image requirements,
  HKRL restrictions and freezing semantics, and that `LBFGSBOptimizer` is a
  SciPy wrapper rather than a CIL `Algorithm`.

### Added
- An MIT `LICENSE` file (copyright 2026 Sam Porter), now referenced by
  `pyproject.toml` (`license = {file = "LICENSE"}`) so it ships in the
  distribution, and linked from the README.
- Tests for backend selection, blurring, callbacks, L-BFGS-B, NIfTI I/O and the
  public import surface, plus a quickstart test running the README snippet
  against real CIL.
- `RichardsonLucy` is exported from `krl.algorithms` as well as `krl`.
- CI `wheel` job: builds the sdist and wheel, checks the wheel ships only the
  `krl` package (no `examples/`, `tests/` or console scripts), installs the wheel
  non-editable into the CIL environment and imports it from outside the checkout.

## [0.2.0] - 2026-08-23

Plugin release: the package is restructured for distribution as a CIL plugin.

### Added
- Package renamed to `cil-krl` (import as `krl`), Python >= 3.10
- `RichardsonLucy`, callbacks and all operators are now part of a clean public API (`krl.__init__`)
- Regression test ensuring `import krl` never pulls in torch (OpenMP safety)
- End-to-end integration tests covering RL, KRL, HKRL, MAP-RL, L-BFGS-B and
  callbacks against real CIL
- GitHub Actions CI (Python 3.10-3.13, CPU, CIL 25.0.0/26.0.0) and an opt-in
  GPU job; a tag-triggered PyPI release workflow

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
- The torch kernel operator runs on CUDA when available and delegates to the
  numba implementation when its device resolves to CPU
- Tests run against real CIL (the previous conftest injected fake `cil`, `torch`
  and `numba` stubs, hiding real defects)
- Research pipelines, scripts, configs and Docker environment moved to `examples/`
  (not shipped in the wheel); dead code removed (`krl.operators.Gradient`,
  `kernel_operator_backup.py`)
- The `krl-deconv`, `krl-compare` and `krl-sweep` console scripts and the
  `[accelerators]` extra were removed (the `gpu` extra replaced it)
- torch-touching tests are gated behind `KRL_RUN_GPU_TESTS=1`

## [0.1.0]

Initial research version: KRL/HKRL/DTV methods, numba CPU backend, PyTorch CUDA
backend with automatic backend selection, sparse masking optimisations.
