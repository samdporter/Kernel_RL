# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-10-13

### Fixed
- `save_image` now rejects non-`ImageData` input with `TypeError` instead of
  silently transposing a raw ndarray.
- NIfTI loading preserves the translation through the CIL geometry centre
  (previously silently discarded) and rejects affines it cannot represent
  (flips, axis permutations, rotation, shear) with a clear error; saving writes
  a geometry-derived origin instead of a zero-origin affine.
- `load_nifti_as_imagedata` preserves the source floating dtype (float32→float32,
  float64→float64; integer→float64) instead of always downcasting to float32;
  complex/RGB/structured inputs are rejected; out-of-range int64 warns about
  possible rounding.
- The scipy blur backend now zero-pads at boundaries like the numba and torch
  backends (previously reflected), so all three backends share a common
  zero-padding boundary condition (numerical results still depend on the
  backend's computational dtype).
- `DirectionalOperator.dot` returns an independent result instead of a shared
  internal buffer.
- Toggling `normalize_features` via `set_parameters` now rebuilds the processed
  anatomy from an owned raw snapshot instead of being a silent no-op.
- `LBFGSBOptimizer` records its first objective at the bound-projected start for
  every run length, so negative initial estimates no longer produce an
  inconsistent first history entry.

### Changed
- `GaussianBlurringOperator` scipy results change near boundaries (now
  zero-padded); `save_image` writes a geometry-derived origin; loaded image
  storage uses the source dtype (float64/int inputs use 8 bytes/voxel instead
  of 4).

### Deprecated
- `MAPRL(eps_preconditioner=...)` is deprecated and has no effect; a
  `FutureWarning` is emitted if a value is supplied (apply stabilisation inside
  the preconditioner instead).

## [0.4.0] - 2026-10-12

### Fixed
- Numba normalised `adjoint` used a `norm_arr.shape[0] > 1` sentinel that
  skipped the per-voxel normalisation when the first array dimension is 1; it
  now uses the array size (matching the torch backend).
- `backend='auto'` now also catches `OSError`/`AttributeError` from a broken
  `import torch` and falls back to numba (kernel and blur operators).
- `MAPRL` rejects a non-positive `step_size` or negative `relaxation_eta`
  (which produced a negative step), never applies a candidate worse than the
  reference after an exhausted Armijo search, and ignores non-finite trial
  losses when selecting the best candidate.
- `LBFGSBOptimizer.run(iterations<=0)` is now a true no-op and clears any
  stale solver `result`.
- Removed the invalid `disp` SciPy option from `LBFGSBOptimizer` (it emitted
  an `OptimizeWarning` and made `verbose` a no-op); `verbose>0` now prints a
  result summary.
- The torch sparse mask uses `int32` when `n**3 > 32767` (previously `int16`
  overflowed for `num_neighbours >= 33`).
- `GaussianBlurringOperator` now validates single-channel 3-D input
  (previously a 2-D multichannel image was silently convolved across
  channels).
- Sigma validation now also rejects values whose squared denominator
  underflows in float32 (native torch would otherwise produce NaNs).
- CI's wheel `entry_points` assertion is fixed for Python >= 3.12.
- `MAPRL` no longer attaches its own logging handler or reads a
  `MAPRL_LOG_LEVEL` environment variable, so it follows the host application's
  logging configuration. It also initialises the preconditioner image during
  `__init__` when an initial line search is configured, so the first line search
  uses the supplied preconditioner instead of an unset one.
- `BaseKernelOperator.adjoint` now raises `NotImplementedError`; concrete
  subclasses (numba and torch) provide the adjoint, rather than the base class
  silently returning the forward result.
- Removed dead `sigma_anat <= 0` branches from the torch kernel operator (a
  non-positive `sigma_anat` is rejected by validation), and corrected the class
  docstring to describe the on-the-fly anatomical-weight computation.

### Changed
- `SaveIterationCallback` docs now describe its actual save schedule;
  `DirectionalOperator` warns on a non-positive `eta`; `utils` docs/errors
  clarified; `pyproject.toml` uses SPDX `license = "MIT"` (setuptools >= 77).
- Documentation: `cil-krl` is now documented as published on PyPI (0.3.3) with
  `pip install cil-krl`, while CIL itself must be installed separately first
  (conda or from source — it is not on PyPI).
- `master` is now a plugin-only checkout: the research pipelines, benchmark
  scripts and BrainWeb data preparation have moved to the
  `paper/brainweb-deconvolution` branch.
- `examples/rl_krl_dtv.py` now reports five variants — RL, KRL, `RL-dTV`,
  L-BFGS-B and L-BFGS-B-precond (the same L-BFGS-B run through the EM-type
  diagonal change of variables), all on the same fidelity/prior for the two
  L-BFGS-B runs, with `examples/README.md` updated accordingly.

### Added
- Optional fixed diagonal linear preconditioner for `LBFGSBOptimizer`
  (`preconditioner=`, an `ImageData` or `ndarray` with strictly positive
  entries, fixed for the run), applied as a change of variables `x = M z`
  (`f̃(z) = F(M z)`, `∇_z = m ⊙ ∇F(M z)`, `x* = M z*`); the objective history
  and `solution` stay in image space and the default is the identity.
- Maintained example `examples/rl_krl_dtv.py` comparing RL, KRL,
  directional-TV-regularised RL (`RL-dTV`) and L-BFGS-B on a synthetic phantom,
  where `RL-dTV` now uses a properly tuned preconditioned update
  (sensitivity-normalised EM preconditioner, step 1.0 with Armijo backtracking
  on every update) on the same objective L-BFGS-B minimises, with
  `examples/README.md`, tests in `tests/test_example_rl_krl_dtv.py` and ruff
  coverage of `examples/` in CI and the Makefile.

## [0.3.3] - 2026-10-09

Metadata-only update; no code changes.

### Changed
- Added ORCIDs for Sam Porter (0000-0002-1815-4716) and Kjell Erlandsson
  (0000-0002-1072-4382) to `CITATION.cff`, alongside the existing Daniel
  Deidda (0000-0002-2766-4339) and Kris Thielemans (0000-0001-6353-9538).

## [0.3.2] - 2026-10-09

Metadata-only update; no code changes.

### Changed
- Added Daniel Deidda (ORCID 0000-0002-2766-4339) as third author and
  Kris Thielemans (ORCID 0000-0001-6353-9538) as last author in
  `pyproject.toml`, `CITATION.cff`, `src/krl/__init__.py` and the README
  citation.

## [0.3.1] - 2026-10-09

Metadata-only correction; no code changes.

### Changed
- Corrected author order so Sam Porter is listed first in `pyproject.toml`,
  `CITATION.cff`, `src/krl/__init__.py` and the README citation.

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
