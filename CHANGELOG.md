# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **GPU SUPPORT**: PyTorch GPU backend for kernel operator enables processing of large volumes (256³) on consumer GPUs
  - Automatic backend selection: `backend='auto'` tries torch (GPU) → numba (CPU)
  - float32 precision option reduces memory by 50% (e.g., 256³ volumes: 3.6-8.9 GB depending on config)
  - Memory requirements for 256³ volumes:
    - Medium config (n=7, k=48): **8.8 GB** - fits RTX 3080 (12GB)
  - Full API compatibility with CPU backend
  - Automatic GPU memory management with cache clearing
  - Example: `get_kernel_operator(geometry, backend='torch', dtype='float32')`

### Changed
- **PERFORMANCE**: Optimized kernel operator with sparse indexing and precomputation - **3-7x faster** and **84-86% less memory**
  - Pre-compute anatomical weights once instead of recalculating every iteration
  - Sparse indexing over k selected neighbors instead of all n³ neighbors
  - Enabled Numba fastmath mode for faster exponential calculations
  - Benchmark results (32³ volume, n=5, k=20):
    - Forward pass: **7.45x faster** (36.49ms → 4.90ms)
    - Adjoint pass: **6.88x faster** (32.29ms → 4.70ms)
    - Precomputation: **12.46x faster** (64.14ms → 5.15ms)
    - Memory: **84% reduction** (31.25MB → 5.00MB)
  - Benchmark results (64³ volume, n=7, k=48):
    - Forward pass: **3.65x faster** (534.86ms → 146.54ms)
    - Adjoint pass: **3.09x faster** (547.35ms → 177.27ms)
    - Precomputation: **3.55x faster** (912.65ms → 256.88ms)
    - Memory: **86% reduction** (686.00MB → 96.00MB)

### Fixed
- **CRITICAL: HKRL convergence bug** - Fixed three critical bugs preventing Hybrid KRL objective function convergence:
  1. **Emission reference freezing logic**: `_update_hybrid_reference()` was incorrectly updating the frozen emission reference even when `freeze_emission_kernel=True`. Fixed to ensure once frozen, the reference never changes.
  2. **Freeze timing**: Clarified that when `freeze_iteration=N`, the kernel freezes at the END of iteration N, so iteration N+1 onwards use iteration N's emission state as the frozen reference.
  3. **Sensitivity recomputation for hybrid kernels**: For HKRL with `normalize_kernel=True`, the kernel operator's adjoint changes every iteration (due to changing emission reference), requiring sensitivity to be recomputed each iteration until freezing. The sensitivity is now:
     - Recomputed every iteration before freezing (when kernel is changing)
     - Recomputed once when freezing (with the frozen kernel)
     - Held constant after freezing (no more recomputation needed)
  - Added comprehensive test suite (`tests/test_hybrid_freezing.py`) with 9 tests verifying correct freezing behavior
  - These fixes are essential for HKRL convergence - without them, the forward and adjoint operators use different emission references, breaking self-adjointness and causing exponential divergence
- Kernel operator now properly utilizes sparse masking to skip masked-out neighbors in computation loops
