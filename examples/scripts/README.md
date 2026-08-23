# Scripts

Executable scripts for experiments and examples.

## Reconstruction Scripts

- **run_deconv_sweeps.py**: Hyperparameter sweeps for DTV/KRL/HKRL
- **example_reconstruction.py**: Example reconstruction demo
- **find_lowest_nrmse.py**: Find best parameters from sweep results

## Benchmarking

- **benchmark_kernel_performance.py**: Comprehensive performance benchmarking tool

  Benchmarks kernel operator across multiple backends and volume sizes:
  - **Backends**: Optimized CPU (Numba), GPU (PyTorch CUDA), Original CPU (backup)
  - **Volume sizes**: 64³, 128³, 256³ (configurable)
  - **Metrics**: Forward/adjoint timing, precomputation, memory usage, speedups

  **Basic usage:**
  ```bash
  # Run all benchmarks (CPU + GPU if available)
  python scripts/benchmark_kernel_performance.py

  # Benchmark specific sizes
  python scripts/benchmark_kernel_performance.py --sizes 64 128

  # GPU-only benchmarking (fast)
  python scripts/benchmark_kernel_performance.py --gpu-only --sizes 256

  # Skip backup comparison (faster)
  python scripts/benchmark_kernel_performance.py --skip-backup
  ```

  **Recent results:**
  - CPU optimization: **3-7x speedup**, **84-86% memory reduction** vs original
  - GPU acceleration: Enables 256³ volumes on consumer GPUs (RTX 3060+)
  - Memory (256³ with float32): 3.6 GB (n=5, k=20) to 8.8 GB (n=7, k=48)

See [../docs/GETTING-STARTED.md](../docs/GETTING-STARTED.md) for usage.
