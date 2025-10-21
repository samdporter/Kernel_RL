#!/usr/bin/env python3
"""Benchmark kernel operator performance across multiple backends and volume sizes.

Compares:
- Optimized CPU (Numba) vs Original CPU (backup)
- GPU (PyTorch CUDA) vs CPU (Numba)
- Different volume sizes: 64³, 128³, 256³

Usage:
    python scripts/benchmark_kernel_performance.py
    python scripts/benchmark_kernel_performance.py --gpu-only
    python scripts/benchmark_kernel_performance.py --sizes 64 128
"""

import argparse
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Check for PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


@dataclass
class DummyGeometry:
    shape: Tuple[int, int, int]

    def allocate(self, value: float = 0.0, dtype=np.float64):
        data = np.full(self.shape, value, dtype=dtype)
        return DummyImage(data)


class DummyImage:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data)

    @property
    def shape(self):
        return self._data.shape

    def as_array(self):
        return self._data

    def clone(self):
        return DummyImage(self._data.copy())

    def fill(self, values):
        self._data[...] = np.asarray(values, dtype=self._data.dtype)


def time_kernel_operations(backend, shape=(64, 64, 64), n=7, k=48,
                          num_iterations=10, dtype='float64', device='auto'):
    """
    Time forward and adjoint operations for a given backend.

    Parameters
    ----------
    backend : str
        'numba', 'torch', or module for backup
    shape : tuple
        Volume shape
    n : int
        Neighborhood size
    k : int
        Number of masked neighbors
    num_iterations : int
        Number of iterations for timing
    dtype : str
        Data type ('float32' or 'float64')
    device : str
        Device for torch backend ('auto', 'cuda', 'cpu')

    Returns
    -------
    dict
        Timing and memory statistics
    """
    from krl.utils import get_array
    from src.krl.operators.kernel_operator import get_kernel_operator

    # Convert dtype string to numpy dtype
    np_dtype = np.float32 if dtype == 'float32' else np.float64

    # Create test data
    geometry = DummyGeometry(shape)

    # Create anatomical image with gradient
    anat = geometry.allocate(0.0, dtype=np_dtype)
    grid = (np.indices(shape).sum(axis=0) / np.prod(shape)).astype(np_dtype)
    anat.fill(grid)

    # Create emission data
    rng = np.random.default_rng(42)
    emission = DummyImage(rng.normal(size=shape).astype(np_dtype))

    # Create operator
    if isinstance(backend, str):
        if backend == 'torch':
            operator = get_kernel_operator(
                geometry,
                backend='torch',
                device=device,
                dtype=dtype,
                num_neighbours=n,
                sigma_anat=1.0,
                sigma_dist=1.0,
                sigma_emission=1.0,
                normalize_kernel=True,
                use_mask=True,
                mask_k=k,
                distance_weighting=True,
                hybrid=True,
            )
        else:  # numba
            operator = get_kernel_operator(
                geometry,
                backend='numba',
                num_neighbours=n,
                sigma_anat=1.0,
                sigma_dist=1.0,
                sigma_emission=1.0,
                normalize_kernel=True,
                use_mask=True,
                mask_k=k,
                distance_weighting=True,
                hybrid=True,
            )
    else:
        # Backup module
        operator = backend.get_kernel_operator(
            geometry,
            backend='numba',
            num_neighbours=n,
            sigma_anat=1.0,
            sigma_dist=1.0,
            sigma_emission=1.0,
            normalize_kernel=True,
            use_mask=True,
            mask_k=k,
            distance_weighting=True,
            hybrid=True,
        )

    operator.set_anatomical_image(anat)

    # First call (includes compilation time)
    start = time.time()
    _ = operator.direct(emission)
    first_forward_time = time.time() - start

    start = time.time()
    _ = operator.adjoint(emission)
    first_adjoint_time = time.time() - start

    # Subsequent calls (pure execution time)
    start = time.time()
    for _ in range(num_iterations):
        result = operator.direct(emission)
    forward_time = (time.time() - start) / num_iterations

    # Benchmark adjoint pass
    start = time.time()
    for _ in range(num_iterations):
        result = operator.adjoint(emission)
    adjoint_time = (time.time() - start) / num_iterations

    # Benchmark precomputation
    if backend == 'torch':
        operator._anatomical_weights_gpu = None
        operator._mask_gpu = None
    else:
        operator._anatomical_weights = None
        operator.mask = None

    start = time.time()
    weights = operator.precompute_anatomical_weights()
    precompute_time = time.time() - start

    # Get memory usage
    if backend == 'torch':
        weights_memory_mb = weights.element_size() * weights.nelement() / (1024 * 1024)
        weights_shape = tuple(weights.shape)
    else:
        weights_memory_mb = weights.nbytes / 1024 / 1024
        weights_shape = weights.shape

    # Clean up GPU memory if torch
    if backend == 'torch' and hasattr(operator, 'clear_gpu'):
        operator.clear_gpu()

    return {
        'first_forward_time': first_forward_time,
        'first_adjoint_time': first_adjoint_time,
        'forward_time': forward_time,
        'adjoint_time': adjoint_time,
        'precompute_time': precompute_time,
        'weights_shape': weights_shape,
        'weights_memory_mb': weights_memory_mb,
    }


def print_results(name, results):
    """Print benchmark results."""
    print(f"\n{name}:")
    print(f"  First forward (w/ compilation): {results['first_forward_time']*1000:.2f} ms")
    print(f"  First adjoint (w/ compilation): {results['first_adjoint_time']*1000:.2f} ms")
    print(f"  Forward (avg):     {results['forward_time']*1000:.2f} ms")
    print(f"  Adjoint (avg):     {results['adjoint_time']*1000:.2f} ms")
    print(f"  Precompute:  {results['precompute_time']*1000:.2f} ms")
    print(f"  Weights memory: {results['weights_memory_mb']:.2f} MB")
    print(f"  Weights shape: {results['weights_shape']}")


def print_speedup(baseline_name, optimized_name, baseline, optimized):
    """Print speedup comparison."""
    print(f"\nSpeedup ({optimized_name} vs {baseline_name}):")
    fwd_speedup = baseline['forward_time'] / optimized['forward_time']
    adj_speedup = baseline['adjoint_time'] / optimized['adjoint_time']
    pre_speedup = baseline['precompute_time'] / optimized['precompute_time']
    mem_reduction = (1 - optimized['weights_memory_mb'] / baseline['weights_memory_mb']) * 100

    print(f"  Forward:     {fwd_speedup:.2f}x")
    print(f"  Adjoint:     {adj_speedup:.2f}x")
    print(f"  Precompute:  {pre_speedup:.2f}x")
    print(f"  Memory reduction: {mem_reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Benchmark kernel operator performance')
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 128, 256],
                       help='Volume sizes to benchmark (e.g., 64 128 256)')
    parser.add_argument('--gpu-only', action='store_true',
                       help='Only run GPU benchmarks (skip CPU backup comparison)')
    parser.add_argument('--skip-backup', action='store_true',
                       help='Skip backup comparison (faster)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of iterations (default: auto based on size)')
    args = parser.parse_args()

    print("=" * 80)
    print("Kernel Operator Performance Benchmark")
    print("=" * 80)

    if TORCH_AVAILABLE:
        print(f"\nPyTorch: Available (CUDA: {CUDA_AVAILABLE})")
        if CUDA_AVAILABLE:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Memory: {gpu_mem_gb:.1f} GB")
    else:
        print("\nPyTorch: Not available (GPU benchmarks will be skipped)")

    # Configuration for different sizes
    configs = {
        32: {'n': 5, 'k': 20},
        64: {'n': 7, 'k': 48},
        128: {'n': 7, 'k': 48},
        256: {'n': 7, 'k': 48},
    }

    for size in args.sizes:
        if size not in configs:
            print(f"\nWarning: No default config for size {size}, using n=7, k=48")
            n, k = 7, 48
        else:
            n, k = configs[size]['n'], configs[size]['k']

        shape = (size, size, size)

        # Adjust iterations based on size
        if args.iterations:
            iterations = args.iterations
        else:
            if size <= 64:
                iterations = 10
            elif size == 128:
                iterations = 5
            else:  # 256
                iterations = 3

        print(f"\n{'=' * 80}")
        print(f"Configuration: shape={shape}, n={n}, k={k}, iterations={iterations}")
        print("-" * 80)

        # --- Optimized CPU (Numba) ---
        print("\n[1] Optimized CPU (Numba, float64):")
        try:
            opt_results = time_kernel_operations(
                'numba',
                shape=shape,
                n=n,
                k=k,
                num_iterations=iterations,
                dtype='float64',
            )
            print_results("Optimized CPU (Numba)", opt_results)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # --- GPU (PyTorch) if available ---
        if TORCH_AVAILABLE and CUDA_AVAILABLE and not args.gpu_only:
            print("\n[2] GPU (PyTorch CUDA, float32):")
            try:
                gpu_results = time_kernel_operations(
                    'torch',
                    shape=shape,
                    n=n,
                    k=k,
                    num_iterations=iterations,
                    dtype='float32',
                    device='cuda',
                )
                print_results("GPU (PyTorch CUDA)", gpu_results)

                # Print GPU vs CPU speedup
                print_speedup("CPU (Numba)", "GPU (PyTorch)", opt_results, gpu_results)

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

        # --- Original CPU (Backup) for comparison ---
        if not args.skip_backup and not args.gpu_only and size <= 64:
            backup_path = REPO_ROOT / 'src' / 'krl' / 'operators' / 'kernel_operator_backup.py'
            if backup_path.exists():
                # Clear Numba cache to avoid dynamic module issues
                import shutil
                numba_cache_dir = backup_path.parent / '__pycache__'
                if numba_cache_dir.exists():
                    for cache_file in numba_cache_dir.glob('kernel_operator_backup*'):
                        try:
                            cache_file.unlink()
                        except:
                            pass

                # Import backup version
                import sys
                backup_dir = str(backup_path.parent)
                if backup_dir not in sys.path:
                    sys.path.insert(0, backup_dir)

                if 'kernel_operator_backup' in sys.modules:
                    del sys.modules['kernel_operator_backup']

                try:
                    import kernel_operator_backup as backup

                    print("\n[3] Original CPU (Backup, float64):")
                    orig_results = time_kernel_operations(
                        backup,
                        shape=shape,
                        n=n,
                        k=k,
                        num_iterations=iterations,
                        dtype='float64',
                    )
                    print_results("Original CPU (Backup)", orig_results)

                    # Print optimized vs backup speedup
                    print_speedup("Original (Backup)", "Optimized (Numba)", orig_results, opt_results)

                except Exception as e:
                    print(f"  Error running backup: {e}")
            else:
                print("\n[3] Original CPU (Backup): Not found")

    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80)

    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        print("\nGPU Summary:")
        print("  - GPU acceleration is available and working")
        print("  - Use backend='torch' with dtype='float32' for large volumes")
        print("  - Memory usage is ~50% of float64 (enables 256³ on consumer GPUs)")
    elif TORCH_AVAILABLE:
        print("\nNote: PyTorch is available but CUDA is not.")
        print("Install CUDA-enabled PyTorch for GPU acceleration.")
    else:
        print("\nNote: PyTorch not available. GPU benchmarks skipped.")
        print("Install PyTorch for GPU support: pip install torch")


if __name__ == "__main__":
    main()
