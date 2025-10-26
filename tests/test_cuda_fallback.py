"""Test CUDA availability fallback behavior for operators.

This test ensures that when PyTorch is installed but CUDA is not available,
the operators correctly fall back to CPU backends (numba/scipy) instead of
crashing with "No CUDA GPUs are available" errors.
"""
import pytest
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MockGeometry:
    """Mock geometry for testing blurring operators."""
    voxel_size_x: float = 1.0
    voxel_size_y: float = 1.0
    voxel_size_z: float = 1.0
    shape: Tuple[int, int, int] = (10, 10, 10)

    def allocate(self, value: float = 0.0):
        data = np.full(self.shape, value, dtype=np.float64)
        return MockImage(data)


class MockImage:
    """Mock image for testing operators."""
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=np.float64)

    @property
    def shape(self):
        return self._data.shape

    def as_array(self):
        return self._data

    def clone(self):
        return MockImage(self._data.copy())

    def fill(self, values):
        self._data[...] = np.asarray(values, dtype=np.float64)


def test_blurring_auto_backend_without_cuda():
    """Test that GaussianBlurringOperator with backend='auto' falls back to CPU when CUDA unavailable."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pytest.skip("PyTorch not installed, skipping CUDA fallback test")

    from krl.operators.blurring import GaussianBlurringOperator

    geometry = MockGeometry()

    # Create operator with auto backend - should not crash
    op = GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend='auto')

    if cuda_available:
        # If CUDA is available, torch backend should be selected
        assert op.backend == 'torch', "Expected torch backend when CUDA available"
    else:
        # If CUDA is not available, should fall back to numba or scipy
        assert op.backend in ['numba', 'scipy'], (
            f"Expected numba or scipy backend when CUDA unavailable, got {op.backend}"
        )
        print(f"✓ Auto backend correctly fell back to: {op.backend}")


def test_blurring_explicit_torch_backend_without_cuda():
    """Test that explicitly requesting torch backend raises clear error when CUDA unavailable."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pytest.skip("PyTorch not installed, skipping CUDA fallback test")

    if cuda_available:
        pytest.skip("CUDA is available, skipping no-CUDA error test")

    from krl.operators.blurring import GaussianBlurringOperator

    geometry = MockGeometry()

    # Explicitly requesting torch backend should raise RuntimeError
    with pytest.raises(RuntimeError, match="no CUDA GPUs available"):
        GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend='torch')

    print("✓ Torch backend correctly raises error when CUDA unavailable")


def test_kernel_operator_auto_backend_without_cuda():
    """Test that kernel operator with backend='auto' falls back to CPU when CUDA unavailable."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pytest.skip("PyTorch not installed, skipping CUDA fallback test")

    from krl.operators.kernel_operator import get_kernel_operator

    geometry = MockGeometry()

    # Create operator with auto backend - should not crash
    try:
        op = get_kernel_operator(geometry, backend='auto')

        if cuda_available:
            # If CUDA is available, torch backend might be selected
            assert op.backend in ['torch', 'numba'], f"Unexpected backend: {op.backend}"
        else:
            # If CUDA is not available, should use numba
            assert op.backend == 'numba', (
                f"Expected numba backend when CUDA unavailable, got {op.backend}"
            )
            print(f"✓ Kernel operator correctly fell back to: {op.backend}")
    except RuntimeError as e:
        if "Numba backend not available" in str(e):
            pytest.skip("Numba not available, skipping kernel operator test")
        raise


if __name__ == "__main__":
    # Run tests when executed directly
    print("Testing CUDA fallback behavior...")
    print()

    try:
        test_blurring_auto_backend_without_cuda()
    except Exception as e:
        print(f"✗ test_blurring_auto_backend_without_cuda failed: {e}")

    try:
        test_blurring_explicit_torch_backend_without_cuda()
    except Exception as e:
        print(f"✗ test_blurring_explicit_torch_backend_without_cuda failed: {e}")

    try:
        test_kernel_operator_auto_backend_without_cuda()
    except Exception as e:
        print(f"✗ test_kernel_operator_auto_backend_without_cuda failed: {e}")

    print()
    print("All tests completed!")
