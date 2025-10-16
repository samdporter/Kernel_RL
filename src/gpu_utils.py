"""
GPU detection and device management utilities.
Automatically detects GPU availability and provides fallback to CPU.
"""

import warnings
from typing import Tuple, Optional


class DeviceManager:
    """Manages device selection (GPU/CPU) with automatic detection and fallback."""

    def __init__(self):
        self._torch_available = False
        self._torch_cuda_available = False
        self._cupy_available = False
        self._numba_cuda_available = False
        self._device = "cpu"

        self._detect_capabilities()

    def _detect_capabilities(self):
        """Detect available GPU libraries and capabilities."""
        # Check PyTorch
        try:
            import torch
            self._torch_available = True
            self._torch_cuda_available = torch.cuda.is_available()
            if self._torch_cuda_available:
                self._device = "cuda"
        except ImportError:
            pass

        # Check CuPy
        try:
            import cupy as cp
            # Try to access GPU to verify it works
            _ = cp.cuda.Device(0)
            self._cupy_available = True
        except (ImportError, Exception):
            pass

        # Check Numba CUDA
        try:
            from numba import cuda
            if cuda.is_available():
                self._numba_cuda_available = True
        except (ImportError, Exception):
            pass

    @property
    def has_gpu(self) -> bool:
        """Returns True if any GPU acceleration is available."""
        return self._torch_cuda_available or self._cupy_available or self._numba_cuda_available

    @property
    def has_torch_gpu(self) -> bool:
        """Returns True if PyTorch GPU is available."""
        return self._torch_cuda_available

    @property
    def has_cupy(self) -> bool:
        """Returns True if CuPy is available."""
        return self._cupy_available

    @property
    def has_numba_cuda(self) -> bool:
        """Returns True if Numba CUDA is available."""
        return self._numba_cuda_available

    @property
    def device(self) -> str:
        """Returns the default device string ('cuda' or 'cpu')."""
        return self._device

    def get_torch_device(self):
        """Get PyTorch device object."""
        if not self._torch_available:
            raise ImportError("PyTorch is not installed")
        import torch
        return torch.device(self._device)

    def get_array_module(self, prefer_gpu: bool = True):
        """
        Get the appropriate array module (cupy or numpy).

        Args:
            prefer_gpu: If True and GPU is available, return cupy. Otherwise return numpy.

        Returns:
            cupy or numpy module
        """
        if prefer_gpu and self._cupy_available:
            import cupy as cp
            return cp
        else:
            import numpy as np
            return np

    def print_capabilities(self):
        """Print available GPU capabilities."""
        print("=" * 60)
        print("GPU CAPABILITIES")
        print("=" * 60)

        if self.has_gpu:
            print(f"✓ GPU acceleration: AVAILABLE")
            print(f"  Default device: {self._device}")
        else:
            print(f"✗ GPU acceleration: NOT AVAILABLE")
            print(f"  Default device: {self._device}")

        print()
        print("Libraries:")
        print(f"  PyTorch:      {'✓ ' if self._torch_available else '✗ '}installed", end="")
        if self._torch_available:
            print(f" | GPU: {'✓ available' if self._torch_cuda_available else '✗ not available'}")
            if self._torch_cuda_available:
                import torch
                print(f"    - CUDA version: {torch.version.cuda}")
                print(f"    - GPU count: {torch.cuda.device_count()}")
                print(f"    - GPU name: {torch.cuda.get_device_name(0)}")
        else:
            print()

        print(f"  CuPy:         {'✓ available' if self._cupy_available else '✗ not available'}")
        if self._cupy_available:
            import cupy as cp
            print(f"    - CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
            print(f"    - GPU name: {cp.cuda.Device(0).compute_capability}")

        print(f"  Numba CUDA:   {'✓ available' if self._numba_cuda_available else '✗ not available'}")
        if self._numba_cuda_available:
            from numba import cuda
            print(f"    - GPU count: {len(cuda.gpus)}")

        print(f"  Numba (CPU):  ", end="")
        try:
            import numba
            print(f"✓ installed (v{numba.__version__})")
        except ImportError:
            print("✗ not installed")

        print("=" * 60)

        if not self.has_gpu:
            print("\nℹ To enable GPU acceleration:")
            print("  1. Ensure NVIDIA GPU drivers are installed")
            print("  2. Install CUDA toolkit")
            print("  3. Rebuild Docker container: ./docker-run.sh build")
            print("=" * 60)


# Global device manager instance
_device_manager = None


def get_device_manager() -> DeviceManager:
    """Get the global DeviceManager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def get_device() -> str:
    """Get the default device string ('cuda' or 'cpu')."""
    return get_device_manager().device


def has_gpu() -> bool:
    """Check if GPU acceleration is available."""
    return get_device_manager().has_gpu


def get_array_module(prefer_gpu: bool = True):
    """
    Get the appropriate array module (cupy or numpy).

    Args:
        prefer_gpu: If True and GPU is available, return cupy. Otherwise return numpy.

    Returns:
        cupy or numpy module
    """
    return get_device_manager().get_array_module(prefer_gpu)


def print_gpu_info():
    """Print GPU capabilities and configuration."""
    get_device_manager().print_capabilities()


if __name__ == "__main__":
    # Print GPU info when run as a script
    print_gpu_info()
