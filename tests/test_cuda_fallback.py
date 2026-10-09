"""CUDA availability tests for operator backend selection.

These tests confirm fallback behaviour when CUDA is unavailable and GPU
preference when it is available. Importing torch alongside CIL's native
libraries breaks OpenMP on some platforms, so torch-touching tests are opt-in
via this environment variable.
"""
import os

import pytest

if os.environ.get("KRL_RUN_GPU_TESTS") != "1":
    pytest.skip(
        "GPU tests disabled; set KRL_RUN_GPU_TESTS=1 to run them",
        allow_module_level=True,
    )

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from krl.operators.blurring import GaussianBlurringOperator
from krl.operators.kernel_operator import get_kernel_operator

CUDA_AVAILABLE = torch.cuda.is_available()


@dataclass
class MockGeometry:
    """Mock geometry for testing blurring operators."""
    voxel_size_x: float = 1.0
    voxel_size_y: float = 1.0
    voxel_size_z: float = 1.0
    shape: Tuple[int, int, int] = (10, 10, 10)

    def allocate(self, value: float = 0.0):
        return MockImage(np.full(self.shape, value, dtype=np.float64))


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


def test_blurring_auto_backend_prefers_torch_only_with_cuda():
    """backend='auto' selects torch on CUDA and falls back to CPU otherwise."""
    op = GaussianBlurringOperator((1.0, 1.0, 1.0), MockGeometry(), backend='auto')

    if CUDA_AVAILABLE:
        assert op.backend == 'torch'
    else:
        assert op.backend in ('numba', 'scipy')


@pytest.mark.skipif(CUDA_AVAILABLE, reason="CUDA available; no-CUDA fallback not exercised")
def test_blurring_explicit_torch_backend_falls_back_to_non_cuda_device():
    """Explicitly requesting torch must not require CUDA.

    The torch backend resolves cuda -> mps -> cpu, so construction should
    succeed and use a non-CUDA device when no GPU is present.
    """
    op = GaussianBlurringOperator((1.0, 1.0, 1.0), MockGeometry(), backend='torch')

    assert op.backend == 'torch'
    assert op.device in ('cpu', 'mps')

    result = op.direct(MockGeometry().allocate(1.0))
    assert result.shape == MockGeometry().shape
    assert np.all(np.isfinite(result.as_array()))


def test_resolve_device_prefers_cuda(monkeypatch):
    """CUDA wins over MPS when both are available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert GaussianBlurringOperator._resolve_device() == 'cuda'


def test_resolve_device_uses_mps_without_cuda(monkeypatch):
    """MPS is selected when CUDA is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert GaussianBlurringOperator._resolve_device() == 'mps'


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    """CPU is selected when neither CUDA nor MPS is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert GaussianBlurringOperator._resolve_device() == 'cpu'


def test_kernel_operator_auto_backend_without_cuda():
    """backend='auto' selects torch on CUDA and numba otherwise."""
    op = get_kernel_operator(MockGeometry(), backend='auto')

    if CUDA_AVAILABLE:
        assert op.backend in ('torch', 'numba')
    else:
        assert op.backend == 'numba'


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_blurring_auto_backend_with_cuda():
    """Test Gaussian blurring prefers torch backend when CUDA is available."""
    op = GaussianBlurringOperator((1.0, 1.0, 1.0), MockGeometry(), backend='auto')

    assert op.backend == 'torch', "Auto backend should select torch when CUDA is available"
    assert hasattr(op, "psf_t")
    assert op.psf_t.is_cuda, "Torch PSF tensor should be allocated on CUDA device"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_kernel_operator_auto_backend_with_cuda():
    """Test kernel operator uses GPU backend when CUDA is available."""
    op = get_kernel_operator(MockGeometry(), backend='auto')

    assert op.backend == 'torch', "Auto backend should choose torch when CUDA is available"
    assert hasattr(op, "device")
    assert op.device.type == 'cuda', "Torch kernel operator should target CUDA device"
