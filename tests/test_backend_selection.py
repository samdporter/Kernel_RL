import builtins
import platform
import sys
import types

import pytest
from cil.framework import ImageGeometry

from krl.operators.blurring import GaussianBlurringOperator, create_gaussian_blur
from krl.operators.kernel_operator import get_kernel_operator


def make_geometry():
    return ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=4)


@pytest.fixture
def torch_import_attempts(monkeypatch):
    """Forbid and record any attempt to import torch."""
    attempts = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            attempts.append(name)
            raise ImportError("torch import forbidden for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return attempts


def test_macos_auto_kernel_uses_numba_without_torch(monkeypatch, torch_import_attempts):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    operator = get_kernel_operator(make_geometry())

    assert operator.backend == "numba"
    assert torch_import_attempts == []


def test_macos_auto_blur_uses_numba_without_torch(monkeypatch, torch_import_attempts):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    operator = create_gaussian_blur((1.0, 1.0, 1.0), make_geometry())
    assert operator.backend == "numba"

    direct_operator = GaussianBlurringOperator((1.0, 1.0, 1.0), make_geometry())
    assert direct_operator.backend == "numba"

    assert torch_import_attempts == []


def test_linux_auto_kernel_probes_torch_then_falls_back_to_numba(monkeypatch, torch_import_attempts):
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    operator = get_kernel_operator(make_geometry())

    assert operator.backend == "numba"
    assert "torch" in torch_import_attempts


def test_linux_auto_blur_probes_torch_then_falls_back_to_numba(monkeypatch, torch_import_attempts):
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    operator = create_gaussian_blur((1.0, 1.0, 1.0), make_geometry())

    assert operator.backend == "numba"
    assert "torch" in torch_import_attempts


def test_linux_auto_kernel_uses_torch_when_cuda_available(monkeypatch):
    class StubTorchOperator:
        def __init__(self, domain_geometry, **kwargs):
            self.backend = "torch"

    fake_gpu = types.ModuleType("krl.operators.gpu_kernel_operator")
    fake_gpu.TorchKernelOperator = StubTorchOperator
    monkeypatch.setitem(sys.modules, "krl.operators.gpu_kernel_operator", fake_gpu)

    fake_torch = types.ModuleType("torch")
    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda.is_available = lambda: True
    fake_torch.cuda = fake_cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    operator = get_kernel_operator(make_geometry())

    assert operator.backend == "torch"


def test_linux_auto_blur_uses_torch_when_cuda_available(monkeypatch):
    class FakeTensor:
        def unsqueeze(self, *args, **kwargs):
            return self

        def to(self, *args, **kwargs):
            return self

    fake_torch = types.ModuleType("torch")
    fake_torch.tensor = lambda *args, **kwargs: FakeTensor()
    fake_torch.float32 = "float32"
    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda.is_available = lambda: True
    fake_torch.cuda = fake_cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    operator = create_gaussian_blur((1.0, 1.0, 1.0), make_geometry())

    assert operator.backend == "torch"
