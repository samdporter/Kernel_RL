"""CPU tests for examples/scripts/benchmark_kernel_performance.py."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "scripts"
    / "benchmark_kernel_performance.py"
)

SHAPE = (8, 8, 8)
N = 3
K = 8


def load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_kernel_performance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_anatomical_gradient(geometry):
    anat = geometry.allocate(0.0, dtype=np.float64)
    grid = (np.indices(SHAPE).sum(axis=0) / np.prod(SHAPE)).astype(np.float64)
    anat.fill(grid)
    return anat


def run_numba_benchmark():
    bench = load_benchmark()
    return bench, bench.time_kernel_operations(
        "numba", shape=SHAPE, n=N, k=K, num_iterations=1, dtype="float64"
    )


def test_numba_benchmark_reports_weights():
    _, results = run_numba_benchmark()

    assert results["weights_shape"] == SHAPE + (K,)
    expected_mb = np.prod(SHAPE) * K * np.dtype(np.float64).itemsize / 1024**2
    assert results["weights_memory_mb"] == pytest.approx(expected_mb)


def test_anatomy_reapplied_before_precomputation(monkeypatch):
    from krl.operators import kernel_operator

    class SpyOperator:
        """Record cache state around each set_anatomical_image call."""

        def __init__(self, real):
            self._real = real
            self.set_calls = 0
            self.states_before_set = []
            self.state_at_precompute = None

        def __getattr__(self, name):
            return getattr(self._real, name)

        def set_anatomical_image(self, image):
            self.states_before_set.append(
                (self._real.mask, self._real._anatomical_weights)
            )
            self.set_calls += 1
            self._real.set_anatomical_image(image)

        def precompute_anatomical_weights(self):
            self.state_at_precompute = (
                self._real.mask,
                self._real._anatomical_weights,
            )
            return self._real.precompute_anatomical_weights()

    spies = []
    original = kernel_operator.get_kernel_operator

    def factory(*args, **kwargs):
        spy = SpyOperator(original(*args, **kwargs))
        spies.append(spy)
        return spy

    monkeypatch.setattr(kernel_operator, "get_kernel_operator", factory)
    run_numba_benchmark()

    assert len(spies) == 1
    spy = spies[0]
    # Once when the operator is created, once to reset it before timing.
    assert spy.set_calls == 2
    assert spy.states_before_set[0] == (None, None)
    mask_before, weights_before = spy.states_before_set[1]
    assert mask_before is not None
    assert weights_before is not None
    assert spy.state_at_precompute == (None, None)


def test_import_does_not_load_torch():
    code = (
        "import importlib.util\n"
        "import sys\n"
        f"spec = importlib.util.spec_from_file_location('bench', {str(SCRIPT)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "print('TORCH_LOADED=' + str('torch' in sys.modules))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert proc.returncode == 0, proc.stderr
    assert "TORCH_LOADED=False" in proc.stdout


@pytest.mark.skipif(
    os.environ.get("KRL_RUN_GPU_TESTS") != "1",
    reason="torch tests disabled; set KRL_RUN_GPU_TESTS=1",
)
def test_set_anatomical_image_resets_torch_cpu_caches():
    # torch and CIL's native libraries each load libomp; on macOS the second
    # copy aborts the process unless this escape hatch is set before torch import.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from krl.operators.gpu_kernel_operator import TorchKernelOperator

    bench = load_benchmark()
    geometry = bench.DummyGeometry(SHAPE)
    operator = TorchKernelOperator(
        geometry,
        device="cpu",
        dtype="float64",
        num_neighbours=N,
        sigma_anat=1.0,
        sigma_dist=1.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        use_mask=True,
        mask_k=K,
        distance_weighting=True,
        hybrid=True,
    )
    operator.set_anatomical_image(make_anatomical_gradient(geometry))
    operator.precompute_mask()
    operator.precompute_anatomical_weights()

    assert operator.mask is not None
    assert operator._anatomical_weights is not None
    assert operator._mask_cpu is not None

    operator.set_anatomical_image(make_anatomical_gradient(geometry))

    assert operator.mask is None
    assert operator._anatomical_weights is None
    assert operator._mask_cpu is None
    assert operator._mask_gpu is None
    assert operator._cpu_operator.mask is None
    assert operator._cpu_operator._anatomical_weights is None
