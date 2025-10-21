"""Tests for GPU kernel operator (PyTorch backend)."""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

# Try importing torch to check if GPU tests should run
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Import after torch check
from src.krl.operators.kernel_operator import get_kernel_operator

CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()

# Skip all tests if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available - GPU kernel operator tests skipped"
)


@dataclass
class DummyGeometry:
    shape: Tuple[int, int, int]

    def allocate(self, value: float = 0.0):
        data = np.full(self.shape, value, dtype=np.float32)
        return DummyImage(data)


class DummyImage:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self._data.shape

    def as_array(self):
        return self._data

    def clone(self):
        return DummyImage(self._data.copy())

    def fill(self, values):
        self._data[...] = np.asarray(values, dtype=np.float32)


@pytest.fixture
def small_geometry():
    """Small test geometry (8x8x8)."""
    return DummyGeometry((8, 8, 8))


@pytest.fixture
def medium_geometry():
    """Medium test geometry (16x16x16)."""
    return DummyGeometry((16, 16, 16))


@pytest.fixture
def anatomical_uniform(small_geometry):
    return small_geometry.allocate(1.0)


@pytest.fixture
def anatomical_gradient(small_geometry):
    img = small_geometry.allocate(0.0)
    grad = np.indices(small_geometry.shape).sum(axis=0).astype(np.float32)
    # Normalize to [0, 1]
    grad = grad / grad.max()
    img.fill(grad)
    return img


@pytest.fixture
def emission_spike(small_geometry):
    arr = np.zeros(small_geometry.shape, dtype=np.float32)
    arr[4, 4, 4] = 10.0
    return DummyImage(arr)


@pytest.fixture
def emission_uniform(small_geometry):
    return small_geometry.allocate(1.0)


@pytest.fixture
def emission_random(small_geometry):
    rng = np.random.default_rng(42)
    arr = rng.normal(size=small_geometry.shape).astype(np.float32)
    return DummyImage(arr)

class TestBasicImports:
    """Test basic imports and availability of GPU operator."""

    def test_torch_import(self):
        """Test PyTorch is imported."""
        assert TORCH_AVAILABLE, "PyTorch should be available for GPU tests."
    
    def test_cuda_availability(self):
        """Test CUDA availability."""
        if TORCH_AVAILABLE:
            assert CUDA_AVAILABLE

    def test_gpu_operator_import(self):
        """Test GPU kernel operator can be imported."""
        op = get_kernel_operator(
            DummyGeometry((4, 4, 4)),
            backend='torch',
            dtype='float32',
            num_neighbours=3,
            mask_k=5,
        )
        assert op is not None
        assert op.backend == 'torch'


class TestGPUKernelOperatorBasics:
    """Test basic GPU kernel operator functionality."""

    def test_gpu_operator_creation(self, small_geometry):
        """Test GPU operator can be created."""
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            dtype='float32',
            num_neighbours=3,
            mask_k=10,
        )
        assert op.backend == 'torch'
        assert op.torch_dtype == torch.float32

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_device_selection(self, small_geometry):
        """Test GPU device is selected when available."""
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='auto',
        )
        assert op.device.type == 'cuda'

    def test_cpu_fallback(self, small_geometry):
        """Test CPU device works when forced."""
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='cpu',
        )
        assert op.device.type == 'cpu'

    def test_anatomical_image_setting(self, small_geometry, anatomical_gradient):
        """Test anatomical image can be set."""
        op = get_kernel_operator(small_geometry, backend='torch')
        op.set_anatomical_image(anatomical_gradient)
        assert op.anatomical_image is not None


class TestGPUMaskPrecomputation:
    """Test GPU mask precomputation."""

    def test_mask_precomputation_shape(self, small_geometry, anatomical_gradient):
        """Test mask has correct shape."""
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            num_neighbours=3,
            mask_k=10,
            use_mask=True,
        )
        op.set_anatomical_image(anatomical_gradient)
        mask = op.precompute_mask()

        s0, s1, s2 = small_geometry.shape
        k = 10
        assert mask.shape == (s0, s1, s2, k)
        assert mask.dtype == torch.int32

    def test_mask_indices_range(self, small_geometry, anatomical_gradient):
        """Test mask indices are within valid range."""
        n = 3
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            num_neighbours=n,
            mask_k=10,
            use_mask=True,
        )
        op.set_anatomical_image(anatomical_gradient)
        mask = op.precompute_mask()

        # Indices should be in [0, n³)
        total = n ** 3
        assert (mask >= 0).all()
        assert (mask < total).all()


class TestGPUWeightPrecomputation:
    """Test GPU weight precomputation."""

    def test_sparse_weights_shape(self, small_geometry, anatomical_gradient):
        """Test sparse weights have correct shape."""
        k = 10
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            num_neighbours=3,
            mask_k=k,
            use_mask=True,
        )
        op.set_anatomical_image(anatomical_gradient)
        weights = op.precompute_anatomical_weights()

        s0, s1, s2 = small_geometry.shape
        assert weights.shape == (s0, s1, s2, k)

    def test_dense_weights_shape(self, small_geometry, anatomical_gradient):
        """Test dense weights have correct shape."""
        n = 3
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            num_neighbours=n,
            use_mask=False,
        )
        op.set_anatomical_image(anatomical_gradient)
        weights = op.precompute_anatomical_weights()

        s0, s1, s2 = small_geometry.shape
        total = n ** 3
        assert weights.shape == (s0, s1, s2, total)

    def test_weights_positive(self, small_geometry, anatomical_gradient):
        """Test weights are non-negative."""
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            num_neighbours=3,
            mask_k=10,
        )
        op.set_anatomical_image(anatomical_gradient)
        weights = op.precompute_anatomical_weights()

        assert (weights >= 0).all()


class TestGPUForwardAdjoint:
    """Test GPU forward and adjoint operations."""

    def test_forward_pass_shape(self, small_geometry, anatomical_gradient, emission_uniform):
        """Test forward pass produces correct output shape."""
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            num_neighbours=3,
            mask_k=10,
        )
        op.set_anatomical_image(anatomical_gradient)

        result = op.direct(emission_uniform)
        assert result.shape == emission_uniform.shape

    def test_adjoint_pass_shape(self, small_geometry, anatomical_gradient, emission_uniform):
        """Test adjoint pass produces correct output shape."""
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            num_neighbours=3,
            mask_k=10,
        )
        op.set_anatomical_image(anatomical_gradient)

        # Need to call direct first for normalize_kernel
        _ = op.direct(emission_uniform)
        result = op.adjoint(emission_uniform)
        assert result.shape == emission_uniform.shape

    def test_uniform_kernel_identity(self, small_geometry, anatomical_uniform, emission_uniform):
        """Test uniform anatomical image with normalized kernel acts as identity."""
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            num_neighbours=3,
            normalize_kernel=True,
            use_mask=False,
        )
        op.set_anatomical_image(anatomical_uniform)

        result = op.direct(emission_uniform)
        result_arr = result.as_array()
        expected = emission_uniform.as_array()

        # Should be very close to identity
        np.testing.assert_allclose(result_arr, expected, rtol=1e-4)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGPUvsCPUConsistency:
    """Test GPU results match CPU results."""

    def test_forward_consistency(self, small_geometry, anatomical_gradient, emission_random):
        """Test GPU forward matches CPU forward."""
        # Create CPU operator
        op_cpu = get_kernel_operator(
            small_geometry,
            backend='numba',
            num_neighbours=3,
            mask_k=10,
            sigma_anat=0.1,
            sigma_emission=0.1,
        )
        op_cpu.set_anatomical_image(anatomical_gradient)

        # Create GPU operator
        op_gpu = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='cpu',  # Use CPU for deterministic comparison
            dtype='float32',
            num_neighbours=3,
            mask_k=10,
            sigma_anat=0.1,
            sigma_emission=0.1,
        )

        # Need to convert anatomical to float32 for GPU
        anat_f32 = DummyImage(anatomical_gradient.as_array().astype(np.float32))
        op_gpu.set_anatomical_image(anat_f32)

        # Run forward pass
        result_cpu = op_cpu.direct(emission_random)
        result_gpu = op_gpu.direct(emission_random)

        # Compare (allow some tolerance for float32 vs float64)
        cpu_arr = result_cpu.as_array().astype(np.float32)
        gpu_arr = result_gpu.as_array()

        np.testing.assert_allclose(gpu_arr, cpu_arr, rtol=1e-3, atol=1e-5)

    def test_adjoint_consistency(self, small_geometry, anatomical_gradient, emission_random):
        """Test GPU adjoint matches CPU adjoint."""
        # Create CPU operator
        op_cpu = get_kernel_operator(
            small_geometry,
            backend='numba',
            num_neighbours=3,
            mask_k=10,
            sigma_anat=0.1,
            sigma_emission=0.1,
        )
        op_cpu.set_anatomical_image(anatomical_gradient)

        # Create GPU operator
        op_gpu = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='cpu',
            dtype='float32',
            num_neighbours=3,
            mask_k=10,
            sigma_anat=0.1,
            sigma_emission=0.1,
        )
        anat_f32 = DummyImage(anatomical_gradient.as_array().astype(np.float32))
        op_gpu.set_anatomical_image(anat_f32)

        # Run forward first (needed for normalization)
        _ = op_cpu.direct(emission_random)
        _ = op_gpu.direct(emission_random)

        # Run adjoint pass
        result_cpu = op_cpu.adjoint(emission_random)
        result_gpu = op_gpu.adjoint(emission_random)

        # Compare
        cpu_arr = result_cpu.as_array().astype(np.float32)
        gpu_arr = result_gpu.as_array()

        np.testing.assert_allclose(gpu_arr, cpu_arr, rtol=1e-3, atol=1e-5)

    def test_hybrid_mode_consistency(self, small_geometry, anatomical_gradient, emission_random):
        """Test GPU hybrid mode matches CPU hybrid mode."""
        # Create CPU operator
        op_cpu = get_kernel_operator(
            small_geometry,
            backend='numba',
            num_neighbours=3,
            mask_k=10,
            hybrid=True,
            sigma_anat=0.1,
            sigma_emission=0.1,
        )
        op_cpu.set_anatomical_image(anatomical_gradient)

        # Create GPU operator
        op_gpu = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='cpu',
            dtype='float32',
            num_neighbours=3,
            mask_k=10,
            hybrid=True,
            sigma_anat=0.1,
            sigma_emission=0.1,
        )
        anat_f32 = DummyImage(anatomical_gradient.as_array().astype(np.float32))
        op_gpu.set_anatomical_image(anat_f32)

        # Run forward pass (HKRL)
        result_cpu = op_cpu.direct(emission_random)
        result_gpu = op_gpu.direct(emission_random)

        # Compare
        cpu_arr = result_cpu.as_array().astype(np.float32)
        gpu_arr = result_gpu.as_array()

        np.testing.assert_allclose(gpu_arr, cpu_arr, rtol=1e-3, atol=1e-5)


class TestKernelParameterEffectsGPU:
    """Ensure sigma parameters modulate the GPU kernel behaviour."""

    def test_sigma_anatomical_parameter_changes_weights(
        self, small_geometry, anatomical_gradient, emission_random
    ):
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='cpu',
            dtype='float32',
            num_neighbours=3,
            sigma_anat=0.1,
            sigma_dist=1.0,
            normalize_kernel=False,
            use_mask=False,
            distance_weighting=False,
            hybrid=False,
        )
        op.set_anatomical_image(anatomical_gradient)

        res_narrow = op.direct(emission_random).as_array()
        op.set_parameters({"sigma_anat": 5.0})
        res_wide = op.direct(emission_random).as_array()

        assert not np.allclose(res_narrow, res_wide, atol=1e-6, rtol=1e-5)
        assert float(np.linalg.norm(res_narrow - res_wide)) > 1e-3

    def test_sigma_distance_parameter_requires_distance_weighting(
        self, small_geometry, anatomical_gradient, emission_random
    ):
        op_no_dist = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='cpu',
            dtype='float32',
            num_neighbours=3,
            sigma_anat=0.2,
            sigma_dist=0.1,
            normalize_kernel=False,
            use_mask=False,
            distance_weighting=False,
            hybrid=False,
        )
        op_no_dist.set_anatomical_image(anatomical_gradient)

        res_no_dist_tight = op_no_dist.direct(emission_random).as_array()
        op_no_dist.set_parameters({"sigma_dist": 5.0})
        res_no_dist_wide = op_no_dist.direct(emission_random).as_array()

        assert np.allclose(res_no_dist_tight, res_no_dist_wide, atol=1e-7, rtol=1e-6)

        op_dist = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='cpu',
            dtype='float32',
            num_neighbours=3,
            sigma_anat=0.2,
            sigma_dist=0.1,
            normalize_kernel=False,
            use_mask=False,
            distance_weighting=True,
            hybrid=False,
        )
        op_dist.set_anatomical_image(anatomical_gradient)

        res_dist_tight = op_dist.direct(emission_random).as_array()
        op_dist.set_parameters({"sigma_dist": 5.0})
        res_dist_wide = op_dist.direct(emission_random).as_array()

        assert not np.allclose(res_dist_tight, res_dist_wide, atol=1e-6, rtol=1e-5)
        assert float(np.linalg.norm(res_dist_tight - res_dist_wide)) > 1e-3

    def test_sigma_emission_parameter_affects_hybrid_kernel(
        self, small_geometry, anatomical_gradient, emission_random
    ):
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='cpu',
            dtype='float32',
            num_neighbours=3,
            sigma_anat=0.2,
            sigma_dist=1.0,
            sigma_emission=0.1,
            normalize_kernel=False,
            use_mask=False,
            distance_weighting=False,
            hybrid=True,
        )
        op.set_anatomical_image(anatomical_gradient)

        res_emission_tight = op.direct(emission_random).as_array()
        op.set_parameters({"sigma_emission": 5.0})
        res_emission_wide = op.direct(emission_random).as_array()

        assert not np.allclose(res_emission_tight, res_emission_wide, atol=1e-6, rtol=1e-5)
        assert float(np.linalg.norm(res_emission_tight - res_emission_wide)) > 1e-3


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUMemoryManagement:
    """Test GPU memory management."""

    def test_memory_cleanup(self, medium_geometry, anatomical_gradient):
        """Test GPU memory is released after operation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Record initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated()

        # Create operator and run operations
        op = get_kernel_operator(
            medium_geometry,
            backend='torch',
            device='cuda',
            dtype='float32',
            num_neighbours=5,
            mask_k=20,
        )

        # Create larger anatomical for medium geometry
        anat = medium_geometry.allocate(0.0)
        grad = np.indices(medium_geometry.shape).sum(axis=0).astype(np.float32)
        grad = grad / grad.max()
        anat.fill(grad)

        op.set_anatomical_image(anat)

        # Run operations
        emission = medium_geometry.allocate(1.0)
        _ = op.direct(emission)

        # Clear GPU
        op.clear_gpu()

        # Memory should be released
        final_mem = torch.cuda.memory_allocated()
        # Some memory may remain but should be much less than peak
        peak_mem = torch.cuda.max_memory_allocated()
        assert final_mem < peak_mem * 0.5  # At least 50% should be freed


class TestAutoBackendSelection:
    """Test automatic backend selection."""

    def test_auto_backend_selection(self, small_geometry):
        """Test auto backend selects appropriate backend."""
        op = get_kernel_operator(
            small_geometry,
            backend='auto',
        )

        # Should select torch if available, otherwise numba
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            assert op.backend == 'torch'
        else:
            # Will fall back to numba or torch CPU
            assert op.backend in ['torch', 'numba']

    def test_explicit_backend_override(self, small_geometry):
        """Test explicit backend selection overrides auto."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='cpu',
        )
        assert op.backend == 'torch'
        assert op.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
