"""Tests for GPU kernel operator (PyTorch backend)."""

import os

import pytest

# Importing torch alongside CIL's native libraries breaks OpenMP on some
# platforms, so torch-touching tests are opt-in via this environment variable.
if os.environ.get("KRL_RUN_GPU_TESTS") != "1":
    pytest.skip(
        "GPU tests disabled; set KRL_RUN_GPU_TESTS=1 to run them",
        allow_module_level=True,
    )

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from cil.framework import ImageGeometry

from krl.operators.kernel_operator import get_kernel_operator

CUDA_AVAILABLE = torch.cuda.is_available()

DEVICE_PARAMS = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
    ),
]


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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
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


class TestGPUAdjointCorrectness:
    """Adjoint-specific correctness checks."""

    @pytest.mark.parametrize("device", DEVICE_PARAMS)
    @pytest.mark.parametrize("use_mask", [True, False])
    def test_inner_product_matches_adjoint(
        self, small_geometry, anatomical_gradient, use_mask, device
    ):
        """Check ⟨Ax, y⟩ equals ⟨x, Aᵀy⟩ for masked and dense modes."""
        params = dict(
            backend="torch",
            device=device,
            dtype="float32",
            num_neighbours=5,
            sigma_anat=0.2,
            sigma_emission=0.1,
            normalize_kernel=True,
            use_mask=use_mask,
        )
        if use_mask:
            params["mask_k"] = 20

        op = get_kernel_operator(small_geometry, **params)
        op.set_anatomical_image(anatomical_gradient)

        rng = np.random.default_rng(123)
        x_arr = rng.normal(size=small_geometry.shape).astype(np.float32)
        y_arr = rng.normal(size=small_geometry.shape).astype(np.float32)

        x_img = DummyImage(x_arr)
        y_img = DummyImage(y_arr)

        ax = op.direct(x_img)
        lhs = float(np.vdot(ax.as_array().ravel(), y_arr.ravel()))

        adj_y = op.adjoint(y_img)
        rhs = float(np.vdot(x_arr.ravel(), adj_y.as_array().ravel()))

        denom = max(abs(lhs), abs(rhs), 1.0)
        assert abs(lhs - rhs) / denom < 5e-4

    @pytest.mark.parametrize("device", DEVICE_PARAMS)
    def test_normalisation_map_precision(self, small_geometry, anatomical_gradient, device):
        """Normalization map should preserve forward precision (no float16 downcast)."""
        op = get_kernel_operator(
            small_geometry,
            backend="torch",
            device=device,
            dtype="float32",
            num_neighbours=5,
            mask_k=20,
            use_mask=True,
            normalize_kernel=True,
            sigma_anat=0.2,
            sigma_emission=0.1,
        )
        op.set_anatomical_image(anatomical_gradient)

        img = small_geometry.allocate(1.0)
        _ = op.direct(img)

        assert op._normalisation_map is not None
        assert op._normalisation_map.dtype == np.float32


class TestBoundedDenseWeightSum:
    """Regression for adjoint-first normalisation in dense mode.

    The adjoint bootstrap must compute the per-voxel weight sum without
    materialising the full (s0, s1, s2, n³) weight tensor. These run on the
    torch CPU device; the CUDA path is covered by the CUDA-only tests.
    """

    def test_bounded_dense_sum_matches_materialised(self, small_geometry, anatomical_gradient):
        op = get_kernel_operator(
            small_geometry,
            backend="torch",
            device="cpu",
            dtype="float32",
            num_neighbours=3,
            use_mask=False,
            sigma_anat=0.5,
            sigma_dist=1.0,
            distance_weighting=True,
        )
        op.set_anatomical_image(anatomical_gradient)
        anat = op._validate_anatomical_image()
        anat_tensor = torch.from_numpy(
            np.ascontiguousarray(anat, dtype=op.numpy_dtype)
        ).to(op.device)

        bounded = op._torch_precompute_anatomical_weight_sum_dense(
            anat_tensor,
            3,
            op.parameters["sigma_anat"],
            op.parameters["sigma_dist"],
            op.parameters["distance_weighting"],
        )
        full = op._torch_precompute_anatomical_weights_dense(
            anat_tensor,
            3,
            op.parameters["sigma_anat"],
            op.parameters["sigma_dist"],
            op.parameters["distance_weighting"],
        ).sum(dim=-1)

        assert bounded.shape == small_geometry.shape
        torch.testing.assert_close(bounded, full, rtol=1e-5, atol=1e-6)

    def test_fixed_dense_weight_sum_avoids_full_weight_tensor(
        self, small_geometry, anatomical_gradient, monkeypatch
    ):
        op = get_kernel_operator(
            small_geometry,
            backend="torch",
            device="cpu",
            dtype="float32",
            num_neighbours=3,
            use_mask=False,
        )
        op.set_anatomical_image(anatomical_gradient)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("dense full weight tensor was materialised")

        monkeypatch.setattr(op, "precompute_anatomical_weights", fail_if_called)

        wsum = op._torch_anatomical_weight_sum_fixed()
        assert wsum.shape == small_geometry.shape
        assert torch.isfinite(wsum).all()


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


def make_cil_geometry(shape=(6, 6, 6), dtype=np.float64):
    z, y, x = shape
    return ImageGeometry(voxel_num_x=x, voxel_num_y=y, voxel_num_z=z, dtype=dtype)


def make_cil_image(geometry, array):
    image = geometry.allocate()
    image.fill(np.asarray(array, dtype=geometry.dtype))
    return image


class TestTorchCPUFallbackDelegate:
    """Regressions for the device='cpu' delegate to the numba KernelOperator.

    These run on the torch CPU delegate and do not exercise the CUDA kernels.
    """

    def test_mixed_dtype_forward_preserves_input_dtype(self):
        """float64 data with a float32 anatomy must round-trip through the delegate."""
        geometry = make_cil_geometry(dtype=np.float32)
        operator = get_kernel_operator(
            geometry,
            backend='torch',
            device='cpu',
            dtype='float32',
            num_neighbours=3,
            sigma_anat=0.4,
            normalize_kernel=True,
            normalize_features=False,
            use_mask=True,
            mask_k=10,
        )
        anatomy = geometry.allocate()
        anatomy.fill(np.random.default_rng(3).normal(size=geometry.shape).astype(np.float32))
        operator.set_anatomical_image(anatomy)

        data_geometry = make_cil_geometry(shape=geometry.shape, dtype=np.float64)
        rng = np.random.default_rng(5)
        x = make_cil_image(data_geometry, rng.normal(size=geometry.shape))
        y = make_cil_image(data_geometry, rng.normal(size=geometry.shape))

        forward = operator.direct(x).as_array()
        assert forward.dtype == np.float64

        dot_forward = float(np.sum(forward * y.as_array()))
        dot_adjoint = float(np.sum(x.as_array() * operator.adjoint(y).as_array()))
        assert np.isclose(dot_forward, dot_adjoint, atol=1e-10, rtol=1e-8)

    def test_normalized_hybrid_forward_adjoint_dot_product(self):
        """Normalized hybrid forward/adjoint must satisfy the dot-product identity."""
        geometry = make_cil_geometry()
        operator = get_kernel_operator(
            geometry,
            backend='torch',
            device='cpu',
            dtype='float64',
            num_neighbours=5,
            sigma_anat=0.5,
            sigma_dist=1.0,
            sigma_emission=0.5,
            normalize_kernel=True,
            use_mask=False,
            hybrid=True,
        )
        grid = np.indices(geometry.shape).sum(axis=0) / float(np.prod(geometry.shape))
        operator.set_anatomical_image(make_cil_image(geometry, grid))

        rng = np.random.default_rng(7)
        x = make_cil_image(geometry, rng.normal(size=geometry.shape))
        y = make_cil_image(geometry, rng.normal(size=geometry.shape))

        forward = operator.direct(x).as_array()
        adjoint = operator.adjoint(y).as_array()

        dot_forward = float(np.sum(forward * y.as_array()))
        dot_adjoint = float(np.sum(x.as_array() * adjoint))
        assert np.isclose(dot_forward, dot_adjoint, atol=1e-6, rtol=1e-5)

    def test_hybrid_frozen_reference_round_trips(self):
        """The frozen emission reference set via the delegate survives forward/adjoint."""
        geometry = make_cil_geometry()
        operator = get_kernel_operator(
            geometry,
            backend='torch',
            device='cpu',
            dtype='float64',
            num_neighbours=3,
            sigma_anat=1.0,
            sigma_emission=1.0,
            normalize_kernel=True,
            use_mask=False,
            hybrid=True,
        )
        operator.set_anatomical_image(make_cil_image(geometry, np.indices(geometry.shape).sum(axis=0)))

        rng = np.random.default_rng(11)
        emission_v1 = make_cil_image(geometry, rng.uniform(50, 150, geometry.shape))
        emission_v2 = make_cil_image(geometry, rng.uniform(50, 150, geometry.shape))

        operator.freeze_emission_kernel = True
        operator.direct(emission_v1)
        frozen = operator.frozen_emission_kernel.copy()
        np.testing.assert_array_equal(frozen, emission_v1.as_array())

        operator.direct(emission_v2)
        np.testing.assert_array_equal(operator.frozen_emission_kernel, frozen)

        operator.adjoint(geometry.allocate(1.0))
        np.testing.assert_array_equal(operator.frozen_emission_kernel, frozen)
        assert not np.allclose(frozen, emission_v2.as_array(), rtol=1e-10)

    def test_masked_anatomical_weights_on_cpu(self):
        """Masked weight precomputation on the CPU delegate must work and stay cached."""
        geometry = make_cil_geometry()
        operator = get_kernel_operator(
            geometry,
            backend='torch',
            device='cpu',
            dtype='float64',
            num_neighbours=3,
            sigma_anat=0.4,
            use_mask=True,
            mask_k=10,
        )
        operator.set_anatomical_image(make_cil_image(geometry, np.indices(geometry.shape).sum(axis=0)))

        weights = operator.precompute_anatomical_weights()

        s0, s1, s2 = geometry.shape
        assert weights.shape == (s0, s1, s2, 10)
        assert weights.dtype == torch.float64
        assert torch.isfinite(weights).all()

        # The caches must stay consistent for subsequent forward/adjoint calls.
        emission = make_cil_image(geometry, np.ones(geometry.shape))
        assert np.all(np.isfinite(operator.direct(emission).as_array()))
        assert np.all(np.isfinite(operator.adjoint(emission).as_array()))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_dtype_forward_adjoint_dot_product(dtype):
    """CUDA-only dtype regression for the real torch kernels (not the CPU delegate)."""
    geometry = make_cil_geometry(dtype=np.float64)
    operator = get_kernel_operator(
        geometry,
        backend='torch',
        device='cuda',
        dtype=dtype,
        num_neighbours=5,
        sigma_anat=0.5,
        sigma_dist=1.0,
        normalize_kernel=True,
        use_mask=False,
        hybrid=False,
    )
    assert operator.device.type == 'cuda'
    assert operator.torch_dtype == getattr(torch, dtype)

    grid = np.indices(geometry.shape).sum(axis=0) / float(np.prod(geometry.shape))
    operator.set_anatomical_image(make_cil_image(geometry, grid))

    rng = np.random.default_rng(21)
    x = make_cil_image(geometry, rng.normal(size=geometry.shape))
    y = make_cil_image(geometry, rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.isclose(dot_forward, dot_adjoint, atol=1e-6, rtol=1e-5)


class TestAutoBackendSelection:
    """Test automatic backend selection."""

    def test_auto_backend_selection(self, small_geometry):
        """Test auto backend selects appropriate backend."""
        op = get_kernel_operator(
            small_geometry,
            backend='auto',
        )

        # Should select torch if CUDA is available, otherwise numba
        if CUDA_AVAILABLE:
            assert op.backend == 'torch'
        else:
            assert op.backend == 'numba'

    def test_explicit_backend_override(self, small_geometry):
        """Test explicit backend selection overrides auto."""
        op = get_kernel_operator(
            small_geometry,
            backend='torch',
            device='cpu',
        )
        assert op.backend == 'torch'
        assert op.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
