"""MPS regression tests: the torch backend on Apple Silicon vs the numba CPU reference.

Runs only when GPU tests are opt-in and MPS is available, so it skips cleanly
on Linux/CI.
"""

import os

import pytest

# Importing torch alongside CIL's native libraries breaks OpenMP on some
# platforms, so torch-touching tests are opt-in via this environment variable.
if os.environ.get("KRL_RUN_GPU_TESTS") != "1":
    pytest.skip(
        "GPU tests disabled; set KRL_RUN_GPU_TESTS=1 to run them",
        allow_module_level=True,
    )

# torch and CIL's native libraries each load libomp; on macOS the second copy
# aborts the process (OMP Error #15) unless this escape hatch is set before
# torch is imported. Only set it for the gated run so a normal suite run is
# not mutated.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from cil.framework import ImageGeometry

from krl.operators.kernel_operator import KernelOperator, get_kernel_operator

if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

# float32 cross-backend tolerance: observed MPS-vs-numba max errors are ~2e-6.
RTOL = 1e-5
ATOL = 1e-5
# Explicit bound matching docs/MACOS-ARM.md; worst observed case is ~1.8e-6.
MAX_ABS_ERROR = 2e-6


def make_geometry():
    return ImageGeometry(voxel_num_x=16, voxel_num_y=16, voxel_num_z=8, dtype=np.float32)


def make_image(geometry, array):
    image = geometry.allocate()
    image.fill(np.asarray(array, dtype=geometry.dtype))
    return image


@pytest.fixture
def geometry():
    return make_geometry()


@pytest.fixture
def anatomy(geometry):
    rng = np.random.default_rng(7)
    grid = np.indices(geometry.shape, dtype=np.float32).sum(axis=0)
    arr = grid / grid.max() + 0.1 * rng.normal(size=geometry.shape)
    return make_image(geometry, arr)


@pytest.fixture
def emission(geometry):
    rng = np.random.default_rng(11)
    return make_image(geometry, rng.normal(size=geometry.shape))


@pytest.fixture
def adjoint_input(geometry):
    rng = np.random.default_rng(23)
    return make_image(geometry, rng.normal(size=geometry.shape))


def make_operators(geometry, use_mask, hybrid):
    params = dict(
        num_neighbours=5,
        sigma_anat=0.3,
        sigma_emission=0.5,
        normalize_kernel=True,
        mask_k=16,
        use_mask=use_mask,
        hybrid=hybrid,
    )
    mps = get_kernel_operator(
        geometry, backend="torch", device="auto", dtype="float32", **params
    )
    assert mps.device.type == "mps"
    reference = KernelOperator(geometry, **params)
    return mps, reference


@pytest.mark.parametrize("use_mask", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("hybrid", [False, True], ids=["normal", "hybrid"])
def test_forward_and_adjoint_match_numba(geometry, anatomy, emission, adjoint_input,
                                         use_mask, hybrid):
    mps, reference = make_operators(geometry, use_mask, hybrid)
    mps.set_anatomical_image(anatomy)
    reference.set_anatomical_image(anatomy)

    mps_forward = mps.direct(emission).as_array()
    reference_forward = reference.direct(emission).as_array()
    np.testing.assert_allclose(mps_forward, reference_forward, rtol=RTOL, atol=ATOL)
    assert np.max(np.abs(mps_forward - reference_forward)) <= MAX_ABS_ERROR

    mps_adjoint = mps.adjoint(adjoint_input).as_array()
    reference_adjoint = reference.adjoint(adjoint_input).as_array()
    np.testing.assert_allclose(mps_adjoint, reference_adjoint, rtol=RTOL, atol=ATOL)
    assert np.max(np.abs(mps_adjoint - reference_adjoint)) <= MAX_ABS_ERROR


@pytest.mark.parametrize("use_mask", [False, True], ids=["dense", "sparse"])
def test_adjoint_first_lazy_normalisation(geometry, anatomy, adjoint_input, use_mask):
    """A fresh MPS operator must bootstrap its normalisation map lazily on the
    first adjoint() call (bounded-memory weight sum) and match numba."""
    mps, reference = make_operators(geometry, use_mask, hybrid=False)
    mps.set_anatomical_image(anatomy)
    reference.set_anatomical_image(anatomy)

    mps_adjoint = mps.adjoint(adjoint_input).as_array()
    reference_adjoint = reference.adjoint(adjoint_input).as_array()

    assert mps._normalisation_map is not None
    assert np.isfinite(mps_adjoint).all()
    np.testing.assert_allclose(mps_adjoint, reference_adjoint, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("use_mask", [False, True], ids=["dense", "sparse"])
def test_adjoint_dot_product_identity(geometry, anatomy, emission, adjoint_input, use_mask):
    """<A x, y> == <x, A^T y> for the MPS implementation of a normalized kernel."""
    mps, _ = make_operators(geometry, use_mask, hybrid=False)
    mps.set_anatomical_image(anatomy)

    x = emission.as_array().astype(np.float64)
    y = adjoint_input.as_array().astype(np.float64)
    ax = mps.direct(emission).as_array().astype(np.float64)
    aty = mps.adjoint(adjoint_input).as_array().astype(np.float64)

    lhs = float(np.vdot(ax, y))
    rhs = float(np.vdot(x, aty))
    denom = max(abs(lhs), abs(rhs), 1.0)
    assert abs(lhs - rhs) / denom < 1e-6


@pytest.mark.parametrize("use_mask", [False, True], ids=["dense", "sparse"])
def test_toggle_normalize_features_rebuilds_and_invalidates_caches(
    geometry, anatomy, emission, adjoint_input, use_mask
):
    """Toggling normalize_features must rebuild the MPS features from the raw
    snapshot and match the numba reference, clearing the device caches."""
    mps, reference = make_operators(geometry, use_mask, hybrid=False)
    mps.set_anatomical_image(anatomy)
    reference.set_anatomical_image(anatomy)

    direct_true = mps.direct(emission).as_array()
    np.testing.assert_allclose(
        direct_true, reference.direct(emission).as_array(), rtol=RTOL, atol=ATOL
    )
    adjoint_true = mps.adjoint(adjoint_input).as_array()
    np.testing.assert_allclose(
        adjoint_true, reference.adjoint(adjoint_input).as_array(), rtol=RTOL, atol=ATOL
    )

    assert mps._normalisation_map_cpu is not None
    if use_mask:
        assert mps._mask_cpu is not None

    mps.set_parameters({"normalize_features": False})
    reference.set_parameters({"normalize_features": False})

    assert mps._mask_gpu is None
    assert mps._mask_cpu is None
    assert mps._normalisation_map_gpu is None
    assert mps._normalisation_map_cpu is None

    direct_false = mps.direct(emission).as_array()
    np.testing.assert_allclose(
        direct_false, reference.direct(emission).as_array(), rtol=RTOL, atol=ATOL
    )
    adjoint_false = mps.adjoint(adjoint_input).as_array()
    np.testing.assert_allclose(
        adjoint_false, reference.adjoint(adjoint_input).as_array(), rtol=RTOL, atol=ATOL
    )
    assert not np.allclose(direct_true, direct_false)

    mps.set_parameters({"normalize_features": True})
    reference.set_parameters({"normalize_features": True})

    direct_back = mps.direct(emission).as_array()
    np.testing.assert_allclose(direct_back, direct_true, rtol=RTOL, atol=ATOL)
    adjoint_back = mps.adjoint(adjoint_input).as_array()
    np.testing.assert_allclose(adjoint_back, adjoint_true, rtol=RTOL, atol=ATOL)
