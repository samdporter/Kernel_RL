"""Package-level sensitivity regressions for the Richardson-Lucy algorithm.

These exercise the package's per-iteration sensitivity handling for hybrid
kernels directly, without the example reconstruction wrapper.
"""

import numpy as np
import pytest
from cil.framework import ImageGeometry

from krl.algorithms.richardson_lucy import RichardsonLucy
from krl.operators.blurring import create_gaussian_blur
from krl.operators.kernel_operator import get_kernel_operator


@pytest.fixture
def geometry():
    return ImageGeometry(voxel_num_x=16, voxel_num_y=16, voxel_num_z=8, dtype=np.float64)


@pytest.fixture
def phantom(geometry):
    """Smooth two-blob phantom."""
    x = np.zeros((8, 16, 16), dtype=np.float64)
    z, y, xx = np.indices(x.shape)
    x += 50 * np.exp(-((z - 2) ** 2 + (y - 5) ** 2 + (xx - 5) ** 2) / 4.0)
    x += 30 * np.exp(-((z - 5) ** 2 + (y - 10) ** 2 + (xx - 11) ** 2) / 6.0)
    img = geometry.allocate(0.0)
    img.fill(x)
    return img


@pytest.fixture
def blur(geometry):
    # 'numba' uses zero padding, so the sensitivity A^T(1) varies near the boundary.
    return create_gaussian_blur(sigma=(1.0, 1.0, 1.0), geometry=geometry, backend="numba")


@pytest.fixture
def observed(phantom, blur):
    return blur.direct(phantom)


def make_kernel(geometry, anatomical, hybrid):
    """Fresh kernel operator so each run owns its mutable state."""
    kernel = get_kernel_operator(
        geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_dist=3.0,
        sigma_emission=1.0,
        normalize_kernel=True,
        hybrid=hybrid,
        use_mask=True,
        mask_k=10,
    )
    kernel.set_anatomical_image(anatomical)
    return kernel


def test_plain_rl_sensitivity_is_adjoint_of_ones_near_boundaries(phantom, blur, observed):
    rl = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
    )
    ones = observed.geometry.allocate(value=1.0)
    sensitivity = rl.sensitivity.as_array()

    np.testing.assert_allclose(sensitivity, blur.adjoint(ones).as_array(), rtol=1e-12, atol=1e-12)

    # Zero-padded blur: interior voxels see the full kernel, boundary voxels lose mass.
    assert sensitivity[4, 8, 8] == pytest.approx(1.0, abs=1e-9)
    assert sensitivity[0, 0, 0] < 0.5
    assert not np.allclose(sensitivity, 1.0)


def test_hkrl_sensitivity_tracks_reference_and_stays_consistent_after_freeze(
    phantom, blur, observed
):
    ones = observed.geometry.allocate(value=1.0)

    # Adaptive kernel: sensitivity is recomputed for the current reference.
    adaptive_kernel = make_kernel(observed.geometry, phantom, hybrid=True)
    adaptive = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
        kernel_operator=adaptive_kernel,
        freeze_iteration=0,
    )
    next(adaptive)  # records the initial objective, no update yet
    stale = adaptive.sensitivity.as_array().copy()
    next(adaptive)  # update 1
    after_one = adaptive.sensitivity.as_array().copy()
    fresh_one = adaptive.forward_operator.adjoint(ones).as_array()
    next(adaptive)  # update 2
    after_two = adaptive.sensitivity.as_array().copy()
    fresh_two = adaptive.forward_operator.adjoint(ones).as_array()

    assert not adaptive_kernel.freeze_emission_kernel
    np.testing.assert_allclose(after_one, fresh_one, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(after_two, fresh_two, rtol=1e-10, atol=1e-12)
    # The reference changed, so a sensitivity computed once at construction is stale.
    assert np.abs(after_two - stale).max() > 1e-6 * np.abs(after_two).max()
    assert np.abs(after_two - after_one).max() > 1e-6 * np.abs(after_two).max()

    # Frozen kernel: sensitivity stops following the reference but stays consistent
    # with the frozen forward operator.
    frozen_kernel = make_kernel(observed.geometry, phantom, hybrid=True)
    frozen = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
        kernel_operator=frozen_kernel,
        freeze_iteration=1,
    )
    next(frozen)
    next(frozen)  # update 1 freezes the kernel
    assert frozen_kernel.freeze_emission_kernel
    at_freeze = frozen.sensitivity.as_array().copy()
    fresh_at_freeze = frozen.forward_operator.adjoint(ones).as_array()
    next(frozen)  # update 2 with a frozen kernel
    after_freeze = frozen.sensitivity.as_array().copy()
    fresh_after_freeze = frozen.forward_operator.adjoint(ones).as_array()

    np.testing.assert_allclose(at_freeze, fresh_at_freeze, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(after_freeze, at_freeze, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(after_freeze, fresh_after_freeze, rtol=1e-10, atol=1e-12)
