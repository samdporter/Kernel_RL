"""Tests for the Richardson-Lucy wrapper in examples/scripts/example_reconstruction.py.

The wrapper must be a faithful thin wrapper around the package
``RichardsonLucy``: same latent iterate, one objective value per completed
update (excluding CIL's initial objective entry), and the package's
per-iteration sensitivity handling for hybrid kernels.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from cil.framework import ImageGeometry

from krl.algorithms.richardson_lucy import RichardsonLucy
from krl.operators.blurring import create_gaussian_blur
from krl.operators.kernel_operator import get_kernel_operator

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "scripts" / "example_reconstruction.py"
)

# None means "do not pass freeze_iteration", i.e. the wrapper's historical default.
FREEZE_CASES = [None, 0, 1, 2]


def load_example_module():
    spec = importlib.util.spec_from_file_location("example_reconstruction", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = load_example_module()


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


def run_wrapper_and_package(observed, blur, anatomical, iterations, freeze, hybrid=None):
    """Run the example wrapper and a separately constructed package algorithm.

    Both sides get their own kernel operator instance so that shared mutable
    state (frozen emission reference, cached weights) cannot make the
    comparison vacuous.
    """

    def new_kernel():
        return None if hybrid is None else make_kernel(observed.geometry, anatomical, hybrid)

    wrapper_kernel = new_kernel()
    wrapper_kwargs = {} if freeze is None else {"freeze_iteration": freeze}
    latent, objectives = example.richardson_lucy(
        observed,
        blur,
        iterations,
        kernel_operator=wrapper_kernel,
        **wrapper_kwargs,
    )

    package_kernel = new_kernel()
    package_freeze = 0 if freeze is None or freeze == 0 else freeze
    rl = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
        kernel_operator=package_kernel,
        freeze_iteration=package_freeze,
    )
    rl.run(iterations=iterations, verbose=0)

    return latent, objectives, wrapper_kernel, rl, package_kernel


def assert_matches_package(latent, objectives, wrapper_kernel, rl, package_kernel, iterations):
    # Latent estimate, not the kernel-applied reconstruction.
    np.testing.assert_allclose(latent.as_array(), rl.x.as_array(), rtol=1e-10, atol=1e-12)
    if wrapper_kernel is not None:
        assert not np.allclose(latent.as_array(), rl.get_output().as_array())

    # Exactly one objective per completed update, excluding CIL's initial entry.
    assert len(objectives) == iterations
    assert len(rl.loss) == iterations + 1
    np.testing.assert_allclose(
        np.asarray(objectives, dtype=np.float64),
        np.asarray(rl.loss[1:], dtype=np.float64),
        rtol=1e-10,
        atol=1e-12,
    )

    if wrapper_kernel is not None:
        assert wrapper_kernel.freeze_emission_kernel == package_kernel.freeze_emission_kernel


@pytest.mark.parametrize("freeze", FREEZE_CASES)
def test_plain_rl_matches_package(observed, blur, phantom, freeze):
    latent, objectives, wrapper_kernel, rl, package_kernel = run_wrapper_and_package(
        observed, blur, phantom, iterations=4, freeze=freeze
    )
    assert wrapper_kernel is None
    assert_matches_package(latent, objectives, wrapper_kernel, rl, package_kernel, 4)


@pytest.mark.parametrize("freeze", FREEZE_CASES)
def test_fixed_kernel_krl_matches_package(observed, blur, phantom, freeze):
    latent, objectives, wrapper_kernel, rl, package_kernel = run_wrapper_and_package(
        observed, blur, phantom, iterations=4, freeze=freeze, hybrid=False
    )
    assert wrapper_kernel.freeze_emission_kernel is (freeze not in (None, 0))
    assert_matches_package(latent, objectives, wrapper_kernel, rl, package_kernel, 4)


@pytest.mark.parametrize("freeze", FREEZE_CASES)
def test_adaptive_hkrl_matches_package(observed, blur, phantom, freeze):
    latent, objectives, wrapper_kernel, rl, package_kernel = run_wrapper_and_package(
        observed, blur, phantom, iterations=4, freeze=freeze, hybrid=True
    )
    assert wrapper_kernel.freeze_emission_kernel is (freeze not in (None, 0))
    assert_matches_package(latent, objectives, wrapper_kernel, rl, package_kernel, 4)


def test_zero_updates_returns_initial_latent(observed, blur, phantom):
    plain_latent, plain_objectives = example.richardson_lucy(observed, blur, iterations=0)
    assert plain_objectives == []
    np.testing.assert_array_equal(plain_latent.as_array(), observed.as_array())

    kernel = make_kernel(observed.geometry, phantom, hybrid=True)
    latent, objectives = example.richardson_lucy(
        observed, blur, iterations=0, freeze_iteration=1, kernel_operator=kernel
    )
    assert objectives == []
    # Latent stays the initial estimate: the kernel is never applied for the caller.
    np.testing.assert_array_equal(latent.as_array(), observed.as_array())


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


def test_example_module_imports_without_plotting_dependencies():
    """matplotlib is imported inside main(), so the numeric helper stays importable."""
    code = (
        "import importlib.util, sys;\n"
        "sys.modules['matplotlib'] = None;\n"
        f"spec = importlib.util.spec_from_file_location('example_reconstruction', {str(EXAMPLE_PATH)!r});\n"
        "module = importlib.util.module_from_spec(spec);\n"
        "spec.loader.exec_module(module);\n"
        "assert hasattr(module, 'richardson_lucy');\n"
        "print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
