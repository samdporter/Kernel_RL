"""Tests for the L-BFGS-B optimiser and its ImageData/array adapter."""

import numpy as np
import pytest
from cil.framework import ImageGeometry
from cil.optimisation.functions import L2NormSquared

from krl.algorithms.lbfgsb import LBFGSBOptimizer, LBFGSBOptions, _ImageArrayAdapter


def make_image(geometry, data):
    image = geometry.allocate(0.0, dtype=np.float64)
    image.fill(np.asarray(data, dtype=np.float64))
    return image


def power_of_two_diag(shape):
    """Heterogeneous positive diagonal whose values are exact powers of two.

    Powers of two make ``m * (x / m)`` round-trip exactly, so the helper
    tests can assert bitwise equality.
    """
    count = int(np.prod(shape))
    return (2.0 ** ((np.arange(count) % 5) - 2)).reshape(shape)


def make_mismatched_template():
    """A float64 image whose geometry still defaults to float32 allocation.

    CIL normally syncs ``geometry.dtype`` to the data dtype, so the divergence
    is forced here to pin the adapter to the template dtype rather than the
    geometry's default allocation dtype.
    """
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2, dtype=np.float32)
    template = geometry.allocate(0.0)
    template.array = template.array.astype(np.float64)
    return template


def test_lbfgsb_reduces_simple_objective():
    geometry = ImageGeometry(voxel_num_x=8, voxel_num_y=8, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(0).uniform(1.0, 2.0, geometry.shape))
    initial = make_image(geometry, np.full(geometry.shape, 0.5))

    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
        options=LBFGSBOptions(ftol=1e-12, gtol=1e-12),
    )
    optimizer.run(iterations=20, verbose=0)

    assert float(np.all(np.isfinite(optimizer.solution.as_array())))
    assert optimizer.objective[-1] < optimizer.objective[0]
    assert optimizer.objective[-1] == pytest.approx(0.0, abs=1e-6)


class PrecisionProbe:
    """Quadratic functional whose value and gradient are sensitive to float64."""

    def __init__(self, target):
        self.target = target

    def __call__(self, x):
        diff = x.as_array() - self.target.as_array()
        return float(np.sum(diff * diff))

    def gradient(self, x):
        return x - self.target


def test_adapter_preserves_template_dtype():
    template = make_mismatched_template()

    assert template.dtype == np.float64
    assert template.geometry.allocate(value=0).dtype == np.float32

    adapter = _ImageArrayAdapter(template)
    assert adapter._working_image.dtype == np.float64


def test_adapter_round_trip_preserves_float64():
    template = make_mismatched_template()
    adapter = _ImageArrayAdapter(template)

    # A value that is not exactly representable as float32.
    precise = 1.23456789012345e-12
    flat = np.full(adapter.size, precise, dtype=np.float64)
    image = adapter.array_to_image(flat)

    assert image.dtype == np.float64
    assert np.array_equal(image.as_array().ravel(order="C"), flat)


def test_adapter_evaluation_and_gradient_preserve_float64():
    template = make_mismatched_template()
    precise = 1.23456789012345e-12
    target = template.clone()
    target.fill(0.0)

    probe = PrecisionProbe(target)
    optimizer = LBFGSBOptimizer(initial_estimate=template.clone(), data_fidelity=probe)
    # Cloning a template syncs its geometry dtype, so install the mismatched
    # adapter explicitly to exercise the float64 evaluation/gradient path.
    optimizer._adapter = _ImageArrayAdapter(template)

    flat = np.full(optimizer._adapter.size, precise, dtype=np.float64)
    image_array = flat.reshape(template.shape)

    value = optimizer._objective_from_array(flat)
    expected_value = float(np.sum((image_array - target.as_array()) ** 2))
    assert value == expected_value

    gradient = optimizer._gradient_from_array(flat)
    expected_gradient = (image_array - target.as_array()).ravel(order="C")
    assert np.array_equal(gradient, expected_gradient)


def test_preconditioner_round_trip_and_identity_helpers():
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2, dtype=np.float64)
    initial = make_image(geometry, np.ones(geometry.shape))
    fidelity = L2NormSquared(b=initial)
    diag = power_of_two_diag(geometry.shape)

    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=fidelity,
        preconditioner=make_image(geometry, diag),
    )

    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, optimizer._adapter.size)
    assert np.array_equal(optimizer._z_to_x(optimizer._x_to_z(x)), x)
    z = rng.uniform(0.0, 1.0, optimizer._adapter.size)
    assert np.array_equal(optimizer._x_to_z(optimizer._z_to_x(z)), z)
    assert optimizer._diag.dtype == np.float64
    assert optimizer._diag.flags["C_CONTIGUOUS"]

    plain = LBFGSBOptimizer(initial_estimate=initial, data_fidelity=fidelity)
    assert plain._diag is None
    probe = np.arange(optimizer._adapter.size, dtype=np.float64)
    assert plain._z_to_x(probe) is probe
    assert plain._x_to_z(probe) is probe
    assert plain._gradient_in_z(probe) is probe


def test_preconditioner_gradient_matches_chain_rule_and_finite_difference():
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(1).uniform(0.5, 2.0, geometry.shape))
    initial = make_image(geometry, np.ones(geometry.shape))
    diag = power_of_two_diag(geometry.shape)
    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
        preconditioner=make_image(geometry, diag),
    )

    z = np.random.default_rng(2).uniform(0.5, 1.5, optimizer._adapter.size)

    grad_x = optimizer._gradient_from_array(optimizer._z_to_x(z))
    grad_z = optimizer._gradient_in_z(grad_x)
    assert np.array_equal(grad_z, optimizer._diag * grad_x)

    h = 1e-6
    fd = np.empty_like(z)
    for i in range(z.size):
        step = np.zeros_like(z)
        step[i] = h
        plus = optimizer._objective_from_array(optimizer._z_to_x(z + step))
        minus = optimizer._objective_from_array(optimizer._z_to_x(z - step))
        fd[i] = (plus - minus) / (2.0 * h)
    assert np.allclose(fd, grad_z, rtol=1e-6, atol=1e-8)


def test_preconditioner_bounds_mapping():
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(3).uniform(0.5, 2.0, geometry.shape))
    initial = make_image(geometry, np.full(geometry.shape, 0.5))
    diag = power_of_two_diag(geometry.shape)
    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
        preconditioner=make_image(geometry, diag),
    )

    rng = np.random.default_rng(4)
    x = np.abs(rng.normal(size=optimizer._adapter.size))
    assert np.all(optimizer._x_to_z(x) >= 0.0)
    z = np.abs(rng.normal(size=optimizer._adapter.size))
    assert np.all(optimizer._z_to_x(z) >= 0.0)

    optimizer.run(iterations=100, verbose=0)
    solution = optimizer.solution.as_array()
    assert float(solution.min()) >= 0.0
    assert float(np.all(np.isfinite(solution)))


def test_preconditioner_identity_matches_unpreconditioned():
    geometry = ImageGeometry(voxel_num_x=8, voxel_num_y=8, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(5).uniform(1.0, 2.0, geometry.shape))
    initial = make_image(geometry, np.full(geometry.shape, 0.5))
    options = LBFGSBOptions(ftol=1e-12, gtol=1e-12)

    plain = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
        options=options,
    )
    preconditioned = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
        options=options,
        preconditioner=make_image(geometry, np.ones(geometry.shape)),
    )

    plain.run(iterations=30, verbose=0)
    preconditioned.run(iterations=30, verbose=0)

    assert plain.objective == pytest.approx(preconditioned.objective, rel=1e-12, abs=1e-12)
    assert np.allclose(
        plain.solution.as_array(),
        preconditioned.solution.as_array(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_preconditioner_minimiser_invariance():
    geometry = ImageGeometry(voxel_num_x=8, voxel_num_y=8, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(6).uniform(1.0, 2.0, geometry.shape))
    initial = make_image(geometry, np.full(geometry.shape, 0.5))
    options = LBFGSBOptions(ftol=1e-15, gtol=1e-15)
    diag = np.linspace(0.5, 2.0, int(np.prod(geometry.shape))).reshape(geometry.shape)

    plain = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
        options=options,
    )
    preconditioned = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
        options=options,
        preconditioner=make_image(geometry, diag),
    )

    plain.run(iterations=200, verbose=0)
    preconditioned.run(iterations=200, verbose=0)

    assert plain.objective[-1] == pytest.approx(0.0, abs=1e-6)
    assert preconditioned.objective[-1] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(plain.solution.as_array(), target.as_array(), atol=1e-6)
    assert np.allclose(preconditioned.solution.as_array(), target.as_array(), atol=1e-6)


def test_preconditioner_monotone_image_space_history():
    geometry = ImageGeometry(voxel_num_x=8, voxel_num_y=8, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(7).uniform(1.0, 2.0, geometry.shape))
    initial = make_image(geometry, np.full(geometry.shape, 0.5))
    options = LBFGSBOptions(ftol=1e-12, gtol=1e-12)
    diag = np.linspace(0.5, 2.0, int(np.prod(geometry.shape))).reshape(geometry.shape)

    for preconditioner in (None, make_image(geometry, diag)):
        optimizer = LBFGSBOptimizer(
            initial_estimate=initial,
            data_fidelity=L2NormSquared(b=target),
            options=options,
            preconditioner=preconditioner,
        )
        optimizer.run(iterations=30, verbose=0)

        history = np.asarray(optimizer.objective)
        assert history[-1] < history[0]
        assert np.all(np.diff(history) <= 1e-8 * max(1.0, abs(history[0])))
        assert optimizer.objective[-1] == pytest.approx(
            float(optimizer.data_fidelity(optimizer.solution)), rel=1e-6, abs=1e-9
        )


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan])
def test_preconditioner_rejects_invalid_entries(bad_value):
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2, dtype=np.float64)
    initial = make_image(geometry, np.ones(geometry.shape))
    bad = np.ones(geometry.shape)
    bad.flat[0] = bad_value

    with pytest.raises(ValueError):
        LBFGSBOptimizer(
            initial_estimate=initial,
            data_fidelity=L2NormSquared(b=initial),
            preconditioner=bad,
        )


def test_preconditioner_rejects_wrong_shape():
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2, dtype=np.float64)
    initial = make_image(geometry, np.ones(geometry.shape))

    with pytest.raises(ValueError):
        LBFGSBOptimizer(
            initial_estimate=initial,
            data_fidelity=L2NormSquared(b=initial),
            preconditioner=np.ones((2, 3, 5)),
        )


def test_preconditioner_rejects_non_array():
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2, dtype=np.float64)
    initial = make_image(geometry, np.ones(geometry.shape))

    with pytest.raises(TypeError):
        LBFGSBOptimizer(
            initial_estimate=initial,
            data_fidelity=L2NormSquared(b=initial),
            preconditioner=lambda x: x,
        )


def test_preconditioner_rejects_complex():
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2, dtype=np.float64)
    initial = make_image(geometry, np.ones(geometry.shape))
    complex_array = np.full(geometry.shape, 1.0 + 2.0j)

    complex_image = geometry.allocate(0.0)
    complex_image.array = complex_array

    for preconditioner in (complex_array, complex_image):
        with pytest.raises(ValueError):
            LBFGSBOptimizer(
                initial_estimate=initial,
                data_fidelity=L2NormSquared(b=initial),
                preconditioner=preconditioner,
            )
