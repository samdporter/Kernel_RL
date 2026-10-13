"""Tests for the L-BFGS-B optimiser and its ImageData/array adapter."""

import warnings

import numpy as np
import pytest
from cil.framework import ImageGeometry
from cil.optimisation.functions import L2NormSquared
from scipy.optimize import Bounds, OptimizeWarning, minimize

from krl.algorithms import lbfgsb as lbfgsb_module
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


def mixed_sign_problem():
    """Anisotropic quadratic whose start straddles zero so the bounds bind."""
    geometry = ImageGeometry(voxel_num_x=6, voxel_num_y=5, voxel_num_z=2, dtype=np.float64)
    rng = np.random.default_rng(17)
    target = make_image(geometry, rng.uniform(-1.0, 2.0, geometry.shape))
    data = rng.uniform(-1.5, 1.5, geometry.shape)
    data.flat[0] = -0.75
    data.flat[1] = 1.25
    return geometry, target, make_image(geometry, data)


def quadratic_weights(shape):
    """Positive weights spanning a factor 64, so L-BFGS-B needs real iterations."""
    count = int(np.prod(shape))
    return (2.0 ** ((np.arange(count) % 7) - 3)).reshape(shape)


def quadratic_value(array, target_array, weights=None):
    squared = (array - target_array) ** 2
    if weights is not None:
        squared = weights * squared
    return float(np.sum(squared))


def quadratic_gradient(array, target_array, weights=None):
    gradient = 2.0 * (array - target_array)
    if weights is not None:
        gradient = weights * gradient
    return gradient


class RecordingQuadratic:
    """Quadratic fidelity that records every image it is asked to evaluate.

    With ``reject_negative`` it refuses negative entries, which proves a
    bounded run never evaluates outside the non-negative orthant.
    """

    def __init__(self, target, weights=None, reject_negative=False):
        self.target = target
        self.weights = weights
        self.reject_negative = reject_negative
        self.evaluated = []

    def _record(self, array):
        if self.reject_negative and np.any(array < 0.0):
            raise AssertionError(f"evaluated a negative image (min={array.min()})")
        self.evaluated.append(array.ravel(order="C").copy())

    def __call__(self, image):
        array = np.asarray(image.as_array(), dtype=np.float64)
        self._record(array)
        weights = None if self.weights is None else self.weights.as_array()
        return quadratic_value(array, self.target.as_array(), weights)

    def gradient(self, image):
        self._record(np.asarray(image.as_array(), dtype=np.float64))
        gradient = image - self.target
        if self.weights is not None:
            gradient = self.weights * gradient
        return 2.0 * gradient


def preconditioner_for(kind, geometry):
    if kind == "none":
        return None
    if kind == "identity":
        return make_image(geometry, np.ones(geometry.shape))
    return make_image(geometry, power_of_two_diag(geometry.shape))


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


@pytest.mark.parametrize("iterations", [0, -1])
def test_nonpositive_iterations_perform_no_step(iterations):
    """SciPy treats maxiter=0 as one iteration, so run(<=0) must short-circuit."""
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=2, dtype=np.float64)
    initial = make_image(geometry, np.full(geometry.shape, 0.5))
    target = make_image(geometry, np.full(geometry.shape, 1.5))

    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
    )

    calls = []
    optimizer.run(iterations=iterations, callbacks=[calls.append], verbose=0)

    assert np.array_equal(optimizer.solution.as_array(), initial.as_array())
    assert optimizer.objective == [float(L2NormSquared(b=target)(initial))]
    assert calls == []


def test_zero_iteration_run_after_valid_run_clears_result():
    geometry = ImageGeometry(voxel_num_x=8, voxel_num_y=8, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(0).uniform(1.0, 2.0, geometry.shape))
    initial = make_image(geometry, np.full(geometry.shape, 0.5))

    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
    )
    optimizer.run(iterations=20, verbose=0)
    assert optimizer.result is not None

    optimizer.run(iterations=0, verbose=0)

    assert optimizer.result is None
    assert np.array_equal(optimizer.solution.as_array(), initial.as_array())
    assert optimizer.objective == [float(L2NormSquared(b=target)(initial))]


def test_normal_run_sets_result_matching_solution():
    geometry = ImageGeometry(voxel_num_x=8, voxel_num_y=8, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(0).uniform(1.0, 2.0, geometry.shape))
    initial = make_image(geometry, np.full(geometry.shape, 0.5))

    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
    )
    optimizer.run(iterations=5, verbose=0)

    assert optimizer.result is not None
    assert np.allclose(
        optimizer.solution.as_array(),
        optimizer.result.x.reshape(geometry.shape),
        rtol=1e-12,
        atol=1e-12,
    )
    assert optimizer.result.fun == pytest.approx(optimizer.objective[-1])


def test_run_emits_no_unknown_solver_warning():
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(0).uniform(1.0, 2.0, geometry.shape))
    initial = make_image(geometry, np.full(geometry.shape, 0.5))

    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        warnings.simplefilter("error", DeprecationWarning)
        optimizer.run(iterations=3, verbose=0)


def test_verbose_prints_scipy_summary(capsys):
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=2, dtype=np.float64)
    target = make_image(geometry, np.random.default_rng(1).uniform(1.0, 2.0, geometry.shape))
    initial = make_image(geometry, np.full(geometry.shape, 0.5))

    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=L2NormSquared(b=target),
    )
    optimizer.run(iterations=3, verbose=1)

    assert "L-BFGS-B" in capsys.readouterr().out


@pytest.mark.parametrize("enforce_non_negativity", [True, False], ids=["bounded", "unbounded"])
@pytest.mark.parametrize("preconditioner_kind", ["none", "identity", "heterogeneous"])
def test_initial_history_contract(monkeypatch, enforce_non_negativity, preconditioner_kind):
    """objective[0] is F(x_start) for every run length, and x_start is projected."""
    geometry, target, initial = mixed_sign_problem()
    weights = make_image(geometry, quadratic_weights(geometry.shape))
    initial_array = np.asarray(initial.as_array(), dtype=np.float64)
    target_array = np.asarray(target.as_array(), dtype=np.float64)
    if enforce_non_negativity:
        assert np.any(initial_array < 0.0), "the projection must actually bind"
        x_start = np.maximum(initial_array, 0.0)
    else:
        x_start = initial_array

    preconditioner = preconditioner_for(preconditioner_kind, geometry)
    diag = None if preconditioner is None else np.asarray(preconditioner.as_array(), dtype=np.float64).ravel(order="C")

    real_minimize = lbfgsb_module.minimize
    solver_starts = []

    def recording_minimize(*args, **kwargs):
        solver_starts.append(np.array(kwargs["x0"], copy=True))
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(lbfgsb_module, "minimize", recording_minimize)

    first_objectives = []
    for iterations in (-1, 0, 5):
        fidelity = RecordingQuadratic(target, weights, reject_negative=enforce_non_negativity)
        optimizer = LBFGSBOptimizer(
            initial_estimate=initial,
            data_fidelity=fidelity,
            options=LBFGSBOptions(enforce_non_negativity=enforce_non_negativity, ftol=1e-12, gtol=1e-12),
            preconditioner=preconditioner,
        )
        callbacks = []
        solver_starts.clear()
        optimizer.run(iterations=iterations, callbacks=[callbacks.append], verbose=0)

        # The first image evaluated is exactly the (possibly projected) start.
        assert np.array_equal(fidelity.evaluated[0], x_start.ravel(order="C"))
        expected = quadratic_value(x_start, target_array, weights.as_array())
        assert optimizer.objective[0] == pytest.approx(expected, rel=1e-12, abs=0)
        assert optimizer.loss[0] == optimizer.objective[0]
        first_objectives.append(optimizer.objective[0])

        if enforce_non_negativity:
            assert min(array.min() for array in fidelity.evaluated) >= 0.0

        if iterations <= 0:
            assert solver_starts == []
            assert optimizer.result is None
            assert optimizer.iteration == 0
            assert callbacks == []
            assert optimizer.objective == [pytest.approx(expected, rel=1e-12, abs=0)]
            assert np.array_equal(optimizer.solution.as_array(), x_start)
        else:
            assert len(solver_starts) == 1
            expected_start = x_start.ravel(order="C") if diag is None else x_start.ravel(order="C") / diag
            assert np.array_equal(solver_starts[0], expected_start)
            assert optimizer.result is not None

    assert first_objectives[0] == first_objectives[1] == first_objectives[2]


def test_unbounded_run_retains_original_start():
    geometry, target, initial = mixed_sign_problem()
    weights = make_image(geometry, quadratic_weights(geometry.shape))
    initial_array = np.asarray(initial.as_array(), dtype=np.float64)
    assert np.any(initial_array < 0.0)

    fidelity = RecordingQuadratic(target, weights, reject_negative=False)
    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=fidelity,
        options=LBFGSBOptions(enforce_non_negativity=False, ftol=1e-12, gtol=1e-12),
    )

    optimizer.run(iterations=0, verbose=0)
    assert np.array_equal(fidelity.evaluated[0], initial_array.ravel(order="C"))
    assert np.array_equal(optimizer.solution.as_array(), initial_array)
    assert float(optimizer.solution.as_array().min()) < 0.0

    fidelity.evaluated.clear()
    optimizer.run(iterations=5, verbose=0)
    assert np.array_equal(fidelity.evaluated[0], initial_array.ravel(order="C"))
    assert optimizer.objective[0] == pytest.approx(
        quadratic_value(initial_array, np.asarray(target.as_array(), dtype=np.float64), weights.as_array()),
        rel=1e-12,
        abs=0,
    )


@pytest.mark.parametrize("preconditioner_kind", ["none", "identity", "heterogeneous"])
def test_mixed_sign_bounded_run_matches_direct_scipy(preconditioner_kind):
    """A real bounded run is coherent and reproduces plain bounded SciPy."""
    geometry, target, initial = mixed_sign_problem()
    weights_image = make_image(geometry, quadratic_weights(geometry.shape))
    weights = weights_image.as_array()
    target_array = np.asarray(target.as_array(), dtype=np.float64)
    initial_array = np.asarray(initial.as_array(), dtype=np.float64)
    x_start = np.maximum(initial_array, 0.0)
    preconditioner = preconditioner_for(preconditioner_kind, geometry)
    diag = None if preconditioner is None else np.asarray(preconditioner.as_array(), dtype=np.float64).ravel(order="C")
    options = LBFGSBOptions(ftol=1e-14, gtol=1e-14, max_linesearch=20)

    fidelity = RecordingQuadratic(target, weights_image, reject_negative=True)
    optimizer = LBFGSBOptimizer(
        initial_estimate=initial,
        data_fidelity=fidelity,
        options=options,
        preconditioner=preconditioner,
    )
    optimizer.run(iterations=100, verbose=0)

    history = np.asarray(optimizer.objective, dtype=np.float64)
    solution = optimizer.solution.as_array()

    assert optimizer.result is not None
    assert optimizer.result.nit > 1
    assert np.array_equal(fidelity.evaluated[0], x_start.ravel(order="C"))
    assert min(array.min() for array in fidelity.evaluated) >= 0.0
    assert history[0] == pytest.approx(quadratic_value(x_start, target_array, weights), rel=1e-12, abs=0)
    assert np.all(np.isfinite(history))
    assert history[-1] < history[0]
    assert np.all(np.diff(history) <= 1e-8 * max(1.0, abs(history[0])))
    assert float(solution.min()) >= 0.0
    assert history[-1] == pytest.approx(quadratic_value(solution, target_array, weights), rel=1e-9, abs=1e-12)

    # The same problem handed straight to bounded SciPy, in the run's own variables.
    weights_flat = weights.ravel(order="C")
    target_flat = target_array.ravel(order="C")
    start_flat = x_start.ravel(order="C")
    if diag is None:

        def fun(z):
            return quadratic_value(z, target_flat, weights_flat)

        def jac(z):
            return quadratic_gradient(z, target_flat, weights_flat)

        z_start = start_flat
    else:

        def fun(z):
            return quadratic_value(diag * z, target_flat, weights_flat)

        def jac(z):
            return diag * quadratic_gradient(diag * z, target_flat, weights_flat)

        z_start = start_flat / diag

    reference = minimize(
        fun=fun,
        jac=jac,
        x0=z_start,
        method="L-BFGS-B",
        bounds=Bounds(lb=0, ub=np.inf),
        options={
            "maxiter": 100,
            "maxls": options.max_linesearch,
            "ftol": options.ftol,
            "gtol": options.gtol,
        },
    )
    reference_solution = reference.x if diag is None else diag * reference.x
    assert solution.ravel(order="C") == pytest.approx(reference_solution, rel=1e-9, abs=1e-11)
    assert history[-1] == pytest.approx(reference.fun, rel=1e-9, abs=1e-12)


def test_caller_image_untouched_and_repeat_runs_are_deterministic():
    geometry, target, initial = mixed_sign_problem()
    weights = make_image(geometry, quadratic_weights(geometry.shape))
    initial_array = np.asarray(initial.as_array(), dtype=np.float64)
    options = LBFGSBOptions(ftol=1e-12, gtol=1e-12)

    def make_optimizer():
        return LBFGSBOptimizer(
            initial_estimate=initial,
            data_fidelity=RecordingQuadratic(target, weights, reject_negative=True),
            options=options,
        )

    optimizer = make_optimizer()
    optimizer.run(iterations=10, verbose=0)
    first_history = list(optimizer.objective)
    first_solution = np.array(optimizer.solution.as_array(), copy=True)

    assert np.array_equal(initial.as_array(), initial_array)
    assert np.array_equal(optimizer.initial_estimate.as_array(), initial_array)

    optimizer.run(iterations=10, verbose=0)
    assert optimizer.objective == first_history
    assert np.array_equal(optimizer.solution.as_array(), first_solution)
    assert np.array_equal(initial.as_array(), initial_array)
    assert np.array_equal(optimizer.initial_estimate.as_array(), initial_array)

    optimizer.run(iterations=0, verbose=0)
    assert optimizer.result is None
    assert np.array_equal(optimizer.solution.as_array(), np.maximum(initial_array, 0.0))
    assert np.array_equal(initial.as_array(), initial_array)
    assert np.array_equal(optimizer.initial_estimate.as_array(), initial_array)

    fresh = make_optimizer()
    fresh.run(iterations=10, verbose=0)
    assert fresh.objective == first_history
    assert np.array_equal(fresh.solution.as_array(), first_solution)
    assert np.array_equal(initial.as_array(), initial_array)
