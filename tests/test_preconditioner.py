"""Tests for MAPRL preconditioner and Armijo scheduling functionality."""

import warnings

import numpy as np
import pytest
from cil.framework import ImageGeometry

from krl.algorithms.maprl import MAPRL

EPS_PRECONDITIONER_DEPRECATION = (
    "eps_preconditioner is deprecated and has no effect. Apply stabilisation "
    "inside the supplied preconditioner. The eps parameter affects only the "
    "unpreconditioned EM scaling."
)


def make_image(geometry, value):
    image = geometry.allocate(0.0, dtype=np.float64)
    image.fill(value)
    return image


@pytest.fixture
def geometry():
    return ImageGeometry(voxel_num_x=4, voxel_num_y=4, voxel_num_z=1, dtype=np.float64)


class ConstantFunctional:
    """Simple functional with a constant gradient and constant value."""

    def __init__(self, gradient, value=0.0):
        self._gradient = gradient
        self._value = value

    def gradient(self, x):
        image = x.geometry.allocate(0.0, dtype=np.float64)
        image.fill(self._gradient)
        return image

    def __call__(self, x):
        return float(self._value)


class QuadraticFunctional:
    """Isotropic quadratic 0.5 * ||x - target||^2 with gradient x - target."""

    def __init__(self, target):
        self.target = target

    def gradient(self, x):
        return x - self.target

    def __call__(self, x):
        diff = x.as_array() - self.target.as_array()
        return 0.5 * float(np.sum(diff ** 2))


class NaNFirstTrialFunctional:
    """Functional whose first Armijo trial lands on a non-finite loss.

    A constant gradient of -1 and ``eps=0`` make the trial points ``1 + step``
    for an initial estimate of 1. ``nan_value`` yields NaN, ``finite_value``
    yields ``finite_loss`` and any other point yields 1.0 (the reference).
    """

    def __init__(self, nan_value, finite_value, finite_loss):
        self.nan_value = nan_value
        self.finite_value = finite_value
        self.finite_loss = finite_loss

    def gradient(self, x):
        image = x.geometry.allocate(0.0, dtype=np.float64)
        image.fill(-1.0)
        return image

    def __call__(self, x):
        arr = x.as_array()
        if np.allclose(arr, self.nan_value):
            return np.nan
        if np.allclose(arr, self.finite_value):
            return self.finite_loss
        return 1.0


class RecordingMAPRL(MAPRL):
    """MAPRL that records every preconditioner refresh via actual update count."""

    def __init__(self, *args, **kwargs):
        self.__dict__["preconditioner_updates"] = []
        super().__init__(*args, **kwargs)

    @property
    def _preconditioner_image(self):
        return self.__dict__.get("_preconditioner_store")

    @_preconditioner_image.setter
    def _preconditioner_image(self, value):
        self.__dict__["_preconditioner_store"] = value
        if value is not None:
            self.__dict__["preconditioner_updates"].append(self._update_count)


def make_maprl(geometry, **kwargs):
    defaults = dict(
        initial_estimate=make_image(geometry, 1.0),
        data_fidelity=ConstantFunctional(0.1, 4.0),
        prior=ConstantFunctional(0.0, 1.0),
        step_size=1.0,
        initial_line_search=False,
        armijo_iterations=0,
    )
    defaults.update(kwargs)
    return MAPRL(**defaults)


def test_maprl_preconditioner_initialization(geometry):
    """Test that MAPRL can be initialized with preconditioner parameters."""
    def test_preconditioner(x):
        return make_image(x.geometry, 0.5)

    maprl = make_maprl(
        geometry,
        preconditioner=test_preconditioner,
        preconditioner_update_initial=5,
        preconditioner_update_interval=10,
    )

    assert maprl.preconditioner is not None
    assert maprl.preconditioner_update_initial == 5
    assert maprl.preconditioner_update_interval == 10
    assert callable(maprl.preconditioner)
    assert maprl._preconditioner_image is None  # Not computed yet


def test_maprl_preconditioner_update_initial_iterations(geometry):
    """The preconditioner is refreshed on the first N actual updates."""
    calls = []
    holder = {}

    def test_preconditioner(x):
        calls.append(holder["algorithm"]._update_count)
        return make_image(x.geometry, 0.5)

    maprl = make_maprl(
        geometry,
        preconditioner=test_preconditioner,
        preconditioner_update_initial=3,
        preconditioner_update_interval=10,
    )
    holder["algorithm"] = maprl
    maprl.run(iterations=5, verbose=0)

    assert maprl._update_count == 5
    assert calls == [1, 2, 3]


def test_maprl_preconditioner_update_periodic(geometry):
    """The preconditioner is refreshed periodically after the initial window."""
    calls = []
    holder = {}

    def test_preconditioner(x):
        calls.append(holder["algorithm"]._update_count)
        return make_image(x.geometry, 0.5)

    maprl = make_maprl(
        geometry,
        preconditioner=test_preconditioner,
        preconditioner_update_initial=2,
        preconditioner_update_interval=5,
    )
    holder["algorithm"] = maprl
    maprl.run(iterations=15, verbose=0)

    assert calls == [1, 2, 5, 10, 15]


def test_maprl_preconditioner_zero_interval_only_initialises(geometry):
    """A zero interval disables periodic refresh but still initialises on first use."""
    calls = []
    holder = {}

    def test_preconditioner(x):
        calls.append(holder["algorithm"]._update_count)
        return make_image(x.geometry, 0.5)

    maprl = make_maprl(
        geometry,
        preconditioner=test_preconditioner,
        preconditioner_update_initial=0,
        preconditioner_update_interval=0,
    )
    holder["algorithm"] = maprl
    maprl.run(iterations=6, verbose=0)

    assert calls == [1]
    assert maprl._preconditioner_image is not None


def test_maprl_static_preconditioner_schedule(geometry):
    """A static preconditioner is applied on the same schedule as a callable one."""
    static_preconditioner = make_image(geometry, 2.0)
    maprl = RecordingMAPRL(
        initial_estimate=make_image(geometry, 1.0),
        data_fidelity=ConstantFunctional(0.1, 4.0),
        prior=ConstantFunctional(0.0, 1.0),
        step_size=1.0,
        initial_line_search=False,
        armijo_iterations=0,
        preconditioner=static_preconditioner,
        preconditioner_update_initial=2,
        preconditioner_update_interval=5,
    )

    maprl.run(iterations=12, verbose=0)

    assert maprl.preconditioner_updates == [1, 2, 5, 10]
    assert maprl._preconditioner_image is static_preconditioner


def test_maprl_preconditioner_application(geometry):
    """The preconditioner multiplies the gradient in the update step."""
    maprl = make_maprl(
        geometry,
        data_fidelity=ConstantFunctional(1.0, 4.0),
        prior=ConstantFunctional(0.0, 1.0),
        step_size=0.1,
        eps=1e-8,
        relaxation_eta=0.01,
        preconditioner=lambda x: make_image(x.geometry, 2.0),
        preconditioner_update_initial=1,
    )

    initial_values = maprl.x.as_array().copy()
    maprl.run(iterations=1, verbose=0)

    step_actual = 0.1 / (1 + 0.01 * 1)
    expected = initial_values - 1.0 * 2.0 * step_actual
    expected = np.maximum(expected, 0.0)

    assert np.allclose(maprl.x.as_array(), expected, rtol=1e-5)


def test_maprl_without_preconditioner(geometry):
    """MAPRL works correctly without a preconditioner."""
    maprl = make_maprl(
        geometry,
        data_fidelity=ConstantFunctional(0.5, 4.0),
        prior=ConstantFunctional(0.0, 1.0),
        step_size=0.1,
        eps=1e-8,
        relaxation_eta=0.01,
        preconditioner=None,
    )

    initial_values = maprl.x.as_array().copy()
    maprl.run(iterations=1, verbose=0)

    step_actual = 0.1 / (1 + 0.01 * 1)
    expected = initial_values - (initial_values + 1e-8) * 0.5 * step_actual
    expected = np.maximum(expected, 0.0)

    assert np.allclose(maprl.x.as_array(), expected, rtol=1e-5)


def test_maprl_preconditioner_static_image(geometry):
    """MAPRL works with a static ImageData preconditioner."""
    static_preconditioner = make_image(geometry, 2.0)
    maprl = make_maprl(
        geometry,
        data_fidelity=ConstantFunctional(1.0, 4.0),
        prior=ConstantFunctional(0.0, 1.0),
        step_size=0.1,
        eps=1e-8,
        preconditioner=static_preconditioner,
    )

    maprl.run(iterations=1, verbose=0)

    assert maprl._preconditioner_image is not None
    assert np.allclose(maprl._preconditioner_image.as_array(), static_preconditioner.as_array())


def construct_recording_warnings(geometry, **kwargs):
    """Build a plain MAPRL through ``make_maprl`` and collect every warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maprl = make_maprl(geometry, **kwargs)
    return maprl, caught


def test_eps_preconditioner_omitted_emits_no_warning(geometry):
    """Omitting the argument stays silent and keeps the legacy stored value."""
    maprl, caught = construct_recording_warnings(geometry)

    assert caught == []
    assert maprl.eps_preconditioner == 1e-6


@pytest.mark.parametrize("value", [1e-6, 0, 1e-3, 1e6])
def test_eps_preconditioner_explicit_value_warns_once(geometry, value):
    """An explicit value warns exactly once, from the caller's frame."""
    maprl, caught = construct_recording_warnings(geometry, eps_preconditioner=value)

    assert len(caught) == 1
    assert issubclass(caught[0].category, FutureWarning)
    assert EPS_PRECONDITIONER_DEPRECATION in str(caught[0].message)
    # stacklevel=2 points at the constructing frame, not inside maprl.py.
    assert caught[0].filename == __file__
    assert maprl.eps_preconditioner == value

    with warnings.catch_warnings(record=True) as during_run:
        warnings.simplefilter("always")
        maprl.run(iterations=2, verbose=0)

    assert during_run == []


@pytest.mark.parametrize("initial_line_search", [False, True])
@pytest.mark.parametrize("preconditioner_kind", ["absent", "static", "callable"])
def test_eps_preconditioner_changes_no_behaviour(
    geometry, preconditioner_kind, initial_line_search
):
    """The deprecated value leaves the update, the search and the map alone."""
    supplied_map = np.array([[0.0, 0.5, 1.0, 2.0]] * 4, dtype=np.float64)
    recorded_maps = []

    def callable_preconditioner(x):
        image = x.geometry.allocate(0.0, dtype=np.float64)
        image.as_array()[...] = supplied_map
        recorded_maps.append(image.as_array().copy())
        return image

    static_preconditioner = geometry.allocate(0.0, dtype=np.float64)
    static_preconditioner.as_array()[...] = supplied_map
    preconditioner = {
        "absent": None,
        "static": static_preconditioner,
        "callable": callable_preconditioner,
    }[preconditioner_kind]

    def build(eps_preconditioner):
        kwargs = dict(
            initial_estimate=make_image(geometry, 1.0),
            data_fidelity=QuadraticFunctional(make_image(geometry, 2.0)),
            prior=ConstantFunctional(0.0, 1.0),
            step_size=1.0,
            eps=0.0,
            initial_line_search=initial_line_search,
            armijo_iterations=0,
            preconditioner=preconditioner,
            preconditioner_update_initial=2,
            preconditioner_update_interval=1,
        )
        if eps_preconditioner is not None:
            kwargs["eps_preconditioner"] = eps_preconditioner
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            maprl = MAPRL(**kwargs)
        return maprl, caught

    baseline, baseline_warnings = build(None)
    deprecated, deprecated_warnings = build(1e-3)

    assert baseline_warnings == []
    assert len(deprecated_warnings) == 1
    assert issubclass(deprecated_warnings[0].category, FutureWarning)
    assert EPS_PRECONDITIONER_DEPRECATION in str(deprecated_warnings[0].message)
    assert deprecated.eps_preconditioner == 1e-3

    baseline.run(iterations=3, verbose=0)
    deprecated.run(iterations=3, verbose=0)

    # The run must actually move x, otherwise the comparison is vacuous.
    assert not np.array_equal(baseline.x.as_array(), make_image(geometry, 1.0).as_array())
    assert np.array_equal(deprecated.x.as_array(), baseline.x.as_array())
    assert deprecated._current_step_size == baseline._current_step_size
    assert deprecated.initial_step_size == baseline.initial_step_size

    # The supplied map is applied verbatim: zeros included, no floor.
    assert np.array_equal(static_preconditioner.as_array(), supplied_map)
    if preconditioner_kind == "callable":
        assert recorded_maps
        for recorded in recorded_maps:
            assert np.array_equal(recorded, supplied_map)
        assert np.array_equal(
            deprecated._preconditioner_image.as_array(), supplied_map
        )


def test_preconditioner_map_with_zeros_gets_no_floor(geometry):
    """A supplied map containing exact zeros is applied without any floor."""
    supplied_map = np.array([[0.0, 1.0, 0.0, 2.0]] * 4, dtype=np.float64)

    def preconditioner(x):
        image = x.geometry.allocate(0.0, dtype=np.float64)
        image.as_array()[...] = supplied_map
        return image

    maprl = make_maprl(
        geometry,
        data_fidelity=ConstantFunctional(0.5, 4.0),
        prior=ConstantFunctional(0.0, 1.0),
        step_size=0.1,
        relaxation_eta=0.0,
        preconditioner=preconditioner,
        preconditioner_update_initial=1,
    )

    initial = maprl.x.as_array().copy()
    maprl.run(iterations=1, verbose=0)

    stored = maprl._preconditioner_image.as_array()
    assert np.array_equal(stored, supplied_map)

    expected = initial - (0.5 * supplied_map) * 0.1
    np.testing.assert_allclose(maprl.x.as_array(), expected, rtol=1e-15, atol=1e-15)

    zero_mask = stored == 0.0
    assert np.any(zero_mask)
    assert np.array_equal(maprl.x.as_array()[zero_mask], initial[zero_mask])


def test_parallel_sum_preconditioner(geometry):
    """The parallel-sum preconditioner is applied correctly."""
    initial = make_image(geometry, 1.0)
    initial.fill(np.array([[1.0, 2.0, 3.0, 4.0]] * 4))

    def parallel_preconditioner(x):
        D = x.as_array()
        R = 0.5
        return make_image(x.geometry, (D * R) / (D + R))

    maprl = MAPRL(
        initial_estimate=initial,
        data_fidelity=ConstantFunctional(0.5, 4.0),
        prior=ConstantFunctional(0.0, 1.0),
        step_size=0.1,
        eps=1e-8,
        relaxation_eta=0.0,
        initial_line_search=False,
        armijo_iterations=0,
        preconditioner=parallel_preconditioner,
        preconditioner_update_initial=1,
    )

    initial_values = initial.as_array().copy()
    maprl.run(iterations=1, verbose=0)

    D = initial_values
    R = 0.5
    P = (D * R) / (D + R)
    expected = initial_values - 0.5 * P * 0.1
    expected = np.maximum(expected, 0.0)

    np.testing.assert_allclose(
        maprl._preconditioner_image.as_array(), P, rtol=1e-15, atol=0.0
    )
    np.testing.assert_allclose(
        maprl.x.as_array(), expected, rtol=1e-15, atol=1e-15
    )


def test_maprl_armijo_update_periodic(geometry):
    """Armijo line searches fire on the initial and periodic update schedule."""
    armijo_updates = []

    original_armijo_step = MAPRL._armijo_step

    def tracked_armijo_step(self, suggested_step):
        armijo_updates.append(self._update_count)
        return original_armijo_step(self, suggested_step)

    MAPRL._armijo_step = tracked_armijo_step

    try:
        maprl = make_maprl(
            geometry,
            step_size=1.0,
            armijo_iterations=25,
            armijo_update_initial=3,
            armijo_update_interval=5,
        )
        maprl.run(iterations=20, verbose=0)
    finally:
        MAPRL._armijo_step = original_armijo_step

    assert armijo_updates == [1, 2, 3, 5, 10, 15, 20]


def test_maprl_armijo_zero_interval_disables_periodic(geometry):
    """A zero Armijo interval keeps only the initial window."""
    armijo_updates = []

    original_armijo_step = MAPRL._armijo_step

    def tracked_armijo_step(self, suggested_step):
        armijo_updates.append(self._update_count)
        return original_armijo_step(self, suggested_step)

    MAPRL._armijo_step = tracked_armijo_step

    try:
        maprl = make_maprl(
            geometry,
            step_size=1.0,
            armijo_iterations=25,
            armijo_update_initial=2,
            armijo_update_interval=0,
        )
        maprl.run(iterations=10, verbose=0)
    finally:
        MAPRL._armijo_step = original_armijo_step

    assert armijo_updates == [1, 2]


def test_maprl_schedules_resume_across_runs(geometry):
    """Preconditioner and Armijo schedules keep counting across separate runs."""
    preconditioner_calls = []
    armijo_updates = []
    holder = {}

    def test_preconditioner(x):
        preconditioner_calls.append(holder["algorithm"]._update_count)
        return make_image(x.geometry, 0.5)

    original_armijo_step = MAPRL._armijo_step

    def tracked_armijo_step(self, suggested_step):
        armijo_updates.append(self._update_count)
        return original_armijo_step(self, suggested_step)

    MAPRL._armijo_step = tracked_armijo_step

    try:
        maprl = make_maprl(
            geometry,
            step_size=1.0,
            armijo_iterations=25,
            armijo_update_initial=1,
            armijo_update_interval=3,
            preconditioner=test_preconditioner,
            preconditioner_update_initial=1,
            preconditioner_update_interval=3,
        )
        holder["algorithm"] = maprl

        maprl.run(iterations=2, verbose=0)
        assert maprl._update_count == 2
        assert preconditioner_calls == [1]
        assert armijo_updates == [1]

        maprl.run(iterations=4, verbose=0)
        assert maprl._update_count == 6
        assert preconditioner_calls == [1, 3, 6]
        assert armijo_updates == [1, 3, 6]
    finally:
        MAPRL._armijo_step = original_armijo_step


def assert_initial_armijo_inequality(
    initial, data_fidelity, prior, accepted_step, sigma, preconditioner_image=None, eps=0.0
):
    """Independently verify the Armijo condition at the accepted initial step."""
    grad = data_fidelity.gradient(initial) + prior.gradient(initial)
    if preconditioner_image is not None:
        direction = grad * preconditioner_image
    else:
        direction = (initial + eps) * grad
    slope = float(np.sum(direction.as_array() * grad.as_array()))
    reference_loss = float(data_fidelity(initial) + prior(initial))
    candidate = initial - direction * accepted_step
    candidate.maximum(0, out=candidate)
    candidate_loss = float(data_fidelity(candidate) + prior(candidate))
    assert candidate_loss <= reference_loss - sigma * accepted_step * slope + 1e-8


def make_initial_search_maprl(geometry, preconditioner=None):
    return MAPRL(
        initial_estimate=make_image(geometry, 1.0),
        data_fidelity=QuadraticFunctional(make_image(geometry, 2.0)),
        prior=ConstantFunctional(0.0, 1.0),
        step_size=1.0,
        eps=0.0,
        initial_line_search=True,
        armijo_beta=0.5,
        armijo_sigma=1e-4,
        preconditioner=preconditioner,
    )


def test_static_preconditioner_sets_initial_armijo_direction(geometry):
    """A static preconditioner changes the constructor's accepted Armijo step."""
    initial = make_image(geometry, 1.0)
    preconditioner = make_image(geometry, 10.0)
    data_fidelity = QuadraticFunctional(make_image(geometry, 2.0))
    prior = ConstantFunctional(0.0, 1.0)

    em = make_initial_search_maprl(geometry)
    preconditioned = make_initial_search_maprl(geometry, preconditioner=preconditioner)

    # EM direction accepts the suggested step; the scaled preconditioner
    # direction backtracks two halvings further.
    assert em._current_step_size == pytest.approx(1.0)
    assert preconditioned._current_step_size == pytest.approx(0.125)

    assert_initial_armijo_inequality(
        initial, data_fidelity, prior, em._current_step_size, sigma=1e-4
    )
    assert_initial_armijo_inequality(
        initial,
        data_fidelity,
        prior,
        preconditioned._current_step_size,
        sigma=1e-4,
        preconditioner_image=preconditioner,
    )


def test_callable_preconditioner_sets_initial_armijo_direction(geometry):
    """A callable preconditioner is used for the constructor's initial search."""
    calls = []

    def preconditioner(x):
        calls.append(x.as_array().copy())
        return make_image(x.geometry, 10.0)

    maprl = make_initial_search_maprl(geometry, preconditioner=preconditioner)

    assert maprl._current_step_size == pytest.approx(0.125)
    assert len(calls) == 1
    assert np.allclose(calls[0], maprl.x.as_array())

    assert_initial_armijo_inequality(
        make_image(geometry, 1.0),
        QuadraticFunctional(make_image(geometry, 2.0)),
        ConstantFunctional(0.0, 1.0),
        maprl._current_step_size,
        sigma=1e-4,
        preconditioner_image=make_image(geometry, 10.0),
    )


def test_constructor_initial_search_preserves_initial_state(geometry):
    """Construction leaves x at the initial estimate and performs no updates."""
    initial = make_image(geometry, 1.0)
    maprl = make_initial_search_maprl(geometry, preconditioner=make_image(geometry, 10.0))

    assert np.allclose(maprl.x.as_array(), initial.as_array())
    assert maprl._update_count == 0


def test_constructor_leaves_lazy_preconditioner_untouched(geometry):
    """Without initial line search the callable preconditioner is not called."""
    calls = []

    def preconditioner(x):
        calls.append(x)
        return make_image(x.geometry, 10.0)

    maprl = make_maprl(
        geometry,
        preconditioner=preconditioner,
        preconditioner_update_initial=5,
    )

    assert calls == []
    assert maprl._preconditioner_image is None


def test_armijo_rejection_does_not_move_to_worse_point(geometry):
    """A failed in-run line search must not apply the rejected candidate.

    With a 10x preconditioner every backtrack from x=1 is uphill, so the
    Armijo search finds no acceptable step and ``x`` must be left unchanged.
    """
    initial = make_image(geometry, 1.0)
    data_fidelity = QuadraticFunctional(make_image(geometry, 2.0))
    prior = ConstantFunctional(0.0, 1.0)

    maprl = MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fidelity,
        prior=prior,
        step_size=1.0,
        eps=0.0,
        initial_line_search=False,
        armijo_beta=0.75,
        armijo_sigma=1e-1,
        armijo_max_iter=1,
        armijo_iterations=25,
        preconditioner=make_image(geometry, 10.0),
        preconditioner_update_initial=1,
    )

    reference_loss = data_fidelity(initial) + prior(initial)
    maprl.run(iterations=1, verbose=0)

    assert np.allclose(maprl.x.as_array(), initial.as_array())
    assert data_fidelity(maprl.x) + prior(maprl.x) <= reference_loss
    assert maprl._current_step_size == pytest.approx(1.0)


def test_initial_armijo_rejection_does_not_seed_rejected_step(geometry):
    """The constructor must not adopt a rejected backtrack as its initial step.

    Two backtracks (1.0 and 0.75) are tried and both are uphill, so the
    constructor should keep the suggested step rather than the last rejected
    one.
    """
    initial = make_image(geometry, 1.0)
    data_fidelity = QuadraticFunctional(make_image(geometry, 2.0))
    prior = ConstantFunctional(0.0, 1.0)

    maprl = MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fidelity,
        prior=prior,
        step_size=1.0,
        eps=0.0,
        initial_line_search=True,
        armijo_beta=0.75,
        armijo_sigma=1e-1,
        armijo_max_iter=2,
        preconditioner=make_image(geometry, 10.0),
    )

    reference_loss = data_fidelity(initial) + prior(initial)

    assert maprl.initial_step_size == pytest.approx(1.0)
    assert maprl._current_step_size == pytest.approx(1.0)
    assert np.allclose(maprl.x.as_array(), initial.as_array())
    assert data_fidelity(maprl.x) + prior(maprl.x) <= reference_loss


def test_armijo_accepts_improving_candidate(geometry):
    """A candidate satisfying the Armijo test is still applied."""
    initial = make_image(geometry, 1.0)
    data_fidelity = QuadraticFunctional(make_image(geometry, 2.0))
    prior = ConstantFunctional(0.0, 1.0)

    maprl = MAPRL(
        initial_estimate=initial,
        data_fidelity=data_fidelity,
        prior=prior,
        step_size=1.0,
        eps=0.0,
        initial_line_search=False,
        armijo_beta=0.5,
        armijo_sigma=1e-1,
        armijo_max_iter=20,
        armijo_iterations=25,
    )

    maprl.run(iterations=1, verbose=0)

    assert maprl._current_step_size == pytest.approx(1.0)
    assert np.allclose(maprl.x.as_array(), 2.0)
    assert data_fidelity(maprl.x) + prior(maprl.x) < data_fidelity(initial) + prior(initial)


def test_armijo_skips_nan_trial_and_applies_finite_candidate(geometry):
    """A non-finite first trial must not mask a later usable candidate.

    Trial 1 (step 1.0) lands on NaN; trial 2 (step 0.5) lands on a finite loss
    that beats the reference but misses the Armijo margin. The finite trial
    must be applied rather than discarded because NaN was tracked as best.
    """
    data_fidelity = NaNFirstTrialFunctional(nan_value=2.0, finite_value=1.5, finite_loss=0.5)
    prior = ConstantFunctional(0.0, 0.0)

    maprl = MAPRL(
        initial_estimate=make_image(geometry, 1.0),
        data_fidelity=data_fidelity,
        prior=prior,
        step_size=1.0,
        eps=0.0,
        initial_line_search=False,
        armijo_beta=0.5,
        armijo_sigma=1e-1,
        armijo_max_iter=2,
        armijo_iterations=25,
    )

    maprl.run(iterations=1, verbose=0)

    assert maprl._current_step_size == pytest.approx(0.5)
    assert np.allclose(maprl.x.as_array(), 1.5)
    assert data_fidelity(maprl.x) == pytest.approx(0.5)


def test_initial_armijo_skips_nan_trial_and_applies_finite_candidate(geometry):
    """The constructor's initial search must also skip a non-finite trial."""
    data_fidelity = NaNFirstTrialFunctional(nan_value=2.0, finite_value=1.5, finite_loss=0.5)
    prior = ConstantFunctional(0.0, 0.0)

    maprl = MAPRL(
        initial_estimate=make_image(geometry, 1.0),
        data_fidelity=data_fidelity,
        prior=prior,
        step_size=1.0,
        eps=0.0,
        initial_line_search=True,
        armijo_beta=0.5,
        armijo_sigma=1e-1,
        armijo_max_iter=2,
    )

    assert maprl.initial_step_size == pytest.approx(0.5)
    assert maprl._current_step_size == pytest.approx(0.5)
    assert np.allclose(maprl.x.as_array(), 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relaxation_eta": -0.01},
        {"relaxation_eta": np.nan},
        {"relaxation_eta": np.inf},
        {"step_size": 0.0},
        {"step_size": -1.0},
        {"step_size": np.inf},
    ],
)
def test_maprl_rejects_invalid_step_parameters(geometry, kwargs):
    """A negative relaxation_eta makes the decay denominator cross zero and
    step_size() return a negative step, so both parameters must be validated."""
    with pytest.raises(ValueError):
        make_maprl(geometry, **kwargs)


def test_maprl_positive_relaxation_eta_keeps_step_non_negative(geometry):
    maprl = make_maprl(geometry, step_size=1.0, relaxation_eta=0.01)

    steps = []
    original_step_size = MAPRL.step_size

    def tracked_step_size(self):
        value = original_step_size(self)
        steps.append(value)
        return value

    MAPRL.step_size = tracked_step_size

    try:
        maprl.run(iterations=100, verbose=0)
    finally:
        MAPRL.step_size = original_step_size

    assert steps
    assert all(step >= 0.0 for step in steps)

