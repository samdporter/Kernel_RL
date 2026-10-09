"""Tests for MAPRL preconditioner and Armijo scheduling functionality."""

import numpy as np
import pytest
from cil.framework import ImageGeometry

from krl.algorithms.maprl import MAPRL


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
    P = (D * R) / (D + R + 1e-6)
    P = np.maximum(P, 1e-6)
    expected = initial_values - 0.5 * P * 0.1
    expected = np.maximum(expected, 0.0)

    assert np.allclose(maprl.x.as_array(), expected, rtol=1e-5)


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

