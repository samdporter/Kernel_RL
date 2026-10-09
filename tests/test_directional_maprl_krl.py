"""Tests for the directional operator, MAPRL updates and RL freezing.

The operator and algorithm tests use real CIL containers (``ImageData`` and
``BlockDataContainer``) so that the algorithms are exercised on the same data
structures they use in production.
"""

import numpy as np
import pytest
from cil.framework import BlockDataContainer, ImageGeometry

from krl.algorithms.maprl import MAPRL
from krl.algorithms.richardson_lucy import RichardsonLucy
from krl.operators.blurring import create_gaussian_blur
from krl.operators.directional import DirectionalOperator
from krl.operators.kernel_operator import get_kernel_operator


def make_image(geometry, data):
    image = geometry.allocate(0.0, dtype=np.float64)
    image.fill(np.asarray(data, dtype=np.float64))
    return image


@pytest.fixture
def directional_geometry():
    return ImageGeometry(voxel_num_x=2, voxel_num_y=2, voxel_num_z=1, dtype=np.float64)


def test_directional_operator_xi_normalization(directional_geometry):
    anatomical = BlockDataContainer(
        make_image(directional_geometry, [[3.0, 4.0], [0.0, 5.0]]),
        make_image(directional_geometry, [[4.0, 0.0], [0.0, 0.0]]),
    )
    operator = DirectionalOperator(anatomical, gamma=0.5, eta=0.2)

    sum_squares = sum(container.as_array() ** 2 for container in anatomical.containers)
    expected_norm = np.sqrt(sum_squares + 0.2**2)

    for xi_component, anatomical_component in zip(operator.xi.containers, anatomical.containers):
        assert np.allclose(
            xi_component.as_array(),
            anatomical_component.as_array() / expected_norm,
        )


def test_directional_operator_direct_writes_to_out(directional_geometry):
    anatomical = BlockDataContainer(
        make_image(directional_geometry, [[1.0, 2.0], [0.0, 1.0]]),
        make_image(directional_geometry, [[0.5, 0.5], [0.5, 0.5]]),
    )
    operator = DirectionalOperator(anatomical, gamma=0.3, eta=0.1)

    test_block = BlockDataContainer(
        make_image(directional_geometry, [[2.0, -1.0], [0.5, 0.0]]),
        make_image(directional_geometry, [[0.0, 1.5], [-0.5, 1.0]]),
    )
    out = test_block.clone()
    operator.direct(test_block, out=out)

    dot_val = operator.dot(operator.xi, test_block)
    expected = test_block - operator.gamma * operator.xi * dot_val

    for out_comp, exp_comp in zip(out.containers, expected.containers):
        assert np.allclose(out_comp.as_array(), exp_comp.as_array())


def test_directional_operator_dot_resets_accumulator(directional_geometry):
    anatomical = BlockDataContainer(
        make_image(directional_geometry, [[1.0, 1.0], [1.0, 1.0]]),
        make_image(directional_geometry, [[1.0, 1.0], [1.0, 1.0]]),
    )
    operator = DirectionalOperator(anatomical, gamma=1.0, eta=0.01)

    block_a = BlockDataContainer(
        make_image(directional_geometry, [[2.0, 3.0], [0.0, 1.0]]),
        make_image(directional_geometry, [[4.0, 5.0], [1.0, 0.0]]),
    )
    block_b = BlockDataContainer(
        make_image(directional_geometry, [[0.5, 1.0], [1.0, 0.5]]),
        make_image(directional_geometry, [[1.5, 2.0], [0.0, 1.0]]),
    )

    dot_a = operator.dot(block_a, block_a).as_array()
    expected_a = (
        block_a.containers[0].as_array() * block_a.containers[0].as_array()
        + block_a.containers[1].as_array() * block_a.containers[1].as_array()
    )
    assert np.allclose(dot_a, expected_a)

    dot_b = operator.dot(block_a, block_b).as_array()
    expected_b = (
        block_a.containers[0].as_array() * block_b.containers[0].as_array()
        + block_a.containers[1].as_array() * block_b.containers[1].as_array()
    )
    assert np.allclose(dot_b, expected_b)


class LinearGradientFunctional:
    def __init__(self, gradient, value):
        self._gradient = gradient
        self._value = value

    def gradient(self, x):
        return make_image(x.geometry, self._gradient)

    def __call__(self, x):
        return float(self._value)


def test_maprl_update_combines_data_and_prior_gradients():
    geometry = ImageGeometry(voxel_num_x=2, voxel_num_y=2, voxel_num_z=1, dtype=np.float64)
    initial = make_image(geometry, [[1.0, 2.0], [0.5, 1.5]])

    algorithm = MAPRL(
        initial_estimate=initial,
        data_fidelity=LinearGradientFunctional([[0.2, -0.1], [0.1, 0.3]], 3.0),
        prior=LinearGradientFunctional([[-0.3, 0.5], [0.0, -0.2]], 5.0),
        step_size=0.4,
        relaxation_eta=0.0,
        eps=0.2,
        initial_line_search=False,
        armijo_iterations=0,
    )
    algorithm.run(iterations=1, verbose=0)

    combined_grad = np.array([[0.2, -0.1], [0.1, 0.3]]) + np.array([[-0.3, 0.5], [0.0, -0.2]])
    expected = initial.as_array() - (initial.as_array() + 0.2) * combined_grad * 0.4
    expected = np.maximum(expected, 0.0)

    assert algorithm._update_count == 1
    assert np.allclose(algorithm.x.as_array(), expected)
    assert np.allclose(initial.as_array(), np.array([[1.0, 2.0], [0.5, 1.5]]))


def test_maprl_update_projects_negative_values():
    geometry = ImageGeometry(voxel_num_x=2, voxel_num_y=2, voxel_num_z=1, dtype=np.float64)
    initial = make_image(geometry, [[0.1, 0.1], [0.1, 0.1]])

    algorithm = MAPRL(
        initial_estimate=initial,
        data_fidelity=LinearGradientFunctional([[5.0, 5.0], [5.0, 5.0]], 1.0),
        prior=LinearGradientFunctional([[0.0, 0.0], [0.0, 0.0]], 0.0),
        step_size=1.0,
        relaxation_eta=0.0,
        eps=0.0,
        initial_line_search=False,
        armijo_iterations=0,
    )

    algorithm.run(iterations=1, verbose=0)
    assert np.allclose(algorithm.x.as_array(), np.zeros_like(initial.as_array()))


class FreezeRecorder:
    def __init__(self):
        self.records = []

    def __call__(self, algorithm):
        self.records.append(
            (algorithm._update_count, algorithm.kernel_operator.freeze_emission_kernel)
        )


@pytest.fixture
def freeze_setup():
    geometry = ImageGeometry(voxel_num_x=12, voxel_num_y=12, voxel_num_z=6, dtype=np.float32)
    phantom = geometry.allocate(0.0)
    array = np.zeros((6, 12, 12), dtype=np.float32)
    z, y, x = np.indices(array.shape)
    array += 50 * np.exp(-((z - 2) ** 2 + (y - 4) ** 2 + (x - 4) ** 2) / 4.0)
    phantom.fill(array)

    blur = create_gaussian_blur(sigma=(1.0, 1.0, 1.0), geometry=geometry, backend="numba")
    observed = blur.direct(phantom)
    return geometry, phantom, blur, observed


def make_kernel(geometry, anatomical):
    kernel = get_kernel_operator(
        geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        use_mask=True,
        mask_k=10,
        normalize_kernel=True,
        hybrid=False,
    )
    kernel.set_anatomical_image(anatomical)
    return kernel


@pytest.mark.parametrize("update_objective_interval", [0, 1, 3])
@pytest.mark.parametrize("freeze_iteration", [0, 1, 2])
def test_rl_freeze_schedule(freeze_setup, freeze_iteration, update_objective_interval):
    geometry, phantom, blur, observed = freeze_setup
    kernel = make_kernel(geometry, phantom)
    recorder = FreezeRecorder()

    rl = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
        kernel_operator=kernel,
        freeze_iteration=freeze_iteration,
        update_objective_interval=update_objective_interval,
    )
    rl.run(iterations=4, verbose=0, callbacks=[recorder])

    assert rl._update_count == 4
    frozen_at = [count for count, frozen in recorder.records if frozen]

    if freeze_iteration == 0:
        assert frozen_at == []
        assert not kernel.freeze_emission_kernel
    else:
        assert frozen_at == list(range(freeze_iteration, 5))
        assert kernel.freeze_emission_kernel
        assert np.all(np.isfinite(rl.x.as_array()))


def test_rl_freeze_resumes_across_runs(freeze_setup):
    geometry, phantom, blur, observed = freeze_setup
    kernel = make_kernel(geometry, phantom)
    recorder = FreezeRecorder()

    rl = RichardsonLucy(
        initial_estimate=observed,
        blurring_operator=blur,
        observed_data=observed,
        kernel_operator=kernel,
        freeze_iteration=2,
    )

    rl.run(iterations=1, verbose=0, callbacks=[recorder])
    assert rl._update_count == 1
    assert not kernel.freeze_emission_kernel

    rl.run(iterations=3, verbose=0, callbacks=[recorder])
    assert rl._update_count == 4
    assert kernel.freeze_emission_kernel
    assert [count for count, frozen in recorder.records if frozen] == [2, 3, 4]
