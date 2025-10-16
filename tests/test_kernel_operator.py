import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

from src.my_kem import (
    DEFAULT_PARAMETERS,
    NUMBA_AVAIL,
    SLIDING_WINDOW_AVAIL,
    KernelOperator,
    get_kernel_operator,
)


@dataclass
class DummyGeometry:
    shape: Tuple[int, int, int]

    def allocate(self, value: float = 0.0):
        data = np.full(self.shape, value, dtype=np.float64)
        return DummyImage(data)


class DummyImage:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=np.float64)

    @property
    def shape(self):
        return self._data.shape

    def as_array(self):
        return self._data

    def clone(self):
        return DummyImage(self._data.copy())

    def fill(self, values):
        self._data[...] = np.asarray(values, dtype=np.float64)


def available_backends():
    backends = ["python"]
    if NUMBA_AVAIL:
        backends.append("numba")
    return backends


@pytest.fixture
def geometry():
    return DummyGeometry((5, 5, 5))


@pytest.fixture
def anatomical_uniform(geometry):
    return geometry.allocate(1.0)


@pytest.fixture
def emission_spike(geometry):
    arr = np.zeros(geometry.shape, dtype=np.float64)
    arr[2, 2, 2] = 10.0
    return DummyImage(arr)


@pytest.fixture
def emission_random(geometry):
    rng = np.random.default_rng(42)
    arr = rng.normal(size=geometry.shape)
    return DummyImage(arr)


@pytest.fixture
def anatomical_image_gradient(geometry):
    img = geometry.allocate(0.0)
    grad = np.indices(geometry.shape).sum(axis=0)
    img.fill(grad)
    return img


@pytest.fixture
def emission_image_uniform(geometry):
    return geometry.allocate(1.0)


@pytest.mark.parametrize("backend", available_backends())
def test_uniform_identity(anatomical_uniform, geometry, backend):
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=1.0,
        sigma_dist=1.0,
        normalize_kernel=True,
    )
    operator.set_anatomical_image(anatomical_uniform)
    emission = geometry.allocate(5.0)

    result = operator.direct(emission)
    assert np.allclose(result.as_array(), 5.0, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("backend", available_backends())
def test_smoothing_effect(anatomical_uniform, emission_spike, geometry, backend):
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        sigma_dist=1.0,
        normalize_kernel=True,
    )
    operator.set_anatomical_image(anatomical_uniform)

    result = operator.direct(emission_spike)
    result_arr = result.as_array()
    assert result_arr.max() < emission_spike.as_array().max()
    assert np.var(result_arr) < np.var(emission_spike.as_array())


@pytest.mark.parametrize("backend", available_backends())
def test_adjoint_dot_product(geometry, backend):
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.5,
        sigma_dist=1.0,
        normalize_kernel=False,
        use_mask=False,
        hybrid=False,
    )
    anat = geometry.allocate(0.0)
    grid = np.indices(geometry.shape).sum(axis=0) / math.prod(geometry.shape)
    anat.fill(grid)
    operator.set_anatomical_image(anat)

    rng = np.random.default_rng(7)
    x = DummyImage(rng.normal(size=geometry.shape))
    y = DummyImage(rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.allclose(dot_forward, dot_adjoint, atol=1e-6, rtol=1e-5)

@pytest.mark.parametrize("backend", available_backends())
def test_hybrid_adjoint_dot_product(geometry, backend):
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.5,
        sigma_dist=1.0,
        normalize_kernel=False,
        use_mask=False,
        hybrid=True,
    )
    anat = geometry.allocate(0.0)
    grid = np.indices(geometry.shape).sum(axis=0) / math.prod(geometry.shape)
    anat.fill(grid)
    operator.set_anatomical_image(anat)

    rng = np.random.default_rng(7)
    x = DummyImage(rng.normal(size=geometry.shape))
    y = DummyImage(rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.allclose(dot_forward, dot_adjoint, atol=1e-6, rtol=1e-5)

@pytest.mark.parametrize("backend", available_backends())
def test_hybrid_adjoint_mormalized_dot_product(geometry, backend):
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.5,
        sigma_dist=1.0,
        normalize_kernel=True,
        use_mask=False,
        hybrid=True,
    )
    anat = geometry.allocate(0.0)
    grid = np.indices(geometry.shape).sum(axis=0) / math.prod(geometry.shape)
    anat.fill(grid)
    operator.set_anatomical_image(anat)

    rng = np.random.default_rng(7)
    x = DummyImage(rng.normal(size=geometry.shape))
    y = DummyImage(rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.allclose(dot_forward, dot_adjoint, atol=1e-6, rtol=1e-5)

def test_mask_requires_sliding_window(geometry):
    operator = KernelOperator(geometry)
    operator.set_parameters({**DEFAULT_PARAMETERS, "use_mask": True})
    operator.set_anatomical_image(geometry.allocate(0.0))

    if SLIDING_WINDOW_AVAIL:
        result = operator.direct(geometry.allocate(1.0))
        assert isinstance(result, DummyImage)
    else:
        with pytest.raises(RuntimeError):
            operator.direct(geometry.allocate(1.0))


@pytest.mark.skipif(not SLIDING_WINDOW_AVAIL, reason="Masking requires sliding_window_view")
@pytest.mark.parametrize("backend", available_backends())
def test_mask_selects_expected_neighbours(geometry, backend, anatomical_image_gradient, emission_image_uniform):
    mask_k = 5
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.2,
        sigma_dist=0.5,
        normalize_kernel=False,
        use_mask=True,
        mask_k=mask_k,
        recalc_mask=False,
    )
    operator.set_anatomical_image(anatomical_image_gradient)

    operator.direct(emission_image_uniform)
    assert operator.mask is not None
    mask = operator.mask
    assert mask.shape[-1] == operator.parameters["num_neighbours"] ** 3
    counts = mask.reshape(-1, mask.shape[-1]).sum(axis=-1)
    assert np.all(counts >= mask_k)
    assert np.all(counts <= mask.shape[-1])
    assert counts.mean() < mask.shape[-1]


@pytest.mark.parametrize("backend", available_backends())
def test_normalized_kernel_bounds_output(geometry, backend, emission_spike):
    spike = emission_spike
    anat = geometry.allocate(1.0)

    base_kwargs = dict(
        sigma_anat=0.3,
        sigma_dist=1.0,
        num_neighbours=5,
        use_mask=False,
    )
    operator_raw = get_kernel_operator(geometry, backend=backend, normalize_kernel=False, **base_kwargs)
    operator_raw.set_anatomical_image(anat)
    raw = operator_raw.direct(spike).as_array()

    operator_norm = get_kernel_operator(geometry, backend=backend, normalize_kernel=True, **base_kwargs)
    operator_norm.set_anatomical_image(anat)
    norm = operator_norm.direct(spike).as_array()

    assert np.all(norm <= raw + 1e-8)
    assert np.isclose(norm.sum(), spike.as_array().sum(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("backend", available_backends())
def test_neighbourhood_size_adjusts_smoothing(geometry, backend, emission_spike):
    spike = emission_spike
    anat = geometry.allocate(1.0)

    operator_small = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.3,
        sigma_dist=1.0,
        normalize_kernel=True,
    )
    operator_small.set_anatomical_image(anat)
    res_small = operator_small.direct(spike).as_array()

    operator_large = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.3,
        sigma_dist=1.0,
        normalize_kernel=True,
    )
    operator_large.set_anatomical_image(anat)
    res_large = operator_large.direct(spike).as_array()

    assert res_large.max() < res_small.max()


@pytest.mark.parametrize("backend", available_backends())
def test_normalize_features_scales_anatomical_image(geometry, backend):
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        normalize_features=True,
        normalize_kernel=False,
    )

    # create a strongly varying anatomical image
    coords = np.indices(geometry.shape).astype(np.float64)
    anat_arr = (coords[0] * 5.0) + (coords[1] * 2.0) + coords[2]
    anat = geometry.allocate(0.0)
    anat.fill(anat_arr)

    operator.set_anatomical_image(anat)
    stored = operator.anatomical_image.as_array()

    std = anat_arr.std()
    assert std > 1e-12
    expected = anat_arr / std
    assert np.allclose(stored, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", available_backends())
def test_distance_weighting_emphasises_near_voxels(geometry, backend):
    spike = geometry.allocate(0.0)
    center = tuple(idx // 2 for idx in geometry.shape)
    spike_arr = spike.as_array()
    spike_arr.fill(0.0)
    spike_arr[center] = 1.0
    spike.fill(spike_arr)

    anat = geometry.allocate(1.0)

    base_kwargs = dict(
        num_neighbours=5,
        sigma_anat=0.5,
        sigma_dist=0.5,
        normalize_kernel=True,
        hybrid=False,
        use_mask=False,
    )

    op_no_dist = get_kernel_operator(geometry, backend=backend, distance_weighting=False, **base_kwargs)
    op_no_dist.set_anatomical_image(anat)
    res_no = op_no_dist.direct(spike).as_array()

    op_dist = get_kernel_operator(geometry, backend=backend, distance_weighting=True, **base_kwargs)
    op_dist.set_anatomical_image(anat)
    res_dist = op_dist.direct(spike).as_array()

    assert res_dist[center] > res_no[center]
    assert np.isclose(res_dist.sum(), spike_arr.sum(), rtol=1e-5, atol=1e-8)


@pytest.mark.skipif(not SLIDING_WINDOW_AVAIL, reason="Masking requires sliding_window_view")
@pytest.mark.parametrize("backend", available_backends())
def test_mask_k_picks_most_similar_neighbours(geometry, backend):
    mask_k = 7
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.5,
        sigma_dist=1.0,
        normalize_kernel=False,
        use_mask=True,
        mask_k=mask_k,
        recalc_mask=False,
    )

    # Assign unique intensities so absolute differences are unique
    anat = geometry.allocate(0.0)
    anat_arr = np.zeros(geometry.shape, dtype=np.float64)
    for i in range(geometry.shape[0]):
        for j in range(geometry.shape[1]):
            for k in range(geometry.shape[2]):
                anat_arr[i, j, k] = i * 100.0 + j * 10.0 + k
    anat.fill(anat_arr)

    operator.set_anatomical_image(anat)
    operator.direct(geometry.allocate(1.0))
    mask = operator.mask
    assert mask is not None

    center = tuple(idx // 2 for idx in geometry.shape)
    mask_vec = mask[center]
    assert mask_vec.shape[0] == operator.parameters["num_neighbours"] ** 3
    assert mask_vec.sum() == mask_k

    # Build the absolute intensity differences within the neighbourhood
    n = operator.parameters["num_neighbours"]
    half = n // 2
    diffs = []
    idx = 0
    ci, cj, ck = center
    center_val = anat_arr[center]
    for di in range(-half, half + 1):
        ii = ci + di
        for dj in range(-half, half + 1):
            jj = cj + dj
            for dk in range(-half, half + 1):
                kk = ck + dk
                diff = abs(anat_arr[ii, jj, kk] - center_val)
                diffs.append((diff, idx))
                idx += 1

    diffs.sort(key=lambda x: x[0])
    expected_indices = {idx for _, idx in diffs[:mask_k]}
    selected_indices = {i for i, active in enumerate(mask_vec) if active}
    assert selected_indices == expected_indices


@pytest.mark.skipif(not SLIDING_WINDOW_AVAIL, reason="Full feature adjoint test requires masking support")
@pytest.mark.parametrize("backend", available_backends())
def test_adjoint_with_all_features(geometry, backend):
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=5,
        sigma_anat=0.4,
        sigma_dist=0.75,
        sigma_emission=0.6,
        normalize_kernel=True,
        normalize_features=True,
        use_mask=True,
        mask_k=20,
        recalc_mask=False,
        distance_weighting=True,
        hybrid=True,
    )

    # Anatomical image with wide dynamic range
    coords = np.indices(geometry.shape).astype(np.float64)
    anat_arr = coords[0] * 3.0 + coords[1] ** 2 * 0.1 + np.sin(coords[2])
    anat = geometry.allocate(0.0)
    anat.fill(anat_arr)
    operator.set_anatomical_image(anat)

    rng = np.random.default_rng(21)
    x = DummyImage(rng.normal(size=geometry.shape))
    y = DummyImage(rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.allclose(dot_forward, dot_adjoint, atol=1e-6, rtol=1e-5)
