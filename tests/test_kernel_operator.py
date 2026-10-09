import math

import numpy as np
import pytest
from cil.framework import BlockDataContainer, ImageGeometry
from cil.optimisation.operators import ScaledOperator

from krl.operators.directional import DirectionalOperator
from krl.operators.kernel_operator import (
    KernelOperator,
    get_kernel_operator,
)


def make_geometry(shape=(5, 5, 5), dtype=np.float64):
    z, y, x = shape
    return ImageGeometry(voxel_num_x=x, voxel_num_y=y, voxel_num_z=z, dtype=dtype)


def make_image(geometry, array):
    image = geometry.allocate()
    image.fill(np.asarray(array, dtype=geometry.dtype))
    return image


def available_backends():
    return ["numba"]


@pytest.fixture
def geometry():
    return make_geometry((5, 5, 5))


@pytest.fixture
def anatomical_uniform(geometry):
    return geometry.allocate(1.0)


@pytest.fixture
def emission_spike(geometry):
    arr = np.zeros(geometry.shape, dtype=np.float64)
    arr[2, 2, 2] = 10.0
    return make_image(geometry, arr)


@pytest.fixture
def emission_random(geometry):
    rng = np.random.default_rng(42)
    return make_image(geometry, rng.normal(size=geometry.shape))


@pytest.fixture
def anatomical_image_gradient(geometry):
    grad = np.indices(geometry.shape).sum(axis=0)
    return make_image(geometry, grad)


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
def test_adjoint_dot_product_float64_data_with_float32_anatomy(geometry, backend):
    """Regression: forward output must follow the input dtype, not the
    anatomical image's. A float32 anatomy previously truncated the forward
    result to float32, breaking adjointness against float64 data."""
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.4,
        normalize_kernel=True,
        normalize_features=False,
        use_mask=True,
        mask_k=10,
    )
    anatomy_f32 = make_geometry(geometry.shape, dtype=np.float32).allocate()
    anatomy_f32.fill(np.random.default_rng(3).normal(size=geometry.shape).astype(np.float32))
    operator.set_anatomical_image(anatomy_f32)

    rng = np.random.default_rng(5)
    x = make_image(geometry, rng.normal(size=geometry.shape))  # float64
    y = make_image(geometry, rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    assert forward.dtype == np.float64

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * operator.adjoint(y).as_array()))
    assert np.isclose(dot_forward, dot_adjoint, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("anat_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("input_dtype", [np.float32, np.float64])
def test_input_dtype_preserved(geometry, anat_dtype, input_dtype):
    operator = get_kernel_operator(
        geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        normalize_kernel=True,
        normalize_features=False,
        use_mask=False,
    )
    anatomy = make_geometry(geometry.shape, dtype=anat_dtype).allocate()
    anatomy.fill(np.random.default_rng(1).normal(size=geometry.shape).astype(anat_dtype))
    operator.set_anatomical_image(anatomy)

    input_geometry = make_geometry(geometry.shape, dtype=input_dtype)
    x = input_geometry.allocate()
    x.fill(np.random.default_rng(2).normal(size=geometry.shape).astype(input_dtype))

    result = operator.direct(x)
    assert result.dtype == input_dtype

    out = input_geometry.allocate()
    returned = operator.direct(x, out=out)
    assert returned is out
    assert out.dtype == input_dtype

    adjoint_result = operator.adjoint(x)
    assert adjoint_result.dtype == input_dtype

    adjoint_out = input_geometry.allocate()
    returned = operator.adjoint(x, out=adjoint_out)
    assert returned is adjoint_out
    assert adjoint_out.dtype == input_dtype


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
    grid = np.indices(geometry.shape).sum(axis=0) / math.prod(geometry.shape)
    operator.set_anatomical_image(make_image(geometry, grid))

    rng = np.random.default_rng(7)
    x = make_image(geometry, rng.normal(size=geometry.shape))
    y = make_image(geometry, rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.allclose(dot_forward, dot_adjoint, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("normalize_kernel", [False, True])
def test_forward_adjoint_dot_dense_sparse(geometry, use_mask, normalize_kernel):
    operator = get_kernel_operator(
        geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        sigma_dist=1.0,
        normalize_kernel=normalize_kernel,
        use_mask=use_mask,
        mask_k=10 if use_mask else None,
        distance_weighting=True,
        hybrid=False,
    )
    grid = np.indices(geometry.shape).sum(axis=0) / math.prod(geometry.shape)
    operator.set_anatomical_image(make_image(geometry, grid))

    rng = np.random.default_rng(11)
    x = make_image(geometry, rng.normal(size=geometry.shape))
    y = make_image(geometry, rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.isclose(dot_forward, dot_adjoint, atol=1e-10, rtol=1e-8)


def test_fresh_fixed_kernel_adjoint_matches_direct_initialised(geometry):
    kwargs = dict(
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        normalize_kernel=True,
        normalize_features=False,
        use_mask=True,
        mask_k=10,
        hybrid=False,
    )
    grid = np.indices(geometry.shape).sum(axis=0) / math.prod(geometry.shape)
    anat = make_image(geometry, grid)
    rng = np.random.default_rng(13)
    x = make_image(geometry, rng.normal(size=geometry.shape))
    y = make_image(geometry, rng.normal(size=geometry.shape))

    fresh = get_kernel_operator(geometry, **kwargs)
    fresh.set_anatomical_image(anat)
    adjoint_fresh = fresh.adjoint(y).as_array()

    initialised = get_kernel_operator(geometry, **kwargs)
    initialised.set_anatomical_image(anat)
    initialised.direct(x)
    adjoint_direct = initialised.adjoint(y).as_array()

    assert np.allclose(adjoint_fresh, adjoint_direct, atol=1e-12, rtol=1e-10)


def test_adaptive_hybrid_adjoint_without_reference_raises(geometry):
    operator = get_kernel_operator(
        geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        sigma_emission=0.5,
        normalize_kernel=True,
        use_mask=False,
        hybrid=True,
    )
    operator.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(RuntimeError, match="emission reference"):
        operator.adjoint(geometry.allocate(1.0))


def test_setters_invalidate_derived_caches(geometry):
    operator = get_kernel_operator(
        geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        normalize_kernel=True,
        use_mask=True,
        mask_k=10,
    )
    operator.set_anatomical_image(make_image(geometry, np.indices(geometry.shape).sum(axis=0)))
    operator.direct(geometry.allocate(1.0))

    assert operator.mask is not None
    assert operator._anatomical_weights is not None
    assert operator._normalisation_map is not None
    operator.set_norm(1.0)

    operator.set_parameters({"sigma_anat": 0.9})
    assert operator.mask is None
    assert operator._anatomical_weights is None
    assert operator._normalisation_map is None
    assert operator._norm is None

    operator.direct(geometry.allocate(1.0))
    operator.set_norm(1.0)
    operator.set_anatomical_image(make_image(geometry, np.ones(geometry.shape)))
    assert operator.mask is None
    assert operator._anatomical_weights is None
    assert operator._normalisation_map is None
    assert operator._norm is None


def test_recalc_mask_rebuilds_mask(geometry, monkeypatch):
    operator = get_kernel_operator(
        geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        normalize_kernel=False,
        use_mask=True,
        mask_k=5,
        recalc_mask=True,
    )
    operator.set_anatomical_image(make_image(geometry, np.indices(geometry.shape).sum(axis=0)))

    calls = []
    original = operator.precompute_mask

    def counting_precompute_mask():
        calls.append(1)
        return original()

    monkeypatch.setattr(operator, "precompute_mask", counting_precompute_mask)

    emission = geometry.allocate(1.0)
    operator.direct(emission)
    operator.direct(emission)

    assert len(calls) == 2


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
    grid = np.indices(geometry.shape).sum(axis=0) / math.prod(geometry.shape)
    operator.set_anatomical_image(make_image(geometry, grid))

    rng = np.random.default_rng(7)
    x = make_image(geometry, rng.normal(size=geometry.shape))
    y = make_image(geometry, rng.normal(size=geometry.shape))

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
    grid = np.indices(geometry.shape).sum(axis=0) / math.prod(geometry.shape)
    operator.set_anatomical_image(make_image(geometry, grid))

    rng = np.random.default_rng(7)
    x = make_image(geometry, rng.normal(size=geometry.shape))
    y = make_image(geometry, rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.allclose(dot_forward, dot_adjoint, atol=1e-6, rtol=1e-5)


def test_mask_available_with_numba(geometry):
    operator = KernelOperator(geometry, use_mask=True, mask_k=3)
    operator.set_anatomical_image(geometry.allocate(0.0))

    result = operator.direct(geometry.allocate(1.0))
    assert isinstance(result, type(geometry.allocate()))
    assert operator.mask is not None
    # With sparse indexing, mask shape is (..., k) not (..., n³)
    assert operator.mask.shape[-1] == operator.parameters["mask_k"]


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
    # With sparse indexing, mask is now integer indices of shape (..., k)
    assert mask.shape[-1] == mask_k
    # Check that indices are valid
    n_cubed = operator.parameters["num_neighbours"] ** 3
    assert np.all(mask >= 0)
    assert np.all(mask < n_cubed)


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

    # Larger neighborhood should spread the spike more (lower peak or higher variance)
    # Use variance as a more robust measure of smoothing
    assert np.var(res_large) >= np.var(res_small) * 0.9  # Allow 10% tolerance


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
    operator.set_anatomical_image(make_image(geometry, anat_arr))
    stored = operator.anatomical_image.as_array()

    std = anat_arr.std()
    assert std > 1e-12
    expected = anat_arr / std
    assert np.allclose(stored, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", available_backends())
def test_distance_weighting_emphasises_near_voxels(geometry, backend):
    center = tuple(idx // 2 for idx in geometry.shape)
    spike_arr = np.zeros(geometry.shape, dtype=np.float64)
    spike_arr[center] = 1.0
    spike = make_image(geometry, spike_arr)

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
    anat_arr = np.zeros(geometry.shape, dtype=np.float64)
    for i in range(geometry.shape[0]):
        for j in range(geometry.shape[1]):
            for k in range(geometry.shape[2]):
                anat_arr[i, j, k] = i * 100.0 + j * 10.0 + k

    operator.set_anatomical_image(make_image(geometry, anat_arr))
    operator.direct(geometry.allocate(1.0))
    mask = operator.mask
    assert mask is not None

    center = tuple(idx // 2 for idx in geometry.shape)
    mask_vec = mask[center]
    # With sparse indexing, mask_vec is now integer indices
    assert mask_vec.shape[0] == mask_k

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
    # mask_vec now contains the indices directly
    selected_indices = set(mask_vec)
    assert selected_indices == expected_indices


@pytest.mark.parametrize("backend", available_backends())
def test_sigma_anatomical_parameter_changes_weights(
    geometry,
    backend,
    anatomical_image_gradient,
    emission_random,
):
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.1,
        sigma_dist=1.0,
        normalize_kernel=False,
        use_mask=False,
        distance_weighting=False,
        hybrid=False,
    )
    operator.set_anatomical_image(anatomical_image_gradient)

    res_narrow = operator.direct(emission_random).as_array()
    operator.set_parameters({"sigma_anat": 5.0})
    res_wide = operator.direct(emission_random).as_array()

    assert not np.allclose(res_narrow, res_wide, atol=1e-6, rtol=1e-5)
    assert np.linalg.norm(res_narrow - res_wide) > 1e-3


@pytest.mark.parametrize("backend", available_backends())
def test_sigma_distance_parameter_requires_distance_weighting(
    geometry,
    backend,
    anatomical_image_gradient,
    emission_random,
):
    # When distance weighting is disabled, sigma_dist should have no effect
    operator_no_dist = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.2,
        sigma_dist=0.1,
        normalize_kernel=False,
        use_mask=False,
        distance_weighting=False,
        hybrid=False,
    )
    operator_no_dist.set_anatomical_image(anatomical_image_gradient)

    res_no_dist_tight = operator_no_dist.direct(emission_random).as_array()
    operator_no_dist.set_parameters({"sigma_dist": 5.0})
    res_no_dist_wide = operator_no_dist.direct(emission_random).as_array()

    assert np.allclose(res_no_dist_tight, res_no_dist_wide, atol=1e-7, rtol=1e-6)

    # With distance weighting enabled, sigma_dist must influence the kernel
    operator_dist = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.2,
        sigma_dist=0.1,
        normalize_kernel=False,
        use_mask=False,
        distance_weighting=True,
        hybrid=False,
    )
    operator_dist.set_anatomical_image(anatomical_image_gradient)

    res_dist_tight = operator_dist.direct(emission_random).as_array()
    operator_dist.set_parameters({"sigma_dist": 5.0})
    res_dist_wide = operator_dist.direct(emission_random).as_array()

    assert not np.allclose(res_dist_tight, res_dist_wide, atol=1e-6, rtol=1e-5)
    assert np.linalg.norm(res_dist_tight - res_dist_wide) > 1e-3


@pytest.mark.parametrize("backend", available_backends())
def test_sigma_emission_parameter_affects_hybrid_kernel(
    geometry,
    backend,
    anatomical_image_gradient,
    emission_random,
):
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.2,
        sigma_dist=1.0,
        sigma_emission=0.1,
        normalize_kernel=False,
        use_mask=False,
        distance_weighting=False,
        hybrid=True,
    )
    operator.set_anatomical_image(anatomical_image_gradient)

    res_emission_tight = operator.direct(emission_random).as_array()
    operator.set_parameters({"sigma_emission": 5.0})
    res_emission_wide = operator.direct(emission_random).as_array()

    assert not np.allclose(res_emission_tight, res_emission_wide, atol=1e-6, rtol=1e-5)
    assert np.linalg.norm(res_emission_tight - res_emission_wide) > 1e-3


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
    operator.set_anatomical_image(make_image(geometry, anat_arr))

    rng = np.random.default_rng(21)
    x = make_image(geometry, rng.normal(size=geometry.shape))
    y = make_image(geometry, rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.allclose(dot_forward, dot_adjoint, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("backend", available_backends())
def test_precompute_anatomical_weights(geometry, backend):
    """Test that anatomical weights are pre-computed and cached correctly."""
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.5,
        sigma_dist=1.0,
        use_mask=False,
        distance_weighting=True,
    )

    grid = np.indices(geometry.shape).sum(axis=0) / math.prod(geometry.shape)
    operator.set_anatomical_image(make_image(geometry, grid))

    # Before any operation, weights should be None
    assert operator._anatomical_weights is None

    # Pre-compute weights manually
    weights = operator.precompute_anatomical_weights()

    # Check shape and properties
    n = operator.parameters["num_neighbours"]
    expected_shape = (geometry.shape[0], geometry.shape[1], geometry.shape[2], n**3)
    assert weights.shape == expected_shape
    assert weights.dtype == np.float64

    # All weights should be positive (Gaussian weights)
    assert np.all(weights >= 0.0)
    # At least some weights should be non-zero
    assert np.any(weights > 0.0)

    # Weights should be cached after first direct call
    emission = geometry.allocate(1.0)
    operator.direct(emission)
    assert operator._anatomical_weights is not None
    assert operator._anatomical_weights.shape == expected_shape


@pytest.mark.parametrize("backend", available_backends())
def test_precomputed_weights_invalidation(geometry, backend):
    """Test that anatomical weights cache is invalidated when parameters change."""
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.5,
        use_mask=False,
    )

    anat = geometry.allocate(1.0)
    operator.set_anatomical_image(anat)

    # Trigger weight computation
    emission = geometry.allocate(1.0)
    operator.direct(emission)
    assert operator._anatomical_weights is not None

    # Changing anatomical image should invalidate cache
    new_anat = geometry.allocate(2.0)
    operator.set_anatomical_image(new_anat)
    assert operator._anatomical_weights is None

    # Trigger weight computation again
    operator.direct(emission)
    assert operator._anatomical_weights is not None

    # Changing parameters should invalidate cache
    operator.set_parameters({"sigma_anat": 0.8})
    assert operator._anatomical_weights is None


@pytest.mark.parametrize("backend", available_backends())
def test_precomputed_weights_with_mask(geometry, backend):
    """Test that anatomical weights are pre-computed correctly with masking."""
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.5,
        use_mask=True,
        mask_k=10,
    )

    # Use gradient anatomical image
    grid = np.indices(geometry.shape).sum(axis=0)
    operator.set_anatomical_image(make_image(geometry, grid))

    # Pre-compute weights
    weights = operator.precompute_anatomical_weights()

    # Check shape - with sparse indexing, shape is (..., k) not (..., n³)
    mask_k = operator.parameters["mask_k"]
    expected_shape = (geometry.shape[0], geometry.shape[1], geometry.shape[2], mask_k)
    assert weights.shape == expected_shape

    # With sparse indexing, all weights should be non-zero (we only store valid ones)
    # But they should vary in magnitude
    assert np.all(weights >= 0.0)
    assert np.any(weights > 0.0)


@pytest.mark.parametrize("backend", available_backends())
def test_adjoint_with_precomputed_anatomical_weights(geometry, backend):
    """
    Test that the adjoint property holds when using pre-computed anatomical weights.
    This is critical because pre-computation changes the kernel implementation.
    """
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
    operator.set_anatomical_image(make_image(geometry, anat_arr))

    # Explicitly pre-compute anatomical weights
    weights = operator.precompute_anatomical_weights()
    assert weights is not None
    assert operator._anatomical_weights is None  # Not cached yet until first use

    rng = np.random.default_rng(21)
    x = make_image(geometry, rng.normal(size=geometry.shape))
    y = make_image(geometry, rng.normal(size=geometry.shape))

    # Run forward and adjoint (this will trigger caching)
    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    # Verify weights are now cached
    assert operator._anatomical_weights is not None

    # Verify adjoint property: <Ax, y> = <x, A*y>
    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.allclose(dot_forward, dot_adjoint, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("backend", available_backends())
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("hybrid", [True, False])
@pytest.mark.parametrize("distance_weighting", [True, False])
def test_precomputed_weights_consistency(geometry, backend, use_mask, hybrid, distance_weighting):
    """
    Test that using pre-computed weights produces consistent results across multiple calls.
    The anatomical weights should be cached and reused, producing identical results.
    """
    operator = get_kernel_operator(
        geometry,
        backend=backend,
        num_neighbours=3,
        sigma_anat=0.5,
        sigma_dist=1.0,
        sigma_emission=0.3,
        normalize_kernel=True,
        use_mask=use_mask,
        mask_k=10 if use_mask else None,
        distance_weighting=distance_weighting,
        hybrid=hybrid,
    )

    # Set anatomical image
    grid = np.indices(geometry.shape).sum(axis=0) / math.prod(geometry.shape)
    operator.set_anatomical_image(make_image(geometry, grid))

    # Create test emission data
    rng = np.random.default_rng(42)
    emission = make_image(geometry, rng.normal(size=geometry.shape))

    # First call - weights will be computed and cached
    result1 = operator.direct(emission).as_array()
    assert operator._anatomical_weights is not None

    # Second call - should use cached weights
    result2 = operator.direct(emission).as_array()

    # Results should be identical (using cached weights)
    assert np.allclose(result1, result2, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("num_neighbours", [0, -1, 2, 4, 3.5, True])
def test_invalid_num_neighbours_rejected(geometry, num_neighbours):
    operator = get_kernel_operator(geometry, backend="numba", num_neighbours=num_neighbours)
    operator.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="num_neighbours"):
        operator.direct(geometry.allocate(1.0))


def test_missing_anatomical_image_rejected(geometry):
    operator = get_kernel_operator(geometry, backend="numba")

    with pytest.raises(ValueError, match="anatomical image"):
        operator.direct(geometry.allocate(1.0))


def test_shape_mismatched_anatomical_image_rejected(geometry):
    operator = get_kernel_operator(geometry, backend="numba")
    operator.set_anatomical_image(make_geometry((3, 3, 3)).allocate(1.0))

    with pytest.raises(ValueError, match="shape"):
        operator.direct(geometry.allocate(1.0))


def test_non_3d_input_rejected(geometry):
    operator = get_kernel_operator(geometry, backend="numba")
    operator.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="3-D"):
        operator.direct(np.zeros((5, 5)))


def test_multichannel_2d_geometry_rejected():
    """A 2-D geometry with channels allocates (channels, y, x); ndim==3 alone
    must not let it be treated as a single-channel 3-D volume."""
    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=4, channels=2)
    operator = get_kernel_operator(geometry, backend="numba", num_neighbours=3)
    operator.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="single-channel 3-D"):
        operator.direct(geometry.allocate(1.0))


def test_reflection_radius_larger_than_volume_rejected():
    small_geometry = make_geometry((3, 3, 3))
    operator = get_kernel_operator(small_geometry, backend="numba", num_neighbours=9)
    operator.set_anatomical_image(small_geometry.allocate(1.0))

    with pytest.raises(ValueError, match="reflection radius"):
        operator.direct(small_geometry.allocate(1.0))


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_invalid_sigma_anat_rejected(geometry, sigma):
    operator = get_kernel_operator(geometry, backend="numba", sigma_anat=sigma)
    operator.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="sigma_anat"):
        operator.direct(geometry.allocate(1.0))


def test_invalid_sigma_emission_rejected(geometry):
    operator = get_kernel_operator(geometry, backend="numba", sigma_emission=0.0, hybrid=True)
    operator.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="sigma_emission"):
        operator.direct(geometry.allocate(1.0))


def test_invalid_sigma_dist_rejected(geometry):
    operator = get_kernel_operator(geometry, backend="numba", sigma_dist=0.0, distance_weighting=True)
    operator.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="sigma_dist"):
        operator.direct(geometry.allocate(1.0))


def test_validation_fails_before_compiled_kernel(geometry, monkeypatch):
    import krl.operators.kernel_operator as kernel_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("compiled kernel was entered")

    monkeypatch.setattr(kernel_module, "_nb_kernel_precomputed", fail_if_called)

    operator = get_kernel_operator(geometry, backend="numba", num_neighbours=4)
    operator.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="num_neighbours"):
        operator.direct(geometry.allocate(1.0))


@pytest.mark.parametrize("num_neighbours", [0, 4, 3.5, True])
def test_precompute_helpers_reject_invalid_neighbourhood(geometry, num_neighbours):
    operator = get_kernel_operator(geometry, backend="numba", num_neighbours=num_neighbours)
    operator.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="num_neighbours"):
        operator.precompute_mask()
    with pytest.raises(ValueError, match="num_neighbours"):
        operator.precompute_anatomical_weights()


def test_precompute_helpers_require_anatomical_image(geometry):
    operator = get_kernel_operator(geometry, backend="numba")

    with pytest.raises(ValueError, match="anatomical image"):
        operator.precompute_mask()
    with pytest.raises(ValueError, match="anatomical image"):
        operator.precompute_anatomical_weights()


def test_precompute_helpers_reject_shape_mismatch(geometry):
    operator = get_kernel_operator(geometry, backend="numba")
    operator.set_anatomical_image(make_geometry((3, 3, 3)).allocate(1.0))

    with pytest.raises(ValueError, match="shape"):
        operator.precompute_mask()
    with pytest.raises(ValueError, match="shape"):
        operator.precompute_anatomical_weights()


def test_precompute_helpers_reject_oversized_neighbourhood():
    small_geometry = make_geometry((3, 3, 3))
    operator = get_kernel_operator(small_geometry, backend="numba", num_neighbours=9)
    operator.set_anatomical_image(small_geometry.allocate(1.0))

    with pytest.raises(ValueError, match="reflection radius"):
        operator.precompute_mask()
    with pytest.raises(ValueError, match="reflection radius"):
        operator.precompute_anatomical_weights()


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan])
def test_precompute_weights_reject_invalid_sigma(geometry, sigma):
    operator = get_kernel_operator(geometry, backend="numba", sigma_anat=sigma)
    operator.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="sigma_anat"):
        operator.precompute_anatomical_weights()


@pytest.mark.parametrize("use_mask", [True, False])
def test_precompute_validation_fails_before_jit(geometry, monkeypatch, use_mask):
    import krl.operators.kernel_operator as kernel_module

    entered = []

    def fail_if_called(*args, **kwargs):
        entered.append(True)
        raise AssertionError("compiled kernel was entered")

    monkeypatch.setattr(kernel_module, "_nb_precompute_mask", fail_if_called)
    monkeypatch.setattr(kernel_module, "_nb_precompute_anatomical_weights", fail_if_called)
    monkeypatch.setattr(kernel_module, "_nb_precompute_anatomical_weights_mask", fail_if_called)

    invalid_neighbourhood = get_kernel_operator(
        geometry, backend="numba", num_neighbours=4, use_mask=use_mask
    )
    invalid_neighbourhood.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="num_neighbours"):
        invalid_neighbourhood.precompute_mask()
    with pytest.raises(ValueError, match="num_neighbours"):
        invalid_neighbourhood.precompute_anatomical_weights()
    assert entered == []

    invalid_sigma = get_kernel_operator(
        geometry, backend="numba", sigma_anat=0.0, use_mask=use_mask
    )
    invalid_sigma.set_anatomical_image(geometry.allocate(1.0))

    with pytest.raises(ValueError, match="sigma_anat"):
        invalid_sigma.precompute_anatomical_weights()
    assert entered == []


def test_directional_operator_scaled_out(geometry):
    component_a = make_image(geometry, np.random.default_rng(0).normal(size=geometry.shape))
    component_b = make_image(geometry, np.random.default_rng(1).normal(size=geometry.shape))
    anatomical_gradient = BlockDataContainer(component_a, component_b)

    operator = DirectionalOperator(anatomical_gradient, gamma=0.3, eta=0.1)
    scaled = ScaledOperator(operator, 2.0)

    x = BlockDataContainer(component_a.clone(), component_b.clone())
    out = x.clone()
    returned = scaled.direct(x, out=out)

    assert returned is out

    expected = scaled.direct(x)
    for result, reference in zip(out.containers, expected.containers):
        assert np.allclose(result.as_array(), reference.as_array())
