import numpy as np
import pytest
from cil.framework import ImageGeometry
from scipy.ndimage import convolve

from krl.operators.blurring import GaussianBlurringOperator, create_gaussian_blur

CPU_BACKENDS = ["numba", "scipy"]
STORAGE_DTYPES = [np.float32, np.float64]


def make_geometry(shape=(7, 7, 7), dtype=np.float64):
    z, y, x = shape
    return ImageGeometry(
        voxel_num_x=x, voxel_num_y=y, voxel_num_z=z, dtype=dtype
    )


def make_image(geometry, array):
    image = geometry.allocate()
    image.fill(np.asarray(array, dtype=geometry.dtype))
    return image


def zero_padded_direct(data, psf):
    """Independent reference: convolution with a zero-extended volume."""
    return convolve(data, psf, mode="constant", cval=0.0)


def zero_padded_adjoint(data, psf):
    """Independent reference: correlation with a zero-extended volume."""
    return convolve(data, psf[::-1, ::-1, ::-1], mode="constant", cval=0.0)


def tolerances(dtype):
    if np.dtype(dtype) == np.float32:
        return {"atol": 1e-6, "rtol": 1e-6}
    return {"atol": 1e-10, "rtol": 1e-10}


def apply_operator(backend, geometry, sigma, data, method):
    operator = GaussianBlurringOperator(sigma, geometry, backend=backend)
    return getattr(operator, method)(make_image(geometry, data)).as_array()


@pytest.fixture
def geometry():
    return make_geometry((7, 7, 7))


@pytest.mark.parametrize("dtype", STORAGE_DTYPES)
@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_direct_matches_independent_reference(backend, dtype):
    geometry = make_geometry((7, 7, 7), dtype)
    operator = GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend=backend)
    rng = np.random.default_rng(0)
    data = rng.normal(size=geometry.shape)
    image = make_image(geometry, data)

    result = operator.direct(image).as_array()

    expected = zero_padded_direct(data, operator.psf)
    assert np.allclose(result, expected, **tolerances(dtype))


@pytest.mark.parametrize("dtype", STORAGE_DTYPES)
@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_adjoint_matches_independent_reference(backend, dtype):
    geometry = make_geometry((7, 7, 7), dtype)
    operator = GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend=backend)
    rng = np.random.default_rng(1)
    data = rng.normal(size=geometry.shape)
    image = make_image(geometry, data)

    result = operator.adjoint(image).as_array()

    expected = zero_padded_adjoint(data, operator.psf)
    assert np.allclose(result, expected, **tolerances(dtype))


@pytest.mark.parametrize("dtype", STORAGE_DTYPES)
@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_adjointness_symmetric_psf(backend, dtype):
    geometry = make_geometry((7, 7, 7), dtype)
    operator = GaussianBlurringOperator((1.2, 0.9, 1.2), geometry, backend=backend)
    rng = np.random.default_rng(2)
    x = make_image(geometry, rng.normal(size=geometry.shape))
    y = make_image(geometry, rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    atol = 1e-5 if np.dtype(dtype) == np.float32 else 1e-10
    assert np.isclose(dot_forward, dot_adjoint, atol=atol, rtol=1e-5)


@pytest.mark.parametrize("dtype", STORAGE_DTYPES)
@pytest.mark.parametrize("method", ["direct", "adjoint"])
def test_direct_and_adjoint_write_into_provided_out(dtype, method):
    geometry = make_geometry((7, 7, 7), dtype)
    rng = np.random.default_rng(3)
    data = rng.normal(size=geometry.shape)
    image = make_image(geometry, data)
    reference = zero_padded_direct if method == "direct" else zero_padded_adjoint

    for backend in CPU_BACKENDS:
        operator = GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend=backend)
        out = make_image(geometry, np.zeros(geometry.shape))
        returned = getattr(operator, method)(image, out=out)

        assert returned is out
        assert out.as_array().dtype == geometry.dtype
        assert np.allclose(
            out.as_array(), reference(data, operator.psf), **tolerances(dtype)
        )


@pytest.mark.parametrize("dtype", STORAGE_DTYPES)
def test_numba_scipy_parity_seeded_random(dtype):
    geometry = make_geometry((7, 7, 7), dtype)
    data = np.random.default_rng(11).normal(size=geometry.shape)

    for method in ("direct", "adjoint"):
        numba_result = apply_operator("numba", geometry, (1.0, 1.0, 1.0), data, method)
        scipy_result = apply_operator("scipy", geometry, (1.0, 1.0, 1.0), data, method)
        assert np.allclose(numba_result, scipy_result, **tolerances(dtype))


@pytest.mark.parametrize("dtype", STORAGE_DTYPES)
def test_numba_scipy_parity_corner_and_edge_impulses(dtype):
    shape = (7, 7, 7)
    geometry = make_geometry(shape, dtype)
    sigma = (1.0, 1.0, 1.0)
    impulses = [
        (0, 0, 0),
        (0, 0, shape[2] - 1),
        (0, shape[1] - 1, 0),
        (shape[0] - 1, 0, 0),
        (shape[0] - 1, shape[1] - 1, shape[2] - 1),
        (0, shape[1] // 2, shape[2] // 2),
        (shape[0] // 2, 0, shape[2] // 2),
        (shape[0] // 2, shape[1] // 2, 0),
    ]

    for position in impulses:
        data = np.zeros(shape)
        data[position] = 1.0
        for method in ("direct", "adjoint"):
            numba_result = apply_operator("numba", geometry, sigma, data, method)
            scipy_result = apply_operator("scipy", geometry, sigma, data, method)
            assert np.allclose(numba_result, scipy_result, **tolerances(dtype))


@pytest.mark.parametrize("dtype", STORAGE_DTYPES)
def test_numba_scipy_parity_constant_volume_boundary_attenuation(dtype):
    shape = (7, 7, 7)
    geometry = make_geometry(shape, dtype)
    data = np.ones(shape)
    operator = GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend="numba")
    expected = zero_padded_direct(data, operator.psf)

    for backend in CPU_BACKENDS:
        result = apply_operator(backend, geometry, (1.0, 1.0, 1.0), data, "direct")
        assert np.allclose(result, expected, **tolerances(dtype))
        # zero-padding attenuates the constant volume away from the interior
        assert result[0, 0, 0] < 0.5
        assert np.isclose(result[3, 3, 3], 1.0, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", STORAGE_DTYPES)
def test_numba_scipy_parity_sigma_1_5_on_8_cubed(dtype):
    geometry = make_geometry((8, 8, 8), dtype)
    data = np.random.default_rng(12).normal(size=geometry.shape)
    sigma = (1.5, 1.5, 1.5)

    for method in ("direct", "adjoint"):
        numba_result = apply_operator("numba", geometry, sigma, data, method)
        scipy_result = apply_operator("scipy", geometry, sigma, data, method)
        assert np.allclose(numba_result, scipy_result, **tolerances(dtype))


@pytest.mark.parametrize("dtype", STORAGE_DTYPES)
def test_numba_scipy_parity_anisotropic_psf_non_cubic_volume(dtype):
    geometry = make_geometry((5, 7, 11), dtype)
    data = np.random.default_rng(13).normal(size=geometry.shape)
    sigma = (0.7, 1.6, 2.4)

    for method in ("direct", "adjoint"):
        numba_result = apply_operator("numba", geometry, sigma, data, method)
        scipy_result = apply_operator("scipy", geometry, sigma, data, method)
        assert np.allclose(numba_result, scipy_result, **tolerances(dtype))


def test_create_gaussian_blur_resolves_backend(geometry):
    operator = create_gaussian_blur((1.0, 1.0, 1.0), geometry, backend="scipy")
    assert operator.backend == "scipy"


@pytest.mark.parametrize(
    "bad_sigma",
    [
        (0.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, np.nan),
        (1.0, 1.0, np.inf),
        (1.0, 1.0),
    ],
)
def test_sigma_validation(geometry, bad_sigma):
    with pytest.raises(ValueError, match="sigma"):
        GaussianBlurringOperator(bad_sigma, geometry, backend="numba")


def test_unknown_backend_raises(geometry):
    with pytest.raises(ValueError, match="not supported"):
        GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend="bogus")


def test_multichannel_geometry_rejected_at_construction():
    geometry = ImageGeometry(voxel_num_x=5, voxel_num_y=5, channels=3)
    with pytest.raises(ValueError, match="single-channel 3-D volumes"):
        GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend="numba")


def test_two_dimensional_input_raises_clear_error():
    geometry = ImageGeometry(voxel_num_x=5, voxel_num_y=5)
    operator = GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend="numba")
    image = make_image(geometry, np.ones(geometry.shape))

    with pytest.raises(ValueError, match="single-channel 3-D volumes"):
        operator.direct(image)
    with pytest.raises(ValueError, match="single-channel 3-D volumes"):
        operator.adjoint(image)


def test_multichannel_input_image_rejected_at_apply(geometry):
    operator = GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend="numba")
    multichannel = ImageGeometry(
        voxel_num_x=5, voxel_num_y=5, channels=3, dtype=np.float64
    )
    image = make_image(multichannel, np.ones(multichannel.shape))
    assert image.as_array().ndim == 3

    with pytest.raises(ValueError, match="single-channel 3-D volumes"):
        operator.direct(image)
    with pytest.raises(ValueError, match="single-channel 3-D volumes"):
        operator.adjoint(image)


def test_three_dimensional_image_still_works(geometry):
    operator = GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend="scipy")
    data = np.indices(geometry.shape).sum(axis=0).astype(np.float64)
    image = make_image(geometry, data)

    result = operator.direct(image).as_array()
    assert result.shape == geometry.shape
    assert np.all(np.isfinite(result))
