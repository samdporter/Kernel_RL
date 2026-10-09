import numpy as np
import pytest
from cil.framework import ImageGeometry
from scipy.ndimage import convolve

from krl.operators.blurring import GaussianBlurringOperator, create_gaussian_blur


def make_geometry(shape=(7, 7, 7)):
    z, y, x = shape
    return ImageGeometry(
        voxel_num_x=x, voxel_num_y=y, voxel_num_z=z, dtype=np.float64
    )


def make_image(geometry, array):
    image = geometry.allocate()
    image.fill(np.asarray(array, dtype=geometry.dtype))
    return image


@pytest.fixture
def geometry():
    return make_geometry((7, 7, 7))


@pytest.mark.parametrize("backend", ["numba", "scipy"])
def test_direct_matches_independent_reference(geometry, backend):
    operator = GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend=backend)
    rng = np.random.default_rng(0)
    data = rng.normal(size=geometry.shape)
    image = make_image(geometry, data)

    result = operator.direct(image).as_array()

    if backend == "numba":
        # numba uses zero-padding at the boundaries
        expected = convolve(data, operator.psf, mode="constant", cval=0.0)
    else:
        expected = convolve(data, operator.psf, mode="reflect")

    assert np.allclose(result, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("backend", ["numba", "scipy"])
def test_adjoint_matches_independent_reference(geometry, backend):
    operator = GaussianBlurringOperator((1.0, 1.0, 1.0), geometry, backend=backend)
    rng = np.random.default_rng(1)
    data = rng.normal(size=geometry.shape)
    image = make_image(geometry, data)

    result = operator.adjoint(image).as_array()

    if backend == "numba":
        expected = convolve(data, operator.psf, mode="constant", cval=0.0)
    else:
        expected = convolve(data, operator.psf, mode="reflect")

    assert np.allclose(result, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("backend", ["numba", "scipy"])
def test_adjointness_symmetric_psf(geometry, backend):
    operator = GaussianBlurringOperator((1.2, 0.9, 1.2), geometry, backend=backend)
    rng = np.random.default_rng(2)
    x = make_image(geometry, rng.normal(size=geometry.shape))
    y = make_image(geometry, rng.normal(size=geometry.shape))

    forward = operator.direct(x).as_array()
    adjoint = operator.adjoint(y).as_array()

    dot_forward = float(np.sum(forward * y.as_array()))
    dot_adjoint = float(np.sum(x.as_array() * adjoint))
    assert np.isclose(dot_forward, dot_adjoint, atol=1e-10, rtol=1e-8)


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
