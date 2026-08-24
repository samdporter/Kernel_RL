import numpy as np
import pytest

try:
    import sirf.STIR  # noqa: F401

    HAS_SIRF = True
except ImportError:
    HAS_SIRF = False

pytestmark = [
    pytest.mark.sirf,
    pytest.mark.skipif(not HAS_SIRF, reason="SIRF not available"),
]

# Reduced-span mMR geometry: real mMR scanner object, small enough to run the
# suite under amd64 emulation (docs/reference/SIRF_API_NOTES.md).
REDUCED_MMR = {"span": 1, "max_ring_diff": 1, "num_views": 42, "num_tangential": 64}


def test_forward_project_and_reconstruct_roundtrip():
    from krl_studies.simulation import (
        acquisition_template,
        forward_project,
        make_acquisition_model,
        reconstruct_osem,
    )

    acq = acquisition_template("Siemens mMR", **REDUCED_MMR)
    image = acq.create_uniform_image(1000.0)
    am = make_acquisition_model(acq, image)
    prompts = forward_project(image, am)
    recon = reconstruct_osem(prompts, am, image, n_subiterations=2)
    arr = np.asarray(recon.as_array())
    assert arr.shape == np.asarray(image.as_array()).shape
    assert arr.max() > 0


def test_poisson_sample_is_deterministic_per_seed():
    from krl_studies.simulation import acquisition_template, poisson_sample

    acq = acquisition_template("Siemens mMR", **REDUCED_MMR)
    prompts = acq.copy()
    prompts.fill(np.full(np.asarray(acq.as_array()).shape, 50.0, dtype=np.float32))

    a = np.asarray(poisson_sample(prompts, seed=1337).as_array())
    b = np.asarray(poisson_sample(prompts, seed=1337).as_array())
    c = np.asarray(poisson_sample(prompts, seed=1338).as_array())

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(a == np.round(a))


def test_gaussian_smooth_image_preserves_shape_and_order():
    from krl_studies.simulation import acquisition_template, gaussian_smooth_image

    acq = acquisition_template("Siemens mMR", **REDUCED_MMR)
    image = acq.create_uniform_image(1.0)
    blurred = gaussian_smooth_image(image, (5.7, 5.7, 7.8))

    arr = np.asarray(blurred.as_array())
    assert arr.shape == np.asarray(image.as_array()).shape
    # normalised kernel, small mass loss where the Gaussian clips the FOV edge
    assert abs(arr.sum() - np.asarray(image.as_array()).sum()) < 0.05 * np.abs(
        np.asarray(image.as_array()).sum()
    )
