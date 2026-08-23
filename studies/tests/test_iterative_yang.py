import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from krl_studies.methods.iterative_yang import IterativeYangMethod


def _two_box_world():
    """1D-like 3D phantom: two compartments, blurred observation."""
    shape = (8, 40, 40)
    truth = np.full(shape, 2.0, dtype=np.float32)
    truth[:, :, 5:15] = 8.0
    sigma = 2.0
    blurred = gaussian_filter(truth, sigma=sigma)
    return truth, blurred, sigma


def _regions(truth):
    return [truth > 4.0, truth <= 4.0]


def test_iy_recovers_region_means_better_than_observed():
    truth, blurred, sigma = _two_box_world()
    regions = _regions(truth)
    true_means = np.array([truth[r].mean() for r in regions])
    init_err = np.abs(np.array([blurred[r].mean() for r in regions]) - true_means).max()

    iters = list(
        IterativeYangMethod().run(
            observed=blurred,
            guidance=None,
            params={
                "region_masks": regions,
                "psf_sigma_vox": (sigma,) * 3,
                "brain_mask": np.ones_like(blurred, dtype=bool),
            },
            n_iterations=30,
        )
    )
    final_err = np.abs(np.array([iters[-1].image[r].mean() for r in regions]) - true_means).max()
    assert final_err < init_err
    assert final_err < 0.25 * float(true_means.max())


def test_iy_streams_requested_iterations():
    truth, blurred, sigma = _two_box_world()
    iters = list(
        IterativeYangMethod().run(
            observed=blurred,
            guidance=None,
            params={
                "region_masks": _regions(truth),
                "psf_sigma_vox": (sigma,) * 3,
                "brain_mask": np.ones_like(blurred, dtype=bool),
            },
            n_iterations=5,
        )
    )
    assert [it.iteration for it in iters] == [1, 2, 3, 4, 5]


def test_iy_requires_regions():
    with pytest.raises(ValueError, match="region_masks"):
        IterativeYangMethod().run(observed=np.zeros((4, 4, 4)), guidance=None, params={}, n_iterations=1).__next__()
