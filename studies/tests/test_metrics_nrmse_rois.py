import numpy as np

from krl_studies.metrics.nrmse import nrmse
from krl_studies.metrics.rois import background_vois, derive_lesion_rois


def test_nrmse_matches_definition():
    gt = np.full((10, 10, 10), 2.0, dtype=np.float32)
    img = gt + 0.5
    expected = np.sqrt(np.mean((0.5) ** 2)) / gt.max()
    assert float(nrmse(img, gt)) == float(expected)


def test_nrmse_zero_for_identical():
    gt = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
    assert nrmse(gt.copy(), gt) == 0.0


def _blob_gt():
    gt = np.zeros((50, 50, 50), dtype=np.float32)
    gt[24:27, 24:27, 24:27] = 10.0
    gt[10:14, 10:14, 10:14] = 6.0
    return gt


def test_derive_lesion_rois_finds_components_above_threshold():
    from scipy.ndimage import gaussian_filter

    gt = gaussian_filter(_blob_gt(), sigma=1.2)
    rois = derive_lesion_rois(gt, min_volume_vox=5)
    assert len(rois) >= 2
    volumes = sorted(int(r.sum()) for r in rois)
    assert volumes[0] >= 5


def test_background_vois_are_disjoint_from_lesions_and_deterministic():
    from scipy.ndimage import gaussian_filter

    gt = gaussian_filter(_blob_gt(), sigma=1.2)
    lesions = derive_lesion_rois(gt, min_volume_vox=5)
    exclusion = np.logical_or.reduce(lesions)
    v1 = background_vois(gt.shape, exclude_mask=exclusion, n_vois=4, radius_vox=3, seed=1)
    v2 = background_vois(gt.shape, exclude_mask=exclusion, n_vois=4, radius_vox=3, seed=1)
    assert len(v1) == 4
    assert all(np.array_equal(a, b) for a, b in zip(v1, v2))
    for voi in v1:
        assert not (voi & exclusion).any()
    union = np.logical_or.reduce(v1)
    assert union.sum() > 0
