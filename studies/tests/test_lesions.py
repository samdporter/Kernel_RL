import numpy as np

from krl_studies.datasets.lesions import (
    DEFAULT_TUMOUR_DIAMETERS_MM,
    default_tumour_specs,
    place_tumours,
    sphere_mask,
)


def test_default_specs_cover_four_sizes_at_distinct_positions():
    specs = default_tumour_specs(shape=(100, 200, 200), voxel_mm=(1.0, 1.0, 1.0))
    assert len(specs) == len(DEFAULT_TUMOUR_DIAMETERS_MM) == 4
    centres = [s["centre_zyx"] for s in specs]
    assert len({tuple(np.round(c, 2)) for c in centres}) == 4
    diameters = sorted(2 * s["radius_mm"] for s in specs)
    assert diameters == sorted(DEFAULT_TUMOUR_DIAMETERS_MM)


def test_sphere_mask_volume_close_to_analytic(rng):
    shape = (60, 60, 60)
    mask = sphere_mask(shape, centre_zyx=(30.0, 30.0, 30.0), radius_vox=8.0)
    analytic = 4.0 / 3.0 * np.pi * 8.0**3
    assert abs(mask.sum() - analytic) / analytic < 0.05


def test_place_tumours_multiplies_only_inside_masks():
    pet = np.full((80, 80, 80), 1.0, dtype=np.float32)
    specs = [{"centre_zyx": (40.0, 40.0, 40.0), "radius_mm": 6.0}]
    with_lesions, masks = place_tumours(pet, specs, contrast=4.0, voxel_mm=(1.0,) * 3)
    assert len(masks) == 1
    m = masks[0]
    assert np.allclose(with_lesions[m], 4.0)
    assert np.allclose(with_lesions[~m], 1.0)
    assert not np.allclose(pet, with_lesions)


def test_place_tumours_does_not_modify_input():
    pet = np.ones((30, 30, 30), dtype=np.float32)
    snapshot = pet.copy()
    place_tumours(pet, [{"centre_zyx": (15.0, 15.0, 15.0), "radius_mm": 3.0}], 2.0, (1.0,) * 3)
    assert np.array_equal(pet, snapshot)


def test_default_masks_do_not_overlap():
    specs = default_tumour_specs(shape=(181, 217, 181), voxel_mm=(1.0, 1.0, 1.0))
    _, masks = place_tumours(np.zeros((181, 217, 181), dtype=np.float32), specs,
                             contrast=4.0, voxel_mm=(1.0, 1.0, 1.0))
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert not (masks[i] & masks[j]).any()
