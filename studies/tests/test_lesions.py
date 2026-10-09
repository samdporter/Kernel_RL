import numpy as np
import pytest

from krl_studies.datasets.lesions import (
    DEFAULT_TUMOUR_DIAMETERS_MM,
    default_tumour_specs,
    place_tumours,
    plan_tumour_layout,
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


def test_plan_tumour_layout_is_deterministic_and_valid():
    labels = np.zeros((80, 80, 80), dtype=np.int16)
    labels[10:70, 10:70, 10:70] = 3
    pet = labels.astype(np.float32) * 2.0
    voxel_mm = (2.0, 2.0, 2.0)

    pet1, specs1, masks1, meta1 = plan_tumour_layout(pet, labels, voxel_mm, 4)
    pet2, specs2, masks2, meta2 = plan_tumour_layout(pet, labels, voxel_mm, 4)

    assert meta1["layout_hash"] == meta2["layout_hash"]
    assert np.array_equal(pet1, pet2)
    assert [s["centre_zyx"] for s in specs1] == [s["centre_zyx"] for s in specs2]
    assert len(masks1) == 4

    brain = labels > 0
    for mask in masks1:
        assert mask.ndim == 3
        assert mask.dtype == bool
        assert brain[mask].all()
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (masks1[i] & masks1[j]).any()

    assert [round(d) for d in meta1["diameters_mm"]] == [8, 12, 16, 24]
    assert all(v > 0 for v in meta1["realised_volumes_mm3"])
    assert pet1[masks1[0]].mean() == pytest.approx(float(pet[masks1[0]].mean()) * 4.0, rel=1e-5)


def test_plan_tumour_layout_reports_no_space():
    labels = np.zeros((20, 20, 20), dtype=np.int16)
    labels[9:11, 9:11, 9:11] = 3
    pet = labels.astype(np.float32)
    with pytest.raises(ValueError, match="could not fit"):
        plan_tumour_layout(pet, labels, (1.0, 1.0, 1.0), 1)


def test_layout_hash_includes_voxel_spacing_and_realised_masks():
    from krl_studies.datasets.lesions import build_layout_metadata

    specs = [{"centre_zyx": (10.0, 10.0, 10.0), "radius_mm": 4.0, "contrast": 4.0}]
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[8:12, 8:12, 8:12] = True

    base = build_layout_metadata(4, specs, [mask], (1.0, 1.0, 1.0), 4.0, 5.0)
    assert base["voxel_mm"] == [1.0, 1.0, 1.0]

    spaced = build_layout_metadata(4, specs, [mask], (2.0, 2.0, 2.0), 4.0, 5.0)
    assert spaced["layout_hash"] != base["layout_hash"]

    altered_mask = mask.copy()
    altered_mask[0, 0, 0] = True
    altered = build_layout_metadata(4, specs, [altered_mask], (1.0, 1.0, 1.0), 4.0, 5.0)
    assert altered["layout_hash"] != base["layout_hash"]
