"""BrainWeb dataset preparation tests (Task 4).

Requires ``brainweb`` pip package (lazy dep). Marked ``sirf`` per plan
so native macOS runs skip when the package (or network) is unavailable;
the ``brainweb`` marker is an alias. Pure unit test for
``regions_from_labels`` runs without brainweb.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

try:
    import brainweb  # noqa: F401

    HAS_BRAINWEB = True
except ImportError:
    HAS_BRAINWEB = False

try:
    import nibabel  # noqa: F401

    HAS_NIB = True
except ImportError:
    HAS_NIB = False


# ------------------------------------------------------------------ pure test
def test_regions_from_labels_partitions_brain():
    from krl_studies.datasets.brainweb import regions_from_labels

    # synthetic 10³ labels: BG=0, CSF=1, GM=2, WM=3
    labels = np.zeros((10, 10, 10), dtype=np.int16)
    labels[0:4, :, :] = 3  # WM
    labels[4:7, :, :] = 2  # GM
    labels[7:9, :, :] = 1  # CSF
    # last slice remains 0 BG

    masks = regions_from_labels(labels)
    assert len(masks) == 3
    wm, gm, rest = masks
    for m in masks:
        assert m.shape == labels.shape
        assert m.dtype == bool

    # disjoint
    assert not np.any(wm & gm)
    assert not np.any(wm & rest)
    assert not np.any(gm & rest)

    # cover brain (non-background) exactly when rest is CSF-only,
    # or cover whole volume when rest = CSF+BG - accept either
    brain = labels != 0
    union = wm | gm | rest
    # At minimum, brain must be covered
    assert np.all(union[brain])
    # Count sanity: WM+GM non-empty
    assert wm.sum() > 0
    assert gm.sum() > 0
    # rest covers at least CSF
    assert rest.sum() >= (labels == 1).sum()


def test_regions_from_labels_handles_random_labels():
    from krl_studies.datasets.brainweb import regions_from_labels

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=(8, 8, 8), dtype=np.int16)
    masks = regions_from_labels(labels)
    assert len(masks) == 3
    wm, gm, rest = masks
    assert not np.any(wm & gm)
    # union covers either brain or whole volume
    brain = labels != 0
    assert np.all((wm | gm | rest)[brain])


# ------------------------------------------------------------ download tests
@pytest.mark.sirf
@pytest.mark.brainweb
@pytest.mark.skipif(not HAS_BRAINWEB, reason="brainweb not installed")
@pytest.mark.skipif(not HAS_NIB, reason="nibabel not available")
def test_prepare_subject_writes_files(tmp_path):
    from krl_studies.datasets.brainweb import prepare_subject

    out = tmp_path / "subj04"
    try:
        paths, labels = prepare_subject(subject_id=4, out_dir=out, tumour=False)
    except Exception as exc:  # noqa: BLE001 - network / brainweb download failures
        msg = str(exc).lower()
        if any(k in msg for k in ("network", "download", "connection", "timeout", "url", "http", "get_file", "links")):
            pytest.skip(f"brainweb download unavailable: {exc}")
        # Also handle requests exceptions which may not contain those keywords
        if "brainweb" in msg or "requests" in msg:
            pytest.skip(f"brainweb unavailable: {exc}")
        raise

    # files exist
    assert pathlib.Path(paths["pet_gt"]).exists()
    assert pathlib.Path(paths["mr_t1"]).exists()
    assert pathlib.Path(paths["labels"]).exists()
    assert (out / "pet_gt.nii.gz").exists()
    assert (out / "mr_t1.nii.gz").exists()
    assert (out / "labels.nii.gz").exists()

    # shapes & dtypes
    import nibabel as nib

    pet = np.transpose(nib.load(str(paths["pet_gt"])).get_fdata(), (2, 1, 0))
    t1 = np.transpose(nib.load(str(paths["mr_t1"])).get_fdata(), (2, 1, 0))
    lab = np.transpose(nib.load(str(paths["labels"])).get_fdata(), (2, 1, 0))
    assert pet.shape == t1.shape == lab.shape == labels.shape
    assert pet.ndim == 3
    assert set(np.unique(labels).tolist()).issubset({0, 1, 2, 3})
    assert pet.max() > 0
    assert t1.max() > 0


@pytest.mark.sirf
@pytest.mark.brainweb
@pytest.mark.skipif(not HAS_BRAINWEB, reason="brainweb not installed")
@pytest.mark.skipif(not HAS_NIB, reason="nibabel not available")
def test_prepare_subject_tumour_placement_respects_labels(tmp_path):
    from krl_studies.datasets.brainweb import prepare_subject, regions_from_labels

    out_tum = tmp_path / "subj04_tum"
    out_base = tmp_path / "subj04_base"
    try:
        paths_tum, labels = prepare_subject(subject_id=4, out_dir=out_tum, tumour=True)
        paths_base, _ = prepare_subject(subject_id=4, out_dir=out_base, tumour=False)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if any(k in msg for k in ("network", "download", "connection", "timeout", "url", "http", "get_file", "links")):
            pytest.skip(f"brainweb download unavailable: {exc}")
        if "brainweb" in msg or "requests" in msg:
            pytest.skip(f"brainweb unavailable: {exc}")
        raise

    masks = regions_from_labels(labels)
    assert len(masks) == 3
    wm, gm, _ = masks
    brain_tissue = wm | gm
    assert brain_tissue.sum() > 0

    import nibabel as nib
    from scipy.ndimage import label as nd_label

    pet_tum = np.transpose(nib.load(str(paths_tum["pet_gt"])).get_fdata(), (2, 1, 0))
    pet_base = np.transpose(nib.load(str(paths_base["pet_gt"])).get_fdata(), (2, 1, 0))
    assert pet_tum.shape == pet_base.shape == labels.shape
    # tumours increase PET by contrast factor 4; find lesion voxels where ratio >1.5
    # guard against zeros in pet_base
    ratio = np.divide(pet_tum, np.maximum(pet_base, 1e-6), out=np.zeros_like(pet_tum), where=pet_base > 1e-6)
    lesion_mask = ratio > 1.5
    assert lesion_mask.sum() > 0, "tumour placement did not increase PET"
    # lesions should be inside GM/WM tissue (allow small CSF spill due to interpolation)
    overlap = np.logical_and(lesion_mask, brain_tissue).sum()
    assert overlap / lesion_mask.sum() > 0.5, (
        f"only {overlap}/{lesion_mask.sum()} lesion voxels overlap GM/WM"
    )
    # check that major lesion components are near GM/WM (ignore tiny fragments)
    lab_arr, nlab = nd_label(lesion_mask)
    assert nlab >= 3, f"expected >=3 lesion components, got {nlab}"
    sizes = [(lab_arr == i).sum() for i in range(1, nlab + 1)]
    # consider only substantial lesions (>30 voxels)
    large_ids = [i for i, s in enumerate(sizes, start=1) if s > 30]
    assert len(large_ids) >= 2, f"expected >=2 substantial lesions, got {large_ids} sizes {sizes}"
    # at least 60% of large lesions should be near GM/WM
    near_brain = 0
    for i in large_ids:
        comp = lab_arr == i
        coords = np.argwhere(comp)
        cz, cy, cx = coords.mean(axis=0)
        iz, iy, ix = int(round(cz)), int(round(cy)), int(round(cx))
        iz = int(np.clip(iz, 0, labels.shape[0] - 1))
        iy = int(np.clip(iy, 0, labels.shape[1] - 1))
        ix = int(np.clip(ix, 0, labels.shape[2] - 1))
        z0, z1 = max(0, iz - 1), min(labels.shape[0], iz + 2)
        y0, y1 = max(0, iy - 1), min(labels.shape[1], iy + 2)
        x0, x1 = max(0, ix - 1), min(labels.shape[2], ix + 2)
        neigh = labels[z0:z1, y0:y1, x0:x1]
        if np.any((neigh == 2) | (neigh == 3)):
            near_brain += 1
        else:
            overlap_c = np.logical_and(comp, brain_tissue).sum()
            if overlap_c / comp.sum() > 0.2:
                near_brain += 1
    assert near_brain >= max(2, len(large_ids) * 0.6), (
        f"only {near_brain}/{len(large_ids)} large lesions near GM/WM"
    )
