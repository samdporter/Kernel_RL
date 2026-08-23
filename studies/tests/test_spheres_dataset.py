import numpy as np
import pytest

from krl_studies.datasets.spheres import SphereDataset, quick_sim


@pytest.fixture(scope="module")
def spheres_dir(tmp_path_factory):
    """Tiny stand-in phantom with the three canonical files."""
    from conftest import write_test_nifti

    d = tmp_path_factory.mktemp("spheres")
    gt = np.zeros((40, 60, 60), dtype=np.float32)
    gt[10:30, 25:35, 25:37] = 8.0
    gt += 1.0
    mr = np.full((40, 60, 60), 0.5, dtype=np.float32)
    write_test_nifti(d / "phant_orig.nii", gt)
    write_test_nifti(d / "phant_mri.nii", mr)
    write_test_nifti(d / "phant_pet.nii", gt * 0.9)
    return d


def test_dataset_loads_images_and_geometry(spheres_dir):
    ds = SphereDataset(root=spheres_dir)
    assert ds.ground_truth.shape == (40, 60, 60)
    assert ds.guidance.shape == (40, 60, 60)
    assert ds.reference_pet.shape == (40, 60, 60)
    assert ds.voxel_mm == pytest.approx((1.0, 1.0, 1.0))
    assert np.allclose(ds.reference_pet, ds.ground_truth * 0.9)


def test_dataset_requires_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="phant_orig"):
        SphereDataset(root=tmp_path)


def test_quick_sim_is_deterministic_and_adds_noise(spheres_dir):
    ds = SphereDataset(root=spheres_dir)
    a = quick_sim(ds.ground_truth, fwhm_mm=3.0, counts=1e4, realisation=0, voxel_mm=ds.voxel_mm)
    b = quick_sim(ds.ground_truth, fwhm_mm=3.0, counts=1e4, realisation=0, voxel_mm=ds.voxel_mm)
    c = quick_sim(ds.ground_truth, fwhm_mm=3.0, counts=1e4, realisation=7, voxel_mm=ds.voxel_mm)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.min() >= 0.0
    # blur must reduce peak compared to sharp gt; needs negligible noise, so
    # use high counts -- a single Poisson count maps to ~15 image units here,
    # dwarfing the phantom peak
    hi = quick_sim(ds.ground_truth, fwhm_mm=3.0, counts=1e12, realisation=0, voxel_mm=ds.voxel_mm)
    assert hi.max() <= ds.ground_truth.max() + 1e-2
