import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


def write_test_nifti(path, array_zyx, voxel_mm=(1.0, 1.0, 1.0)):
    """Helper used by several test modules; transposes (z,y,x)->(x,y,z)."""
    import nibabel as nib

    data_xyz = np.transpose(array_zyx.astype(np.float32), (2, 1, 0))
    affine = np.diag([voxel_mm[2], voxel_mm[1], voxel_mm[0], 1.0])
    nib.save(nib.Nifti1Image(data_xyz, affine), str(path))
