"""Tests for NIfTI I/O functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_nifti_load_save_roundtrip(suffix):
    """Test loading and saving NIfTI files with CIL ImageData."""
    import nibabel as nib

    from krl.utils import load_image, save_image

    # Create test data
    data = np.random.rand(10, 12, 8).astype(np.float32)
    voxel_sizes = (2.0, 2.0, 3.0)

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save as NIfTI directly with nibabel
        affine = np.diag([voxel_sizes[0], voxel_sizes[1], voxel_sizes[2], 1.0])
        nii = nib.Nifti1Image(data, affine)
        nifti_path = tmpdir / f"test{suffix}"
        nib.save(nii, str(nifti_path))

        # Load with our utility
        img = load_image(nifti_path)

        # Check that it's a CIL ImageData
        assert hasattr(img, 'as_array'), "Should be a CIL ImageData"
        assert hasattr(img, 'geometry'), "Should have geometry attribute"

        # Check shape (should be transposed: x,y,z -> z,y,x)
        loaded_data = img.as_array()
        assert loaded_data.shape == (8, 12, 10), f"Expected (8, 12, 10), got {loaded_data.shape}"

        # Check data values (accounting for transpose)
        np.testing.assert_allclose(
            loaded_data,
            np.transpose(data, (2, 1, 0)),
            rtol=1e-5
        )

        # Save with our utility
        output_path = tmpdir / f"output{suffix}"
        save_image(img, output_path)

        # Load back with nibabel to verify
        nii_output = nib.load(str(output_path))
        output_data = nii_output.get_fdata()

        # Should match original data
        np.testing.assert_allclose(output_data, data, rtol=1e-5)

        # Voxel spacing should survive the round trip
        np.testing.assert_allclose(nii_output.header.get_zooms()[:3], voxel_sizes, rtol=1e-5)


def test_anisotropic_voxel_spacing_roundtrip(tmp_path):
    """Distinct x/y/z voxel sizes survive load and save without axis mix-ups."""
    import nibabel as nib

    from krl.utils import load_image, save_image

    data = np.random.rand(6, 7, 5).astype(np.float32)
    voxel_sizes = (2.0, 2.5, 3.5)
    affine = np.diag([voxel_sizes[0], voxel_sizes[1], voxel_sizes[2], 1.0])
    nifti_path = tmp_path / "aniso.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(nifti_path))

    img = load_image(nifti_path)
    geom = img.geometry
    assert geom.voxel_size_x == pytest.approx(voxel_sizes[0])
    assert geom.voxel_size_y == pytest.approx(voxel_sizes[1])
    assert geom.voxel_size_z == pytest.approx(voxel_sizes[2])
    assert img.as_array().shape == (5, 7, 6)

    output_path = tmp_path / "aniso_out.nii"
    save_image(img, output_path)

    nii_output = nib.load(str(output_path))
    np.testing.assert_allclose(nii_output.header.get_zooms()[:3], voxel_sizes, rtol=1e-5)
    np.testing.assert_allclose(nii_output.get_fdata(), data, rtol=1e-5)


def test_load_nifti_as_imagedata():
    """Test the load_nifti_as_imagedata function directly."""
    import nibabel as nib

    from krl.utils import load_nifti_as_imagedata

    # Create test NIfTI file
    data = np.ones((5, 6, 7), dtype=np.float32) * 42.0
    affine = np.eye(4)
    affine[0, 0] = 2.0  # voxel size x
    affine[1, 1] = 2.5  # voxel size y
    affine[2, 2] = 3.0  # voxel size z

    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / "test.nii.gz"
        nii = nib.Nifti1Image(data, affine)
        nib.save(nii, str(nifti_path))

        # Load with our function
        img = load_nifti_as_imagedata(nifti_path)

        # Check it's CIL ImageData
        assert hasattr(img, 'geometry')
        assert hasattr(img, 'as_array')

        # Check data values
        loaded = img.as_array()
        assert np.all(loaded == 42.0), "Data values should be preserved"

        # Check voxel sizes
        geom = img.geometry
        assert hasattr(geom, 'voxel_size_x')
        assert abs(geom.voxel_size_x - 2.0) < 1e-5
        assert abs(geom.voxel_size_y - 2.5) < 1e-5
        assert abs(geom.voxel_size_z - 3.0) < 1e-5


def test_load_unsupported_format():
    """Test that loading unsupported formats raises an error."""
    from krl.utils import load_image

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_image("test.txt")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_image("test.hv")

    # A gzipped file is only accepted when it is a real .nii.gz
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_image("foo.txt.gz")


def test_save_unsupported_format():
    """Test that saving to unsupported formats raises an error."""
    from cil.framework import ImageGeometry

    from krl.utils import save_image

    # Create dummy image
    geom = ImageGeometry(voxel_num_x=5, voxel_num_y=5, voxel_num_z=5)
    img = geom.allocate(1.0)

    with pytest.raises(ValueError, match="Unsupported file format"):
        save_image(img, "test.txt")

    with pytest.raises(ValueError, match="Unsupported file format"):
        save_image(img, "foo.txt.gz")


def _write_nifti(tmp_path, shape):
    import nibabel as nib

    path = Path(tmp_path) / "shape.nii"
    data = np.random.rand(*shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


@pytest.mark.parametrize("shape", [(10, 12), (10, 12, 8, 3)], ids=["2d", "4d"])
def test_load_rejects_non_3d_volumes(shape, tmp_path):
    """2-D and 4-D NIfTI files cannot be represented as a CIL ImageData."""
    from krl.utils import load_image

    with pytest.raises(ValueError, match="only 3-D volumes"):
        load_image(_write_nifti(tmp_path, shape))


@pytest.mark.parametrize(
    "shape",
    [(10, 12, 1), (1, 12, 10), (10, 1, 10)],
    ids=["singleton-z", "singleton-x", "singleton-y"],
)
def test_load_rejects_singleton_axes(shape, tmp_path):
    """CIL mishandles size-1 dimensions, so those volumes are rejected explicitly."""
    from krl.utils import load_image

    with pytest.raises(ValueError, match="singleton axes"):
        load_image(_write_nifti(tmp_path, shape))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
