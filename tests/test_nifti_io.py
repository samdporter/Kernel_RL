"""Tests for NIfTI I/O functionality."""

import numpy as np
import pytest
import tempfile
from pathlib import Path


def test_nifti_load_save_roundtrip():
    """Test loading and saving NIfTI files with CIL ImageData."""
    try:
        import nibabel as nib
        from cil.framework import ImageGeometry
        from krl.utils import load_image, save_image
    except ImportError:
        pytest.skip("nibabel or CIL not available")

    # Create test data
    data = np.random.rand(10, 12, 8).astype(np.float32)
    voxel_sizes = (2.0, 2.0, 3.0)

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save as NIfTI directly with nibabel
        affine = np.diag([voxel_sizes[0], voxel_sizes[1], voxel_sizes[2], 1.0])
        nii = nib.Nifti1Image(data, affine)
        nifti_path = tmpdir / "test.nii.gz"
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
        output_path = tmpdir / "output.nii.gz"
        save_image(img, output_path)

        # Load back with nibabel to verify
        nii_output = nib.load(str(output_path))
        output_data = nii_output.get_fdata()

        # Should match original data
        np.testing.assert_allclose(output_data, data, rtol=1e-5)


def test_load_nifti_as_imagedata():
    """Test the load_nifti_as_imagedata function directly."""
    try:
        import nibabel as nib
        from krl.utils import load_nifti_as_imagedata
    except ImportError:
        pytest.skip("nibabel or CIL not available")

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


def test_save_unsupported_format():
    """Test that saving to unsupported formats raises an error."""
    try:
        from cil.framework import ImageGeometry
        from krl.utils import save_image
    except ImportError:
        pytest.skip("CIL not available")

    # Create dummy image
    geom = ImageGeometry(voxel_num_x=5, voxel_num_y=5, voxel_num_z=5)
    img = geom.allocate(1.0)

    with pytest.raises(ValueError, match="Unsupported file format"):
        save_image(img, "test.txt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
