"""Tests for NIfTI I/O functionality."""

import tempfile
import warnings
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


@pytest.mark.parametrize(
    "shape",
    [(0, 12, 10), (10, 0, 10), (10, 12, 0)],
    ids=["empty-z", "empty-y", "empty-x"],
)
def test_load_rejects_zero_size_axes(shape, tmp_path):
    """Zero-size axes cannot be converted to a CIL ImageData either."""
    from krl.utils import load_image

    with pytest.raises(ValueError, match="zero-size axes"):
        load_image(_write_nifti(tmp_path, shape))


def _affine(linear, translation=(0.0, 0.0, 0.0)):
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(linear, dtype=np.float64)
    affine[:3, 3] = np.asarray(translation, dtype=np.float64)
    return affine


def _write_volume(path, data, affine, big_endian=False):
    import nibabel as nib

    header = nib.Nifti1Header()
    header.set_data_dtype(data.dtype)
    if big_endian:
        header = header.as_byteswapped('>')
    nib.save(nib.Nifti1Image(data, affine, header=header), str(path))
    return Path(path)


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_save_rejects_ndarray(suffix, tmp_path):
    """A bare ndarray is not CIL ImageData and must be rejected before writing."""
    from krl.utils import save_image

    output = tmp_path / f"out{suffix}"
    with pytest.raises(TypeError, match="only supports CIL ImageData"):
        save_image(np.zeros((4, 3, 2), dtype=np.float32), output)
    assert not output.exists()


def test_save_rejects_duck_typed_object(tmp_path):
    """Duck-typing on as_array() is not enough: the container must be ImageData."""
    from krl.utils import save_image

    class DuckImage:
        def as_array(self):
            return np.zeros((2, 3, 4), dtype=np.float32)

    output = tmp_path / "out.nii.gz"
    with pytest.raises(TypeError, match="only supports CIL ImageData"):
        save_image(DuckImage(), output)
    assert not output.exists()


def test_save_asymmetric_imagedata_transposes(tmp_path):
    """The documented CIL (z,y,x) -> NIfTI (x,y,z) transpose is preserved."""
    import nibabel as nib
    from cil.framework import ImageGeometry

    from krl.utils import save_image

    geometry = ImageGeometry(voxel_num_x=4, voxel_num_y=3, voxel_num_z=2)
    image = geometry.allocate(0.0)
    image.fill(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

    output = tmp_path / "out.nii"
    save_image(image, output)

    saved = nib.load(str(output)).get_fdata()
    assert saved.shape == (4, 3, 2)
    assert saved[3, 2, 1] == image.as_array()[1, 2, 3]


@pytest.mark.parametrize(
    "shape,spacing,translation",
    [
        ((4, 3, 2), (2.0, 3.0, 4.0), (10.0, -7.0, 5.0)),
        ((5, 4, 3), (1.5, 2.5, 0.5), (-1.0, 2.0, 3.0)),
        ((2, 5, 4), (0.5, 1.0, 2.0), (0.0, 0.0, 0.0)),
        ((3, 2, 3), (3.0, 2.0, 1.0), (4.0, -2.0, 8.0)),
    ],
    ids=["even-odd-even", "odd-even-odd", "zero-origin", "mixed"],
)
def test_spatial_geometry_roundtrip(shape, spacing, translation, tmp_path):
    """Positive axis-aligned spacing and arbitrary translation round-trip."""
    import nibabel as nib

    from krl.utils import load_image, save_image

    data = (np.arange(np.prod(shape), dtype=np.float32) + 1).reshape(shape)
    path = _write_volume(tmp_path / "geom.nii", data, _affine(np.diag(spacing), translation))

    image = load_image(path)
    geometry = image.geometry
    assert image.as_array().shape == (shape[2], shape[1], shape[0])
    np.testing.assert_allclose(
        [geometry.voxel_size_x, geometry.voxel_size_y, geometry.voxel_size_z],
        spacing,
        rtol=1e-6,
    )

    expected_center = np.asarray(translation) + (np.asarray(shape) - 1) * np.asarray(spacing) / 2
    np.testing.assert_allclose(
        [geometry.center_x, geometry.center_y, geometry.center_z],
        expected_center,
        rtol=1e-6,
    )

    # Independent check: the first voxel centre is the affine translation and
    # the last is translation + (N-1)*spacing, compared against CIL's reported
    # extents rather than the centre formula.
    first_voxel = np.array([
        geometry.get_min_x() + geometry.voxel_size_x / 2,
        geometry.get_min_y() + geometry.voxel_size_y / 2,
        geometry.get_min_z() + geometry.voxel_size_z / 2,
    ])
    last_voxel = np.array([
        geometry.get_max_x() - geometry.voxel_size_x / 2,
        geometry.get_max_y() - geometry.voxel_size_y / 2,
        geometry.get_max_z() - geometry.voxel_size_z / 2,
    ])
    np.testing.assert_allclose(first_voxel, translation, rtol=1e-6)
    np.testing.assert_allclose(
        last_voxel,
        np.asarray(translation) + (np.asarray(shape) - 1) * np.asarray(spacing),
        rtol=1e-6,
    )

    output = tmp_path / "geom_out.nii"
    save_image(image, output)
    saved = nib.load(str(output))
    np.testing.assert_allclose(np.diag(saved.affine[:3, :3]), spacing, rtol=1e-6)
    np.testing.assert_allclose(saved.affine[:3, 3], translation, atol=1e-6)
    np.testing.assert_array_equal(saved.get_fdata(), data)


def test_save_nonzero_centred_geometry_affine(tmp_path):
    """A fresh CIL geometry with a nonzero centre writes the matching affine."""
    import nibabel as nib
    from cil.framework import ImageGeometry

    from krl.utils import save_image

    geometry = ImageGeometry(
        voxel_num_x=4,
        voxel_num_y=3,
        voxel_num_z=2,
        voxel_size_x=2.0,
        voxel_size_y=3.0,
        voxel_size_z=4.0,
        center_x=13.0,
        center_y=-4.0,
        center_z=7.0,
    )
    image = geometry.allocate(0.0)
    image.fill(np.arange(24, dtype=np.float32).reshape(2, 3, 4))

    output = tmp_path / "out.nii"
    save_image(image, output)

    saved = nib.load(str(output))
    np.testing.assert_allclose(np.diag(saved.affine[:3, :3]), (2.0, 3.0, 4.0), atol=1e-6)
    np.testing.assert_allclose(saved.affine[:3, 3], (10.0, -7.0, 5.0), atol=1e-6)


def test_save_2d_imagedata_keeps_z_origin(tmp_path):
    """A 2-D geometry has voxel_num_z == 0; its absent z axis must not be
    shifted by the 3-D translation formula, and CIL (y, x) is written as
    NIfTI (x, y)."""
    import nibabel as nib
    from cil.framework import ImageGeometry

    from krl.utils import save_image

    geometry = ImageGeometry(
        voxel_num_x=4, voxel_num_y=3, voxel_size_x=2.0, voxel_size_y=3.0
    )
    assert geometry.voxel_num_z == 0
    image = geometry.allocate(0.0)
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    image.fill(data)

    output = tmp_path / "out2d.nii"
    save_image(image, output)

    saved = nib.load(str(output))
    assert saved.affine[2, 3] == pytest.approx(0.0)
    assert saved.shape == (4, 3)
    np.testing.assert_allclose(
        np.diag(saved.affine[:3, :3]), (2.0, 3.0, 1.0), atol=1e-6
    )
    # CIL (y, x) -> NIfTI (x, y): voxel (y=1, x=2) lands at saved[2, 1].
    saved_data = saved.get_fdata()
    np.testing.assert_array_equal(saved_data, data.T)
    assert saved_data[2, 1] == image.as_array()[1, 2]


def test_save_2d_matches_3d_xy_mapping(tmp_path):
    """The 2-D save applies the same (x, y) transpose as the 3-D save, checked
    against the corresponding z slice."""
    import nibabel as nib
    from cil.framework import ImageGeometry

    from krl.utils import save_image

    data2d = np.arange(12, dtype=np.float32).reshape(3, 4)

    geometry2d = ImageGeometry(
        voxel_num_x=4, voxel_num_y=3, voxel_size_x=2.0, voxel_size_y=3.0
    )
    image2d = geometry2d.allocate(0.0)
    image2d.fill(data2d)

    geometry3d = ImageGeometry(
        voxel_num_x=4,
        voxel_num_y=3,
        voxel_num_z=2,
        voxel_size_x=2.0,
        voxel_size_y=3.0,
        voxel_size_z=5.0,
    )
    image3d = geometry3d.allocate(0.0)
    image3d.fill(np.stack([data2d, data2d + 100.0], axis=0))

    output2d = tmp_path / "out2d.nii"
    output3d = tmp_path / "out3d.nii"
    save_image(image2d, output2d)
    save_image(image3d, output3d)

    saved2d = nib.load(str(output2d)).get_fdata()
    saved3d = nib.load(str(output3d)).get_fdata()
    np.testing.assert_array_equal(saved2d, saved3d[:, :, 0])


@pytest.mark.parametrize(
    "name,linear",
    [
        ("flip", np.diag([2.0, 3.0, -4.0])),
        ("lps-negative-diagonal", np.diag([-2.0, -3.0, -4.0])),
        ("permutation", np.array([[0.0, 0.0, 4.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])),
        (
            "rotation",
            np.array([
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0],
                [0.0, 0.0, 4.0],
            ]),
        ),
        ("shear", np.array([[2.0, 0.5, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])),
    ],
)
def test_load_rejects_unrepresentable_affines(name, linear, tmp_path):
    from krl.utils import load_image

    data = np.ones((4, 3, 2), dtype=np.float32)
    path = _write_volume(tmp_path / "bad.nii", data, _affine(linear))
    with pytest.raises(ValueError, match="Unsupported NIfTI affine"):
        load_image(path)


def test_load_rejects_noncanonical_affine_last_row(tmp_path, monkeypatch):
    """A last row that is within the old default rtol but outside atol=1e-6
    must still be rejected (no relative tolerance on the bottom row)."""
    import nibabel as nib

    from krl.utils import load_image

    data = np.ones((4, 3, 2), dtype=np.float32)
    real = nib.Nifti1Image(data, _affine(np.eye(3)))
    bad_affine = np.array(real.affine, dtype=np.float64)
    bad_affine[3, 3] = 1.000005

    class StubNifti:
        def get_data_dtype(self):
            return real.get_data_dtype()

        def get_fdata(self, dtype=None, caching=None):
            return real.get_fdata(dtype=dtype, caching=caching)

        @property
        def affine(self):
            return bad_affine

    monkeypatch.setattr("krl.utils.nib.load", lambda path: StubNifti())
    with pytest.raises(ValueError, match="Unsupported NIfTI affine"):
        load_image(tmp_path / "bad.nii")


def test_load_accepts_translation_only(tmp_path):
    """Translation alone is representable and must not be rejected."""
    from krl.utils import load_image

    data = np.ones((4, 3, 2), dtype=np.float32)
    path = _write_volume(tmp_path / "translated.nii", data, _affine(np.diag([2.0, 3.0, 4.0]), (10.0, -7.0, 5.0)))

    image = load_image(path)
    np.testing.assert_allclose(
        [image.geometry.center_x, image.geometry.center_y, image.geometry.center_z],
        (13.0, -4.0, 7.0),
        rtol=1e-6,
    )


@pytest.mark.parametrize("nifti_dtype", [np.float32, np.float64])
def test_load_preserves_float_precision(nifti_dtype, tmp_path):
    from krl.utils import load_image

    data = np.random.default_rng(0).random((3, 4, 5)).astype(nifti_dtype)
    path = _write_volume(tmp_path / "f.nii", data, _affine(np.eye(3)))

    image = load_image(path)
    assert image.as_array().dtype == nifti_dtype
    np.testing.assert_array_equal(image.as_array(), np.transpose(data, (2, 1, 0)))


def test_load_keeps_float64_values_unrepresentable_in_float32(tmp_path):
    from krl.utils import load_image

    value = 2**24 + 1
    assert float(np.float32(value)) != value
    data = np.full((2, 3, 4), value, dtype=np.float64)
    path = _write_volume(tmp_path / "f64.nii", data, _affine(np.eye(3)))

    image = load_image(path)
    assert image.as_array().dtype == np.float64
    assert np.all(image.as_array() == value)


@pytest.mark.parametrize(
    "disk_dtype,native_dtype",
    [(">f4", np.float32), (">f8", np.float64)],
    ids=["big-endian-f4", "big-endian-f8"],
)
def test_load_big_endian_float(disk_dtype, native_dtype, tmp_path):
    from krl.utils import load_image

    data = np.arange(24, dtype=native_dtype).reshape(2, 3, 4)
    path = _write_volume(tmp_path / "be.nii", data, _affine(np.eye(3)), big_endian=True)

    image = load_image(path)
    assert image.as_array().dtype == native_dtype
    np.testing.assert_array_equal(image.as_array(), np.transpose(data, (2, 1, 0)))


@pytest.mark.parametrize("integer_dtype", [np.int16, np.int32])
def test_load_integer_volumes_as_float64(integer_dtype, tmp_path):
    from krl.utils import load_image

    value = 2**24 + 1 if integer_dtype is np.int32 else 12345
    data = np.full((2, 3, 4), value, dtype=integer_dtype)
    path = _write_volume(tmp_path / "int.nii", data, _affine(np.eye(3)))

    image = load_image(path)
    assert image.as_array().dtype == np.float64
    assert np.all(image.as_array() == value)


def test_load_int64_within_exact_range_is_exact(tmp_path):
    from krl.utils import load_image

    data = ((np.arange(24, dtype=np.int64) - 12).reshape(2, 3, 4)) * 2**40
    path = _write_volume(tmp_path / "i64.nii", data, _affine(np.eye(3)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image = load_image(path)

    assert not any(issubclass(warning.category, UserWarning) for warning in caught)
    assert image.as_array().dtype == np.float64
    np.testing.assert_array_equal(image.as_array(), np.transpose(data, (2, 1, 0)))


def test_load_int64_out_of_exact_range_warns(tmp_path):
    from krl.utils import load_image

    value = 2**53 + 1
    data = np.full((2, 3, 4), value, dtype=np.int64)
    path = _write_volume(tmp_path / "i64_big.nii", data, _affine(np.eye(3)))

    with pytest.warns(UserWarning, match="float64"):
        image = load_image(path)

    assert image.as_array().dtype == np.float64
    assert image.as_array()[0, 0, 0] == float(2**53)


def test_load_rejects_complex_and_rgb(tmp_path):
    import nibabel as nib

    from krl.utils import load_image

    complex_data = np.zeros((2, 3, 4), dtype=np.complex64)
    complex_path = _write_volume(tmp_path / "complex.nii", complex_data, _affine(np.eye(3)))
    with pytest.raises(ValueError, match="Unsupported NIfTI data type"):
        load_image(complex_path)

    rgb_dtype = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    rgb_data = np.zeros((2, 3, 4), dtype=rgb_dtype)
    rgb_path = tmp_path / "rgb.nii"
    nib.save(nib.Nifti1Image(rgb_data, _affine(np.eye(3))), str(rgb_path))
    with pytest.raises(ValueError, match="Unsupported NIfTI data type"):
        load_image(rgb_path)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_loaded_precision_survives_blur_and_kernel(dtype, tmp_path):
    from krl.operators.blurring import create_gaussian_blur
    from krl.operators.kernel_operator import get_kernel_operator
    from krl.utils import load_image

    data = np.random.default_rng(0).random((5, 6, 4)).astype(dtype)
    path = _write_volume(tmp_path / "img.nii", data, _affine(np.eye(3)))

    image = load_image(path)
    assert image.as_array().dtype == dtype

    blur = create_gaussian_blur((1.0, 1.0, 1.0), image.geometry, backend="numba")
    assert blur.direct(image).as_array().dtype == dtype

    kernel = get_kernel_operator(
        image.geometry,
        backend="numba",
        num_neighbours=3,
        sigma_anat=0.5,
        sigma_dist=1.0,
        normalize_kernel=True,
        use_mask=False,
    )
    anatomy = image.geometry.allocate(0.0)
    anatomy.fill(np.indices(image.geometry.shape).sum(axis=0).astype(dtype))
    kernel.set_anatomical_image(anatomy)

    assert kernel.direct(image).as_array().dtype == dtype
    assert kernel.adjoint(image).as_array().dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
