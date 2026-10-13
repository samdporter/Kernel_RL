"""I/O utilities for KRL: load and save images as CIL ImageData."""

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from cil.framework import ImageData, ImageGeometry

_EXACT_FLOAT64_INT = 2**53


def get_array(x):
    """
    Extract numpy array from various data containers.

    Parameters
    ----------
    x : object
        Data container (CIL ImageData or numpy array)

    Returns
    -------
    np.ndarray
        Numpy array representation
    """
    if hasattr(x, 'asarray'):
        return x.asarray()
    elif hasattr(x, 'as_array'):
        return x.as_array()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x)


def _unsupported_affine_error(filepath):
    return ValueError(
        f"Unsupported NIfTI affine in {filepath}: this conversion supports "
        "positive axis-aligned voxel spacing and arbitrary translation. Flips, "
        "axis permutations, rotation and shear require external "
        "reorientation/resampling before conversion to CIL ImageData."
    )


def _nifti_float_dtype(data_dtype, filepath):
    """Map the on-disk NIfTI dtype to the dtype used for the CIL image."""
    if data_dtype.kind == 'f' and data_dtype.itemsize == 4:
        return np.float32
    if data_dtype.kind == 'f' and data_dtype.itemsize == 8:
        return np.float64
    if data_dtype.kind in 'iu':
        return np.float64
    raise ValueError(
        f"Unsupported NIfTI data type {data_dtype!r} in {filepath}: only "
        "real floating-point (float32/float64) and integer volumes can be "
        "converted to CIL ImageData; complex, RGB and structured data are "
        "not supported."
    )


def _warn_if_int64_rounds(nii, filepath):
    raw = (
        nii.dataobj.get_unscaled()
        if hasattr(nii.dataobj, 'get_unscaled')
        else np.asanyarray(nii.dataobj)
    )
    lo, hi = int(raw.min()), int(raw.max())
    if lo < -_EXACT_FLOAT64_INT or hi > _EXACT_FLOAT64_INT:
        warnings.warn(
            f"64-bit integer values in {filepath} exceed the exact range of "
            f"float64 (+/-2**53); values may be rounded when converted to "
            "CIL ImageData.",
            UserWarning,
            stacklevel=3,
        )


def load_nifti_as_imagedata(filepath):
    """
    Load a NIfTI file and convert to CIL ImageData.

    The stored floating-point precision is preserved: float32 volumes load as
    float32 and float64 volumes as float64. Integer volumes load as float64.
    Complex, RGB and structured volumes are rejected.

    Only affines with positive, axis-aligned voxel spacing and arbitrary
    translation are supported; flips, axis permutations, rotation and shear
    are rejected.

    Parameters
    ----------
    filepath : str or Path
        Path to a 3-D .nii or .nii.gz file

    Returns
    -------
    ImageData
        CIL ImageData container with loaded image

    Raises
    ------
    ValueError
        If the file is not a 3-D volume, has a singleton or zero-size axis,
        uses an unsupported data type, or has an affine that CIL cannot
        represent.
    """
    nii = nib.load(str(filepath))

    data_dtype = nii.get_data_dtype()
    chosen_dtype = _nifti_float_dtype(data_dtype, filepath)
    if data_dtype.kind in 'iu' and data_dtype.itemsize == 8:
        _warn_if_int64_rounds(nii, filepath)

    data = nii.get_fdata(dtype=chosen_dtype, caching="unchanged")

    if data.ndim != 3:
        raise ValueError(
            f"Unsupported NIfTI shape {data.shape} in {filepath}: "
            "only 3-D volumes can be converted to CIL ImageData"
        )
    if 0 in data.shape:
        raise ValueError(
            f"Unsupported NIfTI shape {data.shape} in {filepath}: "
            "zero-size axes cannot be converted to CIL ImageData "
            "(the container would have no voxels)"
        )
    if 1 in data.shape:
        raise ValueError(
            f"Unsupported NIfTI shape {data.shape} in {filepath}: "
            "singleton axes cannot be converted to CIL ImageData "
            "(CIL drops size-1 dimensions or fails to fill the container)"
        )

    affine = np.asarray(nii.affine, dtype=np.float64)
    if not np.all(np.isfinite(affine)):
        raise _unsupported_affine_error(filepath)
    if not np.allclose(affine[3, :], (0.0, 0.0, 0.0, 1.0), rtol=0, atol=1e-6):
        raise _unsupported_affine_error(filepath)

    linear = affine[:3, :3]
    spacing = np.diag(linear).copy()
    if np.any(spacing <= 0):
        raise _unsupported_affine_error(filepath)
    if not np.allclose(linear / spacing[:, None], np.eye(3), rtol=0, atol=1e-6):
        raise _unsupported_affine_error(filepath)

    # Physical axis order is NIfTI (x, y, z); the array is transposed to CIL
    # (z, y, x), but spacing and centres stay in physical order.
    num_x, num_y, num_z = data.shape
    translation = affine[:3, 3]
    center_x = translation[0] + (num_x - 1) * spacing[0] / 2
    center_y = translation[1] + (num_y - 1) * spacing[1] / 2
    center_z = translation[2] + (num_z - 1) * spacing[2] / 2

    data = np.transpose(data, (2, 1, 0))

    geometry = ImageGeometry(
        voxel_num_x=num_x,
        voxel_num_y=num_y,
        voxel_num_z=num_z,
        voxel_size_x=float(spacing[0]),
        voxel_size_y=float(spacing[1]),
        voxel_size_z=float(spacing[2]),
        center_x=float(center_x),
        center_y=float(center_y),
        center_z=float(center_z),
        dtype=chosen_dtype,
    )

    img = geometry.allocate(dtype=chosen_dtype)
    img.fill(data)

    return img


def load_image(filepath):
    """
    Load an image file (supports .nii, .nii.gz).

    Parameters
    ----------
    filepath : str or Path
        Path to image file

    Returns
    -------
    ImageData
        CIL ImageData
    """
    filepath = Path(filepath)

    if filepath.name.endswith('.nii.gz') or filepath.suffix == '.nii':
        return load_nifti_as_imagedata(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: {filepath.suffix}. "
            "Supported formats: .nii, .nii.gz\n"
            "Note: This package only supports NIfTI format. "
            "If you have .hv files, please convert them to NIfTI first."
        )


def save_image(image, filepath):
    """
    Save a CIL ``ImageData`` to a NIfTI file.

    The voxel array and voxel spacing are written, and the CIL volume centre
    (``center_x/y/z``) is encoded as the affine translation. Original NIfTI
    header fields and the qform/sform codes are not retained.

    Parameters
    ----------
    image : ImageData
        CIL ImageData
    filepath : str or Path
        Output file path (.nii, .nii.gz)

    Raises
    ------
    TypeError
        If ``image`` is not a CIL ``ImageData``.
    ValueError
        If the file extension is unsupported, or the geometry has
        non-positive/non-finite spacing or non-finite centres.
    """
    if not isinstance(image, ImageData):
        raise TypeError(
            f"save_image only supports CIL ImageData; got "
            f"{type(image).__name__}. Convert to ImageData with explicit "
            "(z, y, x) ordering and geometry, or use nibabel to save "
            "NumPy/NIfTI data."
        )

    filepath = Path(filepath)

    if not (filepath.name.endswith('.nii.gz') or filepath.suffix == '.nii'):
        raise ValueError(
            f"Unsupported file format: {filepath.suffix}. "
            "Supported formats: .nii, .nii.gz"
        )

    geometry = image.geometry
    spacing = np.array(
        [geometry.voxel_size_x, geometry.voxel_size_y, geometry.voxel_size_z],
        dtype=np.float64,
    )
    counts = np.array(
        [geometry.voxel_num_x, geometry.voxel_num_y, geometry.voxel_num_z],
        dtype=np.float64,
    )
    centers = np.array(
        [geometry.center_x, geometry.center_y, geometry.center_z],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(
            f"save_image requires positive finite voxel spacing, got "
            f"{tuple(spacing)}"
        )
    if not np.all(np.isfinite(centers)):
        raise ValueError(
            f"save_image requires finite volume centres, got {tuple(centers)}"
        )

    # CIL (z,y,x) -> NIfTI (x,y,z); a 2-D CIL image is (y,x) -> NIfTI (x,y)
    data = get_array(image)
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))
    elif data.ndim == 2:
        data = np.transpose(data, (1, 0))

    # An absent axis (voxel_num == 0, e.g. z for a 2-D image) has no voxel
    # centre, so its translation stays at the origin instead of using the
    # centre formula.
    translation = np.where(
        counts > 0,
        centers - (counts - 1) * spacing / 2,
        0.0,
    )
    affine = np.diag(np.append(spacing, 1.0))
    affine[:3, 3] = translation

    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, str(filepath))
