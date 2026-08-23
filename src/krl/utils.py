"""I/O utilities for KRL: load and save images as CIL ImageData."""

from pathlib import Path

import nibabel as nib
import numpy as np
from cil.framework import ImageGeometry


def get_array(x):
    """
    Extract numpy array from various data containers.

    Parameters
    ----------
    x : object
        Data container (CIL ImageData, SIRF ImageData, or numpy array)

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


def load_nifti_as_imagedata(filepath):
    """
    Load a NIfTI file and convert to CIL ImageData.

    Parameters
    ----------
    filepath : str or Path
        Path to .nii or .nii.gz file

    Returns
    -------
    ImageData
        CIL ImageData container with loaded image
    """
    nii = nib.load(str(filepath))
    data = nii.get_fdata().astype(np.float32)

    # Get voxel sizes from affine (in mm)
    voxel_sizes = nib.affines.voxel_sizes(nii.affine)

    # CIL expects (z, y, x) order; NIfTI is typically (x, y, z)
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))

    geometry = ImageGeometry(
        voxel_num_x=data.shape[2],
        voxel_num_y=data.shape[1],
        voxel_num_z=data.shape[0],
        voxel_size_x=float(voxel_sizes[0]),
        voxel_size_y=float(voxel_sizes[1]),
        voxel_size_z=float(voxel_sizes[2]),
    )

    img = geometry.allocate()
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

    if filepath.suffix in ['.nii', '.gz'] or str(filepath).endswith('.nii.gz'):
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
    Save an image to file.

    Parameters
    ----------
    image : ImageData
        CIL ImageData
    filepath : str or Path
        Output file path (.nii, .nii.gz)
    """
    filepath = Path(filepath)

    if not (filepath.suffix in ['.nii', '.gz'] or str(filepath).endswith('.nii.gz')):
        raise ValueError(
            f"Unsupported file format: {filepath.suffix}. "
            "Supported formats: .nii, .nii.gz"
        )

    # Get array and transpose back to NIfTI convention
    data = get_array(image)
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))  # CIL (z,y,x) -> NIfTI (x,y,z)

    voxel_sizes = (1.0, 1.0, 1.0)
    geometry = getattr(image, 'geometry', None)
    if geometry is not None and hasattr(geometry, 'voxel_size_x'):
        voxel_sizes = (
            float(geometry.voxel_size_x),
            float(geometry.voxel_size_y),
            float(geometry.voxel_size_z),
        )
    elif hasattr(image, 'voxel_size_x'):
        voxel_sizes = (
            float(image.voxel_size_x),
            float(image.voxel_size_y),
            float(image.voxel_size_z),
        )

    affine = np.diag([voxel_sizes[0], voxel_sizes[1], voxel_sizes[2], 1.0])

    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, str(filepath))
