"""Utility functions for KRL package."""

import numpy as np
from pathlib import Path


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
    try:
        import nibabel as nib
        from cil.framework import ImageData, ImageGeometry
    except ImportError as e:
        raise ImportError(
            "nibabel and CIL are required to load NIfTI files. Install with:\n"
            "  pip install nibabel\n"
            "  conda install -c conda-forge -c ccpi cil"
        ) from e

    # Load NIfTI file
    nii = nib.load(str(filepath))
    data = nii.get_fdata().astype(np.float32)

    # Get voxel sizes from affine (in mm)
    voxel_sizes = nib.affines.voxel_sizes(nii.affine)

    # CIL expects (z, y, x) order
    # NIfTI is typically (x, y, z), so transpose
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))

    # Create ImageGeometry
    geometry = ImageGeometry(
        voxel_num_x=data.shape[2],
        voxel_num_y=data.shape[1],
        voxel_num_z=data.shape[0],
        voxel_size_x=float(voxel_sizes[0]),
        voxel_size_y=float(voxel_sizes[1]),
        voxel_size_z=float(voxel_sizes[2]),
    )

    # Create ImageData
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

    # Load NIfTI files
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

    if filepath.suffix in ['.nii', '.gz'] or str(filepath).endswith('.nii.gz'):
        try:
            import nibabel as nib
        except ImportError as e:
            raise ImportError("nibabel is required to save NIfTI files") from e

        # Get array and transpose back to NIfTI convention
        data = get_array(image)
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))  # CIL (z,y,x) -> NIfTI (x,y,x)

        # Get voxel sizes if available
        if hasattr(image, 'voxel_size_x'):
            voxel_sizes = (
                image.voxel_size_x,
                image.voxel_size_y,
                image.voxel_size_z
            )
        elif hasattr(image, 'voxel_sizes'):
            vs = image.voxel_sizes()
            voxel_sizes = (vs[2], vs[1], vs[0])  # Reorder for NIfTI
        else:
            voxel_sizes = (1.0, 1.0, 1.0)

        # Create affine matrix
        affine = np.diag([voxel_sizes[0], voxel_sizes[1], voxel_sizes[2], 1.0])

        # Save as NIfTI
        nii = nib.Nifti1Image(data, affine)
        nib.save(nii, str(filepath))

    else:
        raise ValueError(
            f"Unsupported file format: {filepath.suffix}. "
            "Supported formats: .nii, .nii.gz\n"
            "Note: This package only supports NIfTI format."
        )
