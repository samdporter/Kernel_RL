"""Utility functions for KRL package."""

import ctypes
import importlib.util
import sys
from pathlib import Path

import numpy as np


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


def _import_cil_image_classes():
    """
    Import ImageData and ImageGeometry from CIL, attempting to load the C++ backend
    library on-the-fly if it is not yet in the loader path.
    """
    try:
        from cil.framework import ImageData, ImageGeometry  # type: ignore
        return ImageData, ImageGeometry
    except (ImportError, OSError) as first_error:
        root = None
        try:
            spec = importlib.util.find_spec("cil")
            if spec and spec.origin:
                root = Path(spec.origin).resolve()
        except ValueError:
            module = sys.modules.get("cil")
            module_file = getattr(module, "__file__", None) if module else None
            module_paths = list(getattr(module, "__path__", [])) if module else []
            if module_file:
                root = Path(module_file).resolve()
            elif module_paths:
                root = Path(module_paths[0]).resolve()

        if root is None:
            for entry in map(Path, sys.path):
                candidate = entry / "cil" / "__init__.py"
                if candidate.exists():
                    root = candidate.resolve()
                    break

        if root is None:
            search_roots = [
                Path.cwd(),
                Path.cwd().parent,
                Path.home(),
                Path.home() / "devel",
                Path.home() / "devel" / "SIRF_builds",
                Path("/opt/conda"),
            ]
            for base in search_roots:
                if not base.exists():
                    continue
                try:
                    for candidate in base.rglob("cil/__init__.py"):
                        root = candidate.resolve()
                        break
                except (OSError, PermissionError):
                    continue
                if root is not None:
                    break

        if root:
            # CIL installations typically live under either:
            # <prefix>/python/cil  (with libs in <prefix>/lib)
            # or <prefix>/python/cil (with libs in <prefix>/cil/lib)
            package_root = root.parent
            if package_root.parent.exists():
                package_parent = package_root.parent
                if str(package_parent) not in sys.path:
                    sys.path.append(str(package_parent))
            candidates = []
            for parent in root.parents:
                candidates.append(parent / "lib" / "libcilacc.so")
                candidates.append(parent / "cil" / "lib" / "libcilacc.so")
            for candidate in candidates:
                if candidate.exists():
                    try:
                        ctypes.cdll.LoadLibrary(str(candidate))
                        # purge partially imported modules before retry
                        sys.modules.pop("cil.framework", None)
                        if (module := sys.modules.get("cil")) is not None and getattr(module, "__spec__", None) is None:
                            sys.modules.pop("cil", None)
                        from cil.framework import ImageData, ImageGeometry  # type: ignore
                        return ImageData, ImageGeometry
                    except OSError:
                        continue
        raise ImportError(
            "nibabel and CIL are required to load NIfTI files. Install with:\n"
            "  pip install nibabel\n"
            "  conda install -c conda-forge -c ccpi cil\n"
            "Ensure libcilacc.so is discoverable via LD_LIBRARY_PATH."
        ) from first_error


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
    except ImportError as e:
        raise ImportError(
            "nibabel and CIL are required to load NIfTI files. Install with:\n"
            "  pip install nibabel\n"
            "  conda install -c conda-forge -c ccpi cil"
        ) from e

    ImageData, ImageGeometry = _import_cil_image_classes()

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
