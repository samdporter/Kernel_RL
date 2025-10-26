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

def prepare_brainweb_pet_dataset(
    out_dir,
    subject_ids=None,
    mr_modality="T1",                 # "T1" or "T2" (no lesions ever added to MR)
    fwhm_mm=5.0,                      # PET PSF FWHM in mm
    noise_model="poisson",            # "poisson" or "gaussian"
    poisson_scale=1e5,                # counts-per-unit scaling for Poisson
    gaussian_sigma=None,              # absolute σ (same units as PET image); if None use 5% of max
    replicate_subject_id=None,        # e.g. 54 -> make many noise realisations for this subject
    n_realisations=10,                # number of noise realisations for the replicate subject
    seed=1337,
    use_brainweb_lesions=True,        # if False, use custom spherical lesions from lesion_specs
    lesion_specs=None,                # list of dicts: {"centre_mm": (z,y,x), "radius_mm": r, "factor": 2.0}
    voxel_size_mm=(2.0, 2.0, 2.0),    # mMR-like voxel size (z,y,x) mm
):
    """
    Build PET–MR phantom volumes from BrainWeb, add PET lesions (GT), blur, add noise,
    and save NIfTI files.

    Outputs per subject s:
      - pet_gt_lesion_s{s}.nii.gz      (ground-truth PET with lesions)
      - mr_{mr_modality.lower()}_s{s}.nii.gz (MR without lesions)
      - ctlike_umap_s{s}.nii.gz        (attenuation map, CT-like)
      - pet_blur_noisy_{model}_s{s}_rXX.nii.gz  (blurred + noisy PET, multiple realisations for at least one subject)

    Mathematical notes:
      * Gaussian PSF: σ_axis = FWHM / (2√(2 ln 2)) scaled by (1 / voxel_size_axis).
      * Poisson noise: Y ~ Poisson(λ = poisson_scale * B), noisy = Y / poisson_scale (B = blurred PET).
      * Gaussian noise: N(0, σ²) added to B (σ = gaussian_sigma if given, else 0.05 * max(B)).

    Requirements: pip install brainweb nibabel scipy
    """
    import os, re
    from pathlib import Path
    import numpy as np
    import nibabel as nib
    try:
        import brainweb
    except ImportError as e:
        raise ImportError("Please install the BrainWeb helper: pip install brainweb") from e
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as e:
        raise ImportError("Please install SciPy for Gaussian blurring: pip install scipy") from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # -- helpers (kept local to honour single-function request) -----------------
    def _save_nifti_zyx(arr_zyx, vox_mm, path):
        # NIfTI uses (x,y,z); our arrays are (z,y,x) -> transpose
        data_xyz = np.transpose(arr_zyx, (2, 1, 0))
        affine = np.diag([vox_mm[2], vox_mm[1], vox_mm[0], 1.0]).astype(np.float32)
        nib.save(nib.Nifti1Image(data_xyz.astype(np.float32, copy=False), affine), str(path))

    def _sigma_voxels(fwhm_mm, vox_mm):
        c = fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return (c / vox_mm[0], c / vox_mm[1], c / vox_mm[2])  # (z,y,x)

    def _add_spherical_lesions(pet_zyx, specs, vox_mm):
        if not specs:
            return pet_zyx
        pet = pet_zyx.copy()
        Z, Y, X = np.indices(pet.shape, dtype=np.float32)
        for spec in specs:
            if "centre_mm" not in spec or "radius_mm" not in spec or "factor" not in spec:
                raise ValueError("Each lesion spec must include centre_mm=(z,y,x), radius_mm, factor.")
            cz, cy, cx = spec["centre_mm"]
            rz = spec["radius_mm"] / vox_mm[0]
            ry = spec["radius_mm"] / vox_mm[1]
            rx = spec["radius_mm"] / vox_mm[2]
            # convert centre (mm) to voxel indices
            cz_v = cz / vox_mm[0]
            cy_v = cy / vox_mm[1]
            cx_v = cx / vox_mm[2]
            mask = ((Z - cz_v)**2 / (rz**2) + (Y - cy_v)**2 / (ry**2) + (X - cx_v)**2 / (rx**2)) <= 1.0
            pet[mask] *= float(spec["factor"])
        return pet

    def _subject_id_from_path(p):
        m = re.search(r"subject_(\d+)\.bin\.gz$", str(p))
        return int(m.group(1)) if m else None

    # -- download/list BrainWeb subjects (cached under ~/.brainweb) ------------
    files = brainweb.get_files()  # downloads if needed; returns list of subject_XX.bin.gz paths
    if subject_ids is not None:
        subject_ids = set(int(s) for s in subject_ids)
        files = [f for f in files if _subject_id_from_path(f) in subject_ids]
        if not files:
            raise ValueError("No matching BrainWeb subjects found for given subject_ids.")

    # -- process each subject ---------------------------------------------------
    for f in files:
        sid = _subject_id_from_path(f)
        vol = brainweb.get_mmr_fromfile(
            f,
            petNoise=1, t1Noise=0.75, t2Noise=0.75,
            petSigma=1, t1Sigma=1, t2Sigma=1
        )  # returns dict with keys: 'PET','T1','T2','uMap' at ~2 mm, shape (127,344,344)
        pet = vol["PET"].astype(np.float32, copy=False)     # (z,y,x)
        mr = vol[mr_modality].astype(np.float32, copy=False)
        umap = vol["uMap"].astype(np.float32, copy=False)

        # PET ground-truth with lesions (MR remains lesion-free)
        if use_brainweb_lesions and lesion_specs is None:
            brainweb.seed(seed)  # determinism
            pet_gt = brainweb.add_lesions(pet.copy())
        else:
            pet_gt = _add_spherical_lesions(pet, lesion_specs or [], voxel_size_mm)

        # Save GT PET (+lesion), MR (no lesion), CT-like (uMap)
        _save_nifti_zyx(pet_gt, voxel_size_mm, out_dir / f"pet_gt_lesion_s{sid}.nii.gz")
        _save_nifti_zyx(mr,     voxel_size_mm, out_dir / f"mr_{mr_modality.lower()}_s{sid}.nii.gz")
        _save_nifti_zyx(umap,   voxel_size_mm, out_dir / f"ctlike_umap_s{sid}.nii.gz")

        # Blur PET with specified FWHM (per-axis σ in voxel units)
        sigma = _sigma_voxels(float(fwhm_mm), voxel_size_mm)
        pet_blurred = gaussian_filter(pet_gt, sigma=sigma, mode="constant", cval=0.0)

        # Noise: at least one subject gets multiple realisations
        R = n_realisations if (replicate_subject_id is not None and sid == int(replicate_subject_id)) else 1
        for r in range(R):
            if noise_model.lower() == "poisson":
                lam = np.clip(pet_blurred * float(poisson_scale), 0.0, None)
                noisy_counts = rng.poisson(lam).astype(np.float32, copy=False)
                pet_noisy = noisy_counts / float(poisson_scale)
            elif noise_model.lower() == "gaussian":
                sigma_abs = (float(gaussian_sigma) if gaussian_sigma is not None
                             else 0.05 * float(np.max(pet_blurred)))
                pet_noisy = pet_blurred + rng.normal(0.0, sigma_abs, size=pet_blurred.shape).astype(np.float32)
                pet_noisy = np.clip(pet_noisy, 0.0, None)
            else:
                raise ValueError("noise_model must be 'poisson' or 'gaussian'.")

            _save_nifti_zyx(pet_noisy, voxel_size_mm,
                            out_dir / f"pet_blur_noisy_{noise_model.lower()}_s{sid}_r{r:02d}.nii.gz")
