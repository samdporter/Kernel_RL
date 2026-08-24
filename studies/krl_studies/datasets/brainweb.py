"""BrainWeb phantom preparation with tissue labels.

Requires ``brainweb`` pip package (lazy optional dependency):
``pip install brainweb nibabel scipy``. The module itself imports without
``brainweb``; :func:`prepare_subject` raises a clear ``ImportError`` if the
package is missing, allowing test suites to skip gracefully on macOS.

Outputs per subject (in ``out_dir``):
  - ``pet_gt.nii.gz``  ground-truth PET with optional tumours
  - ``mr_t1.nii.gz``   T1-weighted MR (lesion-free)
  - ``labels.nii.gz``  integer labels: 0 background, 1 CSF, 2 GM, 3 WM

``regions_from_labels`` converts the label volume into three boolean masks
[WM, GM, CSF/background] suitable for iY/GMM PVC comparators. The third mask
is CSF inside the brain (``labels==1``); background (0) is outside the brain
mask and left uncovered — callers that need a whole-volume partition can OR
the third mask with ``labels==0``.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

LABEL_BG = 0
LABEL_CSF = 1
LABEL_GM = 2
LABEL_WM = 3


def _subject_fname(subject_id: int | str) -> str:
    s = str(subject_id).strip()
    if s.endswith(".bin.gz"):
        base = s[: -len(".bin.gz")]
        if base.startswith("subject_"):
            suffix = base[len("subject_") :]
            try:
                num = int(suffix)
                return f"subject_{num:02d}.bin.gz"
            except ValueError:
                return s
        return s
    if s.startswith("subject_"):
        suffix = s[len("subject_") :]
        try:
            num = int(suffix)
            return f"subject_{num:02d}.bin.gz"
        except ValueError:
            return f"{s}.bin.gz"
    try:
        num = int(s)
        return f"subject_{num:02d}.bin.gz"
    except ValueError:
        return f"subject_{s}.bin.gz"


def _save_nifti(arr_zyx: np.ndarray, path: Path, voxel_mm_zyx: tuple[float, float, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.issubdtype(arr_zyx.dtype, np.integer):
        data_xyz = np.transpose(arr_zyx, (2, 1, 0))
        affine = np.diag([voxel_mm_zyx[2], voxel_mm_zyx[1], voxel_mm_zyx[0], 1.0]).astype(np.float32)
        nib.save(nib.Nifti1Image(data_xyz, affine), str(path))
    else:
        data_xyz = np.transpose(arr_zyx.astype(np.float32, copy=False), (2, 1, 0))
        affine = np.diag([voxel_mm_zyx[2], voxel_mm_zyx[1], voxel_mm_zyx[0], 1.0]).astype(np.float32)
        nib.save(nib.Nifti1Image(data_xyz, affine), str(path))


def prepare_subject(
    subject_id: int | str,
    out_dir: str | Path,
    tumour: bool = True,
) -> tuple[dict[str, Path], np.ndarray]:
    """Prepare one BrainWeb subject.

    Downloads/caches the BrainWeb volume via the ``brainweb`` pip package,
    builds PET ground truth (with optional tumours via
    :mod:`krl_studies.datasets.lesions`), T1 MR and discrete tissue labels,
    saves ``pet_gt.nii.gz``, ``mr_t1.nii.gz``, ``labels.nii.gz`` under
    ``out_dir`` and returns ``(paths, labels)``.

    Args:
        subject_id: BrainWeb subject identifier (e.g. ``4``, ``"04"``,
            ``"subject_04"`` or ``"subject_04.bin.gz"``). Must be present
            in :data:`brainweb.LINKS` (20 subjects).
        out_dir: Directory to write outputs (created if needed).
        tumour: If True, place the standard tumour set (four spheres,
            diameters 8/12/16/24 mm, contrast 4×) using
            :func:`krl_studies.datasets.lesions.default_tumour_specs`
            and :func:`krl_studies.datasets.lesions.place_tumours`.

    Returns:
        (paths, labels) where ``paths`` is ``{"pet_gt": Path, "mr_t1": Path,
        "labels": Path}`` and ``labels`` is the ``(z,y,x)`` integer label
        array (0 BG, 1 CSF, 2 GM, 3 WM) matching the saved PET/MR geometry.

    Requires:
        ``pip install brainweb nibabel scipy scikit-image requests tqdm``

    Notes:
        The function imports ``brainweb`` lazily; if the package is not
        installed it raises :class:`ImportError` with an actionable message
        so callers/tests can ``pytest.skip`` gracefully.
    """
    try:
        import brainweb  # noqa: WPS433 - lazy optional dep
    except ImportError as exc:
        raise ImportError(
            "brainweb package is required for BrainWeb preparation; "
            "install with: pip install brainweb nibabel scipy"
        ) from exc

    from krl_studies.datasets.lesions import default_tumour_specs, place_tumours

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = _subject_fname(subject_id)
    origin = brainweb.LINKS.get(fname)
    if origin is None:
        available = ", ".join(sorted(brainweb.LINKS.keys()))
        raise ValueError(f"Unknown BrainWeb subject {subject_id!r} ({fname}); available: {available}")

    brainweb_file = brainweb.get_file(fname, origin)

    vol = brainweb.get_mmr_fromfile(
        brainweb_file,
        petNoise=1,
        t1Noise=0.75,
        t2Noise=0.75,
        petSigma=1,
        t1Sigma=1,
        t2Sigma=1,
    )
    pet = np.asarray(vol["PET"], dtype=np.float32)
    t1 = np.asarray(vol["T1"], dtype=np.float32)

    try:
        voxel_mm = tuple(float(v) for v in np.asarray(vol["res"]).ravel().tolist())
        if len(voxel_mm) != 3:
            raise ValueError
    except Exception:
        voxel_mm = tuple(float(v) for v in brainweb.Res.mMR.tolist())

    # discrete labels from tissue probability maps (BG/CSF/GM/WM)
    probs = brainweb.get_label_probabilities(
        brainweb_file,
        labels=["background", "csf", "greyMatter", "whiteMatter"],
        outres="mMR",
        progress=False,
    )
    labels = np.argmax(probs, axis=0).astype(np.int16)
    # probs shape (4,127,344,344) -> labels 0..3 matches LABEL_* constants

    if tumour:
        specs = default_tumour_specs(shape=pet.shape, voxel_mm=voxel_mm)
        # Snap centres to nearest GM/WM voxel when the fractional position
        # lands outside the brain (large FOV padding in mMR geometry).
        gmwm = (labels == LABEL_GM) | (labels == LABEL_WM)
        if np.any(gmwm):
            coords = np.argwhere(gmwm)
            for spec in specs:
                cz, cy, cx = spec["centre_zyx"]
                iz, iy, ix = int(round(cz)), int(round(cy)), int(round(cx))
                iz = int(np.clip(iz, 0, labels.shape[0] - 1))
                iy = int(np.clip(iy, 0, labels.shape[1] - 1))
                ix = int(np.clip(ix, 0, labels.shape[2] - 1))
                if int(labels[iz, iy, ix]) in (LABEL_GM, LABEL_WM):
                    continue
                z0, z1 = max(0, iz - 1), min(labels.shape[0], iz + 2)
                y0, y1 = max(0, iy - 1), min(labels.shape[1], iy + 2)
                x0, x1 = max(0, ix - 1), min(labels.shape[2], ix + 2)
                neigh = labels[z0:z1, y0:y1, x0:x1]
                if np.any((neigh == LABEL_GM) | (neigh == LABEL_WM)):
                    continue
                d2 = np.sum((coords - np.array([cz, cy, cx])) ** 2, axis=1)
                nearest = coords[np.argmin(d2)]
                spec["centre_zyx"] = tuple(float(v) for v in nearest)
        pet_gt, _ = place_tumours(pet, specs, voxel_mm=voxel_mm)
    else:
        pet_gt = pet

    pet_path = out_dir / "pet_gt.nii.gz"
    mr_path = out_dir / "mr_t1.nii.gz"
    label_path = out_dir / "labels.nii.gz"

    _save_nifti(pet_gt, pet_path, voxel_mm)
    _save_nifti(t1, mr_path, voxel_mm)
    _save_nifti(labels, label_path, voxel_mm)

    paths = {"pet_gt": pet_path, "mr_t1": mr_path, "labels": label_path}
    return paths, labels


def regions_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    """Convert discrete labels to three PVC region masks.

    Args:
        labels: Integer array with values 0 BG, 1 CSF, 2 GM, 3 WM as
            produced by :func:`prepare_subject`. Other values (if any)
            are lumped into the CSF/background compartment.

    Returns:
        List ``[WM, GM, CSF_background]`` of boolean arrays of the same
        shape as ``labels``, disjoint and covering the brain mask
        (``labels != 0``). Background voxels (0) are not included in any
        mask when the input follows the standard encoding; callers that
        need a whole-volume partition can OR the third mask with
        ``labels == 0``.

        Example::

            wm, gm, csf = regions_from_labels(labels)
            brain = labels != 0
            assert np.all((wm | gm | csf)[brain])
            assert not np.any(wm & gm)
    """
    arr = np.asarray(labels)
    wm = arr == LABEL_WM
    gm = arr == LABEL_GM
    # CSF/background compartment: CSF voxels only (labels==1). For whole-
    # volume partitioning, background (0) would be included, but PVC
    # comparators use brain-only masks so we restrict to CSF.
    # Any unexpected label values are also mapped here for robustness.
    brain = arr != LABEL_BG
    rest = brain & ~(wm | gm)
    # If no brain voxels (synthetic test with only 0), fall back to complement
    # to ensure partition property for toy arrays.
    if not np.any(brain):
        rest = ~(wm | gm)
    return [wm.astype(bool, copy=False), gm.astype(bool, copy=False), rest.astype(bool, copy=False)]
