"""BrainWeb phantom preparation with tissue labels.

Requires ``brainweb`` pip package (lazy optional dependency):
``pip install brainweb nibabel scipy``. The module itself imports without
``brainweb``; :func:`prepare_subject` raises a clear ``ImportError`` if the
package is missing, allowing test suites to skip gracefully on macOS.

Outputs per subject (in ``out_dir``):
  - ``pet_gt.nii.gz``  ground-truth PET with optional tumours
  - ``pet_tumour_free.nii.gz``  tumour-free PET baseline for QC
  - ``mr_t1_absent.nii.gz``  raw BrainWeb T1 (lesion-free)
  - ``mr_t1_present.nii.gz``  synthetic T1: union of lesion masks scaled by 4×
  - ``mr_t2.nii.gz``   T2-weighted MR (lesion-free)
  - ``labels.nii.gz``  integer labels: 0 background, 1 CSF, 2 GM, 3 WM
  - ``mu_map.nii.gz``  attenuation map (mu in 1/cm) on the same grid as PET
  - ``lesion_masks.npz``  compressed boolean tumour masks (n, z, y, x)
  - ``lesion_diameters_mm.json``  list of tumour diameters in mm
  - ``lesion_layout.json``  centres, volumes, contrast and layout hash
  - ``preparation.json``  texture, seed, cache namespace and guidance metadata

``regions_from_labels`` converts the label volume into three boolean masks
[WM, GM, CSF/background] suitable for iY/GMM PVC comparators. The third mask
is CSF inside the brain (``labels==1``); background (0) is outside the brain
mask and left uncovered — callers that need a whole-volume partition can OR
the third mask with ``labels==0``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

LABEL_BG = 0
LABEL_CSF = 1
LABEL_GM = 2
LABEL_WM = 3

# BrainWeb structural-texture parameters. They are baked into the processed
# ``*.npz`` phantom, so they namespace the campaign cache; the acquisition
# Poisson seed is applied later by the runner and is deliberately excluded.
DEFAULT_TEXTURE = {
    "petNoise": 1.0,
    "t1Noise": 0.75,
    "t2Noise": 0.75,
    "petSigma": 1.0,
    "t1Sigma": 1.0,
    "t2Sigma": 1.0,
}
_TEXTURE_KEYS = tuple(DEFAULT_TEXTURE)
DEFAULT_PREP_SEED = 1337

# Guidance T1 lesion-state assets. ``absent`` is the raw BrainWeb T1; ``present``
# is that T1 with the union of the planned lesion masks scaled by the injection
# multiplier (and nothing else changed).
GUIDANCE_T1_ABSENT = "mr_t1_absent.nii.gz"
GUIDANCE_T1_PRESENT = "mr_t1_present.nii.gz"
GUIDANCE_T1_INJECTION_MULTIPLIER = 4.0
GUIDANCE_T1_INJECTION_RULE = (
    "multiply T1 by injection_multiplier inside the union of the planned lesion masks; "
    "all other voxels unchanged"
)


def subject_inventory() -> tuple[str, ...]:
    """Freeze the BrainWeb subject filenames advertised by the installed package.

    Values are the canonical ``subject_XX.bin.gz`` filenames accepted by
    :func:`prepare_subject` / :func:`prepare_subjects`, so the inventory
    round-trips straight back into preparation.
    """
    import brainweb

    return tuple(sorted(brainweb.LINKS))


def inventory_digest(inventory: list[str] | tuple[str, ...]) -> str:
    """Stable digest of a frozen inventory list."""
    return hashlib.sha256("\n".join(inventory).encode("utf-8")).hexdigest()


def normalize_subject_id(subject_id: int | str) -> str:
    """Canonical ``subject_XX.bin.gz`` filename for any accepted spelling."""
    return _subject_fname(subject_id)


def _subject_number(subject_id: int | str) -> int:
    fname = _subject_fname(subject_id)
    base = fname[: -len(".bin.gz")] if fname.endswith(".bin.gz") else fname
    if base.startswith("subject_"):
        try:
            return int(base[len("subject_") :])
        except ValueError as exc:
            raise ValueError(f"cannot parse BrainWeb subject id from {subject_id!r}") from exc
    raise ValueError(f"cannot parse BrainWeb subject id from {subject_id!r}")


def texture_namespace(texture: dict, seed: int = DEFAULT_PREP_SEED) -> str:
    """Stable short digest of the structural-texture parameters and prep seed."""
    payload = {key: float(texture[key]) for key in _TEXTURE_KEYS}
    payload["seed"] = int(seed)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _namespaced_raw_file(brainweb_file: str | Path, namespace: str) -> Path:
    """Link the raw archive into its parameter namespace so the derived cache
    filename changes whenever the texture parameters change."""
    raw = Path(brainweb_file)
    ns_dir = raw.parent / "campaign_cache" / namespace
    ns_dir.mkdir(parents=True, exist_ok=True)
    namespaced = ns_dir / raw.name
    if not namespaced.exists():
        try:
            namespaced.symlink_to(raw)
        except OSError:
            shutil.copy2(raw, namespaced)
    return namespaced


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


def build_guidance_t1_present(
    t1: np.ndarray,
    masks: list[np.ndarray],
    multiplier: float = GUIDANCE_T1_INJECTION_MULTIPLIER,
) -> np.ndarray:
    """Return a copy of ``t1`` with the lesion-mask union scaled by ``multiplier``.

    The input T1 is never modified; voxels outside the union are unchanged.
    """
    present = np.asarray(t1, dtype=np.float32).copy()
    if masks:
        union = np.zeros(present.shape, dtype=bool)
        for mask in masks:
            union |= np.asarray(mask, dtype=bool)
        present[union] *= float(multiplier)
    return present


def prepare_subject(
    subject_id: int | str,
    out_dir: str | Path,
    tumour: bool = True,
    *,
    seed: int = DEFAULT_PREP_SEED,
    texture: dict | None = None,
) -> tuple[dict[str, Path], np.ndarray]:
    """Prepare one BrainWeb subject.

    Downloads/caches the BrainWeb volume via the ``brainweb`` pip package,
    builds PET ground truth (with an optional subject-specific four-lesion
    layout via :mod:`krl_studies.datasets.lesions`), T1/T2 MR and discrete
    tissue labels, and writes the assets documented at module level.

    Args:
        subject_id: BrainWeb subject identifier (e.g. ``4``, ``"04"``,
            ``"subject_04"`` or ``"subject_04.bin.gz"``). Must be present in
            :func:`subject_inventory` (the installed ``brainweb.LINKS``).
        out_dir: Directory to write outputs (created if needed).
        tumour: If True, place the standard tumour set (diameters 8/12/16/24 mm,
            contrast 4× local baseline PET) fully inside the brain.
        seed: Seed for BrainWeb's structural-texture noise. Recorded separately
            from the acquisition Poisson seed, which the runner owns.
        texture: Structural-texture parameters; defaults to
            :data:`DEFAULT_TEXTURE`. They namespace the processed cache.

    Returns:
        ``(paths, labels)`` where ``labels`` is the ``(z,y,x)`` integer label
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

    from krl_studies.datasets.lesions import (
        build_layout_metadata,
        mask_union_hash,
        plan_tumour_layout,
    )

    texture = dict(DEFAULT_TEXTURE if texture is None else texture)
    missing = [key for key in _TEXTURE_KEYS if key not in texture]
    if missing:
        raise ValueError(f"texture is missing BrainWeb parameters: {missing}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = subject_inventory()
    fname = _subject_fname(subject_id)
    if fname not in inventory:
        raise ValueError(
            f"Unknown BrainWeb subject {subject_id!r} ({fname}); available: {', '.join(inventory)}"
        )
    origin = brainweb.LINKS[fname]

    brainweb_file = brainweb.get_file(fname, origin)
    namespace = texture_namespace(texture, seed)
    namespaced_file = _namespaced_raw_file(brainweb_file, namespace)

    # brainweb.noise() draws from the global numpy RNG, so seed it for
    # reproducible phantoms. The cache is namespaced by the texture
    # parameters, so changing them can never silently reuse an old phantom.
    brainweb.seed(seed)
    vol = brainweb.get_mmr_fromfile(str(namespaced_file), **texture)
    for key in _TEXTURE_KEYS:
        if not np.isclose(float(vol[key]), float(texture[key]), atol=1e-6):
            raise RuntimeError(
                f"BrainWeb cache {namespaced_file} was built with {key}={float(vol[key])!r} "
                f"but {float(texture[key])!r} was requested"
            )

    pet = np.asarray(vol["PET"], dtype=np.float32)
    t1 = np.asarray(vol["T1"], dtype=np.float32)
    t2 = np.asarray(vol["T2"], dtype=np.float32)
    u_map = np.asarray(vol["uMap"], dtype=np.float32)

    try:
        voxel_mm = tuple(float(v) for v in np.asarray(vol["res"]).ravel().tolist())
        if len(voxel_mm) != 3:
            raise ValueError
    except Exception:
        voxel_mm = tuple(float(v) for v in brainweb.Res.mMR.tolist())

    # discrete labels from tissue probability maps (BG/CSF/GM/WM)
    probs = brainweb.get_label_probabilities(
        namespaced_file,
        labels=["background", "csf", "greyMatter", "whiteMatter"],
        outres="mMR",
        progress=False,
    )
    labels = np.argmax(probs, axis=0).astype(np.int16)
    # probs shape (4,127,344,344) -> labels 0..3 matches LABEL_* constants

    if tumour:
        pet_gt, specs, lesion_masks, layout = plan_tumour_layout(
            pet, labels, voxel_mm, subject_id
        )
        diameters = [2.0 * float(spec["radius_mm"]) for spec in specs]
    else:
        pet_gt = pet.copy()
        specs = []
        lesion_masks = []
        diameters = []
        layout = build_layout_metadata(subject_id, [], [], voxel_mm, None, None)

    mask_array = (
        np.asarray(lesion_masks, dtype=bool)
        if lesion_masks
        else np.empty((0, *pet.shape), dtype=bool)
    )
    np.savez_compressed(out_dir / "lesion_masks.npz", masks=mask_array)
    (out_dir / "lesion_diameters_mm.json").write_text(json.dumps(diameters))
    (out_dir / "lesion_layout.json").write_text(json.dumps(layout, indent=2))

    # Two immutable T1 guidance variants: the raw BrainWeb T1 and the synthetic
    # lesion-state T1 built from the union of the planned lesion masks.
    union_hash = mask_union_hash(lesion_masks)
    t1_present = build_guidance_t1_present(t1, lesion_masks)

    preparation = {
        "subject_id": str(subject_id),
        "subject_file": fname,
        "inventory": list(inventory),
        "inventory_count": len(inventory),
        "inventory_digest": inventory_digest(inventory),
        "texture": {key: float(texture[key]) for key in _TEXTURE_KEYS},
        "texture_namespace": namespace,
        "structural_seed": int(seed),
        "layout_hash": layout["layout_hash"],
        "guidance": {
            "lesion_states": ["absent", "present"],
            "injection_multiplier": float(GUIDANCE_T1_INJECTION_MULTIPLIER),
            "injection_rule": GUIDANCE_T1_INJECTION_RULE,
            "mask_union_hash": union_hash,
            "t1_variants": {
                "absent": GUIDANCE_T1_ABSENT,
                "present": GUIDANCE_T1_PRESENT,
            },
        },
    }
    (out_dir / "preparation.json").write_text(json.dumps(preparation, indent=2))

    pet_path = out_dir / "pet_gt.nii.gz"
    t1_absent_path = out_dir / GUIDANCE_T1_ABSENT
    t1_present_path = out_dir / GUIDANCE_T1_PRESENT
    mr_t2_path = out_dir / "mr_t2.nii.gz"
    label_path = out_dir / "labels.nii.gz"
    mu_path = out_dir / "mu_map.nii.gz"
    pet_clean_path = out_dir / "pet_tumour_free.nii.gz"
    lesion_masks_path = out_dir / "lesion_masks.npz"
    lesion_diameters_path = out_dir / "lesion_diameters_mm.json"
    lesion_layout_path = out_dir / "lesion_layout.json"
    preparation_path = out_dir / "preparation.json"

    _save_nifti(pet_gt, pet_path, voxel_mm)
    _save_nifti(pet, pet_clean_path, voxel_mm)
    _save_nifti(t1, t1_absent_path, voxel_mm)
    _save_nifti(t1_present, t1_present_path, voxel_mm)
    _save_nifti(t2, mr_t2_path, voxel_mm)
    _save_nifti(labels, label_path, voxel_mm)
    _save_nifti(u_map, mu_path, voxel_mm)

    paths = {
        "pet_gt": pet_path,
        "pet_tumour_free": pet_clean_path,
        "mr_t1_absent": t1_absent_path,
        "mr_t1_present": t1_present_path,
        "mr_t2": mr_t2_path,
        "labels": label_path,
        "mu_map": mu_path,
        "lesion_masks": lesion_masks_path,
        "lesion_diameters_mm": lesion_diameters_path,
        "lesion_layout": lesion_layout_path,
        "preparation": preparation_path,
    }
    return paths, labels


def prepare_subjects(
    subject_ids: list[int | str],
    out_root: str | Path,
    tumour: bool = True,
    *,
    seed: int = DEFAULT_PREP_SEED,
    texture: dict | None = None,
) -> dict[str, Any]:
    """Prepare a batch of subjects, reporting incomplete downloads explicitly.

    ``subject_ids`` may be ints, numeric strings, bare ``subject_XX`` names or
    ``subject_XX.bin.gz`` filenames (including :func:`subject_inventory`
    output). Returns ``{"inventory": [...], "prepared": {filename: paths},
    "incomplete": {filename: error}}``; a failed download is recorded in
    ``incomplete`` rather than dropped.
    """
    out_root = Path(out_root)
    inventory = subject_inventory()
    prepared: dict[str, dict[str, str]] = {}
    incomplete: dict[str, str] = {}
    for subject_id in subject_ids:
        canonical = _subject_fname(subject_id)
        try:
            number = _subject_number(subject_id)
            out_dir = out_root / f"subject_{number:02d}"
            paths, _ = prepare_subject(subject_id, out_dir, tumour=tumour, seed=seed, texture=texture)
        except Exception as exc:  # noqa: BLE001 - report, never silently exclude
            incomplete[canonical] = f"{type(exc).__name__}: {exc}"
        else:
            prepared[canonical] = {key: str(value) for key, value in paths.items()}
    return {"inventory": list(inventory), "prepared": prepared, "incomplete": incomplete}


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


class BrainWebDataset:
    """Adapter to load a prepared BrainWeb subject directory."""

    _FILES = {
        "PET": "pet_gt.nii.gz",
        "T1_ABSENT": GUIDANCE_T1_ABSENT,
        "T1_PRESENT": GUIDANCE_T1_PRESENT,
        "T2": "mr_t2.nii.gz",
        "LABELS": "labels.nii.gz",
        "MU_MAP": "mu_map.nii.gz",
        "LESION_MASKS": "lesion_masks.npz",
        "LESION_DIAMETERS": "lesion_diameters_mm.json",
        "LESION_LAYOUT": "lesion_layout.json",
        "PREPARATION": "preparation.json",
    }
    _GEOMETRY_KEYS = ("PET", "T1_ABSENT", "T1_PRESENT", "T2", "LABELS", "MU_MAP")
    _ATTRS = {
        "PET": "pet_gt",
        "T1_ABSENT": "mr_t1_absent",
        "T1_PRESENT": "mr_t1_present",
        "T2": "mr_t2",
        "LABELS": "labels",
        "MU_MAP": "mu_map",
    }

    def __init__(self, root: str | Path, subject_id: int | str):
        self.dir = Path(root) / f"subject_{int(subject_id):02d}"
        self.subject_id = int(subject_id)

        pet_path = self.dir / self._FILES["PET"]
        if not pet_path.exists():
            raise FileNotFoundError(
                f"{pet_path} not found. See data/README.md for BrainWeb preparation."
            )

        self._affines: dict[str, np.ndarray] = {}
        self.pet_gt = self._load_asset("PET")
        self.mr_t1_absent = self._load_asset("T1_ABSENT")
        self.mr_t1_present = self._load_asset("T1_PRESENT")
        self.mr_t2 = self._load_asset("T2")
        self.labels = self._load_asset("LABELS")
        self.mu_map = self._load_asset("MU_MAP")
        # Backwards-compatible default anatomy is the raw (lesion-absent) T1.
        self.mr_t1 = self.mr_t1_absent

        self._pet_affine = self._affines["PET"]
        self._pet_voxel_mm = tuple(float(v) for v in nib.affines.voxel_sizes(self._pet_affine))
        # Ensure z,y,x order
        self._pet_voxel_mm_zyx = (self._pet_voxel_mm[2], self._pet_voxel_mm[1], self._pet_voxel_mm[0])

        self.lesion_masks = self._load_lesion_masks()
        self.lesion_diameters_mm = self._load_lesion_diameters()
        self.lesion_layout = self._load_lesion_layout()
        self.preparation = self._load_preparation()
        self._validate_geometry()

    def _load_asset(self, key: str) -> np.ndarray:
        path = self.dir / self._FILES[key]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. See data/README.md for BrainWeb preparation."
            )
        nii = nib.load(str(path))
        self._affines[key] = np.asarray(nii.affine, dtype=float)
        return np.transpose(nii.get_fdata().astype(np.float32), (2, 1, 0))

    def _validate_geometry(self) -> None:
        ref_shape = self.pet_gt.shape
        ref_affine = self._affines["PET"]
        arrays = {key: getattr(self, self._ATTRS[key]) for key in self._GEOMETRY_KEYS}
        for key in self._GEOMETRY_KEYS:
            arr = arrays[key]
            if arr.shape != ref_shape:
                raise ValueError(
                    f"BrainWeb subject {self.subject_id}: {key} shape {arr.shape} "
                    f"does not match PET shape {ref_shape}"
                )
            if not np.allclose(self._affines[key], ref_affine, rtol=0.0, atol=1e-3):
                raise ValueError(
                    f"BrainWeb subject {self.subject_id}: {key} affine does not match PET affine; "
                    "assets are not on a common physical grid"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"BrainWeb subject {self.subject_id}: {key} contains non-finite values"
                )
        brain = self.labels > 0
        if not np.any(brain):
            raise ValueError(f"BrainWeb subject {self.subject_id}: labels contain no brain voxels")
        if float(self.pet_gt.max()) <= 0:
            raise ValueError(f"BrainWeb subject {self.subject_id}: PET has no positive signal")
        if not np.any(brain & (self.pet_gt > 0)):
            raise ValueError(
                f"BrainWeb subject {self.subject_id}: PET does not overlap brain labels"
            )
        if not np.any(self.mu_map > 0):
            raise ValueError(f"BrainWeb subject {self.subject_id}: mu_map has no positive values")
        if len(self.lesion_masks) != len(self.lesion_diameters_mm):
            raise ValueError(
                f"BrainWeb subject {self.subject_id}: {len(self.lesion_masks)} lesion masks "
                f"but {len(self.lesion_diameters_mm)} diameters"
            )
        from krl_studies.datasets.lesions import mask_union_hash, validate_loaded_lesion_masks

        validate_loaded_lesion_masks(self.lesion_masks, brain, self._pet_voxel_mm_zyx)

        # The injected-T1 contract is fixed: exactly the expected union-overlay
        # rule at the expected multiplier, and a union hash that matches the
        # loaded masks (never trust arbitrary preparation metadata).
        guidance = self.preparation.get("guidance", {})
        recorded_multiplier = guidance.get("injection_multiplier")
        if recorded_multiplier is None or float(recorded_multiplier) != GUIDANCE_T1_INJECTION_MULTIPLIER:
            raise ValueError(
                f"BrainWeb subject {self.subject_id}: injection_multiplier must be "
                f"{GUIDANCE_T1_INJECTION_MULTIPLIER}, got {recorded_multiplier!r}"
            )
        if guidance.get("injection_rule") != GUIDANCE_T1_INJECTION_RULE:
            raise ValueError(
                f"BrainWeb subject {self.subject_id}: injection_rule does not match "
                "the expected union-overlay rule"
            )
        computed_union_hash = mask_union_hash(self.lesion_masks)
        if guidance.get("mask_union_hash") != computed_union_hash:
            raise ValueError(
                f"BrainWeb subject {self.subject_id}: mask_union_hash does not match "
                "the union of the loaded lesion masks"
            )

        expected_present = build_guidance_t1_present(
            self.mr_t1_absent, self.lesion_masks, GUIDANCE_T1_INJECTION_MULTIPLIER
        )
        if not np.array_equal(self.mr_t1_present, expected_present):
            raise ValueError(
                f"BrainWeb subject {self.subject_id}: mr_t1_present does not equal "
                "mr_t1_absent with the lesion-mask union scaled by the injection multiplier"
            )

    @property
    def ground_truth(self) -> np.ndarray:
        return self.pet_gt

    @property
    def guidance(self) -> np.ndarray:
        return self.mr_t1_absent

    @property
    def t2(self) -> np.ndarray | None:
        return self.mr_t2

    @property
    def voxel_mm(self) -> tuple[float, float, float]:
        return self._pet_voxel_mm_zyx

    @property
    def affine(self) -> np.ndarray | None:
        return self._pet_affine

    @property
    def guidance_mask_union_hash(self) -> str:
        return str(self.preparation.get("guidance", {}).get("mask_union_hash", ""))

    def guidance_for(self, modality: str, lesion_state: str = "absent") -> np.ndarray:
        """Return the requested guidance volume.

        T1 selects the lesion-state variant; the mu-map case returns a separate
        copy so the attenuation array (``self.mu_map``) can never be mutated by
        guidance conditioning.
        """
        if modality == "t1":
            if lesion_state == "absent":
                return self.mr_t1_absent
            if lesion_state == "present":
                return self.mr_t1_present
            raise ValueError(f"unknown guidance lesion state: {lesion_state!r}")
        if modality == "t2":
            return self.mr_t2
        if modality == "umap":
            return self.mu_map.copy()
        raise ValueError(f"unknown guidance modality: {modality!r}")

    def _load_lesion_masks(self) -> list[np.ndarray]:
        path = self.dir / self._FILES["LESION_MASKS"]
        if not path.exists():
            return []
        data = np.asarray(np.load(path)["masks"])
        if data.size == 0:
            return []
        # Validate the raw values BEFORE bool coercion: NaN is truthy, so a
        # permissive cast would silently turn malformed data into masks.
        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"BrainWeb subject {self.subject_id}: lesion_masks.npz contains non-finite values"
            )
        if not np.all((data == 0) | (data == 1)):
            raise ValueError(
                f"BrainWeb subject {self.subject_id}: lesion_masks.npz must contain 0/1 or boolean values"
            )
        if data.ndim == 4:
            return [data[i].astype(bool, copy=False) for i in range(data.shape[0])]
        if data.ndim == 3:
            return [data.astype(bool, copy=False)]
        raise ValueError(
            f"BrainWeb subject {self.subject_id}: lesion_masks has unsupported shape {data.shape}"
        )

    def _load_lesion_diameters(self) -> list[float]:
        path = self.dir / self._FILES["LESION_DIAMETERS"]
        if path.exists():
            return json.loads(path.read_text())
        return []

    def _load_lesion_layout(self) -> dict:
        path = self.dir / self._FILES["LESION_LAYOUT"]
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _load_preparation(self) -> dict:
        path = self.dir / self._FILES["PREPARATION"]
        if path.exists():
            return json.loads(path.read_text())
        return {}
