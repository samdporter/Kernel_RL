"""Fixed standard tumour set for simulated PET studies.

Tumour positions are expressed as fractional offsets from the volume centre so
the same layout transfers across subjects and voxel sizes. Positions were
chosen (fraction of dimension) to sit in plausible GM / WM / background zones
of BrainWeb brains; validation against tissue labels happens at dataset level.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

DEFAULT_TUMOUR_DIAMETERS_MM = (8.0, 12.0, 16.0, 24.0)
DEFAULT_CONTRAST = 4.0
DEFAULT_MIN_SEPARATION_MM = 5.0
LAYOUT_SCHEMA_VERSION = "lesion_layout_v1"

# (dz, dy, dx) fraction-of-dimension offsets from centre, one per diameter
# (smallest tumour most central).
_POSITION_FRACTIONS = (
    (0.00, 0.00, -0.05),
    (0.12, 0.10, 0.10),
    (-0.10, -0.08, 0.12),
    (0.05, -0.15, -0.10),
)


def default_tumour_specs(
    shape: tuple[int, int, int],
    voxel_mm: tuple[float, float, float],
    diameters_mm: tuple[float, ...] = DEFAULT_TUMOUR_DIAMETERS_MM,
    contrast: float = DEFAULT_CONTRAST,
) -> list[dict[str, Any]]:
    if len(shape) != 3:
        raise ValueError("shape must be 3D (z, y, x)")
    if len(diameters_mm) > len(_POSITION_FRACTIONS):
        raise ValueError(
            f"got {len(diameters_mm)} diameters but only "
            f"{len(_POSITION_FRACTIONS)} fixed positions are defined"
        )
    centre = np.array(shape, dtype=float) / 2.0
    extent = np.array(shape, dtype=float) * np.array(voxel_mm, dtype=float)
    vmm = np.array(voxel_mm, dtype=float)
    specs = []
    for diameter, frac in zip(sorted(diameters_mm), _POSITION_FRACTIONS):
        offset_mm = np.array(frac, dtype=float) * extent
        offset_vox = offset_mm / vmm
        specs.append(
            {
                "centre_zyx": tuple(centre + offset_vox),
                "radius_mm": diameter / 2.0,
                "contrast": contrast,
            }
        )
    return specs


def sphere_mask(
    shape: tuple[int, int, int],
    centre_zyx: tuple[float, float, float],
    radius_vox: float,
) -> np.ndarray:
    z = np.arange(shape[0], dtype=np.float32)
    y = np.arange(shape[1], dtype=np.float32)
    x = np.arange(shape[2], dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    d2 = (zz - centre_zyx[0]) ** 2 + (yy - centre_zyx[1]) ** 2 + (xx - centre_zyx[2]) ** 2
    return d2 <= radius_vox**2


def tumour_mask(
    shape: tuple[int, int, int],
    centre_zyx: tuple[float, float, float],
    radius_mm: float,
    voxel_mm: tuple[float, float, float],
) -> np.ndarray:
    """Boolean mask of one physical sphere sampled on an (anisotropic) grid."""
    vmm = np.asarray(voxel_mm, dtype=float)
    radius_vox = float(radius_mm) / vmm
    if np.ptp(radius_vox) < 1e-6:
        return sphere_mask(shape, centre_zyx, float(radius_vox[0]))
    # anisotropic voxel (e.g. BrainWeb mMR 2.03×2.09×2.09): physical sphere
    # becomes ellipsoid in voxel index space.
    z = np.arange(shape[0], dtype=np.float32)
    y = np.arange(shape[1], dtype=np.float32)
    x = np.arange(shape[2], dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    cz, cy, cx = centre_zyx
    rz, ry, rx = radius_vox
    d2 = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    return d2 <= 1.0


def place_tumours(
    pet: np.ndarray,
    specs: list[dict[str, Any]],
    contrast: float | None = None,
    voxel_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (pet with tumours, per-tumour boolean masks); input untouched."""
    out = pet.astype(np.float32, copy=True)
    masks = []
    for spec in specs:
        c = spec.get("contrast", contrast)
        if c is None:
            raise ValueError("each spec or the call must provide contrast")
        mask = tumour_mask(pet.shape, spec["centre_zyx"], float(spec["radius_mm"]), voxel_mm)
        out[mask] *= float(c)
        masks.append(mask)
    return out, masks


def validate_lesion_layout(
    masks: list[np.ndarray],
    specs: list[dict[str, Any]],
    brain: np.ndarray,
    voxel_mm: tuple[float, float, float],
    min_separation_mm: float = DEFAULT_MIN_SEPARATION_MM,
) -> None:
    """Raise ``ValueError`` unless every lesion is a valid, separated in-brain mask."""
    if len(masks) != len(specs):
        raise ValueError(f"got {len(masks)} masks for {len(specs)} lesion specs")
    brain = np.asarray(brain, dtype=bool)
    vmm = np.asarray(voxel_mm, dtype=float)
    radii = [float(s["radius_mm"]) for s in specs]
    for i, mask in enumerate(masks):
        if mask.ndim != 3 or mask.shape != brain.shape:
            raise ValueError(f"lesion {i} mask shape {mask.shape} does not match volume {brain.shape}")
        if mask.dtype != bool:
            raise ValueError(f"lesion {i} mask dtype is {mask.dtype}, expected bool")
        if not np.any(mask):
            raise ValueError(f"lesion {i} ({2.0 * radii[i]:g} mm) is empty")
        if not np.all(brain[mask]):
            raise ValueError(
                f"lesion {i} ({2.0 * radii[i]:g} mm) is not fully contained in the brain mask"
            )
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.any(masks[i] & masks[j]):
                raise ValueError(f"lesions {i} and {j} overlap")
            ci = np.asarray(specs[i]["centre_zyx"], dtype=float) * vmm
            cj = np.asarray(specs[j]["centre_zyx"], dtype=float) * vmm
            gap = float(np.linalg.norm(ci - cj)) - (radii[i] + radii[j])
            if gap < min_separation_mm - 1e-6:
                raise ValueError(
                    f"lesions {i} and {j} are {gap:.2f} mm apart, "
                    f"less than the {min_separation_mm:g} mm minimum"
                )


def validate_loaded_lesion_masks(
    masks: list[np.ndarray],
    brain: np.ndarray,
    voxel_mm: tuple[float, float, float],
    min_separation_mm: float = DEFAULT_MIN_SEPARATION_MM,
) -> None:
    """Re-check loaded masks: non-empty, in-brain, non-overlapping and separated.

    Unlike :func:`validate_lesion_layout` this needs no lesion specs, so it can
    validate an arbitrary ``lesion_masks.npz``; separation is measured directly
    from the loaded voxel supports in physical units.
    """
    from scipy.ndimage import distance_transform_edt

    brain = np.asarray(brain, dtype=bool)
    vmm = np.asarray(voxel_mm, dtype=float)
    for i, mask in enumerate(masks):
        if mask.ndim != 3 or mask.shape != brain.shape:
            raise ValueError(f"lesion mask {i} shape {mask.shape} does not match volume {brain.shape}")
        if mask.dtype != bool:
            raise ValueError(f"lesion mask {i} dtype is {mask.dtype}, expected bool")
        if not np.any(mask):
            raise ValueError(f"lesion mask {i} is empty")
        if not np.all(brain[mask]):
            raise ValueError(f"lesion mask {i} is not fully contained in the brain mask")
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.any(masks[i] & masks[j]):
                raise ValueError(f"lesion masks {i} and {j} overlap")
    for i, mask in enumerate(masks):
        others = np.zeros(brain.shape, dtype=bool)
        for j, other in enumerate(masks):
            if j != i:
                others |= other
        if not np.any(others):
            continue
        distance_mm = distance_transform_edt(~others, sampling=vmm)
        gap = float(distance_mm[mask].min())
        if gap < min_separation_mm:
            raise ValueError(
                f"lesion mask {i} is only {gap:.2f} mm from another lesion, "
                f"less than the {min_separation_mm:g} mm minimum"
            )


def _mask_digest(masks: list[np.ndarray]) -> str:
    """Canonical digest of the realised boolean masks (shape + packed voxels)."""
    digest = hashlib.sha256()
    for mask in masks:
        arr = np.ascontiguousarray(mask, dtype=bool)
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(np.packbits(arr, axis=None).tobytes())
    return digest.hexdigest()


def mask_union_hash(masks: list[np.ndarray]) -> str:
    """Canonical digest of the union of the given boolean masks, order-independent."""
    digest = hashlib.sha256()
    if masks:
        union = np.zeros(np.asarray(masks[0]).shape, dtype=bool)
        for mask in masks:
            union |= np.asarray(mask, dtype=bool)
        union = np.ascontiguousarray(union, dtype=bool)
        digest.update(np.asarray(union.shape, dtype=np.int64).tobytes())
        digest.update(np.packbits(union, axis=None).tobytes())
    return digest.hexdigest()


def build_layout_metadata(
    subject_id: Any,
    specs: list[dict[str, Any]],
    masks: list[np.ndarray],
    voxel_mm: tuple[float, float, float],
    contrast: float | None,
    min_separation_mm: float | None,
) -> dict[str, Any]:
    """Serialisable layout record: centres, volumes, contrast and a stable hash."""
    vmm = np.asarray(voxel_mm, dtype=float)
    voxel_volume = float(np.prod(vmm))
    nominal = [4.0 / 3.0 * np.pi * float(s["radius_mm"]) ** 3 for s in specs]
    voxel_counts = [int(mask.sum()) for mask in masks]
    realised = [count * voxel_volume for count in voxel_counts]
    payload = {
        "schema": LAYOUT_SCHEMA_VERSION,
        "subject_id": str(subject_id),
        "diameters_mm": [2.0 * float(s["radius_mm"]) for s in specs],
        "contrast": float(contrast) if specs and contrast is not None else None,
        "centres_zyx": [[float(v) for v in s["centre_zyx"]] for s in specs],
        "voxel_mm": [float(v) for v in vmm],
        "realised_voxel_counts": voxel_counts,
        "mask_digest": _mask_digest(masks),
    }
    layout_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **payload,
        "nominal_volumes_mm3": nominal,
        "realised_volumes_mm3": realised,
        "min_separation_mm": float(min_separation_mm) if specs and min_separation_mm is not None else None,
        "layout_hash": layout_hash,
    }


def plan_tumour_layout(
    pet: np.ndarray,
    labels: np.ndarray,
    voxel_mm: tuple[float, float, float],
    subject_id: Any,
    diameters_mm: tuple[float, ...] = DEFAULT_TUMOUR_DIAMETERS_MM,
    contrast: float = DEFAULT_CONTRAST,
    min_separation_mm: float = DEFAULT_MIN_SEPARATION_MM,
) -> tuple[np.ndarray, list[dict[str, Any]], list[np.ndarray], dict[str, Any]]:
    """Place the standard four lesions fully inside one subject's brain.

    Centres are chosen deterministically as the valid voxel nearest each
    standard fractional position, where "valid" means the whole spherical
    lesion fits in ``labels > 0`` and keeps ``min_separation_mm`` from the
    lesions placed before it. Larger lesions are placed first. Returns
    ``(pet_with_lesions, specs, masks, layout_metadata)``.
    """
    from scipy.ndimage import distance_transform_edt

    pet_arr = np.asarray(pet, dtype=np.float32)
    label_arr = np.asarray(labels)
    if pet_arr.ndim != 3:
        raise ValueError("pet must be a 3D (z, y, x) array")
    if pet_arr.shape != label_arr.shape:
        raise ValueError(f"pet shape {pet_arr.shape} does not match labels shape {label_arr.shape}")
    vmm = np.asarray(voxel_mm, dtype=float)
    if vmm.shape != (3,) or not np.all(np.isfinite(vmm)) or np.any(vmm <= 0):
        raise ValueError("voxel_mm must contain three positive finite values")
    brain = label_arr > 0
    if not np.any(brain):
        raise ValueError(f"subject {subject_id!r}: no brain voxels available for lesion placement")

    ideal_specs = default_tumour_specs(
        pet_arr.shape, tuple(float(v) for v in vmm), diameters_mm=diameters_mm, contrast=contrast
    )
    inside_mm = distance_transform_edt(brain, sampling=vmm).astype(np.float32)
    occupied = np.zeros(pet_arr.shape, dtype=bool)
    zz, yy, xx = np.indices(pet_arr.shape, dtype=np.float32)
    centres: list[tuple[float, float, float] | None] = [None] * len(ideal_specs)

    for k in sorted(range(len(ideal_specs)), key=lambda i: (-ideal_specs[i]["radius_mm"], i)):
        spec = ideal_specs[k]
        radius = float(spec["radius_mm"])
        valid = inside_mm > (radius + 1e-6)
        if np.any(occupied):
            occupied_mm = distance_transform_edt(~occupied, sampling=vmm).astype(np.float32)
            valid &= occupied_mm > (radius + min_separation_mm)
        if not np.any(valid):
            raise ValueError(
                f"subject {subject_id!r}: could not fit a {2.0 * radius:g} mm lesion fully inside "
                f"the brain and ≥{min_separation_mm:g} mm from the other lesions"
            )
        ideal = np.asarray(spec["centre_zyx"], dtype=np.float64)
        cost = (
            ((zz - ideal[0]) * vmm[0]) ** 2
            + ((yy - ideal[1]) * vmm[1]) ** 2
            + ((xx - ideal[2]) * vmm[2]) ** 2
        )
        centre = np.unravel_index(int(np.argmin(np.where(valid, cost, np.inf))), pet_arr.shape)
        centres[k] = tuple(float(c) for c in centre)
        occupied |= tumour_mask(pet_arr.shape, centres[k], radius, vmm)

    specs = [
        {"centre_zyx": centres[k], "radius_mm": float(s["radius_mm"]), "contrast": float(contrast)}
        for k, s in enumerate(ideal_specs)
    ]
    pet_gt, masks = place_tumours(pet_arr, specs, voxel_mm=tuple(float(v) for v in vmm))
    validate_lesion_layout(masks, specs, brain, vmm, min_separation_mm)
    metadata = build_layout_metadata(subject_id, specs, masks, vmm, contrast, min_separation_mm)
    return pet_gt, specs, masks, metadata
