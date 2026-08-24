"""Fixed standard tumour set for simulated PET studies.

Tumour positions are expressed as fractional offsets from the volume centre so
the same layout transfers across subjects and voxel sizes. Positions were
chosen (fraction of dimension) to sit in plausible GM / WM / background zones
of BrainWeb brains; validation against tissue labels happens at dataset level.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_TUMOUR_DIAMETERS_MM = (8.0, 12.0, 16.0, 24.0)
DEFAULT_CONTRAST = 4.0

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


def place_tumours(
    pet: np.ndarray,
    specs: list[dict[str, Any]],
    contrast: float | None = None,
    voxel_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (pet with tumours, per-tumour boolean masks); input untouched."""
    out = pet.astype(np.float32, copy=True)
    masks = []
    vmm = np.asarray(voxel_mm, dtype=float)
    for spec in specs:
        c = spec.get("contrast", contrast)
        if c is None:
            raise ValueError("each spec or the call must provide contrast")
        radius_vox = float(spec["radius_mm"]) / vmm
        if np.ptp(radius_vox) < 1e-6:
            mask = sphere_mask(pet.shape, spec["centre_zyx"], float(radius_vox[0]))
        else:
            # anisotropic voxel (e.g. BrainWeb mMR 2.03×2.09×2.09): physical
            # sphere becomes ellipsoid in voxel index space.
            z = np.arange(pet.shape[0], dtype=np.float32)
            y = np.arange(pet.shape[1], dtype=np.float32)
            x = np.arange(pet.shape[2], dtype=np.float32)
            zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
            cz, cy, cx = spec["centre_zyx"]
            rz, ry, rx = radius_vox
            d2 = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
            mask = d2 <= 1.0
        out[mask] *= float(c)
        masks.append(mask)
    return out, masks
