"""Physical guidance transforms for Task 4."""

import numpy as np
from scipy.ndimage import shift


def apply_guidance_condition(
    image: np.ndarray,
    condition: str,
    voxel_mm_zyx: tuple[float, float, float],
    *,
    order: int = 1,
) -> np.ndarray:
    """Apply exact or ±2/±5 mm rigid shifts; return a new float32 array."""
    offsets_mm = {
        "exact": (0.0, 0.0, 0.0),
        "shift_p2": (2.0, 2.0, 2.0),
        "shift_m2": (-2.0, -2.0, -2.0),
        "shift_p5": (5.0, 5.0, 5.0),
        "shift_m5": (-5.0, -5.0, -5.0),
    }
    if condition == "t2":
        raise ValueError("t2 guidance is loaded from the dataset, not shifted")
    if condition not in offsets_mm:
        raise ValueError(f"unknown guidance condition: {condition!r}")
    voxels = np.asarray(voxel_mm_zyx, dtype=float)
    if voxels.shape != (3,) or not np.all(np.isfinite(voxels)) or np.any(voxels <= 0):
        raise ValueError("voxel_mm_zyx must contain three positive finite values")
    if condition == "exact":
        return np.asarray(image, dtype=np.float32).copy()
    voxel_shift = tuple(mm / voxel for mm, voxel in zip(offsets_mm[condition], voxels))
    return shift(np.asarray(image, dtype=np.float32), voxel_shift, order=order, mode="nearest").astype(np.float32)
