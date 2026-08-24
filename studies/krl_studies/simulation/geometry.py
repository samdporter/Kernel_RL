"""Physical voxel-grid utilities; all array axes are (z, y, x)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def _validate_voxels(values: tuple[float, float, float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"{name} must contain three positive finite values")
    return arr


def resample_array_zyx(
    array: np.ndarray,
    source_voxel_mm: tuple[float, float, float],
    target_voxel_mm: tuple[float, float, float],
    *,
    output_shape: tuple[int, int, int] | None = None,
    order: int = 1,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Resample a volume while preserving its physical extent."""
    source = np.asarray(array)
    if source.ndim != 3:
        raise ValueError(f"array must be 3-D (z,y,x), got {source.shape}")
    source_mm = _validate_voxels(source_voxel_mm, "source_voxel_mm")
    target_mm = _validate_voxels(target_voxel_mm, "target_voxel_mm")
    if order not in (0, 1, 2, 3):
        raise ValueError("order must be one of 0, 1, 2, or 3")
    if output_shape is None:
        target = tuple(
            max(1, int(round(n * source_size / target_size)))
            for n, source_size, target_size in zip(source.shape, source_mm, target_mm)
        )
    else:
        target = tuple(int(n) for n in output_shape)
        if len(target) != 3 or any(n < 1 for n in target):
            raise ValueError("output_shape must contain three positive integers")

    factors = tuple(target_size / source_size for target_size, source_size in zip(target, source.shape))
    # grid_mode aligns sample centres between grids; the default corner
    # alignment biases even-ratio resamples by half a voxel.
    result = zoom(
        source.astype(np.float32, copy=False),
        factors,
        order=order,
        mode="nearest",
        prefilter=order > 1,
        grid_mode=True,
    )
    if result.shape != target:
        result = zoom(
            result,
            tuple(target_size / source_size for target_size, source_size in zip(target, result.shape)),
            order=order,
            mode="nearest",
            grid_mode=True,
        )
    return result.astype(np.float32, copy=False), target


def _centred_window(length: int, window: int) -> slice:
    start = max(0, (length - window) // 2)
    return slice(start, min(length, start + window))


def resample_to_fov_zyx(
    array: np.ndarray,
    source_voxel_mm: tuple[float, float, float],
    fov_shape: tuple[int, int, int],
    fov_voxel_mm: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Rescale to the FOV voxel size, then embed centred into the FOV grid.

    Physical rescaling uses the voxel-size ratio only; volume content larger
    than the FOV is centre-cropped, smaller content is zero-padded.
    """
    scaled, natural = resample_array_zyx(array, source_voxel_mm, fov_voxel_mm)
    target = tuple(int(n) for n in fov_shape)
    if len(target) != 3 or any(n < 1 for n in target):
        raise ValueError("fov_shape must contain three positive integers")
    out = np.zeros(target, dtype=np.float32)
    source_index = []
    target_index = []
    for out_len, in_len in zip(target, scaled.shape):
        window = min(out_len, in_len)
        source_index.append(_centred_window(in_len, window))
        target_index.append(_centred_window(out_len, window))
    out[tuple(target_index)] = scaled[tuple(source_index)]
    return out, target


def resample_from_fov_zyx(
    array: np.ndarray,
    fov_voxel_mm: tuple[float, float, float],
    output_shape: tuple[int, int, int],
    output_voxel_mm: tuple[float, float, float],
    *,
    order: int = 1,
) -> np.ndarray:
    """Crop the centred output-extent window from an FOV grid and rescale."""
    fov_array = np.asarray(array)
    if fov_array.ndim != 3:
        raise ValueError(f"array must be 3-D (z,y,x), got {fov_array.shape}")
    target = tuple(int(n) for n in output_shape)
    if len(target) != 3 or any(n < 1 for n in target):
        raise ValueError("output_shape must contain three positive integers")
    fov_mm = _validate_voxels(fov_voxel_mm, "fov_voxel_mm")
    out_mm = _validate_voxels(output_voxel_mm, "output_voxel_mm")
    window = tuple(
        max(1, int(round(n_out * out_size / fov_size)))
        for n_out, out_size, fov_size in zip(target, out_mm, fov_mm)
    )
    source_index = [
        _centred_window(fov_len, min(fov_len, win)) for fov_len, win in zip(fov_array.shape, window)
    ]
    cropped = fov_array[tuple(source_index)]
    rescaled, _ = resample_array_zyx(cropped, fov_mm, out_mm, output_shape=target, order=order)
    return rescaled
