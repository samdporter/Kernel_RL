"""Contrast recovery and noise variability metrics (NEMA-style definitions)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _voi_means(image: np.ndarray, vois: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([float(np.mean(image[m])) for m in vois], dtype=np.float64)


def crc_percent(
    lesion_mask: np.ndarray,
    image: np.ndarray,
    ground_truth: np.ndarray,
    background_vois: Sequence[np.ndarray],
) -> float:
    """CRC = (C_meas/B_meas - 1) / (C_true/B_true - 1) * 100.

    Background levels are measured as the mean over the background VOIs in the
    respective image, so bias common to lesion and background cancels.
    """
    b_meas = float(np.mean(_voi_means(image, background_vois)))
    b_true = float(np.mean(_voi_means(ground_truth, background_vois)))
    c_meas = float(np.mean(image[lesion_mask]))
    c_true = float(np.mean(ground_truth[lesion_mask]))
    denom = (c_true / b_true) - 1.0
    if denom == 0.0 or b_meas == 0.0:
        raise ValueError("degenerate CRC definition (zero contrast or background)")
    return float(100.0 * ((c_meas / b_meas) - 1.0) / denom)


def background_variability(
    image: np.ndarray, vois: Sequence[np.ndarray]
) -> float:
    """Percent coefficient of variation of VOI means (relative to their mean)."""
    means = _voi_means(image, vois)
    overall = float(np.mean(means))
    if overall == 0.0:
        raise ValueError("background mean is zero")
    return float(100.0 * np.std(means, ddof=1) / overall) if len(means) > 1 else 0.0
