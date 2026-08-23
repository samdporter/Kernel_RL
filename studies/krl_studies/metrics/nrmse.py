"""Normalised root-mean-square error, matching krl.callbacks.NRMSECallback."""

from __future__ import annotations

import numpy as np


def nrmse(image: np.ndarray, ground_truth: np.ndarray) -> float:
    diff = np.asarray(image, dtype=np.float64) - np.asarray(ground_truth, dtype=np.float64)
    gt_max = float(np.max(ground_truth))
    if gt_max == 0.0:
        raise ValueError("ground truth max is zero; NRMSE undefined")
    return float(np.sqrt(np.mean(diff**2)) / gt_max)
