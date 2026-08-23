"""Simple comparison baselines."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from krl_studies.methods.base import Iterate, Method

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


class PostSmoothingMethod(Method):
    """Gaussian post-reconstruction smoothing baseline (single step)."""

    name = "post_smoothing"

    def run(self, observed: Any, guidance: Any | None, params: dict[str, Any], n_iterations: int) -> Iterator[Iterate]:
        if n_iterations != 1:
            raise ValueError("post-smoothing is single-step; use n_iterations=1")
        voxel = tuple(float(v) for v in params.get("voxel_mm", (1.0, 1.0, 1.0)))
        fwhm = float(params["sigma_mm"])  # scenario-level smoothing width in mm
        sigma = tuple(fwhm * FWHM_TO_SIGMA / v for v in voxel)
        smoothed = gaussian_filter(np.asarray(observed, dtype=np.float64), sigma=sigma)
        yield Iterate(iteration=1, image=smoothed.astype(np.float32))
