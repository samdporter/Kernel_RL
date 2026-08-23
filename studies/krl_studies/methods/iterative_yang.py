"""Iterative Yang partial volume correction (Yang 1996; Erlandsson PMB 2012 review).

Piecewise-constant anatomical model: regional means are re-estimated each
iteration and the residual between measured and model-simulated PET is added
back within the brain mask.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from krl_studies.methods.base import Iterate, Method


class IterativeYangMethod(Method):
    name = "iy"

    def run(self, observed: Any, guidance: Any | None, params: dict[str, Any], n_iterations: int) -> Iterator[Iterate]:
        """Stream PVC iterates.

        Unlike the RL-family wrappers this operates on plain arrays and yields
        exactly n_iterations iterates.
        """
        region_masks = params.get("region_masks")
        if not region_masks:
            raise ValueError("iterative Yang requires region_masks (from segmentation/GT)")
        sigma = tuple(float(s) for s in params["psf_sigma_vox"])
        mask = params.get("brain_mask")
        if mask is None:
            mask = np.ones(np.asarray(observed).shape, dtype=bool)
        damping = float(params.get("damping", 1.0))

        y = np.asarray(observed, dtype=np.float64)
        masked_y = np.where(mask, y, 0.0)
        x = masked_y.copy()

        def step_image(current: np.ndarray) -> np.ndarray:
            s = np.zeros_like(current)
            for m in region_masks:
                s[m] = current[m].mean()
            return s

        for iteration in range(1, int(n_iterations) + 1):
            model = gaussian_filter(step_image(x), sigma=sigma)
            residual = masked_y - model
            x = x + damping * residual * mask
            yield Iterate(iteration=iteration, image=x.astype(np.float32))
