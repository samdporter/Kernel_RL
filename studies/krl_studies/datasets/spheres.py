"""Synthetic spheres phantom dataset (files committed under data/spheres)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

_REQUIRED = {
    "ground_truth": "phant_orig.nii",
    "guidance": "phant_mri.nii",
    "reference_pet": "phant_pet.nii",
}

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass
class SphereDataset:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        missing = [fname for fname in _REQUIRED.values() if not (self.root / fname).exists()]
        if missing:
            raise FileNotFoundError(f"{self.root} is missing spheres files: {missing}")

    def _load(self, fname: str) -> np.ndarray:
        nii = nib.load(str(self.root / fname))
        return np.transpose(nii.get_fdata().astype(np.float32), (2, 1, 0))

    @property
    def ground_truth(self) -> np.ndarray:
        return self._load(_REQUIRED["ground_truth"])

    @property
    def guidance(self) -> np.ndarray:
        return self._load(_REQUIRED["guidance"])

    @property
    def reference_pet(self) -> np.ndarray:
        return self._load(_REQUIRED["reference_pet"])

    @property
    def voxel_mm(self) -> tuple[float, float, float]:
        nii = nib.load(str(self.root / _REQUIRED["ground_truth"]))
        sizes = nib.affines.voxel_sizes(nii.affine)
        return (float(sizes[2]), float(sizes[1]), float(sizes[0]))  # (z, y, x)


def quick_sim(
    gt: np.ndarray,
    fwhm_mm: float,
    counts: float,
    realisation: int,
    voxel_mm: tuple[float, float, float],
    seed: int = 1337,
) -> np.ndarray:
    """Deterministic image-space surrogate for the SIRF simulation (Plan 2).

    gt must be non-negative (emission image).

    Gaussian blur with the given FWHM followed by Poisson noise scaled so that
    `counts` is the total expected count level. Same (gt, fwhm, counts,
    realisation, seed) => identical output.
    """
    if counts <= 0:
        raise ValueError(f"counts must be positive, got {counts}")
    sigma_vox = [(fwhm_mm * FWHM_TO_SIGMA) / v for v in voxel_mm]
    blurred = gaussian_filter(gt.astype(np.float64), sigma=sigma_vox, mode="constant", cval=0.0)
    scale = counts / max(float(blurred.sum()), 1e-12)
    lam = np.clip(blurred * scale, 0.0, None)
    rng = np.random.default_rng(seed + int(realisation) * 7919)
    noisy = rng.poisson(lam).astype(np.float32) / scale
    return noisy
