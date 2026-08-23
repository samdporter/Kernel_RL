"""Patient cohort adapter.

Convention (see data/README.md): one directory per subject under
`data/patients/` containing `PET.nii.gz` and `T1.nii.gz`, optionally
`ROIs.nii.gz`. Ground truth does not exist for patients by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

_FILES = {"PET": "PET.nii.gz", "T1": "T1.nii.gz", "ROIs": "ROIs.nii.gz"}


def _load(path: Path) -> np.ndarray:
    return np.transpose(nib.load(str(path)).get_fdata().astype(np.float32), (2, 1, 0))


def discover_subjects(patients_root: Path) -> dict[str, dict[str, Path]]:
    """Map subject id -> present files; only subjects with PET+T1 qualify."""
    patients_root = Path(patients_root)
    found: dict[str, dict[str, Path]] = {}
    if not patients_root.exists():
        return found
    for d in sorted(patients_root.iterdir()):
        if not d.is_dir():
            continue
        present = {key: d / fname for key, fname in _FILES.items() if (d / fname).exists()}
        if {"PET", "T1"} <= present.keys():
            found[d.name] = present
    return found


@dataclass
class PatientDataset:
    subject_id: str
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.dir = self.root / self.subject_id
        if not (self.dir / _FILES["PET"]).exists() or not (self.dir / _FILES["T1"]).exists():
            raise FileNotFoundError(
                f"{self.dir} must contain PET.nii.gz and T1.nii.gz (see data/README.md)"
            )
        self.pet = _load(self.dir / _FILES["PET"])
        self.guidance = _load(self.dir / _FILES["T1"])
        roi_path = self.dir / _FILES["ROIs"]
        self.rois = _load(roi_path) if roi_path.exists() else None
        self.ground_truth = None
