"""Pure helpers for recording measured simulation resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fwhm_from_profile(profile: np.ndarray, spacing_mm: float) -> float:
    """Return full width at half maximum with linearly interpolated crossings."""
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError(
            "profile must be a 1-D sequence of at least three samples and spacing_mm must be positive"
        )
    peak = float(values.max())
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("profile must contain a positive finite peak")
    half = peak / 2.0
    peak_index = int(np.argmax(values))

    i = peak_index
    while i > 0 and values[i] >= half:
        i -= 1
    if values[i] >= half:
        raise ValueError("profile does not cross half maximum on the left side")
    left = i + (half - values[i]) / (values[i + 1] - values[i])

    j = peak_index
    while j < values.size - 1 and values[j] >= half:
        j += 1
    if values[j] >= half:
        raise ValueError("profile does not cross half maximum on the right side")
    right = (j - 1) + (values[j - 1] - half) / (values[j - 1] - values[j])

    return float((right - left) * spacing_mm)


def write_resolution_calibration(
    records: dict[str, tuple[float, float, float]], path: str | Path
) -> Path:
    """Write sorted condition -> measured (x, y, z) FWHM records as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {key: list(records[key]) for key in sorted(records)}
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return output
