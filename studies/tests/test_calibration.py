import json

import numpy as np
import pytest

from krl_studies.simulation.calibration import fwhm_from_profile, write_resolution_calibration


def test_fwhm_from_profile_uses_half_maximum_crossing():
    profile = np.array([0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.0])
    assert fwhm_from_profile(profile, spacing_mm=1.0) == 2.0


def test_fwhm_from_profile_rejects_non_positive_peak():
    with pytest.raises(ValueError, match="positive finite peak"):
        fwhm_from_profile(np.zeros(3), spacing_mm=1.0)


def test_write_resolution_calibration_is_sorted_and_creates_parent(tmp_path):
    output = write_resolution_calibration(
        {"psf-matched": (4.5, 4.5, 6.4), "psf-none": (5.7, 5.7, 7.8)},
        tmp_path / "nested" / "resolution.json",
    )
    assert output.exists()
    assert list(json.loads(output.read_text())) == ["psf-matched", "psf-none"]


def test_fwhm_from_profile_interpolates_crossings():
    # Peak 4 at index 3, crossings between samples -> sub-sample width.
    profile = np.array([0.0, 0.4, 3.0, 4.0, 3.0, 0.4, 0.0])
    # Left crossing between idx1-2: 1 + (2-0.4)/(3-0.4) = 1.615...
    # Right crossing between idx4-5: 4 + (3-2)/(3-0.4) = 4.384...
    assert fwhm_from_profile(profile, spacing_mm=2.0) == pytest.approx((4.3846153846 - 1.6153846154) * 2.0)
