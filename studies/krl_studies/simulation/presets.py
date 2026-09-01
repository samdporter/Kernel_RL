"""Recon-PSF condition presets (Plan 3 truth/recon split).

Each condition distinguishes THREE widths in distinct metadata keys:

* ``forward_model_fwhm_xyz`` is the FWHM (mm, x/y/z) of the SINGLE Gaussian
  applied as a truth-side pre-blur before every forward projection. It is
  shared across all conditions so that the noisy prompts depend only on
  counts and seed -- the acquisition is identical, only the reconstruction
  model varies.

* ``recon_model_fwhm_xyz`` is the FWHM (mm, x/y/z) passed as
  ``resolution_fwhm=`` to ``make_acquisition_model`` when constructing the
  reconstruction acquisition model. It encodes the in-model Gaussian the
  recon AM uses to compensate for system blur. ``psf-none`` is None (no
  resolution modelling); the others carry the condition's specific kernel.

* ``target_residual_fwhm_xyz`` is the EXPECTED measured residual FWHM
  (mm, x/y/z) on the reconstructed input image after the condition's recon
  runs. These are the Vision 600 Hoffman measurements
  (Vision_resolution.docx) used as the calibration targets; they are NOT
  equal to the truth PSF or the recon PSF and must be populated from the
  calibration output (studies/scenarios/resolution_calibration.yaml).

All widths are PROVISIONAL until calibration signs them off on a realistic
grid (Task 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass

PRESET_NAMES = ("psf-none", "psf-undersized", "psf-matched")

# PROVISIONAL truth-side pre-blur (mm, x/y/z) shared by every condition. The
# ground truth is convolved with this Gaussian before forward projection so the
# prompts carry a single, known system PSF. Replaced by calibrated values from
# studies/scenarios/resolution_calibration.yaml after the Task 4.4 calibration
# run.
_TRUTH_PSF = (5.0, 5.0, 6.0)

# PROVISIONAL per-condition reconstruction-model PSFs (mm, x/y/z) passed as
# `resolution_fwhm=` to `make_acquisition_model`. None disables the in-model
# Gaussian. The unmatched "undersized" half-PSF intentionally mismodels the
# truth PSF to expose sensitivity to PSF miscalibration; "matched" mirrors the
# truth PSF exactly.
_RECON_PSF_NONE: tuple[float, float, float] | None = None
_RECON_PSF_UNDERSIZED = (2.5, 2.5, 3.0)
_RECON_PSF_MATCHED = (5.0, 5.0, 6.0)

# Vision 600 Hoffman measurements (Vision_resolution.docx). These are the
# EXPECTED measured residual FWHMs after reconstruction, not the truth or
# recon PSF -- kept as the calibration targets.
_TARGET_RESIDUAL_NONE = (5.7, 5.7, 7.8)
_TARGET_RESIDUAL_UNDERSIZED = (5.1, 5.1, 7.1)
_TARGET_RESIDUAL_MATCHED = (4.5, 4.5, 6.4)


@dataclass(frozen=True)
class ResolutionCondition:
    name: str
    target_residual_fwhm_xyz: tuple[float, float, float]
    forward_model_fwhm_xyz: tuple[float, float, float]
    recon_model_fwhm_xyz: tuple[float, float, float] | None


CONDITION_SPECS = {
    "psf-none": ResolutionCondition(
        "psf-none", _TARGET_RESIDUAL_NONE, _TRUTH_PSF, _RECON_PSF_NONE
    ),
    "psf-undersized": ResolutionCondition(
        "psf-undersized", _TARGET_RESIDUAL_UNDERSIZED, _TRUTH_PSF, _RECON_PSF_UNDERSIZED
    ),
    "psf-matched": ResolutionCondition(
        "psf-matched", _TARGET_RESIDUAL_MATCHED, _TRUTH_PSF, _RECON_PSF_MATCHED
    ),
}


def condition_spec(condition: str) -> ResolutionCondition:
    try:
        return CONDITION_SPECS[condition]
    except KeyError as exc:
        raise ValueError(f"unknown recon-PSF condition: {condition!r}") from exc


def resolution_for_condition(condition: str) -> tuple[float, float, float]:
    return condition_spec(condition).target_residual_fwhm_xyz


RECON_PSF_CONDITIONS = PRESET_NAMES
