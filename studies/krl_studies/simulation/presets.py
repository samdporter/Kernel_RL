"""Effective-residual-resolution presets anchored to the Vision 600 Hoffman
measurements (Vision_resolution.docx): values are the FWHM of the blur left ON
the reconstructed input image, on top of recon-side resolution modelling."""

from __future__ import annotations

from dataclasses import dataclass

PRESET_NAMES = ("psf-none", "psf-undersized", "psf-matched")

_PSF_MATCHED = (4.5, 4.5, 6.4)  # recon PSF modelled correctly -> clinical-like residual
_PSF_NONE = (5.7, 5.7, 7.8)  # no resolution modelling -> full effective blur


@dataclass(frozen=True)
class ResolutionCondition:
    name: str
    target_residual_fwhm_xyz: tuple[float, float, float]
    forward_model_fwhm_xyz: tuple[float, float, float]
    recon_model_fwhm_xyz: tuple[float, float, float] | None


def _undersized() -> tuple[float, float, float]:
    return tuple((a + b) / 2 for a, b in zip(_PSF_MATCHED, _PSF_NONE))


CONDITION_SPECS = {
    # Conditions are realised by pre-blurring the ground truth to the
    # condition's target residual before a CLEAN acquisition model: attaching
    # a Gaussian image-data processor to the reconstruction AM reduces central
    # recovery instead of sharpening it in the pinned SIRF build (adjoint of
    # the processor is not honoured inside OSMAPOSL; see
    # docs/reference/SIRF_API_NOTES.md). recon_model_fwhm_xyz therefore stays
    # None until a build supports adjoint-correct processors.
    "psf-none": ResolutionCondition(
        "psf-none", _PSF_NONE, _PSF_NONE, None
    ),
    "psf-undersized": ResolutionCondition(
        "psf-undersized", _undersized(), _undersized(), None
    ),
    "psf-matched": ResolutionCondition(
        "psf-matched", _PSF_MATCHED, _PSF_MATCHED, None
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
