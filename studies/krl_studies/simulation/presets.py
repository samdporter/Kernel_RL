"""Effective-residual-resolution presets anchored to the Vision 600 Hoffman
measurements (Vision_resolution.docx): values are the FWHM of the blur left ON
the reconstructed input image, on top of recon-side resolution modelling."""

from __future__ import annotations

PRESET_NAMES = ("psf-none", "psf-undersized", "psf-matched")

_PSF_MATCHED = (4.5, 4.5, 6.4)  # recon PSF modelled correctly -> clinical-like residual
_PSF_NONE = (5.7, 5.7, 7.8)  # no resolution modelling -> full effective blur


def _undersized() -> tuple[float, float, float]:
    return tuple((a + b) / 2 for a, b in zip(_PSF_MATCHED, _PSF_NONE))


def resolution_for_condition(condition: str) -> tuple[float, float, float]:
    if condition == "psf-matched":
        return _PSF_MATCHED
    if condition == "psf-none":
        return _PSF_NONE
    if condition == "psf-undersized":
        return _undersized()
    raise ValueError(f"unknown recon-PSF condition: {condition!r}")


RECON_PSF_CONDITIONS = PRESET_NAMES
