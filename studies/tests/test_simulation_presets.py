import pytest

from krl_studies.simulation import condition_spec as public_condition_spec
from krl_studies.simulation.presets import (
    CONDITION_SPECS,
    PRESET_NAMES,
    RECON_PSF_CONDITIONS,
    condition_spec,
    resolution_for_condition,
)


def test_preset_names_match_spec():
    assert set(PRESET_NAMES) == {"psf-none", "psf-undersized", "psf-matched"}
    assert set(RECON_PSF_CONDITIONS) == set(PRESET_NAMES)


def test_matched_matches_vision_doc():
    assert resolution_for_condition("psf-matched") == pytest.approx((4.5, 4.5, 6.4))
    assert resolution_for_condition("psf-none") == pytest.approx((5.7, 5.7, 7.8))


def test_undersized_is_halfway_by_default():
    u = resolution_for_condition("psf-undersized")
    n = resolution_for_condition("psf-none")
    m = resolution_for_condition("psf-matched")
    assert u == pytest.approx(tuple((a + b) / 2 for a, b in zip(m, n)))


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        resolution_for_condition("psf-wat")


def test_condition_specs_separate_forward_and_reconstruction_models():
    # Conditions differ via the true blur applied in the forward model
    # (calibrated pre-blur route); recon-side processors are disabled in this
    # SIRF build (see docs/reference/SIRF_API_NOTES.md).
    assert condition_spec("psf-none").forward_model_fwhm_xyz == pytest.approx((5.7, 5.7, 7.8))
    assert condition_spec("psf-undersized").forward_model_fwhm_xyz == pytest.approx((5.1, 5.1, 7.1))
    assert condition_spec("psf-matched").forward_model_fwhm_xyz == pytest.approx((4.5, 4.5, 6.4))
    assert all(spec.recon_model_fwhm_xyz is None for spec in CONDITION_SPECS.values())
    for name, spec in CONDITION_SPECS.items():
        assert spec.forward_model_fwhm_xyz == spec.target_residual_fwhm_xyz


def test_condition_spec_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown recon-PSF condition"):
        condition_spec("psf-wat")


def test_condition_spec_is_available_from_simulation_package():
    assert public_condition_spec("psf-matched") == condition_spec("psf-matched")
