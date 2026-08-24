import pytest

from krl_studies.simulation.presets import (
    PRESET_NAMES,
    RECON_PSF_CONDITIONS,
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
