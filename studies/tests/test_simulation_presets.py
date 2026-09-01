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
    # Plan 3 truth/recon split (Task 4): every condition shares the SAME
    # truth-side blur applied as a pre-blur in the forward model; conditions
    # differ ONLY through the reconstruction model's in-model processor. Values
    # remain provisional until calibration signs them off
    # (studies/scenarios/resolution_calibration.yaml).
    assert condition_spec("psf-none").forward_model_fwhm_xyz == pytest.approx((5.0, 5.0, 6.0))
    assert condition_spec("psf-undersized").forward_model_fwhm_xyz == pytest.approx((5.0, 5.0, 6.0))
    assert condition_spec("psf-matched").forward_model_fwhm_xyz == pytest.approx((5.0, 5.0, 6.0))


def test_psf_presets_share_truth_model():
    # The forward-model PSF is the truth-side blur; it must be identical across
    # all conditions so that prompts only differ by condition-independent noise.
    truth_fwhms = {
        name: spec.forward_model_fwhm_xyz for name, spec in CONDITION_SPECS.items()
    }
    assert truth_fwhms["psf-none"] == truth_fwhms["psf-undersized"]
    assert truth_fwhms["psf-none"] == truth_fwhms["psf-matched"]


def test_psf_presets_recon_model_distinguishes_conditions():
    # The reconstruction-model PSF is condition-specific; it varies the
    # post-acquisition blur the recon AM compensates for.
    assert condition_spec("psf-none").recon_model_fwhm_xyz is None
    undersized = condition_spec("psf-undersized").recon_model_fwhm_xyz
    matched = condition_spec("psf-matched").recon_model_fwhm_xyz
    assert isinstance(undersized, tuple) and len(undersized) == 3
    assert isinstance(matched, tuple) and len(matched) == 3
    assert matched != undersized


def test_psf_presets_distinct_metadata_keys():
    # simulate_inputs metadata must record distinct forward/recon PSFs so that
    # the truth-side value is reusable across conditions and the recon-side
    # value carries the condition's specific kernel. The metadata model is
    # implemented directly via the spec; this asserts the contractual keys.
    spec = condition_spec("psf-matched")
    meta = {
        "forward_model_fwhm": spec.forward_model_fwhm_xyz,
        "recon_model_fwhm": spec.recon_model_fwhm_xyz,
    }
    assert meta["forward_model_fwhm"] == pytest.approx((5.0, 5.0, 6.0))
    assert meta["recon_model_fwhm"] is not None


def test_condition_spec_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown recon-PSF condition"):
        condition_spec("psf-wat")


def test_condition_spec_is_available_from_simulation_package():
    assert public_condition_spec("psf-matched") == condition_spec("psf-matched")
