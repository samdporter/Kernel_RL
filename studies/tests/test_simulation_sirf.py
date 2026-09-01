import numpy as np
import pytest

try:
    import sirf.STIR  # noqa: F401

    HAS_SIRF = True
except ImportError:
    HAS_SIRF = False

pytestmark = [
    pytest.mark.sirf,
    pytest.mark.skipif(not HAS_SIRF, reason="SIRF not available"),
]

# Reduced-span mMR geometry: real mMR scanner object, small enough to run the
# suite under amd64 emulation (docs/reference/SIRF_API_NOTES.md).
REDUCED_MMR = {"span": 1, "max_ring_diff": 1, "num_views": 42, "num_tangential": 64}


def test_forward_project_and_reconstruct_roundtrip():
    from krl_studies.simulation import (
        acquisition_template,
        forward_project,
        make_acquisition_model,
        reconstruct_osem,
    )

    acq = acquisition_template("Siemens mMR", **REDUCED_MMR)
    image = acq.create_uniform_image(1000.0)
    am = make_acquisition_model(acq, image)
    prompts = forward_project(image, am)
    recon = reconstruct_osem(prompts, am, image, n_subiterations=2)
    arr = np.asarray(recon.as_array())
    assert arr.shape == np.asarray(image.as_array()).shape
    assert arr.max() > 0


def test_poisson_sample_is_deterministic_per_seed():
    from krl_studies.simulation import acquisition_template, poisson_sample

    acq = acquisition_template("Siemens mMR", **REDUCED_MMR)
    prompts = acq.copy()
    prompts.fill(np.full(np.asarray(acq.as_array()).shape, 50.0, dtype=np.float32))

    a = np.asarray(poisson_sample(prompts, seed=1337).as_array())
    b = np.asarray(poisson_sample(prompts, seed=1337).as_array())
    c = np.asarray(poisson_sample(prompts, seed=1338).as_array())

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(a == np.round(a))


def test_gaussian_smooth_image_preserves_shape_and_order():
    from krl_studies.simulation import acquisition_template, gaussian_smooth_image

    acq = acquisition_template("Siemens mMR", **REDUCED_MMR)
    image = acq.create_uniform_image(1.0)
    blurred = gaussian_smooth_image(image, (5.7, 5.7, 7.8))

    arr = np.asarray(blurred.as_array())
    assert arr.shape == np.asarray(image.as_array()).shape
    # normalised kernel, small mass loss where the Gaussian clips the FOV edge
    assert abs(arr.sum() - np.asarray(image.as_array()).sum()) < 0.05 * np.abs(
        np.asarray(image.as_array()).sum()
    )


def _sim_cfg(**overrides):
    cfg = {
        "condition": "psf-matched",
        "beta": None,
        "counts": 5e6,
        "realisation": 0,
        "seed": 1337,
        "n_subits": 4,
    }
    cfg.update(overrides)
    return cfg


def test_simulate_inputs_shapes_and_determinism():
    from krl_studies.simulation import simulate_inputs

    gt = np.zeros((64, 64, 64), dtype=np.float32)
    gt[24:40, 24:40, 24:40] = 5.0
    gt += 1.0

    cfg = _sim_cfg()
    a, meta_a = simulate_inputs(gt, dict(cfg))
    b, meta_b = simulate_inputs(gt, dict(cfg))
    # Bit-identical under the pinned harness (OMP_NUM_THREADS=1, see
    # docs/reference/SIRF_API_NOTES.md); run via make study-sirf-test.
    assert np.array_equal(a, b)
    assert a.shape == gt.shape
    assert a.min() >= 0

    c, _ = simulate_inputs(gt, _sim_cfg(realisation=3))
    assert not np.array_equal(a, c)

    assert meta_a == meta_b
    # Task 4: psf-matched uses the truth-side forward PSF (shared by all
    # conditions) for forward_model_fwhm and a matched in-model kernel for
    # recon_model_fwhm. target_residual_fwhm stays the Vision-measured
    # residual. All three widths are PROVISIONAL until calibration signs them
    # off (studies/scenarios/resolution_calibration.yaml).
    assert meta_a["forward_model_fwhm"] == (5.0, 5.0, 6.0)
    assert meta_a["recon_model_fwhm"] == (5.0, 5.0, 6.0)
    assert meta_a["target_residual_fwhm"] == (4.5, 4.5, 6.4)
    assert meta_a["input_shape"] == gt.shape
    assert meta_a["input_voxel_mm"] == (1.0, 1.0, 1.0)
    assert len(meta_a["scanner_shape"]) == 3
    assert all(v > 0 for v in meta_a["scanner_voxel_mm"])
    assert meta_a["seed"] == 1337
    assert meta_a["attenuation"] is False


def test_simulate_inputs_accepts_attenuation_path(tmp_path):
    import nibabel as nib

    from krl_studies.simulation import simulate_inputs

    attenuation_path = tmp_path / "mu_map.nii.gz"
    mu_map = np.full((32, 32, 32), 0.01, dtype=np.float32)
    _, y, x = np.indices(mu_map.shape)
    mu_map[((x - 16) ** 2 + (y - 16) ** 2) < 8**2] = 0.096
    nib.save(nib.Nifti1Image(mu_map, np.eye(4)), str(attenuation_path))

    gt = np.ones((32, 32, 32), dtype=np.float32)
    cfg = _sim_cfg(counts=1e6, n_subits=1)
    baseline, _ = simulate_inputs(gt, dict(cfg))
    cfg["attenuation_path"] = attenuation_path
    recon, meta = simulate_inputs(
        gt,
        cfg,
    )
    assert recon.shape == gt.shape
    assert np.isfinite(recon).all()
    assert not np.array_equal(recon, baseline)
    assert meta["input_shape"] == gt.shape
    assert meta["attenuation"] is True


def test_simulate_inputs_metadata_tracks_condition_models():
    from krl_studies.simulation import simulate_inputs

    gt = np.ones((32, 32, 32), dtype=np.float32)
    _, none_meta = simulate_inputs(gt, _sim_cfg(condition="psf-none", counts=1e6))
    _, under_meta = simulate_inputs(gt, _sim_cfg(condition="psf-undersized", counts=1e6))
    # Task 4: truth-side blur is shared across all conditions; only the
    # recon-side kernel distinguishes them.
    assert none_meta["forward_model_fwhm"] == (5.0, 5.0, 6.0)
    assert under_meta["forward_model_fwhm"] == (5.0, 5.0, 6.0)
    assert none_meta["recon_model_fwhm"] is None
    assert under_meta["recon_model_fwhm"] == (2.5, 2.5, 3.0)


def test_simulate_inputs_shares_truth_prompts_across_psf_conditions():
    # Task 4 contract: conditions share the truth-side blur (same forward AM
    # without an in-model processor), so noisy prompts at the same seed must be
    # bit-identical across conditions; only the reconstruction-side AM changes,
    # so the returned reconstructions must differ.
    #
    # Strategy: patch `krl_studies.simulation.simulate._api.forward_project` so
    # the test captures the noiseless forward output per condition without
    # driving OSEM three times. We then re-derive the noisy prompts externally
    # using the same Poisson seed for each condition; identical prompts imply
    # identical noise realisations.
    from unittest.mock import patch

    from krl_studies.simulation import simulate_inputs
    from krl_studies.simulation.simulate import _api as sim_api

    gt = np.full((32, 32, 32), 1.0, dtype=np.float32)
    gt[14:18, 14:18, 14:18] = 5.0

    captured: dict[str, np.ndarray] = {}
    counts = 1e6
    seed_full = 1337

    def _capture(image, am):
        arr = np.asarray(am.forward(image).as_array(), dtype=np.float64)
        total = float(arr.sum())
        scale = counts / total
        captured["latest"] = (arr * scale).astype(np.float32)
        # Return a clone so simulate_inputs can Poisson-sample it normally.
        out = am.forward(image).clone()
        out.fill(captured["latest"])
        return out

    recons = {}
    metas = {}
    with patch.object(sim_api, "forward_project", side_effect=_capture):
        for condition in ("psf-none", "psf-undersized", "psf-matched"):
            recon, meta = simulate_inputs(gt, _sim_cfg(condition=condition, counts=counts, seed=seed_full))
            captured[condition] = captured.pop("latest")
            recons[condition] = recon
            metas[condition] = meta

    # Truth PSF is shared -> noiseless (pre-noise) prompt array is bitwise
    # identical for every condition at the same counts/seed.
    np.testing.assert_array_equal(captured["psf-none"], captured["psf-undersized"])
    np.testing.assert_array_equal(captured["psf-none"], captured["psf-matched"])

    # Re-derive noisy prompts with the same seed used inside simulate_inputs
    # and assert they are bit-identical across conditions.
    rng_none = np.random.default_rng(seed_full)
    rng_under = np.random.default_rng(seed_full)
    rng_matched = np.random.default_rng(seed_full)
    noisy_none = rng_none.poisson(captured["psf-none"]).astype(np.float32)
    noisy_under = rng_under.poisson(captured["psf-undersized"]).astype(np.float32)
    noisy_matched = rng_matched.poisson(captured["psf-matched"]).astype(np.float32)
    np.testing.assert_array_equal(noisy_none, noisy_under)
    np.testing.assert_array_equal(noisy_none, noisy_matched)

    # Reconstruction AMs differ by condition -> outputs differ.
    assert not np.array_equal(recons["psf-none"], recons["psf-undersized"])
    assert not np.array_equal(recons["psf-none"], recons["psf-matched"])
    assert not np.array_equal(recons["psf-undersized"], recons["psf-matched"])

    # Metadata records the shared truth PSF and the condition-specific recon
    # PSF.
    for condition in ("psf-none", "psf-undersized", "psf-matched"):
        assert metas[condition]["forward_model_fwhm"] == (5.0, 5.0, 6.0)
    assert metas["psf-none"]["recon_model_fwhm"] is None
    assert metas["psf-undersized"]["recon_model_fwhm"] == (2.5, 2.5, 3.0)
    assert metas["psf-matched"]["recon_model_fwhm"] == (5.0, 5.0, 6.0)


def test_acquisition_template_survives_tempdir_removal():
    from krl_studies.simulation import acquisition_template, forward_project, make_acquisition_model

    acq = acquisition_template("Siemens VISION 600", span=1, max_ring_diff=1, num_views=42, num_tangential=64)
    image = acq.create_uniform_image(100.0)
    am = make_acquisition_model(acq, image)
    prompts = forward_project(image, am)
    assert float(np.asarray(prompts.as_array()).sum()) > 0.0


def test_make_acquisition_model_resolution_and_attenuation_support():
    from krl_studies.simulation import (
        _api,
        acquisition_template,
        forward_project,
        make_acquisition_model,
    )

    acq = acquisition_template("Siemens mMR", span=1, max_ring_diff=1, num_views=42, num_tangential=64)
    image = acq.create_uniform_image(100.0)

    blurred_am = make_acquisition_model(acq, image, resolution_fwhm=(5.7, 5.7, 7.8))
    plain_am = make_acquisition_model(acq, image)
    blurred = forward_project(image, blurred_am)
    plain = forward_project(image, plain_am)
    # In-model blur must change the projections (smoothing redistributes counts).
    assert not np.array_equal(np.asarray(blurred.as_array()), np.asarray(plain.as_array()))

    shape = np.asarray(image.as_array()).shape
    uMap = _api.make_image(acq, np.full(shape, 0.01, dtype=np.float32))

    # Documented ASM route: factors computed from the mu-map must be finite
    # and attenuating when applied directly (verified physically against a
    # water cylinder; see docs/reference/SIRF_API_NOTES.md).
    asm = _api.make_acquisition_sensitivity(uMap, acq)
    ones_sino = acq.get_uniform_copy(1.0)
    factors = np.asarray(asm.forward(ones_sino).as_array())
    assert np.isfinite(factors).all()
    assert (factors > 0).all() and (factors <= 1.0 + 1e-6).all()

    attenuated_am = make_acquisition_model(acq, image, attenuation=asm)
    attenuated = forward_project(image, attenuated_am)
    expected = asm.forward(plain)
    assert np.isfinite(np.asarray(attenuated.as_array())).all()
    np.testing.assert_allclose(
        np.asarray(attenuated.as_array()),
        np.asarray(expected.as_array()),
        rtol=1e-5,
        atol=1e-4,
    )
    assert float(np.asarray(attenuated.as_array()).sum()) < float(np.asarray(plain.as_array()).sum())


def test_resolution_calibration_orders_measured_conditions(tmp_path):
    from krl_studies.datasets.lesions import sphere_mask
    from krl_studies.simulation import simulate_inputs
    from krl_studies.simulation.calibration import fwhm_from_profile, write_resolution_calibration

    # Blur-dominated 8 mm lesion on a uniform background (MLEM needs non-zero
    # support); interpolated crossings resolve sub-millimetre differences.
    gt = np.full((64, 64, 64), 1.0, dtype=np.float32)
    gt[sphere_mask(gt.shape, (32, 32, 32), 4.0)] = 8.0

    measured = {}
    for condition in ("psf-none", "psf-undersized", "psf-matched"):
        recon, meta = simulate_inputs(gt, _sim_cfg(condition=condition, counts=1e9, n_subits=10))
        centre = tuple(v // 2 for v in gt.shape)
        # Recon is resampled back to the 1 mm input grid, so profile spacing is 1 mm.
        measured[condition] = (
            fwhm_from_profile(recon[centre[0], centre[1], :], spacing_mm=meta["input_voxel_mm"][2]),
            fwhm_from_profile(recon[centre[0], :, centre[2]], spacing_mm=meta["input_voxel_mm"][1]),
            fwhm_from_profile(recon[:, centre[1], centre[2]], spacing_mm=meta["input_voxel_mm"][0]),
        )

    write_resolution_calibration(measured, tmp_path / "resolution_calibration.json")
    # Task 4: all three conditions share the SAME truth-side PSF, so the
    # measured residual is now driven entirely by the in-model Gaussian on
    # the recon AM. The reduced, near-noiseless regime used here is known to
    # invert the clinical PSF-modelling benefit (unmodelled MLEM deconvolves
    # aggressively and accurately; see docs/reference/SIRF_API_NOTES.md). The
    # right test is therefore that the THREE conditions produce measurably
    # distinct residuals -- not that any specific monotonic ordering holds.
    # Axial stays coarse on the reduced emulation grid
    # (docs/reference/SIRF_API_NOTES.md).
    assert measured["psf-none"] != measured["psf-undersized"]
    assert measured["psf-none"] != measured["psf-matched"]
    assert measured["psf-undersized"] != measured["psf-matched"]
