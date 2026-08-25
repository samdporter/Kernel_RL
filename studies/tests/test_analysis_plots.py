"""Tests for Task 5: plotting API and publication figures."""
import numpy as np
import pandas as pd
import pytest

from krl_studies.analysis.plots import (
    plot_crc_by_size,
    plot_mismatch_sensitivity,
    plot_nrmse_convergence,
    plot_profile,
    plot_recovery_vs_cov,
)


def _summary_frame():
    """Construct a small summary DataFrame matching the canonical schema."""
    data = [
        {
            "study": "spheres",
            "method": "rl",
            "condition": "psf-none",
            "beta": None,
            "counts": 5.0e7,
            "realisation": 0,
            "guidance_condition": "exact",
            "assumed_fwhm_mm": 5.0,
            "iteration": 1,
            "metric": "nrmse",
            "value_mean": 0.4,
            "value_std": 0.05,
            "n": 2,
            "forward_model_fwhm_json": "[5.7, 5.7, 7.8]",
            "recon_model_fwhm_json": "null",
            "target_residual_fwhm_json": "[5.7, 5.7, 7.8]",
        },
        {
            "study": "spheres",
            "method": "rl",
            "condition": "psf-none",
            "beta": None,
            "counts": 5.0e7,
            "realisation": 0,
            "guidance_condition": "exact",
            "assumed_fwhm_mm": 5.0,
            "iteration": 2,
            "metric": "nrmse",
            "value_mean": 0.3,
            "value_std": 0.04,
            "n": 2,
            "forward_model_fwhm_json": "[5.7, 5.7, 7.8]",
            "recon_model_fwhm_json": "null",
            "target_residual_fwhm_json": "[5.7, 5.7, 7.8]",
        },
        {
            "study": "spheres",
            "method": "krl",
            "condition": "psf-matched",
            "beta": None,
            "counts": 1.0e8,
            "realisation": 0,
            "guidance_condition": "shift_p2",
            "assumed_fwhm_mm": 5.0,
            "iteration": 1,
            "metric": "nrmse",
            "value_mean": 0.35,
            "value_std": 0.03,
            "n": 2,
            "forward_model_fwhm_json": "[4.5, 4.5, 6.4]",
            "recon_model_fwhm_json": "[4.5, 4.5, 6.4]",
            "target_residual_fwhm_json": "[4.5, 4.5, 6.4]",
        },
    ]
    return pd.DataFrame(data)


def _lesion_summary_frame():
    """Construct a small lesion summary DataFrame."""
    data = [
        {
            "study": "spheres",
            "method": "rl",
            "condition": "psf-none",
            "beta": None,
            "counts": 5.0e7,
            "realisation": 0,
            "guidance_condition": "exact",
            "assumed_fwhm_mm": 5.0,
            "iteration": 1,
            "metric": "crc_percent",
            "lesion_diameter_mm": 8.0,
            "value_mean": 85.0,
            "value_std": 2.0,
            "n": 2,
        },
        {
            "study": "spheres",
            "method": "krl",
            "condition": "psf-matched",
            "beta": None,
            "counts": 1.0e8,
            "realisation": 0,
            "guidance_condition": "shift_p2",
            "assumed_fwhm_mm": 5.0,
            "iteration": 1,
            "metric": "crc_percent",
            "lesion_diameter_mm": 8.0,
            "value_mean": 92.0,
            "value_std": 1.5,
            "n": 2,
        },
        {
            "study": "spheres",
            "method": "rl",
            "condition": "psf-none",
            "beta": None,
            "counts": 5.0e7,
            "realisation": 0,
            "guidance_condition": "exact",
            "assumed_fwhm_mm": 5.0,
            "iteration": 1,
            "metric": "crc_percent",
            "lesion_diameter_mm": 12.0,
            "value_mean": 90.0,
            "value_std": 1.0,
            "n": 2,
        },
    ]
    return pd.DataFrame(data)


def _tradeoff_frame():
    """Construct a small tradeoff frame with BV and CRC."""
    data = [
        {
            "study": "spheres",
            "method": "rl",
            "condition": "psf-none",
            "beta": None,
            "counts": 5.0e7,
            "realisation": 0,
            "guidance_condition": "exact",
            "assumed_fwhm_mm": 5.0,
            "iteration": 1,
            "bv_percent": 4.0,
            "crc_percent": 85.0,
            "nrmse": 0.4,
            "objective": 1.2,
        },
        {
            "study": "spheres",
            "method": "krl",
            "condition": "psf-matched",
            "beta": None,
            "counts": 1.0e8,
            "realisation": 0,
            "guidance_condition": "shift_p2",
            "assumed_fwhm_mm": 5.0,
            "iteration": 1,
            "bv_percent": 3.5,
            "crc_percent": 92.0,
            "nrmse": 0.35,
            "objective": 1.1,
        },
    ]
    return pd.DataFrame(data)


def _image_dict():
    """Return a small dictionary of test images."""
    img = np.zeros((5, 6, 7), dtype=np.float32)
    img[2, 3, 4] = 1.0
    return {"img1": img, "img2": img * 2.0}


def test_plot_nrmse_convergence(tmp_path):
    output = tmp_path / "nrmse_convergence.png"
    plot_nrmse_convergence(_summary_frame(), output, title="Test NRMSE")
    assert output.exists()
    assert output.stat().st_size > 0
    # Second call overwrites deterministically
    plot_nrmse_convergence(_summary_frame(), output, title="Test NRMSE")
    assert output.exists()


def test_plot_recovery_vs_cov(tmp_path):
    output = tmp_path / "recovery_vs_cov.png"
    plot_recovery_vs_cov(_tradeoff_frame(), output, title="Test Recovery vs Cov")
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_crc_by_size(tmp_path):
    output = tmp_path / "crc_by_size.png"
    plot_crc_by_size(_lesion_summary_frame(), output, title="Test CRC by Size")
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_mismatch_sensitivity(tmp_path):
    output = tmp_path / "mismatch_sensitivity.png"
    plot_mismatch_sensitivity(_summary_frame(), output, title="Test Mismatch")
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_profile(tmp_path):
    output = tmp_path / "profile.png"
    plot_profile(_image_dict(), output, axis=0, index=(3, 4))
    assert output.exists()
    assert output.stat().st_size > 0


def test_empty_input_produces_empty_figure(tmp_path):
    """Empty inputs should produce an empty labelled figure, not raise."""
    output = tmp_path / "empty.png"
    plot_nrmse_convergence(pd.DataFrame(), output, title="Empty")
    assert output.exists()


def test_profile_invalid_axis_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_profile(_image_dict(), tmp_path / "bad.png", axis=3, index=(0, 0))


def test_profile_invalid_index_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_profile(_image_dict(), tmp_path / "bad.png", axis=0, index=(10, 10))
