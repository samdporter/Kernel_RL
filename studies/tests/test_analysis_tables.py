"""Tests for Task 5: LaTeX/CSV tables."""
import pandas as pd

from krl_studies.analysis.tables import best_results_table, write_latex_table


def _selected_frame():
    """Construct a small selected DataFrame matching the canonical schema."""
    data = [
        {
            "run_id": "r0",
            "realisation": 0,
            "method": "rl",
            "iteration": 2,
            "metric": "nrmse",
            "value": 0.3,
            "study": "spheres",
            "subject_id": "sph",
            "dataset_kind": "spheres",
            "input_kind": "sirf_sim",
            "scanner": "Siemens mMR",
            "condition": "psf-none",
            "beta": None,
            "counts": 5.0e7,
            "guidance_condition": "exact",
            "assumed_fwhm_mm": 5.0,
            "selection": "oracle_min_nrmse",
        },
        {
            "run_id": "r1",
            "realisation": 1,
            "method": "rl",
            "iteration": 2,
            "metric": "nrmse",
            "value": 0.35,
            "study": "spheres",
            "subject_id": "sph",
            "dataset_kind": "spheres",
            "input_kind": "sirf_sim",
            "scanner": "Siemens mMR",
            "condition": "psf-none",
            "beta": None,
            "counts": 5.0e7,
            "guidance_condition": "exact",
            "assumed_fwhm_mm": 5.0,
            "selection": "oracle_min_nrmse",
        },
        {
            "run_id": "r0",
            "realisation": 0,
            "method": "rl",
            "iteration": 2,
            "metric": "crc_percent",
            "value": 85.0,
            "study": "spheres",
            "subject_id": "sph",
            "dataset_kind": "spheres",
            "input_kind": "sirf_sim",
            "scanner": "Siemens mMR",
            "condition": "psf-none",
            "beta": None,
            "counts": 5.0e7,
            "guidance_condition": "exact",
            "assumed_fwhm_mm": 5.0,
            "selection": "oracle_min_nrmse",
        },
        {
            "run_id": "r0",
            "realisation": 0,
            "method": "rl",
            "iteration": 2,
            "metric": "bv_percent",
            "value": 4.0,
            "study": "spheres",
            "subject_id": "sph",
            "dataset_kind": "spheres",
            "input_kind": "sirf_sim",
            "scanner": "Siemens mMR",
            "condition": "psf-none",
            "beta": None,
            "counts": 5.0e7,
            "guidance_condition": "exact",
            "assumed_fwhm_mm": 5.0,
            "selection": "oracle_min_nrmse",
        },
    ]
    return pd.DataFrame(data)


def test_best_results_table_groups_and_pivots():
    result = best_results_table(_selected_frame())
    # Should have one row per method/condition/beta/guidance/beta combination
    assert len(result) >= 1
    # Should have metric columns as columns after pivot
    assert "value_mean" in result.columns
    # Metrics should be columns after pivot
    assert ("value_mean", "nrmse") in result.columns


def test_best_results_table_preserves_null_beta():
    frame = _selected_frame()
    # Ensure beta is None in input
    result = best_results_table(frame)
    # Null beta should be preserved in output
    assert result["beta"].isna().any() or (result["beta"] == "").any()


def test_best_results_table_one_row_has_zero_std():
    # Create a frame with only one realisation for a given group
    data = [
        {
            "run_id": "r0",
            "realisation": 0,
            "method": "rl",
            "iteration": 2,
            "metric": "nrmse",
            "value": 0.3,
            "study": "spheres",
            "subject_id": "sph",
            "dataset_kind": "spheres",
            "input_kind": "sirf_sim",
            "scanner": "Siemens mMR",
            "condition": "psf-none",
            "beta": None,
            "counts": 5.0e7,
            "guidance_condition": "exact",
            "assumed_fwhm_mm": 5.0,
            "selection": "oracle_min_nrmse",
        },
    ]
    frame = pd.DataFrame(data)
    result = best_results_table(frame)
    # With only one value, std should be 0.0
    assert result.loc[0, ("value_std", "nrmse")] == 0.0


def test_write_latex_table(tmp_path):
    output = tmp_path / "table.tex"
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    write_latex_table(frame, output, caption="Test", label="tab:test")
    assert output.exists()
    content = output.read_text()
    assert "Test" in content
    assert "tab:test" in content
