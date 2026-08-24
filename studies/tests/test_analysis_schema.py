"""Schema tests for Task 2 canonical results tables."""
import json

import pytest

from krl_studies.analysis.schema import (
    ITERATION_COLUMNS,
    LESION_COLUMNS,
    RUN_COLUMNS,
    flatten_manifest,
    melt_metrics,
)


def test_flatten_manifest_normalises_nested_params():
    row = flatten_manifest(
        {
            "run_id": "r1",
            "study": "spheres",
            "dataset": {"kind": "spheres"},
            "input_kind": "sirf_sim",
            "input_params": {"condition": "psf-none", "beta": None, "counts": 1e7},
            "method": "krl",
            "method_params": {"sigma_anat": 0.2},
            "sim": {"seed": 1337},
            "simulation": {"scanner": "Siemens mMR"},
            "status": "complete",
        },
        "/tmp/r1",
    )
    assert row["condition"] == "psf-none"
    assert row["beta"] is None
    assert json.loads(row["method_params_json"]) == {"sigma_anat": 0.2}
    assert set(RUN_COLUMNS) <= row.keys()


def test_melt_metrics_separates_standard_metrics_and_crc():
    iterations, lesions = melt_metrics(
        "r1",
        {"iteration": [1], "nrmse": [0.2], "bv_percent": [4.0], "crc_mm8": [35.0]},
        {"study": "spheres", "method": "rl"},
    )
    assert set(iterations["metric"]) == {"nrmse", "bv_percent"}
    assert iterations["value"].tolist() == [0.2, 4.0]
    assert lesions["lesion_diameter_mm"].tolist() == [8.0]
    assert lesions["metric"].tolist() == ["crc_percent"]
    assert lesions["value"].tolist() == [35.0]
    assert set(ITERATION_COLUMNS) <= set(iterations.columns)
    assert set(LESION_COLUMNS) <= set(lesions.columns)


def test_invalid_metric_file_is_reported(tmp_path):
    (tmp_path / "metrics.csv").write_text("not_iteration,value\n1,2\n")
    with pytest.raises(ValueError, match="iteration"):
        melt_metrics("r1", {"not_iteration": [1], "value": [2]}, {})


def test_melt_metrics_handles_missing_metrics():
    iterations, lesions = melt_metrics(
        "r2",
        {"iteration": [1, 2], "nrmse": [0.3, 0.25]},
        {"study": "patient", "method": "dtv"},
    )
    assert lesions.empty
    assert iterations["metric"].tolist() == ["nrmse", "nrmse"]
    assert iterations["value"].tolist() == [0.3, 0.25]


def test_melt_metrics_crc_column_parses_diameter():
    iterations, lesions = melt_metrics(
        "r3",
        {"iteration": [1], "crc_mm12p5": [42.0]},
        {"study": "spheres", "method": "krl"},
    )
    assert lesions["lesion_diameter_mm"].tolist() == [12.5]
    assert lesions["metric"].tolist() == ["crc_percent"]


def test_flatten_manifest_defaults_guidance_condition():
    row = flatten_manifest(
        {
            "run_id": "r4",
            "study": "spheres",
            "dataset": {"kind": "spheres"},
            "input_kind": "sirf_sim",
            "input_params": {},
            "method": "rl",
            "method_params": {},
            "sim": {},
            "simulation": {},
            "status": "complete",
        },
        "/tmp/r4",
    )
    assert row["guidance_condition"] == "exact"
    assert row["condition"] is None
    assert row["beta"] is None
    assert row["assumed_fwhm_mm"] is None


def test_flatten_manifest_reads_scanner_preference_order():
    row = flatten_manifest(
        {
            "run_id": "r5",
            "study": "spheres",
            "dataset": {"kind": "spheres"},
            "input_kind": "sirf_sim",
            "input_params": {"scanner": "Siemens VISION 600"},
            "method": "rl",
            "method_params": {},
            "sim": {"scanner": "Siemens mMR"},
            "simulation": {"scanner": "Siemens mMR"},
            "status": "complete",
        },
        "/tmp/r5",
    )
    assert row["scanner"] == "Siemens mMR"


def test_flatten_manifest_compact_json_sort_keys():
    row = flatten_manifest(
        {
            "run_id": "r6",
            "study": "spheres",
            "dataset": {"kind": "spheres"},
            "input_kind": "sirf_sim",
            "input_params": {},
            "method": "rl",
            "method_params": {"b": 1, "a": 2},
            "sim": {"z": 3, "y": 4},
            "simulation": {"forward_model_fwhm": (4.5, 4.5, 6.4)},
            "status": "complete",
        },
        "/tmp/r6",
    )
    # JSON should have sorted keys and compact format (no spaces)
    assert row["method_params_json"] == '{"a":2,"b":1}'
    assert row["sim_params_json"] == '{"y":4,"z":3}'
    assert row["forward_model_fwhm_json"] == "[4.5,4.5,6.4]"
