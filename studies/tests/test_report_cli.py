"""CLI tests for Task 5: report CLI."""
import json

import pandas as pd
import pytest

from krl_studies.report import main


def _write_run_dir(tmp_path, run_id):
    """Helper to create a minimal completed run directory."""
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "study": "spheres",
        "dataset": {"kind": "spheres", "subject_id": "sph"},
        "input_kind": "sirf_sim",
        "input_params": {"condition": "psf-none", "counts": 1e7, "realisation": 0},
        "method": "rl",
        "method_params": {"sigma_anat": 0.2},
        "sim": {"seed": 1337},
        "simulation": {
            "scanner": "Siemens mMR",
            "forward_model_fwhm": [4.5, 4.5, 6.4],
            "recon_model_fwhm": None,
            "target_residual_fwhm": [4.5, 4.5, 6.4],
        },
        "status": "complete",
        "git_rev": "abc123",
        "krl_version": "0.2.0",
        "krl_studies_version": "0.1.0",
        "finished_at": "2026-01-01T00:00:00+00:00",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    # Include CRC metrics for lesion summary
    pd.DataFrame({
        "iteration": [1, 2],
        "nrmse": [0.3, 0.25],
        "bv_percent": [4.0, 3.5],
        "crc_mm8": [85.0, 90.0],
    }).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / ".done").write_text("2026-01-01T00:00:00+00:00")
    return run_dir


def test_report_aggregate_creates_canonical_files(tmp_path):
    _write_run_dir(tmp_path, "run1")
    out_dir = tmp_path / "analysis"
    exit_code = main(["aggregate", "--results", str(tmp_path), "--out", str(out_dir)])
    assert exit_code == 0

    # Check all canonical files exist (directly in out_dir, not aggregate subdir)
    for name in ["runs.csv", "iterations.csv", "lesions.csv", "errors.csv",
                 "summary.csv", "lesion_summary.csv", "tradeoff.csv",
                 "oracle.csv", "fixed.csv", "analysis_metadata.json"]:
        assert (out_dir / name).exists(), f"Missing {name}"


def test_report_figures_generates_pngs(tmp_path):
    _write_run_dir(tmp_path, "run1")
    agg_dir = tmp_path / "analysis"
    out_dir = tmp_path / "figures"

    # First run aggregate
    from krl_studies.analysis.report import aggregate_results
    aggregate_results(tmp_path, agg_dir, fixed_iteration=10)

    exit_code = main(["figures", "--analysis", str(agg_dir), "--out", str(out_dir)])
    assert exit_code == 0

    for name in ["nrmse_convergence.png", "recovery_vs_cov.png", "crc_by_size.png", "mismatch_sensitivity.png"]:
        assert (out_dir / name).exists()


def test_report_tables_generates_csv_and_tex(tmp_path):
    _write_run_dir(tmp_path, "run1")
    agg_dir = tmp_path / "analysis"

    # First run aggregate
    from krl_studies.analysis.report import aggregate_results
    aggregate_results(tmp_path, agg_dir, fixed_iteration=2)  # Use iteration 2 which exists in test data

    exit_code = main(["tables", "--analysis", str(agg_dir), "--out", str(tmp_path / "tables")])
    assert exit_code == 0

    for name in ["best_oracle.csv", "best_oracle.tex", "best_fixed.csv", "best_fixed.tex"]:
        assert (tmp_path / "tables" / name).exists()


def test_report_all_runs_full_pipeline(tmp_path):
    _write_run_dir(tmp_path, "run1")
    out_root = tmp_path / "out"

    exit_code = main(["all", "--results", str(tmp_path), "--out", str(out_root), "--fixed-iteration", "2"])
    assert exit_code == 0

    # Check all subdirectories exist
    for subdir in ["aggregate", "figures", "tables"]:
        assert (out_root / subdir).exists()

    # Check key files
    for name in ["runs.csv", "iterations.csv", "lesions.csv", "errors.csv",
                 "summary.csv", "lesion_summary.csv", "tradeoff.csv",
                 "oracle.csv", "fixed.csv"]:
        assert (out_root / "aggregate" / name).exists()


def test_report_help_exits_zero(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_report_aggregate_invalid_fixed_iteration(tmp_path):
    _write_run_dir(tmp_path, "run1")
    exit_code = main(
        ["aggregate", "--results", str(tmp_path), "--out", str(tmp_path / "out"), "--fixed-iteration", "5"]
    )
    assert exit_code == 0  # Should work with any positive integer
