"""Ingestion tests for Task 2 canonical results tables."""
import json

import pandas as pd

from krl_studies.analysis.ingest import (
    ResultsTables,
    discover_completed_runs,
    ingest_results,
    write_tables,
)


def _write_run_dir(tmp_path, run_id, with_crc=False, malformed=False):
    """Helper to create a minimal completed run directory."""
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "study": "spheres",
        "dataset": {"kind": "spheres", "subject_id": "sub01"},
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
    if not malformed:
        metrics_data = {"iteration": [1, 2], "nrmse": [0.3, 0.25], "bv_percent": [4.0, 3.5]}
        if with_crc:
            metrics_data["crc_mm8"] = [35.0, 40.0]
        pd.DataFrame(metrics_data).to_csv(run_dir / "metrics.csv", index=False)
    else:
        (run_dir / "metrics.csv").write_text("not_iteration,value\n1,2\n")
    (run_dir / ".done").write_text("2026-01-01T00:00:00+00:00")
    return run_dir


def test_discover_completed_runs_finds_valid(tmp_path):
    _write_run_dir(tmp_path, "run_valid1")
    _write_run_dir(tmp_path, "run_valid2", with_crc=True)
    # Directory without .done should be skipped
    (tmp_path / "incomplete").mkdir()
    (tmp_path / "incomplete" / "manifest.json").write_text("{}")
    (tmp_path / "incomplete" / "metrics.csv").write_text("iteration,nrmse\n1,0.5\n")
    # No .done file

    found = discover_completed_runs(tmp_path)
    assert len(found) == 2
    assert all(p.name in ("run_valid1", "run_valid2") for p in found)


def test_ingest_results_reads_valid_and_errors(tmp_path):
    _write_run_dir(tmp_path, "run_ok")
    _write_run_dir(tmp_path, "run_ok2", with_crc=True)
    _write_run_dir(tmp_path, "run_bad", malformed=True)

    tables = ingest_results(tmp_path)

    assert isinstance(tables, ResultsTables)
    assert len(tables.runs) == 2
    assert set(tables.runs["run_id"]) == {"run_ok", "run_ok2"}
    # Each run has 2 iterations x 2 standard metrics = 4 rows; run_ok2 also has CRC in lesions
    assert len(tables.iterations) == 8  # run_ok: 4 + run_ok2: 4
    assert len(tables.lesions) == 2     # run_ok2: 1 CRC x 2 iterations
    assert tables.errors.shape == (1, 2)
    assert tables.errors.iloc[0]["run_path"] == str(tmp_path / "run_bad")


def test_ingest_results_skips_incomplete(tmp_path):
    _write_run_dir(tmp_path, "run_ok")
    (tmp_path / "no_done").mkdir()
    (tmp_path / "no_done" / "manifest.json").write_text("{}")
    (tmp_path / "no_done" / "metrics.csv").write_text("iteration,nrmse\n1,0.5\n")

    tables = ingest_results(tmp_path)
    assert len(tables.runs) == 1
    assert tables.errors.empty


def test_write_tables_csv_and_parquet(tmp_path):
    _write_run_dir(tmp_path, "run_ok")
    tables = ingest_results(tmp_path)
    paths = write_tables(tables, tmp_path / "out")

    # CSV always written
    for name in ("runs", "iterations", "lesions", "errors"):
        assert paths[name].exists()
        assert paths[name].suffix == ".csv"

    # Parquet optional - check if pyarrow available
    try:
        import pyarrow  # noqa: F401
        for name in ("runs", "iterations", "lesions", "errors"):
            assert paths[f"{name}_parquet"].exists()
    except ImportError:
        # pyarrow not installed, only CSV expected
        for name in ("runs", "iterations", "lesions", "errors"):
            assert f"{name}_parquet" not in paths


def test_write_tables_deterministic(tmp_path):
    _write_run_dir(tmp_path, "run_ok")
    tables = ingest_results(tmp_path)
    paths1 = write_tables(tables, tmp_path / "out1")
    paths2 = write_tables(tables, tmp_path / "out2")

    for name in ("runs", "iterations", "lesions", "errors"):
        csv1 = (tmp_path / "out1" / f"{name}.csv").read_bytes()
        csv2 = (tmp_path / "out2" / f"{name}.csv").read_bytes()
        assert csv1 == csv2, f"CSV not deterministic for {name}"
