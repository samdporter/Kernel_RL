"""Tests for Task 6: run-plan JSONL serialization and SGE generation."""
import json

import pytest

from krl_studies.config import RunSpec
from krl_studies.runner.plan import (
    PLAN_VERSION,
    read_run_plan,
    write_run_plan,
)


def _make_run_spec(run_id="test_run"):
    return RunSpec(
        run_id=run_id,
        study="spheres",
        dataset={"kind": "spheres", "root": "data/spheres"},
        input_kind="sirf_sim",
        input_params={"condition": "psf-none", "counts": 1e7},
        method_name="rl",
        method_params={"fwhm_mm": 5.0},
        sim={"seed": 1337},
    )


def test_write_run_plan_writes_jsonl(tmp_path):
    runs = [_make_run_spec("run1"), _make_run_spec("run2")]
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 runs
    header = json.loads(lines[0])
    assert header == {"plan_version": PLAN_VERSION}

    run1 = json.loads(lines[1])
    assert run1["run_id"] == "run1"
    assert run1["study"] == "spheres"
    assert run1["input_kind"] == "sirf_sim"
    assert run1["input_params"]["condition"] == "psf-none"


def test_read_run_plan_roundtrip(tmp_path):
    runs = [_make_run_spec("run1"), _make_run_spec("run2")]
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)

    loaded = read_run_plan(path)
    assert len(loaded) == 2
    assert loaded[0].run_id == "run1"
    assert loaded[0].study == "spheres"
    assert loaded[0].input_kind == "sirf_sim"
    assert loaded[0].input_params["condition"] == "psf-none"
    assert loaded[1].run_id == "run2"


def test_read_run_plan_validates_header(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"plan_version": 999}\n{"run_id": "r1"}')
    with pytest.raises(ValueError, match="unsupported run-plan header"):
        read_run_plan(path)


def test_read_run_plan_rejects_blank_row(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"plan_version": 1}\n\n{"run_id": "r1"}')
    with pytest.raises(ValueError, match="blank run-plan row"):
        read_run_plan(path)


def test_read_run_plan_rejects_malformed_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"plan_version": 1}\nnot json')
    with pytest.raises(ValueError, match="invalid JSON at line"):
        read_run_plan(path)


def test_read_run_plan_rejects_missing_fields(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"plan_version": 1}\n{"run_id": "r1"}')  # missing required fields
    with pytest.raises(ValueError, match="invalid run-plan row"):
        read_run_plan(path)
