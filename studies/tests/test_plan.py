import json
import subprocess
import sys

import pytest

from krl_studies.config import RunSpec
from krl_studies.plan import _plan_cli as main
from krl_studies.runner.plan import read_run_plan, write_run_plan

# Skip tests that require CLI subprocess on macOS (no CIL)
# They run in the container via make study-sirf-test
requires_cil = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="requires CIL which is not available on macOS; runs in container"
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


def test_plan_cli_creates_jsonl(tmp_path):
    """Test that plan CLI creates a valid JSONL plan from a scenario."""
    # Create a minimal scenario file
    scenario = {
        "study": "spheres",
        "dataset": {"kind": "spheres", "root": "data/spheres"},
        "inputs": [{"kind": "sirf_sim", "params": {"condition": "psf-none", "counts": 1e7}}],
        "methods": [{"name": "rl", "params": {"fwhm_mm": 5.0}}],
        "sim": {"seed": 1337},
        "output": str(tmp_path / "results"),
    }
    scenario_path = tmp_path / "scenario.yaml"
    import yaml
    scenario_path.write_text(yaml.safe_dump(scenario))

    plan_path = tmp_path / "plan.jsonl"
    rc = main(["--scenario", str(scenario_path), "--out", str(plan_path)])
    assert rc == 0

    # Verify plan file
    lines = plan_path.read_text().strip().splitlines()
    assert len(lines) >= 2  # header + at least 1 run
    header = json.loads(lines[0])
    assert header == {"plan_version": 1}


def test_plan_cli_roundtrip(tmp_path):
    """Test that a plan can be written and read back."""
    runs = [
        RunSpec(
            run_id="run1",
            study="spheres",
            dataset={"kind": "spheres", "root": "data/spheres"},
            input_kind="sirf_sim",
            input_params={"condition": "psf-none", "counts": 1e7},
            method_name="rl",
            method_params={"fwhm_mm": 5.0},
            sim={"seed": 1337},
        ),
        RunSpec(
            run_id="run2",
            study="spheres",
            dataset={"kind": "spheres", "root": "data/spheres"},
            input_kind="sirf_sim",
            input_params={"condition": "psf-matched", "counts": 1e7},
            method_name="krl",
            method_params={"fwhm_mm": 5.0, "sigma_anat": 0.2},
            sim={"seed": 1337},
        ),
    ]
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)

    loaded = read_run_plan(path)
    assert len(loaded) == 2
    assert loaded[0].run_id == "run1"
    assert loaded[1].run_id == "run2"


@requires_cil
def test_plan_cli_invalid_index(tmp_path):
    """Test that plan CLI rejects invalid index."""
    # Create a plan with 1 run
    runs = [
        RunSpec(
            run_id="run1",
            study="spheres",
            dataset={"kind": "spheres", "root": "data/spheres"},
            input_kind="sirf_sim",
            input_params={"condition": "psf-none", "counts": 1e7},
            method_name="rl",
            method_params={"fwhm_mm": 5.0},
            sim={"seed": 1337},
        ),
    ]
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)

    # Test index out of bounds
    result = subprocess.run(
        [sys.executable, "-m", "krl_studies.run", "--plan", str(path), "--index", "2"],
        capture_output=True,
        text=True,
        cwd="/workspace/studies"
    )
    assert result.returncode != 0
    assert "between 1 and 1" in result.stderr


@requires_cil
def test_plan_cli_missing_index_fails(tmp_path):
    """Test that --plan without --index fails."""
    result = subprocess.run(
        [sys.executable, "-m", "krl_studies.run", "--plan", "nonexistent.jsonl"],
        capture_output=True,
        text=True,
        cwd="/workspace/studies"
    )
    assert result.returncode != 0
    assert "requires --index" in result.stderr
