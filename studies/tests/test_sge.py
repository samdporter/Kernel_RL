"""Tests for Task 6: SGE script generation."""
import shlex
from pathlib import Path

import pytest

from krl_studies.cluster.sge import write_sge_array_script


def test_write_sge_array_script_basic(tmp_path):
    plan_path = Path("/absolute/path/plan.jsonl")
    script_path = tmp_path / "run.sge.sh"
    result = write_sge_array_script(plan_path, script_path, n_runs=4, gpu=False, slots=1)

    assert result == script_path
    assert script_path.exists()
    content = script_path.read_text()

    assert "#!/bin/bash" in content
    assert "#$ -cwd" in content
    assert "#$ -V" in content
    assert "#$ -t 1-4" in content
    assert "#$ -pe smp 1" in content
    assert "gpu=true" not in content
    assert shlex.quote(str(plan_path)) in content
    assert "$SGE_TASK_ID" in content
    assert script_path.stat().st_mode & 0o111  # executable


def test_write_sge_array_script_gpu(tmp_path):
    plan_path = Path("/absolute/path/plan.jsonl")
    script_path = tmp_path / "run.sge.sh"
    write_sge_array_script(plan_path, script_path, n_runs=2, gpu=True, slots=4)

    content = script_path.read_text()
    assert "#$ -l gpu=true" in content
    assert "#$ -pe smp 4" in content


def test_write_sge_array_script_validates_inputs(tmp_path):
    plan_path = Path("/absolute/path/plan.jsonl")
    script_path = tmp_path / "run.sge.sh"

    with pytest.raises(ValueError, match="n_runs and slots must be positive"):
        write_sge_array_script(plan_path, tmp_path / "x.sh", n_runs=0, slots=1)

    with pytest.raises(ValueError, match="n_runs and slots must be positive"):
        write_sge_array_script(plan_path, tmp_path / "x.sh", n_runs=1, slots=0)


def test_write_sge_array_script_quotes_path(tmp_path):
    """Paths with spaces should be quoted."""
    plan_path = Path("/path with spaces/plan.jsonl")
    script_path = tmp_path / "run.sge.sh"
    write_sge_array_script(plan_path, tmp_path / "run.sge.sh", n_runs=1)

    content = Path(tmp_path / "run.sge.sh").read_text()
    assert shlex.quote(str(Path("/path with spaces/plan.jsonl"))) in content
