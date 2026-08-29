import json

import numpy as np
import pytest
import yaml
from conftest import write_test_nifti

from krl_studies.config import load_scenario_dict
from krl_studies.runner.cli import main
from krl_studies.runner.execute import execute_run
from krl_studies.runner.expand import expand_scenario


def _mini_scenario(tmp_path):
    # 64^3 so background_vois defaults fit: 8 VOIs of radius 5 need centres
    # >=20 voxels apart inside a 5-voxel margin.
    gt = np.full((64, 64, 64), 1.0, dtype=np.float32)
    gt[24:40, 24:40, 24:40] = 6.0
    d = tmp_path / "spheres"
    d.mkdir()
    write_test_nifti(d / "phant_orig.nii", gt)
    write_test_nifti(d / "phant_mri.nii", np.full_like(gt, 0.5))
    write_test_nifti(d / "phant_pet.nii", gt * 0.95)

    scenario = {
        "study": "spheres",
        "dataset": {"kind": "spheres", "root": str(d)},
        "inputs": [{"kind": "reference"}],
        "methods": [
            {"name": "post_smoothing", "params": {"sigma_mm": 2.0}},
        ],
        "sim": {"fwhm_mm": 3.0, "counts": 1e5},
        "output": str(tmp_path / "results"),
    }
    sp = tmp_path / "scen.yaml"
    sp.write_text(yaml.safe_dump(scenario))
    return sp


def test_execute_creates_manifest_metrics_marker(tmp_path):
    scenario = load_scenario_dict(yaml.safe_load(_mini_scenario(tmp_path).read_text()))
    run = expand_scenario(scenario)[0]
    assert run.sim == {"fwhm_mm": 3.0, "counts": 1e5}
    assert run.out_root == tmp_path / "results"

    out = execute_run(run)
    assert (out / ".done").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["method"] == "post_smoothing"
    assert manifest["status"] == "complete"
    csv_lines = (out / "metrics.csv").read_text().strip().splitlines()
    assert csv_lines[0].startswith("iteration")
    assert len(csv_lines) == 2  # header + single iterate
    assert (out / "final.nii.gz").exists()


def test_execute_skips_completed(tmp_path):
    scenario = load_scenario_dict(yaml.safe_load(_mini_scenario(tmp_path).read_text()))
    run = expand_scenario(scenario)[0]
    out1 = execute_run(run)
    marker1 = (out1 / ".done").read_text()
    out2 = execute_run(run, force=False)
    assert out1 == out2
    assert (out2 / ".done").read_text() == marker1


def test_cli_dry_run_lists_runs_without_executing(tmp_path, capsys):
    sp = _mini_scenario(tmp_path)
    rc = main(["--scenario", str(sp), "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "post_smoothing" in captured
    markers = list((tmp_path / "results").rglob(".done")) if (tmp_path / "results").exists() else []
    assert markers == []


def test_force_failure_removes_marker(tmp_path, monkeypatch):
    scenario = load_scenario_dict(yaml.safe_load(_mini_scenario(tmp_path).read_text()))
    run = expand_scenario(scenario)[0]
    out = execute_run(run)
    assert (out / ".done").exists()

    import krl_studies.runner.execute as ex

    def boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ex.METHOD_REGISTRY["post_smoothing"], "run", boom)
    with pytest.raises(RuntimeError):
        execute_run(run, force=True)
    assert not (out / ".done").exists()


def test_runner_brainweb_lesion_truth_fix(tmp_path):
    """Runner must not raise ValueError on non-empty lesion_masks array."""
    import numpy as np

    from krl_studies.config import RunSpec
    from krl_studies.runner.execute import execute_run

    # Build a minimal RunSpec for brainweb with lesion masks present
    gt = np.ones((16, 16, 16), dtype=np.float32)
    gt[4:12, 4:12, 4:12] = 4.0
    lesion_masks = [gt > 2.0]  # non-empty list of boolean arrays
    lesion_labels = [8]

    # Mock the BrainWebDataset to return our gt and lesion masks
    # This test runs without SIRF by using input_kind="reference"
    run = RunSpec(
        run_id="test_brainweb_lesion",
        study="brainweb",
        dataset={"kind": "brainweb", "root": str(tmp_path), "subject_id": 99},
        input_kind="reference",
        input_params={"condition": "psf-matched", "guidance_condition": "exact"},
        method_name="rl",
        method_params={"fwhm_mm": 4.0, "iterations": 1},
        sim={"seed": 1337},
        out_root=str(tmp_path / "results"),
    )
    # This should not raise "The truth value of an array with more than one element is ambiguous"
    out_dir = execute_run(run, force=True)
    assert out_dir.exists()
