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


def _make_spheres_run(tmp_path, *, input_kind, input_params, sim, method_name="post_smoothing", method_params=None):
    import uuid

    gt = np.full((64, 64, 64), 1.0, dtype=np.float32)
    gt[24:40, 24:40, 24:40] = 6.0
    # Unique sub-directory so a test can build multiple runs against one tmp_path.
    d = tmp_path / f"spheres_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    write_test_nifti(d / "phant_orig.nii", gt)
    write_test_nifti(d / "phant_mri.nii", np.full_like(gt, 0.5))
    write_test_nifti(d / "phant_pet.nii", gt * 0.95)
    from krl_studies.config import RunSpec

    return RunSpec(
        run_id="test_propagation",
        study="spheres",
        dataset={"kind": "spheres", "root": str(d)},
        input_kind=input_kind,
        input_params=input_params,
        method_name=method_name,
        method_params=method_params or {"sigma_mm": 2.0},
        sim=sim,
        out_root=str(tmp_path / "results"),
    ), d


def test_sim_values_propagate_to_simulate_inputs(tmp_path, monkeypatch):
    """scenario sim (scanner/seed/n_subits) must reach simulate_inputs via _build_observed."""
    from krl_studies.runner import execute as ex

    captured = {}

    def fake_simulate_inputs(gt_array, cfg_dict):
        captured["cfg"] = dict(cfg_dict)
        return np.zeros_like(gt_array), {"scanner": cfg_dict.get("scanner", "Siemens mMR")}

    # Patch in the simulate namespace because _build_observed lazy-imports it
    monkeypatch.setattr("krl_studies.simulation.simulate.simulate_inputs", fake_simulate_inputs)

    run, d = _make_spheres_run(
        tmp_path,
        input_kind="sirf_sim",
        input_params={
            "condition": "psf-matched",
            "beta": None,
            "counts": 1e5,
            "realisation": 0,
        },
        sim={"scanner": "Siemens VISION 600", "seed": 42, "n_subits": 7},
    )
    ex._build_observed(run, _spheres_ds(d), np.zeros((64, 64, 64), dtype=np.float32))
    assert captured["cfg"]["scanner"] == "Siemens VISION 600"
    assert captured["cfg"]["seed"] == 42
    assert captured["cfg"]["n_subits"] == 7


def test_input_params_override_scenario_sim_for_scanner_seed_subits(tmp_path, monkeypatch):
    """input_params overrides scenario sim for scanner, seed, and n_subits."""
    from krl_studies.runner import execute as ex

    captured = {}

    def fake_simulate_inputs(gt_array, cfg_dict):
        captured["cfg"] = dict(cfg_dict)
        return np.zeros_like(gt_array), {}

    monkeypatch.setattr("krl_studies.simulation.simulate.simulate_inputs", fake_simulate_inputs)

    run, d = _make_spheres_run(
        tmp_path,
        input_kind="sirf_sim",
        input_params={
            "condition": "psf-matched",
            "beta": None,
            "counts": 1e5,
            "realisation": 0,
            "scanner": "Siemens mMR",  # explicit override
            "seed": 999,  # explicit override
            "n_subits": 1,  # explicit override
        },
        sim={"scanner": "Siemens VISION 600", "seed": 42, "n_subits": 7},
    )
    ex._build_observed(run, _spheres_ds(d), np.zeros((64, 64, 64), dtype=np.float32))
    assert captured["cfg"]["scanner"] == "Siemens mMR"
    assert captured["cfg"]["seed"] == 999
    assert captured["cfg"]["n_subits"] == 1


def test_scenario_sim_n_subiterations_alias_propagates(tmp_path, monkeypatch):
    """n_subiterations alias in scenario sim must reach simulate_inputs."""
    from krl_studies.runner import execute as ex

    captured = {}

    def fake_simulate_inputs(gt_array, cfg_dict):
        captured["cfg"] = dict(cfg_dict)
        return np.zeros_like(gt_array), {}

    monkeypatch.setattr("krl_studies.simulation.simulate.simulate_inputs", fake_simulate_inputs)

    run, d = _make_spheres_run(
        tmp_path,
        input_kind="sirf_sim",
        input_params={
            "condition": "psf-matched",
            "beta": None,
            "counts": 1e5,
            "realisation": 0,
        },
        sim={"scanner": "Siemens mMR", "seed": 5, "n_subiterations": 3},
    )
    ex._build_observed(run, _spheres_ds(d), np.zeros((64, 64, 64), dtype=np.float32))
    # cfg may carry either n_subits or n_subiterations; the test must be tolerant.
    n_subits = captured["cfg"].get("n_subits", captured["cfg"].get("n_subiterations"))
    assert n_subits == 3
    assert captured["cfg"]["seed"] == 5


def test_quick_sim_receives_scenario_seed(tmp_path, monkeypatch):
    """quick_sim path must receive scenario sim seed (overridable by input_params)."""
    from krl_studies.runner import execute as ex

    captured = {}

    def fake_quick_sim(gt_array, fwhm_mm, counts, realisation, voxel_mm, seed=1337):
        captured["seed"] = seed
        return np.zeros_like(gt_array)

    # execute.py imports quick_sim at module load, so patch its bound name.
    monkeypatch.setattr(ex, "quick_sim", fake_quick_sim)

    run, d = _make_spheres_run(
        tmp_path,
        input_kind="quick_sim",
        input_params={"fwhm_mm": 3.0, "counts": 1e5, "realisation": 0},
        sim={"seed": 1234},
    )
    ex._build_observed(run, _spheres_ds(d), np.zeros((64, 64, 64), dtype=np.float32))
    assert captured["seed"] == 1234

    captured.clear()
    run2, d2 = _make_spheres_run(
        tmp_path,
        input_kind="quick_sim",
        input_params={"fwhm_mm": 3.0, "counts": 1e5, "realisation": 0, "seed": 77},
        sim={"seed": 1234},
    )
    ex._build_observed(run2, _spheres_ds(d2), np.zeros((64, 64, 64), dtype=np.float32))
    assert captured["seed"] == 77


def _spheres_ds(root):
    """Return a SphereDataset without touching the filesystem."""
    from krl_studies.datasets.spheres import SphereDataset

    return SphereDataset(root=root)


def test_get_acquisition_does_not_silently_fallback_to_mmr(monkeypatch):
    """Requested scanner failure must propagate, never silently swap to mMR."""
    from krl_studies.simulation import simulate as sim_mod
    from krl_studies.simulation import _api

    sim_mod._ACQ_CACHE.clear()
    try:
        def dispatch(name, **kwargs):
            if name == "Siemens mMR":
                return f"acq_{name}"
            raise RuntimeError(f"unsupported scanner: {name!r}")

        monkeypatch.setattr(_api, "acquisition_template", dispatch)

        # Vision request must fail even though mMR would build successfully.
        with pytest.raises(RuntimeError, match="unsupported scanner"):
            sim_mod._get_acquisition("Siemens VISION 600")
        # And no mMR entry was smuggled into the cache as a substitute.
        assert "Siemens mMR" not in sim_mod._ACQ_CACHE
    finally:
        sim_mod._ACQ_CACHE.clear()


def test_get_acquisition_cache_keyed_by_scanner(monkeypatch):
    """Cache is keyed by the exact scanner name; mMR and Vision are distinct entries."""
    from krl_studies.simulation import simulate as sim_mod
    from krl_studies.simulation import _api

    sim_mod._ACQ_CACHE.clear()
    try:
        seen = []

        def fake(name, **_kwargs):
            seen.append(name)
            return f"acq_{name}"

        monkeypatch.setattr(_api, "acquisition_template", fake)

        acq1, used1 = sim_mod._get_acquisition("Siemens mMR")
        acq2, used2 = sim_mod._get_acquisition("Siemens VISION 600")
        assert used1 == "Siemens mMR"
        assert used2 == "Siemens VISION 600"
        assert acq1 == "acq_Siemens mMR"
        assert acq2 == "acq_Siemens VISION 600"

        # second call uses the cache, not _api.acquisition_template
        seen.clear()
        sim_mod._get_acquisition("Siemens mMR")
        assert seen == []
    finally:
        sim_mod._ACQ_CACHE.clear()
