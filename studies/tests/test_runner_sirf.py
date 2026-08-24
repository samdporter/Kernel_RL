from pathlib import Path

import numpy as np
import pytest
import yaml

try:
    import sirf.STIR  # noqa: F401

    HAS_SIRF = True
except ImportError:
    HAS_SIRF = False

from conftest import write_test_nifti

from krl_studies.config import load_scenario, load_scenario_dict
from krl_studies.runner.cli import main as cli_main
from krl_studies.runner.expand import expand_scenario


def test_sirf_sim_dry_run_expansion_arithmetic_native():
    """Native dry-run: 3 conditions * 3 betas * 2 realisations * N methods."""
    scenario = {
        "study": "spheres",
        "dataset": {"kind": "spheres", "root": "data/spheres"},
        "inputs": [
            {
                "kind": "sirf_sim",
                "params": {
                    "condition": ["psf-none", "psf-undersized", "psf-matched"],
                    "beta": [None, 10.0, 50.0],
                    "counts": [1.0e7],
                    "realisation": [0, 1],
                },
            }
        ],
        "methods": [
            {"name": "rl", "params": {"fwhm_mm": 5.5, "iterations": 5}},
            {"name": "krl", "params": {"fwhm_mm": 5.5, "iterations": 5, "num_neighbours": 9, "sigma_anat": 0.5}},
        ],
        "sim": {"seed": 0},
        "output": "results/spheres_sirf_test",
    }
    sc = load_scenario_dict(scenario)
    runs = expand_scenario(sc)
    # 3*3*1*2 = 18 input combos, 2 method combos => 36 runs
    assert len(runs) == 3 * 3 * 1 * 2 * 2
    # all input kinds are sirf_sim and params cover grid
    assert all(r.input_kind == "sirf_sim" for r in runs)
    assert {r.input_params["condition"] for r in runs} == {"psf-none", "psf-undersized", "psf-matched"}
    assert {r.input_params["beta"] for r in runs} == {None, 10.0, 50.0}


def test_spheres_sirf_yaml_expansion_native():
    """Load the committed spheres_sirf.yaml and verify expansion count."""
    yaml_path = Path("studies/scenarios/spheres_sirf.yaml")
    # When running from repo root, relative path works; when pytest runs from studies, try alternative
    if not yaml_path.exists():
        yaml_path = Path(__file__).parents[1] / "scenarios" / "spheres_sirf.yaml"
    assert yaml_path.exists(), f"missing {yaml_path}"
    sc = load_scenario(yaml_path)
    assert sc.inputs[0].kind == "sirf_sim"
    assert set(sc.inputs[0].params["condition"]) == {"psf-none", "psf-undersized", "psf-matched"}
    assert set(sc.inputs[0].params["beta"]) == {None, 10.0, 50.0}
    # YAML 1.0e7 may load as string '1.0e7' (PyYAML quirk) — compare as float
    assert [float(v) for v in sc.inputs[0].params["counts"]] == [1.0e7]
    assert set(sc.inputs[0].params["realisation"]) == {0, 1}
    runs = expand_scenario(sc)
    # input combos = 3*3*1*2 = 18
    from krl_studies.config import _grid  # internal, but stable

    input_combos = len(_grid(sc.inputs[0].params))
    assert input_combos == 18
    method_combos = sum(len(_grid(m.params)) for m in sc.methods)
    assert len(runs) == input_combos * method_combos
    assert len(runs) == 18 * method_combos


def test_cli_dry_run_sirf_sim_native(tmp_path, capsys):
    """CLI --dry-run for sirf_sim must work without SIRF."""
    scenario = {
        "study": "spheres",
        "dataset": {"kind": "spheres", "root": str(tmp_path / "spheres")},
        "inputs": [
            {
                "kind": "sirf_sim",
                "params": {
                    "condition": ["psf-none", "psf-matched"],
                    "beta": [None, 10.0],
                    "counts": [1e7],
                    "realisation": [0, 1],
                },
            }
        ],
        "methods": [{"name": "post_smoothing", "params": {"sigma_mm": 2.0}}],
        "sim": {"seed": 0},
        "output": str(tmp_path / "results"),
    }
    sp = tmp_path / "sirf_scen.yaml"
    sp.write_text(yaml.safe_dump(scenario))
    rc = cli_main(["--scenario", str(sp), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    # 2*2*2 = 8 runs expected
    assert "-- 8 run(s)" in out
    assert "sirf_sim" in out
    # no .done markers created
    markers = list((tmp_path / "results").rglob(".done")) if (tmp_path / "results").exists() else []
    assert markers == []


@pytest.mark.sirf
@pytest.mark.skipif(not HAS_SIRF, reason="SIRF not available")
def test_sirf_sim_end_to_end_tiny_metrics(tmp_path):
    """Container-marked tiny sirf_sim run: 64^3 fixture, 2 subiterations, tumours → crc columns."""
    from krl_studies.config import load_scenario_dict
    from krl_studies.runner.execute import execute_run
    from krl_studies.runner.expand import expand_scenario

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
        "inputs": [
            {
                "kind": "sirf_sim",
                "params": {
                    "condition": "psf-matched",
                    "beta": None,
                    "counts": 1e5,
                    "realisation": 0,
                    "n_subits": 2,
                },
            }
        ],
        "methods": [{"name": "post_smoothing", "params": {"sigma_mm": 2.0}}],
        "sim": {"seed": 0, "add_tumours": True},
        "output": str(tmp_path / "results"),
    }
    sc = load_scenario_dict(scenario)
    runs = expand_scenario(sc)
    assert len(runs) == 1
    out = execute_run(runs[0])
    assert (out / ".done").exists()
    assert (out / "manifest.json").exists()
    assert (out / "metrics.csv").exists()
    header = (out / "metrics.csv").read_text().splitlines()[0]
    # must have at least nrmse/bv, and with tumours should have crc
    assert "nrmse" in header
    assert "bv_percent" in header
    assert "crc_mm" in header
    # manifest preserves sirf_sim params
    import json

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["input_kind"] == "sirf_sim"
    assert manifest["input_params"]["condition"] == "psf-matched"
