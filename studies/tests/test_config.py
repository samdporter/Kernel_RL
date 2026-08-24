import pytest

from krl_studies.config import expand_scenario, load_scenario_dict

SCENARIO = {
    "study": "spheres",
    "dataset": {"kind": "spheres", "root": "data/spheres"},
    "inputs": [
        {"kind": "preblurred"},
        {"kind": "quick_sim", "params": {"fwhm_mm": [5.0], "counts": [1e5], "realisation": [0, 1]}},
    ],
    "methods": [
        {"name": "post_smoothing", "params": {"sigma_mm": [2.0]}},
        {"name": "rl", "params": {"fwhm_mm": 5.0, "iterations": 10}},
    ],
    "output": "results/test",
}


def test_load_scenario_returns_defaults():
    sc = load_scenario_dict(SCENARIO)
    assert sc.study == "spheres"
    assert len(sc.inputs) == 2
    assert sc.methods[1].name == "rl"


def test_expand_produces_cartesian_product_with_slugs():
    sc = load_scenario_dict(SCENARIO)
    runs = expand_scenario(sc)
    kinds = {(r.input_kind, r.method_name) for r in runs}
    assert ("quick_sim", "rl") in kinds
    # preblurred(1 grid) + quick_sim(2 realisations) = 3 input-grids x 2 methods = 6 runs
    assert len(runs) == 6


def test_expand_respects_scalar_and_grid_params():
    sc = load_scenario_dict(SCENARIO)
    runs = expand_scenario(sc)
    rl_quick = [r for r in runs if r.method_name == "rl" and r.input_kind == "quick_sim"]
    assert all(r.method_params["fwhm_mm"] == 5.0 for r in rl_quick)
    assert {r.input_params["realisation"] for r in rl_quick} == {0, 1}


def test_slug_is_filesystem_safe_and_deterministic():
    sc = load_scenario_dict(SCENARIO)
    runs = expand_scenario(sc)
    slugs = [r.run_id for r in runs]
    assert len(slugs) == len(set(slugs))
    assert all(" " not in s and ".." not in s for s in slugs)
    again = expand_scenario(load_scenario_dict(SCENARIO))
    assert [r.run_id for r in again] == slugs


def test_missing_required_keys_raise():
    with pytest.raises(KeyError):
        load_scenario_dict({"study": "spheres"})


def test_near_identical_float_params_produce_distinct_runs():
    scenario = dict(SCENARIO)
    scenario["methods"] = [
        {"name": "post_smoothing", "params": {"sigma_mm": [2.00001, 2.00002]}},
    ]
    scenario["inputs"] = [{"kind": "preblurred"}]
    runs = expand_scenario(load_scenario_dict(scenario))
    assert len(runs) == 2
    assert len({r.run_id for r in runs}) == 2
