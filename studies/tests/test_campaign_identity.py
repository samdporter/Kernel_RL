"""Campaign configuration and canonical identity contracts (Task T2)."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from krl_studies.config import expand_scenario, load_scenario_dict
from krl_studies.identity import (
    METHOD_DEFAULTS,
    OsemConfig,
    forward_id,
    identity_config,
    input_id,
    resolve_guidance,
    resolve_method_params,
    run_id,
)
from krl_studies.runner import plan as plan_mod
from krl_studies.runner.cli import main as run_main
from krl_studies.runner.plan import (
    INDEX_SCHEMA_VERSION,
    PLAN_VERSION,
    SCHEMA_VERSION,
    count_run_plan,
    read_run_plan,
    read_run_plan_index,
    write_run_plan,
)

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _forward(**overrides):
    forward = {
        "truth_fwhm_mm": 5.0,
        "scanner": "Siemens mMR",
        "projection_geometry": {},
        "image_geometry": {},
    }
    forward.update(overrides)
    return forward


def _base_scenario(**overrides):
    scenario = {
        "study": "spheres",
        "dataset": {"kind": "spheres", "root": "data/spheres"},
        "subjects": [0],
        "forward": _forward(),
        "inputs": [
            {
                "kind": "quick_sim",
                "params": {
                    "counts": 1.0e5,
                    "realisation": 0,
                    "seed": 1,
                    "osem": {"subsets": 7, "full_iterations": 2},
                },
            }
        ],
        "methods": [{"name": "rl", "params": {"fwhm_mm": 5.0, "iterations": 3}}],
        "output": "results/test_contract",
    }
    scenario.update(overrides)
    return scenario


def _expand(scenario):
    return expand_scenario(load_scenario_dict(scenario))


def _only(runs):
    assert len(runs) == 1, f"expected one run, got {len(runs)}"
    return runs[0]


# --- identities ------------------------------------------------------------


def test_ids_are_deterministic_and_unique_across_subjects():
    runs_a = _expand(_base_scenario(subjects=[0]))
    runs_b = _expand(_base_scenario(subjects=[1]))
    again = _expand(_base_scenario(subjects=[0]))

    assert runs_a[0].forward_id == again[0].forward_id
    assert runs_a[0].input_id == again[0].input_id
    assert runs_a[0].run_id == again[0].run_id
    assert runs_a[0].forward_id != runs_b[0].forward_id
    assert runs_a[0].input_id != runs_b[0].input_id
    assert runs_a[0].run_id != runs_b[0].run_id


def test_identical_inputs_share_input_id_across_guidance_and_methods():
    scenario = _base_scenario(
        inputs=[
            {
                "kind": "quick_sim",
                "params": {
                    "counts": 1.0e5,
                    "realisation": 0,
                    "seed": 1,
                    "osem": {"subsets": 7, "full_iterations": 2},
                    "guidance_modality": ["t1", "t2"],
                },
            }
        ],
        methods=[
            {"name": "rl", "params": {"fwhm_mm": 5.0, "iterations": 3}},
            {"name": "krl", "params": {"fwhm_mm": 5.0, "iterations": 3, "sigma_anat": 0.2}},
        ],
    )
    runs = _expand(scenario)
    assert len(runs) == 4
    assert len({r.input_id for r in runs}) == 1
    assert len({r.forward_id for r in runs}) == 1
    assert len({r.run_id for r in runs}) == 4
    assert {r.guidance_modality for r in runs} == {"t1", "t2"}


def test_forward_id_ignores_counts_osem_guidance_and_method():
    base = _only(_expand(_base_scenario()))
    varied = _expand(
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {
                        "counts": 9.0e7,
                        "realisation": 3,
                        "seed": 99,
                        "osem": {"subsets": 14, "full_iterations": 4},
                        "guidance_modality": "t2",
                        "guidance_condition": "shift_p5",
                    },
                }
            ],
            methods=[{"name": "dtv", "params": {"fwhm_mm": 7.5, "iterations": 5}}],
        )
    )[0]

    assert varied.forward_id == base.forward_id
    assert varied.input_id != base.input_id
    assert varied.run_id != base.run_id


def test_truth_psf_counts_and_osem_separate_input_ids():
    base = _only(_expand(_base_scenario()))

    truth = _only(_expand(_base_scenario(forward=_forward(truth_fwhm_mm=6.5))))
    counts = _only(
        _expand(
            _base_scenario(
                inputs=[
                    {
                        "kind": "quick_sim",
                        "params": {
                            "counts": 2.0e5,
                            "realisation": 0,
                            "seed": 1,
                            "osem": {"subsets": 7, "full_iterations": 2},
                        },
                    }
                ]
            )
        )
    )
    osem = _only(
        _expand(
            _base_scenario(
                inputs=[
                    {
                        "kind": "quick_sim",
                        "params": {
                            "counts": 1.0e5,
                            "realisation": 0,
                            "seed": 1,
                            "osem": {"subsets": 7, "full_iterations": 5},
                        },
                    }
                ]
            )
        )
    )

    for changed in (truth, counts, osem):
        assert changed.input_id != base.input_id

    assert truth.forward_id != base.forward_id
    assert counts.forward_id == base.forward_id
    assert osem.forward_id == base.forward_id


def test_guidance_shift_never_touches_forward_or_attenuation():
    scenario = _base_scenario(
        study="brainweb",
        dataset={"kind": "brainweb", "root": "data/brainweb", "subject_id": 4},
        forward=_forward(attenuation_path="data/brainweb/subject_04/mu_map.nii.gz"),
        inputs=[
            {
                "kind": "sirf_sim",
                "params": {
                    "counts": 1.0e7,
                    "realisation": 0,
                    "seed": 1,
                    "osem": {"subsets": 7, "full_iterations": 2},
                    "guidance_condition": ["exact", "shift_p2", "shift_m2"],
                },
            }
        ],
    )
    runs = _expand(scenario)
    assert len(runs) == 3
    assert len({r.forward_id for r in runs}) == 1
    assert len({r.input_id for r in runs}) == 1
    assert len({r.run_id for r in runs}) == 3
    checksums = {identity_config(r).attenuation_checksum for r in runs}
    assert len(checksums) == 1


# --- legacy mapping --------------------------------------------------------


def test_resolve_guidance_lifts_legacy_conditions():
    assert resolve_guidance({}) == ("t1", "exact")
    assert resolve_guidance({"guidance_condition": "exact"}) == ("t1", "exact")
    assert resolve_guidance({"guidance_condition": "t2"}) == ("t2", "exact")
    assert resolve_guidance({"guidance_condition": "shift_p2"}) == ("t1", "shift_p2")
    assert resolve_guidance({"guidance_modality": "umap"}) == ("umap", "exact")
    assert resolve_guidance({"guidance_modality": "t2", "guidance_condition": "shift_m5"}) == (
        "t2",
        "shift_m5",
    )


def test_existing_scenarios_load_and_expand():
    for path in sorted(SCENARIOS_DIR.glob("*.yaml")):
        scenario = load_scenario_dict(yaml.safe_load(path.read_text()))
        assert scenario.protocol_version
        assert scenario.stage in ("development", "evaluation")
        runs = expand_scenario(scenario)
        assert runs, f"{path.name} expanded to no runs"
        first = runs[0]
        assert first.forward_id and first.input_id and first.run_id
        assert first.guidance_modality in ("t1", "t2", "umap")
        assert first.guidance_condition in ("exact", "shift_p2", "shift_m2", "shift_p5", "shift_m5")


def test_legacy_brainweb_scenario_freezes_subject_and_shift_mapping():
    scenario = load_scenario_dict(
        yaml.safe_load((SCENARIOS_DIR / "brainweb_mismatch.yaml").read_text())
    )
    assert scenario.subjects == (4,)
    runs = expand_scenario(scenario)
    shift_runs = [r for r in runs if r.guidance_condition == "shift_p2"]
    assert shift_runs
    assert all(r.guidance_modality == "t1" for r in shift_runs)
    t2_runs = [r for r in runs if r.guidance_modality == "t2"]
    assert t2_runs and all(r.guidance_condition == "exact" for r in t2_runs)


# --- malformed settings ----------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    [
        _base_scenario(forward={"scanner": "Siemens mMR", "projection_geometry": {}, "image_geometry": {}}),
        _base_scenario(forward=_forward(truth_fwhm_mm=[5.0, 6.0])),
        _base_scenario(forward=_forward(truth_fwhm_mm=0.0)),
        _base_scenario(forward=_forward(truth_fwhm_mm=float("nan"))),
        _base_scenario(forward=_forward(truth_fwhm_mm=float("inf"))),
        _base_scenario(forward=_forward(projection_geometry=[1, 2])),
        _base_scenario(
            forward={"truth_fwhm_mm": 5.0, "scanner": "Siemens mMR", "image_geometry": {}}
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {"counts": 1e5, "osem": {"subsets": 7}},
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {
                        "counts": 1e5,
                        "osem": {"subsets": 7, "full_iterations": 2, "subiterations": 9},
                    },
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {"counts": "not-a-number"},
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {"counts": -1.0},
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {"counts": None},
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {"counts": 0.0},
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {"counts": float("nan")},
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {"counts": float("inf")},
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {"counts": float("-inf")},
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {"counts": 1e5, "guidance_modality": "pet"},
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {"counts": 1e5, "guidance_condition": "shift_p3"},
                }
            ]
        ),
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {
                        "counts": 1e5,
                        "guidance_modality": "t1",
                        "guidance_condition": "t2",
                    },
                }
            ]
        ),
    ],
)
def test_malformed_settings_raise_explicitly(scenario):
    with pytest.raises(ValueError):
        _expand(scenario)


def test_malformed_scenario_header_raises():
    with pytest.raises(ValueError):
        load_scenario_dict(_base_scenario(stage="production"))
    with pytest.raises(ValueError):
        load_scenario_dict(_base_scenario(output_grid=[1, 2]))
    with pytest.raises(ValueError):
        load_scenario_dict(_base_scenario(subjects=[1, 1]))


# --- plan round-trip and random access -------------------------------------


def test_plan_roundtrip_preserves_identity(tmp_path):
    runs = _expand(
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {
                        "counts": [1.0e5, 2.0e5],
                        "realisation": 0,
                        "seed": 1,
                        "osem": {"subsets": 7, "full_iterations": 2},
                    },
                }
            ],
            methods=[
                {"name": "rl", "params": {"fwhm_mm": 5.0, "iterations": 3}},
                {"name": "krl", "params": {"fwhm_mm": 5.0, "iterations": 3, "sigma_anat": 0.2}},
            ],
        )
    )
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)

    header = json.loads(path.read_text().splitlines()[0])
    assert header == {"plan_version": PLAN_VERSION, "schema_version": SCHEMA_VERSION}

    loaded = read_run_plan(path)
    assert len(loaded) == len(runs)
    for original, restored in zip(runs, loaded):
        cfg = identity_config(restored)
        assert restored.run_id == original.run_id
        assert restored.forward_id == forward_id(cfg) == original.forward_id
        assert restored.input_id == input_id(cfg) == original.input_id
        assert restored.run_id == run_id(cfg)


def test_plan_random_access_matches_full_load(tmp_path, monkeypatch):
    runs = _expand(
        _base_scenario(
            inputs=[
                {
                    "kind": "quick_sim",
                    "params": {
                        "counts": [1.0e5, 2.0e5, 3.0e5],
                        "realisation": [0, 1],
                        "seed": 1,
                        "osem": {"subsets": 7, "full_iterations": 2},
                    },
                }
            ]
        )
    )
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)
    assert count_run_plan(path) == len(runs)

    calls = {"n": 0}
    real_parse = plan_mod._parse_row

    def counting_parse(line, line_number):
        calls["n"] += 1
        return real_parse(line, line_number)

    monkeypatch.setattr(plan_mod, "_parse_row", counting_parse)
    target = read_run_plan_index(path, 4)
    assert calls["n"] == 1, "indexed read must parse exactly one row"

    full = read_run_plan(path)
    assert target.run_id == full[3].run_id
    assert target.input_id == full[3].input_id

    with pytest.raises(IndexError):
        read_run_plan_index(path, len(runs) + 1)


def test_dry_run_needs_no_sirf_or_brainweb():
    sys.modules.pop("sirf", None)
    path = SCENARIOS_DIR / "brainweb_mismatch.yaml"
    rc = run_main(["--scenario", str(path), "--dry-run"])
    assert rc == 0
    assert "sirf" not in sys.modules
    assert "brainweb" not in sys.modules


def test_equivalent_numeric_representations_share_input_id():
    def with_counts(counts):
        return _only(
            _expand(
                _base_scenario(
                    inputs=[
                        {
                            "kind": "quick_sim",
                            "params": {
                                "counts": counts,
                                "realisation": 0,
                                "seed": 1,
                                "osem": {"subsets": 7, "full_iterations": 2},
                            },
                        }
                    ]
                )
            )
        )

    runs = [with_counts(v) for v in ("5.0e7", "5e7", 50000000.0, 50000000)]
    assert len({r.input_id for r in runs}) == 1


def test_osem_dataclass_identity_defaults_are_canonical():
    cfg = identity_config(_only(_expand(_base_scenario())))
    assert isinstance(cfg.osem, OsemConfig)
    assert cfg.osem.subsets == 7
    assert cfg.osem.full_iterations == 2
    assert cfg.osem.subiterations == 14
    assert cfg.osem.initialisation == "ones"


# --- effective method parameters -------------------------------------------


def test_effective_method_params_include_registry_defaults():
    effective, iterations = resolve_method_params("rl", {"fwhm_mm": 5.0, "iterations": 3})
    assert effective["backend"] == METHOD_DEFAULTS["rl"]["backend"]
    assert effective["epsilon"] == METHOD_DEFAULTS["rl"]["epsilon"]
    assert "iterations" not in effective
    assert iterations == 3

    overridden, _ = resolve_method_params("rl", {"epsilon": 1e-6})
    assert overridden["epsilon"] == 1e-6

    dtv, _ = resolve_method_params("dtv", {"alpha": 0.1})
    assert dtv["lbfgs_ftol"] == 1e-6
    assert dtv["lbfgs_gtol"] == 1e-6
    assert dtv["lbfgs_max_linesearch"] == 20

    iy, _ = resolve_method_params("iy", {})
    assert iy["damping"] == 1.0
    assert iy["fwhm_mm"] == 5.0
    assert len(iy["psf_sigma_vox"]) == 3
    assert iy["psf_sigma_vox"][0] == pytest.approx(5.0 / (2.0 * math.sqrt(2.0 * math.log(2.0))))


def test_kernel_defaults_are_effective_for_krl_and_hkrl():
    krl, _ = resolve_method_params("krl", {"fwhm_mm": 5.0})
    for key, value in METHOD_DEFAULTS["krl"].items():
        assert krl[key] == value

    hkrl, _ = resolve_method_params("hkrl", {"fwhm_mm": 5.0})
    assert hkrl["num_neighbours"] == 5
    assert hkrl["sigma_anat"] == 0.1
    assert hkrl["hybrid"] is False

    hybrid, _ = resolve_method_params("hkrl", {"fwhm_mm": 5.0, "sigma_emission": 0.5})
    assert hybrid["hybrid"] is True


def test_changing_kernel_default_changes_run_id(monkeypatch):
    # The default must be part of the effective params to begin with.
    assert resolve_method_params("krl", {"fwhm_mm": 5.0})[0]["num_neighbours"] == 5

    run = _only(
        _expand(
            _base_scenario(
                methods=[{"name": "krl", "params": {"fwhm_mm": 5.0, "sigma_anat": 0.2, "iterations": 3}}]
            )
        )
    )
    original_run = run.run_id
    original_input = run.input_id

    monkeypatch.setitem(METHOD_DEFAULTS["krl"], "num_neighbours", 7)

    cfg = identity_config(run)
    assert run_id(cfg) != original_run
    assert input_id(cfg) == original_input


def test_explicit_default_equals_omitted_for_run_id():
    omitted = _only(
        _expand(_base_scenario(methods=[{"name": "rl", "params": {"fwhm_mm": 5.0, "iterations": 3}}]))
    )
    explicit = _only(
        _expand(
            _base_scenario(
                methods=[
                    {
                        "name": "rl",
                        "params": {
                            "fwhm_mm": 5.0,
                            "iterations": 3,
                            "epsilon": METHOD_DEFAULTS["rl"]["epsilon"],
                            "backend": METHOD_DEFAULTS["rl"]["backend"],
                        },
                    }
                ]
            )
        )
    )
    assert omitted.run_id == explicit.run_id


def test_changing_runtime_default_changes_run_id(monkeypatch):
    run = _only(_expand(_base_scenario()))
    original_run = run.run_id
    original_input = run.input_id

    monkeypatch.setitem(METHOD_DEFAULTS["rl"], "epsilon", 1e-7)

    cfg = identity_config(run)
    assert run_id(cfg) != original_run
    assert input_id(cfg) == original_input


def test_assumed_psf_only_does_not_change_input_id():
    lower = _only(
        _expand(_base_scenario(methods=[{"name": "rl", "params": {"fwhm_mm": 5.0, "iterations": 3}}]))
    )
    higher = _only(
        _expand(_base_scenario(methods=[{"name": "rl", "params": {"fwhm_mm": 9.0, "iterations": 3}}]))
    )
    assert lower.input_id == higher.input_id
    assert lower.forward_id == higher.forward_id
    assert lower.run_id != higher.run_id


def test_legacy_truth_fwhm_prefers_input_over_sim():
    scenario = _base_scenario(
        forward=None,
        sim={"fwhm_mm": 7.0, "counts": 1.0e5},
        inputs=[
            {
                "kind": "quick_sim",
                "params": {"fwhm_mm": 4.0, "counts": 1.0e5, "realisation": 0, "seed": 1},
            }
        ],
        methods=[{"name": "post_smoothing", "params": {"sigma_mm": 2.0}}],
    )
    run = _only(_expand(scenario))
    assert run.forward is not None
    assert run.forward.truth_fwhm_mm == 4.0
    assert identity_config(run).forward.truth_fwhm_mm == 4.0


# --- sidecar offset index --------------------------------------------------


def _multi_run_scenario():
    return _base_scenario(
        inputs=[
            {
                "kind": "quick_sim",
                "params": {
                    "counts": [1.0e5, 2.0e5, 3.0e5],
                    "realisation": 0,
                    "seed": 1,
                    "osem": {"subsets": 7, "full_iterations": 2},
                },
            }
        ]
    )


def test_write_run_plan_writes_sidecar_index(tmp_path):
    runs = _expand(_multi_run_scenario())
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)

    index_path = plan_mod._index_path(path)
    assert index_path.exists()
    index = json.loads(index_path.read_text())
    assert index["index_schema"] == INDEX_SCHEMA_VERSION
    assert index["row_count"] == len(runs)
    assert index["plan_size"] == path.stat().st_size
    assert len(index["offsets"]) == len(runs)


def test_indexed_read_uses_sidecar_without_rescan(tmp_path, monkeypatch):
    runs = _expand(_multi_run_scenario())
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)

    def boom(*_args, **_kwargs):
        raise AssertionError("sidecar-backed access must not scan the plan")

    # Patch both streaming and rebuild paths: a valid sidecar must avoid either.
    monkeypatch.setattr(plan_mod, "_iter_rows", boom)
    monkeypatch.setattr(plan_mod, "_build_index", boom)
    assert count_run_plan(path) == len(runs)
    target = read_run_plan_index(path, 2)
    assert target.run_id == runs[1].run_id
    assert target.input_id == runs[1].input_id


def test_stale_index_rejected_on_same_size_rewrite(tmp_path):
    runs = _expand(_multi_run_scenario())
    assert len(runs) == 3
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)
    original = path.read_bytes()
    stat = path.stat()

    # Grow row 1 and shrink row 3 by the same amount so the plan keeps its size
    # but the stale offsets for later rows point into the wrong place.
    lines = original.split(b"\n")
    lines[1] = lines[1].replace(b'", "', b'",  "', 1)
    lines[3] = lines[3].replace(b'", "', b'","', 1)
    assert len(lines[1]) == len(original.split(b"\n")[1]) + 1
    path.write_bytes(b"\n".join(lines))
    # Preserve mtime too: only the row digests can detect the rewrite.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert path.stat().st_mtime_ns == stat.st_mtime_ns

    target = read_run_plan_index(path, 2)
    assert target.run_id == runs[1].run_id
    assert count_run_plan(path) == len(runs)


def test_indexed_read_falls_back_without_sidecar(tmp_path):
    runs = _expand(_multi_run_scenario())
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)
    plan_mod._index_path(path).unlink()

    target = read_run_plan_index(path, 3)
    assert target.run_id == runs[2].run_id
    assert plan_mod._index_path(path).exists(), "missing sidecar must be rebuilt"
    assert count_run_plan(path) == len(runs)


def test_v1_plan_indexed_read(tmp_path):
    row = {
        "run_id": "legacy_run",
        "study": "spheres",
        "dataset": {"kind": "spheres", "root": "data/spheres"},
        "input_kind": "reference",
        "input_params": {},
        "method_name": "post_smoothing",
        "method_params": {"sigma_mm": 2.0},
        "sim": {"fwhm_mm": 3.0},
        "out_root": "results/legacy",
    }
    path = tmp_path / "v1.jsonl"
    path.write_text(
        json.dumps({"plan_version": 1}) + "\n" + json.dumps(row, sort_keys=True) + "\n"
    )

    target = read_run_plan_index(path, 1)
    assert target.run_id == "legacy_run"
    assert target.study == "spheres"
    assert count_run_plan(path) == 1


def test_dry_run_works_with_cil_unimportable():
    code = (
        "import builtins, sys\n"
        "real_import = builtins.__import__\n"
        "def guard(name, *args, **kwargs):\n"
        "    if name == 'cil' or name.startswith('cil.'):\n"
        "        raise ImportError('cil blocked for dry-run test')\n"
        "    return real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guard\n"
        "from krl_studies.runner.cli import main\n"
        "rc = main(['--scenario', 'studies/scenarios/spheres_core.yaml', '--dry-run'])\n"
        "assert rc == 0, rc\n"
        "assert 'cil' not in sys.modules, sorted(m for m in sys.modules if m.startswith('cil'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
