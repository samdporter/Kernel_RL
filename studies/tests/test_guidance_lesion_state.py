"""Guided T1 lesion-state plumbing: asset selection and identity placement (T3)."""

import hashlib
import json

import pytest

from krl_studies.config import expand_scenario, load_scenario_dict
from krl_studies.datasets.brainweb import GUIDANCE_T1_ABSENT, GUIDANCE_T1_PRESENT
from krl_studies.identity import forward_id, identity_config, input_id, run_id, run_identity


def _scenario(tmp_path, *, methods, state):
    return {
        "study": "brainweb",
        "dataset": {"kind": "brainweb", "root": str(tmp_path / "data"), "subject_id": 4},
        "subjects": [4],
        "forward": {
            "truth_fwhm_mm": 5.0,
            "scanner": "Siemens mMR",
            "projection_geometry": {},
            "image_geometry": {},
        },
        "inputs": [
            {
                "kind": "quick_sim",
                "params": {
                    "counts": 1.0e5,
                    "realisation": 0,
                    "seed": 1,
                    "osem": {"subsets": 7, "full_iterations": 2},
                    "guidance_modality": "t1",
                    "guidance_lesion_state": state,
                },
            }
        ],
        "methods": methods,
        "output": str(tmp_path / "results"),
    }


def _expand(scenario):
    return expand_scenario(load_scenario_dict(scenario))


def _only(runs):
    assert len(runs) == 1, f"expected one run, got {len(runs)}"
    return runs[0]


def _krl_method():
    return [{"name": "krl", "params": {"fwhm_mm": 5.0, "iterations": 2, "sigma_anat": 0.2}}]


def test_lesion_state_changes_guided_run_id_not_acquisition_ids(tmp_path):
    absent = _only(_expand(_scenario(tmp_path, methods=_krl_method(), state="absent")))
    present = _only(_expand(_scenario(tmp_path, methods=_krl_method(), state="present")))

    assert absent.guidance_lesion_state == "absent"
    assert present.guidance_lesion_state == "present"
    assert absent.forward_id == present.forward_id
    assert absent.input_id == present.input_id
    assert absent.run_id != present.run_id


def test_lesion_state_does_not_change_non_guided_run_id(tmp_path):
    methods = [{"name": "rl", "params": {"fwhm_mm": 5.0, "iterations": 2}}]
    absent = _only(_expand(_scenario(tmp_path, methods=methods, state="absent")))
    present = _only(_expand(_scenario(tmp_path, methods=methods, state="present")))

    assert absent.forward_id == present.forward_id
    assert absent.input_id == present.input_id
    assert absent.run_id == present.run_id


def test_lesion_state_grid_expands_to_distinct_guided_runs(tmp_path):
    runs = _expand(_scenario(tmp_path, methods=_krl_method(), state=["absent", "present"]))
    assert len(runs) == 2
    assert {r.guidance_lesion_state for r in runs} == {"absent", "present"}
    assert len({r.forward_id for r in runs}) == 1
    assert len({r.input_id for r in runs}) == 1
    assert len({r.run_id for r in runs}) == 2


def test_non_guided_identity_has_no_lesion_guidance_block(tmp_path):
    methods = [{"name": "post_smoothing", "params": {"sigma_mm": 2.0}}]
    run = _only(_expand(_scenario(tmp_path, methods=methods, state="present")))
    cfg = identity_config(run)
    assert cfg.guidance_lesion == {}
    assert "lesion" not in run_identity(cfg)["guidance"]


def test_guided_identity_carries_variant_checksum_and_union_hash(tmp_path):
    data = tmp_path / "data" / "subject_04"
    data.mkdir(parents=True)
    (data / GUIDANCE_T1_ABSENT).write_bytes(b"absent-bytes")
    (data / GUIDANCE_T1_PRESENT).write_bytes(b"present-bytes")
    (data / "preparation.json").write_text(
        json.dumps(
            {
                "guidance": {
                    "injection_multiplier": 4.0,
                    "injection_rule": "union x4",
                    "mask_union_hash": "union-hash",
                }
            }
        )
    )

    present = _only(_expand(_scenario(tmp_path, methods=_krl_method(), state="present")))
    cfg = identity_config(present)
    assert cfg.guidance_lesion["t1_checksum"] == hashlib.sha256(b"present-bytes").hexdigest()
    assert cfg.guidance_lesion["mask_union_hash"] == "union-hash"
    assert cfg.guidance_lesion["injection"]["multiplier"] == 4.0
    assert run_identity(cfg)["guidance"]["lesion"]["lesion_state"] == "present"
    # The variant checksum must not leak into acquisition identities.
    assert present.run_id == run_id(cfg)
    assert present.forward_id == forward_id(cfg)
    assert present.input_id == input_id(cfg)


def test_run_plan_round_trips_lesion_state(tmp_path):
    from krl_studies.runner.plan import read_run_plan, write_run_plan

    runs = _expand(_scenario(tmp_path, methods=_krl_method(), state="present"))
    path = tmp_path / "plan.jsonl"
    write_run_plan(runs, path)
    loaded = read_run_plan(path)
    assert len(loaded) == 1
    assert loaded[0].guidance_lesion_state == "present"
    assert loaded[0].run_id == runs[0].run_id
    assert loaded[0].input_id == runs[0].input_id


def test_guidance_does_not_change_acquisition_cache_id(tmp_path):
    from krl_studies.runner.cache import build_input_identity, compute_input_id

    def cache_id(**overrides):
        scenario = _scenario(tmp_path, methods=_krl_method(), state=overrides.pop("state", "absent"))
        scenario["inputs"][0]["params"].update(overrides)
        run = _only(_expand(scenario))
        return compute_input_id(build_input_identity(run)), run

    base_id, base_run = cache_id()
    present_id, present_run = cache_id(state="present")
    t2_id, _ = cache_id(guidance_modality="t2")
    shift_id, _ = cache_id(guidance_condition="shift_p2")

    # Guidance state/modality/condition must not change the acquisition cache.
    assert base_id == present_id == t2_id == shift_id
    # ...but the guided run identity still distinguishes the T1 variant.
    assert base_run.run_id != present_run.run_id


def test_guidance_t1_checksum_cache_invalidates_on_content_change(tmp_path):
    data = tmp_path / "data" / "subject_04"
    data.mkdir(parents=True)
    (data / GUIDANCE_T1_ABSENT).write_bytes(b"absent")
    present_path = data / GUIDANCE_T1_PRESENT
    present_path.write_bytes(b"present-v1")
    (data / "preparation.json").write_text(
        json.dumps(
            {
                "guidance": {
                    "injection_multiplier": 4.0,
                    "injection_rule": "union x4",
                    "mask_union_hash": "hash",
                }
            }
        )
    )

    run = _only(_expand(_scenario(tmp_path, methods=_krl_method(), state="present")))
    first = identity_config(run).guidance_lesion["t1_checksum"]
    assert first == hashlib.sha256(b"present-v1").hexdigest()

    # Replace the variant within the same process; the new checksum must be used.
    present_path.write_bytes(b"present-v2-longer")
    run2 = _only(_expand(_scenario(tmp_path, methods=_krl_method(), state="present")))
    second = identity_config(run2).guidance_lesion["t1_checksum"]
    assert second == hashlib.sha256(b"present-v2-longer").hexdigest()
    assert second != first


@pytest.mark.parametrize("bad", ["bogus", "", 3])
def test_invalid_lesion_state_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        _expand(_scenario(tmp_path, methods=_krl_method(), state=bad))
