"""Scenario configuration: YAML/dict parsing and sweep expansion.

Campaign identity fields (see :mod:`krl_studies.identity`) are resolved here.
Legacy YAMLs without a campaign block are lifted onto the new fields with the
documented mapping in :mod:`krl_studies.identity`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from krl_studies.identity import (
    DEFAULT_SELECTION_POLICY,
    DEFAULT_STAGE,
    LEGACY_PROTOCOL_VERSION,
    ForwardConfig,
    OsemConfig,
    all_identities,
    identity_config,
    identity_digest,
    resolve_forward,
    resolve_guidance,
    resolve_guidance_lesion_state,
    resolve_osem,
    validate_stage,
)

_REQUIRED_KEYS = ("study", "dataset", "inputs", "methods", "output")


@dataclass(frozen=True)
class InputSpec:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    study: str
    dataset: dict[str, Any]
    inputs: tuple[InputSpec, ...]
    methods: tuple[MethodSpec, ...]
    output: Path
    sim: dict[str, Any]
    raw: dict[str, Any]
    subjects: tuple[Any, ...] = (None,)
    forward_raw: dict[str, Any] | None = None
    osem_raw: dict[str, Any] | None = None
    output_grid: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = LEGACY_PROTOCOL_VERSION
    stage: str = DEFAULT_STAGE
    selection_policy: str = DEFAULT_SELECTION_POLICY


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    study: str
    dataset: dict[str, Any]
    input_kind: str
    input_params: dict[str, Any]
    method_name: str
    method_params: dict[str, Any]
    subject: Any = None
    guidance_modality: str = "t1"
    guidance_condition: str = "exact"
    guidance_lesion_state: str = "absent"
    forward: ForwardConfig | None = None
    osem: OsemConfig | None = None
    output_grid: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = LEGACY_PROTOCOL_VERSION
    stage: str = DEFAULT_STAGE
    selection_policy: str = DEFAULT_SELECTION_POLICY
    sim: dict[str, Any] = field(default_factory=dict)
    out_root: Path = Path("results")
    forward_id: str = ""
    input_id: str = ""


def _resolve_subjects(raw: dict[str, Any], dataset: dict[str, Any]) -> tuple[Any, ...]:
    subjects = raw.get("subjects")
    if subjects is not None:
        if not isinstance(subjects, (list, tuple)) or not subjects:
            raise ValueError("scenario 'subjects' must be a non-empty list")
        if len({str(s) for s in subjects}) != len(subjects):
            raise ValueError(f"duplicate subjects in frozen inventory: {list(subjects)!r}")
        return tuple(subjects)
    subject = dataset.get("subject_id", dataset.get("subject"))
    return (subject,) if subject is not None else (None,)


def load_scenario_dict(raw: dict[str, Any]) -> Scenario:
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise KeyError(f"Scenario missing required keys: {missing}")
    dataset = dict(raw["dataset"])
    output_grid = raw.get("output_grid", {})
    if not isinstance(output_grid, dict):
        raise ValueError(f"scenario 'output_grid' must be a mapping, got {output_grid!r}")
    inputs = tuple(InputSpec(kind=i["kind"], params=i.get("params", {})) for i in raw["inputs"])
    methods = tuple(MethodSpec(name=m["name"], params=m.get("params", {})) for m in raw["methods"])
    return Scenario(
        study=str(raw["study"]),
        dataset=dataset,
        inputs=inputs,
        methods=methods,
        output=Path(raw["output"]),
        sim=dict(raw.get("sim", {})),
        raw=raw,
        subjects=_resolve_subjects(raw, dataset),
        forward_raw=raw.get("forward"),
        osem_raw=raw.get("osem"),
        output_grid=dict(output_grid),
        protocol_version=str(raw.get("protocol_version", LEGACY_PROTOCOL_VERSION)),
        stage=validate_stage(str(raw.get("stage", DEFAULT_STAGE))),
        selection_policy=str(raw.get("selection_policy", DEFAULT_SELECTION_POLICY)),
    )


def load_scenario(path: str | Path) -> Scenario:
    with open(path) as f:
        return load_scenario_dict(yaml.safe_load(f))


def _grid(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand scalar/list parameter values into the cartesian product."""
    keys = sorted(params)
    values = [v if isinstance(v, list) else [v] for v in (params[k] for k in keys)]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _dataset_for_subject(dataset: dict[str, Any], subject: Any) -> dict[str, Any]:
    if subject is None:
        return dict(dataset)
    return {**dataset, "subject_id": subject}


def _register(registry: dict[str, str], short: str, identity: dict[str, Any]) -> None:
    digest = identity_digest(identity)
    existing = registry.get(short)
    if existing is not None and existing != digest:
        raise ValueError(f"identity hash collision detected for {short}")
    registry[short] = digest


def expand_scenario(scenario: Scenario) -> list[RunSpec]:
    runs: list[RunSpec] = []
    seen: dict[str, dict[str, str]] = {"forward": {}, "input": {}, "run": {}}

    for subject in scenario.subjects:
        dataset = _dataset_for_subject(scenario.dataset, subject)
        for inp in scenario.inputs:
            for input_params in _grid(inp.params):
                modality, condition = resolve_guidance(input_params)
                lesion_state = resolve_guidance_lesion_state(input_params)
                forward = resolve_forward(scenario.forward_raw, input_params, scenario.sim)
                osem = resolve_osem(input_params, scenario.sim, scenario.osem_raw)
                output_grid = input_params.get("output_grid", scenario.output_grid)
                if not isinstance(output_grid, dict):
                    raise ValueError(f"input output_grid must be a mapping, got {output_grid!r}")

                base = RunSpec(
                    run_id="",
                    study=scenario.study,
                    dataset=dataset,
                    input_kind=inp.kind,
                    input_params=input_params,
                    method_name="",
                    method_params={},
                    subject=subject,
                    guidance_modality=modality,
                    guidance_condition=condition,
                    guidance_lesion_state=lesion_state,
                    forward=forward,
                    osem=osem,
                    output_grid=output_grid,
                    protocol_version=scenario.protocol_version,
                    stage=scenario.stage,
                    selection_policy=scenario.selection_policy,
                    sim=scenario.sim,
                    out_root=scenario.output,
                )

                for method in scenario.methods:
                    for method_params in _grid(method.params):
                        run_spec = replace(base, method_name=method.name, method_params=method_params)
                        cfg = identity_config(run_spec)
                        fid, iid, rid, fwd_dict, inp_dict, run_dict = all_identities(cfg)
                        _register(seen["forward"], fid, fwd_dict)
                        _register(seen["input"], iid, inp_dict)
                        _register(seen["run"], rid, run_dict)
                        runs.append(replace(run_spec, run_id=rid, forward_id=fid, input_id=iid))

    ids = [r.run_id for r in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("run_id collision detected; canonical identity is not injective")
    return runs
