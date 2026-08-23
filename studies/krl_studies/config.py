"""Scenario configuration: YAML/dict parsing and sweep expansion."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
    raw: dict[str, Any]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    scenario_name: str
    study: str
    dataset: dict[str, Any]
    input_kind: str
    input_params: dict[str, Any]
    method_name: str
    method_params: dict[str, Any]


def load_scenario_dict(raw: dict[str, Any]) -> Scenario:
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise KeyError(f"Scenario missing required keys: {missing}")
    inputs = tuple(InputSpec(kind=i["kind"], params=i.get("params", {})) for i in raw["inputs"])
    methods = tuple(MethodSpec(name=m["name"], params=m.get("params", {})) for m in raw["methods"])
    return Scenario(
        study=str(raw["study"]),
        dataset=dict(raw["dataset"]),
        inputs=inputs,
        methods=methods,
        output=Path(raw["output"]),
        raw=raw,
    )


def load_scenario(path: str | Path) -> Scenario:
    with open(path) as f:
        return load_scenario_dict(yaml.safe_load(f))


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return repr(float(value)).replace(".", "p").replace("+", "")
    return str(value)


def _grid(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand scalar/list parameter values into the cartesian product."""
    keys = sorted(params)
    values = [v if isinstance(v, list) else [v] for v in (params[k] for k in keys)]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _input_slug(kind: str, params: dict[str, Any]) -> str:
    parts = [kind] + [f"{k}-{_format_value(params[k])}" for k in sorted(params)]
    return "_".join(parts)


def expand_scenario(scenario: Scenario) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for inp in scenario.inputs:
        for input_params in _grid(inp.params):
            input_slug = _input_slug(inp.kind, input_params)
            for method in scenario.methods:
                for method_params in _grid(method.params):
                    parts = [scenario.study, input_slug, method.name] + [
                        f"{k}-{_format_value(method_params[k])}" for k in sorted(method_params)
                    ]
                    run_id = "__".join(parts)
                    runs.append(
                        RunSpec(
                            run_id=run_id,
                            scenario_name=scenario.output.name,
                            study=scenario.study,
                            dataset=scenario.dataset,
                            input_kind=inp.kind,
                            input_params=input_params,
                            method_name=method.name,
                            method_params=method_params,
                        )
                    )
    ids = [r.run_id for r in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("run_id collision detected; slug formatting is not injective")
    return runs
