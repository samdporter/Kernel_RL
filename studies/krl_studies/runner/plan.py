"""Run-plan JSONL serialization for Task 6."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from krl_studies.config import RunSpec

PLAN_VERSION = 1


def run_to_dict(run: RunSpec) -> dict[str, Any]:
    data = asdict(run)
    data["out_root"] = str(run.out_root)
    return data


def write_run_plan(runs: Sequence[RunSpec], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"plan_version": PLAN_VERSION}) + "\n")
        for run in runs:
            f.write(json.dumps(run_to_dict(run), sort_keys=True) + "\n")
    return path


def read_run_plan(path: str | Path) -> list[RunSpec]:
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError("run plan is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON run-plan header") from exc
    if header != {"plan_version": PLAN_VERSION}:
        raise ValueError(f"unsupported run-plan header: {header!r}")

    runs = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            raise ValueError(f"blank run-plan row at line {line_number}")
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}") from exc
        try:
            runs.append(RunSpec(
                run_id=str(data["run_id"]),
                study=str(data["study"]),
                dataset=dict(data["dataset"]),
                input_kind=str(data["input_kind"]),
                input_params=dict(data["input_params"]),
                method_name=str(data["method_name"]),
                method_params=dict(data["method_params"]),
                sim=dict(data["sim"]),
                out_root=Path(data["out_root"]),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid run-plan row at line {line_number}") from exc
    return runs
