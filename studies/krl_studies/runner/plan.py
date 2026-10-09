"""Run-plan JSONL serialization for Task 6."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from krl_studies.config import RunSpec
from krl_studies.identity import ForwardConfig, OsemConfig

PLAN_VERSION = 2
SCHEMA_VERSION = "run_plan_v2"
INDEX_SCHEMA_VERSION = "run_plan_index_v1"


def run_to_dict(run: RunSpec) -> dict[str, Any]:
    data = asdict(run)
    data["out_root"] = str(run.out_root)
    return data


def _run_from_dict(data: dict[str, Any]) -> RunSpec:
    forward = data.get("forward")
    osem = data.get("osem")
    return RunSpec(
        run_id=str(data["run_id"]),
        study=str(data["study"]),
        dataset=dict(data["dataset"]),
        input_kind=str(data["input_kind"]),
        input_params=dict(data["input_params"]),
        method_name=str(data["method_name"]),
        method_params=dict(data["method_params"]),
        subject=data.get("subject"),
        guidance_modality=str(data.get("guidance_modality", "t1")),
        guidance_condition=str(data.get("guidance_condition", "exact")),
        guidance_lesion_state=str(data.get("guidance_lesion_state", "absent")),
        forward=ForwardConfig(**forward) if forward else None,
        osem=OsemConfig(**osem) if osem else None,
        output_grid=dict(data.get("output_grid", {})),
        protocol_version=str(data.get("protocol_version", "legacy")),
        stage=str(data.get("stage", "development")),
        selection_policy=str(data.get("selection_policy", "oracle_min_nrmse")),
        sim=dict(data.get("sim", {})),
        out_root=Path(data["out_root"]),
        forward_id=str(data.get("forward_id", "")),
        input_id=str(data.get("input_id", "")),
    )


def _validate_header(header_line: str) -> dict[str, Any]:
    if not header_line.strip():
        raise ValueError("run plan is empty")
    try:
        header = json.loads(header_line)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON run-plan header") from exc
    if not isinstance(header, dict) or header.get("plan_version") not in (1, PLAN_VERSION):
        raise ValueError(f"unsupported run-plan header: {header!r}")
    return header


def _iter_rows(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, line)`` for data rows, validating the header first."""
    with path.open() as f:
        _validate_header(f.readline())
        for line_number, line in enumerate(f, start=2):
            if not line.strip():
                raise ValueError(f"blank run-plan row at line {line_number}")
            yield line_number, line


def _index_path(path: Path) -> Path:
    return path.with_name(path.name + ".idx")


def _row_sha(line: bytes) -> str:
    return hashlib.sha256(line).hexdigest()


def _make_payload(path: Path, offsets: list[int], first_sha: str | None, last_sha: str | None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "index_schema": INDEX_SCHEMA_VERSION,
        "plan_size": stat.st_size,
        "plan_mtime_ns": stat.st_mtime_ns,
        "row_count": len(offsets),
        "first_row_sha256": first_sha,
        "last_row_sha256": last_sha,
        "offsets": offsets,
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    try:
        _index_path(path).write_text(json.dumps(payload, separators=(",", ":")))
    except OSError:
        # A read-only plan directory must not prevent writing the plan itself.
        pass


def write_run_plan(runs: Sequence[RunSpec], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = []
    first_sha: str | None = None
    last_sha: str | None = None
    with path.open("wb") as f:
        header = json.dumps({"plan_version": PLAN_VERSION, "schema_version": SCHEMA_VERSION}) + "\n"
        f.write(header.encode("utf-8"))
        offset = len(header.encode("utf-8"))
        for run in runs:
            line = (json.dumps(run_to_dict(run), sort_keys=True) + "\n").encode("utf-8")
            sha = _row_sha(line)
            first_sha = sha if first_sha is None else first_sha
            last_sha = sha
            offsets.append(offset)
            f.write(line)
            offset += len(line)
    _write_payload(path, _make_payload(path, offsets, first_sha, last_sha))
    return path


def _build_index(path: Path) -> dict[str, Any]:
    offsets: list[int] = []
    first_sha: str | None = None
    last_sha: str | None = None
    with path.open("rb") as f:
        _validate_header(f.readline().decode("utf-8"))
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                raise ValueError(f"blank run-plan row at line {len(offsets) + 2}")
            sha = _row_sha(line)
            first_sha = sha if first_sha is None else first_sha
            last_sha = sha
            offsets.append(offset)
    return _make_payload(path, offsets, first_sha, last_sha)


def _signatures_match(path: Path, data: dict[str, Any], offsets: list[int]) -> bool:
    if not offsets:
        return data.get("first_row_sha256") is None and data.get("last_row_sha256") is None
    try:
        with path.open("rb") as f:
            f.seek(offsets[0])
            first = f.readline()
            f.seek(offsets[-1])
            last = f.readline()
    except OSError:
        return False
    return (
        _row_sha(first) == data.get("first_row_sha256")
        and _row_sha(last) == data.get("last_row_sha256")
    )


def _read_index(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(_index_path(path).read_text())
        stat = path.stat()
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("index_schema") != INDEX_SCHEMA_VERSION:
        return None
    if data.get("plan_size") != stat.st_size or data.get("plan_mtime_ns") != stat.st_mtime_ns:
        return None
    offsets = data.get("offsets")
    if not isinstance(offsets, list) or len(offsets) != data.get("row_count"):
        return None
    if not _signatures_match(path, data, offsets):
        return None
    return data


def _ensure_index(path: Path) -> dict[str, Any]:
    data = _read_index(path)
    if data is not None:
        return data
    data = _build_index(path)
    _write_payload(path, data)
    return data


def _parse_row(line: str, line_number: int) -> RunSpec:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at line {line_number}") from exc
    try:
        return _run_from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid run-plan row at line {line_number}") from exc


def read_run_plan(path: str | Path) -> list[RunSpec]:
    path = Path(path)
    return [_parse_row(line, line_number) for line_number, line in _iter_rows(path)]


def read_run_plan_index(path: str | Path, index: int) -> RunSpec:
    """Read a single 1-based row, seeking via the sidecar offset index."""
    path = Path(path)
    if index < 1:
        raise IndexError(f"plan index must be >= 1, got {index}")
    data = _ensure_index(path)
    if index > data["row_count"]:
        raise IndexError(f"plan index {index} out of range")
    with path.open("rb") as f:
        f.seek(data["offsets"][index - 1])
        line = f.readline().decode("utf-8")
    return _parse_row(line, index + 1)


def count_run_plan(path: str | Path) -> int:
    path = Path(path)
    return int(_ensure_index(path)["row_count"])
