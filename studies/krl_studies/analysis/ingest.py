"""Ingestion of completed run directories (Task 2)."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from krl_studies.analysis.schema import (
    ITERATION_COLUMNS,
    LESION_COLUMNS,
    RUN_COLUMNS,
    flatten_manifest,
    melt_metrics,
)


@dataclass(frozen=True)
class ResultsTables:
    runs: pd.DataFrame
    iterations: pd.DataFrame
    lesions: pd.DataFrame
    errors: pd.DataFrame


def discover_completed_runs(results_root: str | Path) -> list[Path]:
    """Find completed run directories (have .done, manifest.json, metrics.csv)."""
    root = Path(results_root)
    return sorted(
        marker.parent for marker in root.rglob(".done")
        if (marker.parent / "manifest.json").exists()
        and (marker.parent / "metrics.csv").exists()
    )


def ingest_results(results_root: str | Path) -> ResultsTables:
    """Read only complete runs; malformed runs are returned in errors."""
    run_rows, iteration_frames, lesion_frames, errors = [], [], [], []
    for run_path in discover_completed_runs(results_root):
        try:
            manifest = json.loads((run_path / "manifest.json").read_text())
            metrics = pd.read_csv(run_path / "metrics.csv")
            run_row = flatten_manifest(manifest, run_path)
            iterations, lesions = melt_metrics(run_row["run_id"], metrics, run_row)
        except (OSError, KeyError, ValueError, json.JSONDecodeError, pd.errors.ParserError,
                pd.errors.EmptyDataError) as exc:
            errors.append({"run_path": str(run_path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        run_rows.append(run_row)
        iteration_frames.append(iterations)
        lesion_frames.append(lesions)
    return ResultsTables(
        runs=pd.DataFrame(run_rows, columns=RUN_COLUMNS),
        iterations=(
            pd.concat(iteration_frames, ignore_index=True)
            if iteration_frames else pd.DataFrame(columns=ITERATION_COLUMNS)
        ),
        lesions=(
            pd.concat(lesion_frames, ignore_index=True)
            if lesion_frames else pd.DataFrame(columns=LESION_COLUMNS)
        ),
        errors=pd.DataFrame(errors, columns=["run_path", "error"]),
    )


def write_tables(tables: ResultsTables, out_dir: str | Path) -> dict[str, Path]:
    """Write CSV always and parquet when pyarrow is installed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "runs": tables.runs.reindex(columns=RUN_COLUMNS),
        "iterations": tables.iterations.reindex(columns=ITERATION_COLUMNS),
        "lesions": tables.lesions.reindex(columns=LESION_COLUMNS),
        "errors": tables.errors.reindex(columns=["run_path", "error"]),
    }
    paths = {}
    for name, frame in frames.items():
        sort_columns = [column for column in frame.columns if column in frame]
        ordered = frame.sort_values(sort_columns, kind="stable", na_position="last") if not frame.empty else frame
        csv_path = out_dir / f"{name}.csv"
        ordered.to_csv(csv_path, index=False)
        paths[name] = csv_path
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return paths
    for name, frame in frames.items():
        parquet_path = out_dir / f"{name}.parquet"
        frame.to_parquet(parquet_path, index=False)
        paths[f"{name}_parquet"] = parquet_path
    return paths
