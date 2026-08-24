"""Schema definitions for canonical results tables (Task 2)."""

import json
import re
from pathlib import Path

import pandas as pd

COMMON_COLUMNS = [
    "run_id", "iteration", "method", "study", "subject_id", "dataset_kind",
    "input_kind", "scanner", "condition", "beta", "counts", "realisation",
    "guidance_condition", "assumed_fwhm_mm", "forward_model_fwhm_json",
    "recon_model_fwhm_json", "target_residual_fwhm_json",
]
RUN_COLUMNS = [
    "run_id", "run_path", "study", "subject_id", "dataset_kind", "input_kind",
    "scanner", "condition", "beta", "counts", "realisation",
    "guidance_condition", "method", "assumed_fwhm_mm", "forward_model_fwhm_json",
    "recon_model_fwhm_json", "target_residual_fwhm_json", "method_params_json",
    "sim_params_json", "status", "git_rev", "krl_version", "krl_studies_version",
    "finished_at",
]
ITERATION_COLUMNS = COMMON_COLUMNS + ["metric", "value"]
LESION_COLUMNS = COMMON_COLUMNS + ["lesion_diameter_mm", "metric", "value"]

STANDARD_METRICS = ("nrmse", "bv_percent", "objective")
CRC_RE = re.compile(r"^crc_mm(?P<diameter>[-+0-9p.e]+)$")


def _compact(value):
    return None if value is None else json.dumps(value, sort_keys=True, separators=(",", ":"))


def flatten_manifest(manifest: dict, run_path: str | Path) -> dict:
    """Flatten a run manifest into a row for the runs table."""
    dataset = dict(manifest.get("dataset", {}))
    input_params = dict(manifest.get("input_params", {}))
    method_params = dict(manifest.get("method_params", {}))
    sim_params = dict(manifest.get("sim", {}))
    simulation = dict(manifest.get("simulation", {}))
    subject_id = dataset.get("subject_id", dataset.get("subject"))
    assumed = method_params.get("fwhm_mm")
    if isinstance(assumed, (list, tuple, dict)):
        assumed = None
    row = {
        "run_id": manifest.get("run_id"),
        "run_path": str(run_path),
        "study": manifest.get("study"),
        "subject_id": subject_id,
        "dataset_kind": dataset.get("kind"),
        "input_kind": manifest.get("input_kind"),
        "scanner": simulation.get("scanner", input_params.get("scanner", sim_params.get("scanner"))),
        "condition": input_params.get("condition"),
        "beta": input_params.get("beta"),
        "counts": input_params.get("counts"),
        "realisation": input_params.get("realisation"),
        "guidance_condition": input_params.get(
            "guidance_condition", method_params.get("guidance_condition", "exact")
        ),
        "method": manifest.get("method"),
        "assumed_fwhm_mm": assumed,
        "forward_model_fwhm_json": _compact(simulation.get("forward_model_fwhm")),
        "recon_model_fwhm_json": _compact(simulation.get("recon_model_fwhm")),
        "target_residual_fwhm_json": _compact(simulation.get("target_residual_fwhm")),
        "method_params_json": _compact(method_params),
        "sim_params_json": _compact(sim_params),
        "status": manifest.get("status"),
        "git_rev": manifest.get("git_rev"),
        "krl_version": manifest.get("krl_version"),
        "krl_studies_version": manifest.get("krl_studies_version"),
        "finished_at": manifest.get("finished_at"),
    }
    return {column: row.get(column) for column in RUN_COLUMNS}


def _frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = None
    return frame.reindex(columns=columns)


def melt_metrics(run_id: str, frame_or_mapping, metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Melt a metrics DataFrame into iteration and lesion tables."""
    frame = frame_or_mapping.copy() if isinstance(frame_or_mapping, pd.DataFrame) else pd.DataFrame(frame_or_mapping)
    if "iteration" not in frame.columns:
        raise ValueError("metrics require an iteration column")
    common = {column: metadata.get(column) for column in COMMON_COLUMNS}
    common["run_id"] = run_id
    standard = [column for column in STANDARD_METRICS if column in frame.columns]
    melted = frame.melt(
        id_vars=["iteration"], value_vars=standard, var_name="metric", value_name="value"
    )
    iteration_rows = []
    for record in melted.to_dict("records"):
        if pd.notna(record["value"]):
            iteration_rows.append(
            {**common, "iteration": record["iteration"], "metric": record["metric"], "value": record["value"]}
        )
    lesion_rows = []
    for column in frame.columns:
        match = CRC_RE.fullmatch(str(column))
        if match is None:
            continue
        diameter = float(match.group("diameter").replace("p", "."))
        for iteration, value in zip(frame["iteration"], frame[column]):
            if pd.notna(value):
                lesion_rows.append({
                    **common,
                    "iteration": iteration,
                    "lesion_diameter_mm": diameter,
                    "metric": "crc_percent",
                    "value": value,
                })
    return _frame(iteration_rows, ITERATION_COLUMNS), _frame(lesion_rows, LESION_COLUMNS)
