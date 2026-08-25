"""Report generation for Task 5: aggregate results, figures, and tables."""

import json
from pathlib import Path

import pandas as pd

from krl_studies.analysis.aggregate import summarize_replicates
from krl_studies.analysis.ingest import discover_completed_runs, ingest_results, write_tables
from krl_studies.analysis.plots import (
    plot_crc_by_size,
    plot_mismatch_sensitivity,
    plot_nrmse_convergence,
    plot_recovery_vs_cov,
)
from krl_studies.analysis.selection import select_fixed_iteration, select_oracle
from krl_studies.analysis.tables import best_results_table, write_latex_table


def _build_tradeoff(iterations: pd.DataFrame) -> pd.DataFrame:
    """Build tradeoff.csv by joining BV and CRC on (run_id, iteration)."""
    if iterations.empty:
        return pd.DataFrame()

    # Get scalar iteration metrics (bv_percent, nrmse, objective)
    scalar_metrics = ["bv_percent", "nrmse", "objective"]
    scalar = iterations[iterations["metric"].isin(scalar_metrics)].copy()

    if scalar.empty:
        return pd.DataFrame()

    # Pivot scalar metrics to wide
    # Find identity columns (all except metric and value)
    id_cols = [c for c in scalar.columns if c not in ("metric", "value")]

    # Fill NaN in index columns with placeholder for pivot
    scalar_pivot = scalar.copy()
    for col in id_cols:
        if scalar_pivot[col].dtype == object:
            scalar_pivot[col] = scalar_pivot[col].fillna("__NA__")
        elif pd.api.types.is_numeric_dtype(scalar_pivot[col]):
            scalar_pivot[col] = scalar_pivot[col].fillna(-999)

    scalar_wide = scalar_pivot.pivot_table(
        index=id_cols,
        columns="metric",
        values="value",
        aggfunc="first"
    ).reset_index()

    # Restore NaN in the index columns
    for col in id_cols:
        if scalar_wide[col].dtype == object:
            scalar_wide[col] = scalar_wide[col].replace("__NA__", pd.NA)
        elif pd.api.types.is_numeric_dtype(scalar_wide[col]):
            scalar_wide[col] = scalar_wide[col].replace(-999, pd.NA)

    # Get CRC/lesion data
    crc = iterations[iterations["metric"] == "crc_percent"].copy()
    if crc.empty:
        return scalar_wide

    # CRC has lesion_diameter_mm, we need to merge on identity columns + iteration + run_id
    merge_cols = [c for c in scalar_wide.columns if c not in ("bv_percent", "nrmse", "objective")]

    # Get CRC values with lesion_diameter_mm
    crc_target_cols = ("run_id", "iteration", "lesion_diameter_mm", "value")
    crc_cols = [c for c in crc.columns if c in merge_cols or c in crc_target_cols]
    crc_info = crc[crc_cols].copy()
    crc_info = crc_info.rename(columns={"value": "crc_percent"})

    # Merge scalar with CRC on merge_cols + run_id + iteration
    join_cols = [c for c in merge_cols if c in crc_info.columns] + ["run_id", "iteration"]
    tradeoff = scalar_wide.merge(crc_info[join_cols + ["crc_percent", "lesion_diameter_mm"]], on=join_cols, how="left")

    return tradeoff


def aggregate_results(results_root: Path, out_dir: Path, fixed_iteration: int = 10) -> None:
    """Ingest results, compute summaries, and write aggregate CSVs."""
    results_root = Path(results_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = ingest_results(results_root)

    # Write raw tables
    write_tables(tables, out_dir)

    # Compute summary (replicate aggregation)
    summary = summarize_replicates(tables.iterations)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # Lesion summary
    lesion_summary = summarize_replicates(tables.lesions)
    lesion_summary_path = out_dir / "lesion_summary.csv"
    lesion_summary.to_csv(lesion_summary_path, index=False)

    # Tradeoff (BV vs CRC)
    tradeoff = _build_tradeoff(tables.iterations)
    tradeoff_path = out_dir / "tradeoff.csv"
    tradeoff.to_csv(tradeoff_path, index=False)

    # Selection: oracle
    oracle = select_oracle(tables.iterations)
    oracle_path = out_dir / "oracle.csv"
    oracle.to_csv(oracle_path, index=False)

    # Selection: fixed iteration
    fixed = select_fixed_iteration(tables.iterations, fixed_iteration)
    fixed_path = out_dir / "fixed.csv"
    fixed.to_csv(fixed_path, index=False)

    # Analysis metadata
    metadata = {
        "fixed_iteration": fixed_iteration,
        "n_runs": len(discover_completed_runs(Path(results_root))),
    }
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))


def generate_figures(analysis_dir: Path, out_dir: Path) -> None:
    """Generate publication figures from analysis CSVs."""
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(analysis_dir / "summary.csv")
    if summary.empty:
        return

    lesion_summary = pd.read_csv(analysis_dir / "lesion_summary.csv")
    tradeoff_path = analysis_dir / "tradeoff.csv"
    tradeoff = (
        pd.read_csv(tradeoff_path) if tradeoff_path.exists() else pd.DataFrame()
    )

    # NRMSE convergence
    plot_nrmse_convergence(summary, out_dir / "nrmse_convergence.png", title="NRMSE Convergence")

    # Recovery vs covariance (tradeoff)
    has_recovery = "bv_percent" in tradeoff.columns and (
        "crc_percent" in tradeoff.columns or "nrmse" in tradeoff.columns
    )
    if not tradeoff.empty and has_recovery:
        plot_recovery_vs_cov(tradeoff, out_dir / "recovery_vs_cov.png", title="Recovery vs. Covariance")

    # CRC by size
    if not lesion_summary.empty:
        plot_crc_by_size(lesion_summary, out_dir / "crc_by_size.png", title="CRC by Lesion Size")

    # Mismatch sensitivity
    plot_mismatch_sensitivity(summary, out_dir / "mismatch_sensitivity.png", title="Mismatch Sensitivity")


def generate_tables(analysis_dir: Path, out_dir: Path) -> None:
    """Generate best-result CSV and LaTeX tables from oracle/fixed CSVs."""
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle = pd.read_csv(analysis_dir / "oracle.csv") if (analysis_dir / "oracle.csv").exists() else pd.DataFrame()
    fixed = pd.read_csv(analysis_dir / "fixed.csv") if (analysis_dir / "fixed.csv").exists() else pd.DataFrame()

    # Oracle best results
    if not oracle.empty:
        oracle_best = best_results_table(oracle)
        oracle_best.to_csv(out_dir / "best_oracle.csv", index=False)
        write_latex_table(oracle_best, out_dir / "best_oracle.tex", caption="Oracle best results", label="tab:oracle")

    # Fixed iteration best results
    if not fixed.empty:
        fixed_best = best_results_table(fixed)
        fixed_best.to_csv(out_dir / "best_fixed.csv", index=False)
        write_latex_table(
            fixed_best,
            out_dir / "best_fixed.tex",
            caption="Fixed-iteration best results",
            label="tab:fixed",
        )
