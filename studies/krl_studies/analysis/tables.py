"""LaTeX/CSV table generation for Task 5."""

from pathlib import Path

import pandas as pd


def best_results_table(selected: pd.DataFrame) -> pd.DataFrame:
    """Return one row per method/input condition with mean/std metric columns."""
    if selected.empty:
        return pd.DataFrame()

    group_columns = [
        column for column in (
            "study", "subject_id", "dataset_kind", "input_kind", "scanner",
            "condition", "beta", "counts", "guidance_condition", "method",
            "assumed_fwhm_mm", "selection",
        ) if column in selected.columns
    ]
    group_columns_with_metric = group_columns + ["metric"]

    result = selected.groupby(group_columns_with_metric, dropna=False, sort=True)["value"].agg(
        value_mean="mean", value_std=lambda values: values.std(ddof=1), n="count"
    ).reset_index()
    result["value_std"] = result["value_std"].fillna(0.0)
    result["n"] = result["n"].astype(int)
    return result.pivot(
        index=group_columns,
        columns="metric",
        values=["value_mean", "value_std", "n"]
    ).reset_index()


def write_latex_table(frame: pd.DataFrame, output: Path, caption: str, label: str) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(frame.to_latex(index=False, escape=True, caption=caption, label=label))
