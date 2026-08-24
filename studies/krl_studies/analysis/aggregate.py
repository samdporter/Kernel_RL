"""Replicate aggregation (Task 3)."""

import pandas as pd


def summarize_replicates(iterations: pd.DataFrame) -> pd.DataFrame:
    """Summarize replicate realisations by computing mean, std, and count per group."""
    frame = iterations.copy()
    if frame.empty:
        return pd.DataFrame(columns=["value_mean", "value_std", "n"])
    group_columns = [
        column for column in frame.columns
        if column not in {"run_id", "realisation", "value", "value_mean", "value_std", "n"}
    ]
    grouped = frame.groupby(group_columns, dropna=False, sort=True)["value"].agg(
        value_mean="mean", value_std=lambda values: values.std(ddof=1), n="count"
    ).reset_index()
    grouped["value_std"] = grouped["value_std"].fillna(0.0)
    grouped["n"] = grouped["n"].astype(int)
    return grouped
