"""Iteration selection policies (Task 3)."""

import pandas as pd


def select_oracle(iterations: pd.DataFrame) -> pd.DataFrame:
    """Select best iteration per run_id by minimum NRMSE, tie-breaking on earliest iteration."""
    nrmse = iterations.loc[
        (iterations["metric"] == "nrmse") & iterations["value"].notna(),
        ["run_id", "iteration", "value"],
    ].sort_values(["run_id", "value", "iteration"], kind="stable")
    best = nrmse.drop_duplicates("run_id", keep="first")[["run_id", "iteration"]]
    selected = iterations.merge(best, on=["run_id", "iteration"], how="inner", validate="many_to_many")
    selected = selected.sort_values(["run_id", "iteration", "metric"], kind="stable").reset_index(drop=True)
    selected["selection"] = "oracle_min_nrmse"
    return selected


def select_fixed_iteration(iterations: pd.DataFrame, iteration: int) -> pd.DataFrame:
    """Select a fixed iteration for all runs."""
    selected = iterations.loc[iterations["iteration"] == int(iteration)].copy()
    selected["selection"] = f"fixed_{int(iteration)}"
    return selected.reset_index(drop=True)
