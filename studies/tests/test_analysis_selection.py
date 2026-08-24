"""Tests for Task 3: replicate summaries and iteration selection."""
import pandas as pd

from krl_studies.analysis.aggregate import summarize_replicates
from krl_studies.analysis.selection import select_fixed_iteration, select_oracle


def _frame():
    return pd.DataFrame(
        [
            {"run_id": "r0", "realisation": 0, "method": "rl", "iteration": 1, "metric": "nrmse", "value": 0.4},
            {"run_id": "r0", "realisation": 0, "method": "rl", "iteration": 2, "metric": "nrmse", "value": 0.3},
            {"run_id": "r1", "realisation": 1, "method": "rl", "iteration": 1, "metric": "nrmse", "value": 0.5},
            {"run_id": "r1", "realisation": 1, "method": "rl", "iteration": 2, "metric": "nrmse", "value": 0.35},
        ]
    )


def test_oracle_selects_lowest_nrmse_and_earliest_tie():
    result = select_oracle(_frame())
    assert result.iloc[0]["iteration"] == 2
    assert result.iloc[0]["selection"] == "oracle_min_nrmse"
    assert len(result) == 2


def test_fixed_selection_returns_empty_for_missing_iteration():
    result = select_fixed_iteration(_frame(), 3)
    assert result.empty


def test_oracle_retains_other_metrics_at_selected_iteration():
    iterations = pd.concat(
        [
            _frame(),
            pd.DataFrame(
                [
                    {
                        "run_id": "r0",
                        "realisation": 0,
                        "method": "rl",
                        "iteration": 2,
                        "metric": "bv_percent",
                        "value": 3.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    result = select_oracle(iterations)
    assert set(result.loc[result["run_id"] == "r0", "metric"]) == {"nrmse", "bv_percent"}


def test_replicate_summary_has_mean_std_count():
    summary = summarize_replicates(_frame())
    row = summary.iloc[0]
    assert row["n"] == 2
    assert row["value_mean"] == 0.45
    assert row["value_std"] > 0


def test_replicate_summary_excludes_missing_values():
    data = _frame()
    data.loc[0, "value"] = None
    summary = summarize_replicates(data)
    assert summary.loc[summary["iteration"] == 1, "n"].iloc[0] == 1
