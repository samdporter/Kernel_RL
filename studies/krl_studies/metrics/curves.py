"""Per-iteration metric rows -> DataFrame/CSV."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


def metrics_to_dataframe(rows: Sequence[Mapping]) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).sort_values("iteration").reset_index(drop=True)


def write_metrics_csv(rows: Sequence[Mapping], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_to_dataframe(rows).to_csv(path, index=False)
