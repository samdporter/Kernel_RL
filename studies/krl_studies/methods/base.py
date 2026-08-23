"""Method contract: every deconvolution method streams per-iteration results."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Iterate:
    iteration: int
    image: np.ndarray  # (z, y, x), emission-domain estimate
    objective: float | None = None


class Method:
    """Subclasses return a lazy iterator of Iterates from `run`."""

    name: str = "method"

    def run(
        self,
        observed: Any,  # CIL ImageData
        guidance: Any | None,  # CIL ImageData or None
        params: dict[str, Any],
        n_iterations: int,
    ) -> Iterator[Iterate]:
        raise NotImplementedError
