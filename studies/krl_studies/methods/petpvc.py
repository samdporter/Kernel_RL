"""GTM PVC via the PETPVC command-line toolbox (Thomas et al., PMB 2016).

Requires the `petpvc` binary on PATH (cluster/docker environments). Runs as a
single-shot correction; exposed through the streaming interface with exactly
one iterate.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from krl_studies.methods.base import Iterate, Method


def build_petpvc_cmd(
    petpvc_bin: str,
    input_path: str | Path,
    output_path: str | Path,
    mode: str,
    pvc_fwhm: tuple[float, float, float],
    mask_path: str | Path | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        petpvc_bin,
        "-i", str(input_path),
        "-o", str(output_path),
        "-p", mode,
        "-f", ",".join(_fmt_fwhm(v) for v in pvc_fwhm),
    ]
    if mask_path is not None:
        cmd += ["-m", str(mask_path)]
    if extra:
        cmd += list(extra)
    return cmd


def _fmt_fwhm(value: float) -> str:
    text = repr(float(value)).replace(".", "p").replace("+", "")
    return text


class GTMMethod(Method):
    name = "gtm"

    def run(self, observed: Any, guidance: Any | None, params: dict[str, Any], n_iterations: int) -> Iterator[Iterate]:
        if n_iterations != 1:
            raise ValueError("GTM is single-step; use n_iterations=1")
        bin_name = str(params.get("petpvc_bin", "petpvc"))
        if shutil.which(bin_name) is None:
            raise FileNotFoundError(
                f"PETPVC binary '{bin_name}' not found on PATH; "
                "install PETPVC (cluster/docker) or skip GTM scenarios"
            )
        cmd = build_petpvc_cmd(
            petpvc_bin=bin_name,
            input_path=params["input_path"],
            output_path=params["output_path"],
            mode="GTM",
            pvc_fwhm=tuple(float(v) for v in params.get("pvc_fwhm", (5.0, 5.0, 5.0))),
            mask_path=params.get("mask_path"),
            extra=params.get("extra"),
        )
        subprocess.run(cmd, check=True)
        arr = np.transpose(
            nib.load(str(params["output_path"])).get_fdata().astype(np.float32), (2, 1, 0)
        )
        yield Iterate(iteration=1, image=arr)
