#!/usr/bin/env python3
"""Run hyperparameter sweeps for multiple PET deconvolution pipelines.

This orchestrates experiments for:
  • Directional total variation MAP-Richardson–Lucy (DTV / MAPRL)
  • Kernelised Richardson–Lucy (KRL)
  • Hybrid kernelised Richardson–Lucy (HKRL)

Two datasets are supported out-of-the-box (spheres, MK-H001).  Each sweep
builds on :mod:`run_deconv` and reuses the same configuration dataclasses.

Examples
--------
Dry-run to inspect the planned experiments::

    python scripts/run_deconv_sweeps.py --dry-run

Run only the hybrid sweep on the spheres dataset::

    python scripts/run_deconv_sweeps.py --datasets spheres --pipelines hkrl
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from krl.cli.config import KernelParameters, PipelineConfig


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.3g}".replace(".", "p")
    return str(value)


def _iter_grid(grid: Mapping[str, Iterable]) -> Iterator[Dict[str, object]]:
    if not grid:
        yield {}
        return

    keys = sorted(grid)
    values = [list(grid[k]) for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def _slugify(config_updates: Mapping[str, object], kernel_updates: Mapping[str, object]) -> str:
    items: List[Tuple[str, object]] = []
    items.extend(sorted(config_updates.items()))
    items.extend(sorted(kernel_updates.items()))
    if not items:
        return "base"
    return "_".join(f"{key}-{_format_value(value)}" for key, value in items)


# Dataset specific defaults ----------------------------------------------------

DATASETS: Dict[str, Dict[str, object]] = {
    "spheres": {
        "config": PipelineConfig(
            data_path=Path("data/spheres"),
            emission_file="phant_pet.nii",
            guidance_file="phant_mri.nii",
            ground_truth_file="phant_orig.nii",
            backend="auto",
            do_rl=False,
            do_krl=False,
            do_drl=False,
            rl_iterations_standard=100,
            rl_iterations_kernel=100,
            dtv_iterations=100,
            fwhm=(5.5, 5.5, 5.5),
            psf_kernel_size=5,
            save_suffix="spheres",
        ),
        "kernel": KernelParameters(
            num_neighbours=5,
            sigma_anat=0.35,
            sigma_dist=0.6,
            sigma_emission=0.8,
            distance_weighting=False,
            normalize_features=True,
            normalize_kernel=True,
            use_mask=True,
            mask_k=48,
            recalc_mask=False,
            hybrid=False,
        ),
    },
    "mk-h001": {
        "config": PipelineConfig(
            data_path=Path("data/MK-H001"),
            emission_file="MK-H001_PET_MNI.nii",
            guidance_file="MK-H001_T1_MNI.nii",
            backend="auto",
            flip_guidance=True,
            do_rl=False,
            do_krl=False,
            do_drl=False,
            rl_iterations_standard=80,
            rl_iterations_kernel=120,
            dtv_iterations=120,
            fwhm=(6.0, 6.0, 6.0),
            psf_kernel_size=7,
            save_suffix="mk_h001",
        ),
        "kernel": KernelParameters(
            num_neighbours=7,
            sigma_anat=0.45,
            sigma_dist=1.0,
            sigma_emission=1.2,
            distance_weighting=True,
            normalize_features=True,
            normalize_kernel=True,
            use_mask=True,
            mask_k=64,
            recalc_mask=False,
            hybrid=False,
        ),
    },
}


# Pipeline sweep definitions ---------------------------------------------------

PIPELINES: Dict[str, Dict[str, object]] = {
    "rl": {
        "description": "Standard Richardson-Lucy",
        "config_overrides": {
            "do_rl": True,
            "do_krl": False,
            "do_drl": False,
            "rl_iterations_standard": 100,
        },
        "kernel_overrides": {},
        "config_grid": {},
        "kernel_grid": {},
    },
    "dtv": {
        "description": "Directional total variation MAP-RL",
        "config_overrides": {
            "do_drl": True,
            "do_rl": False,
            "do_krl": False,
            "dtv_iterations": 100,
        },
        "kernel_overrides": {},
        "config_grid": {
            "alpha": [0.1,0.2,0.28,0.3,0.4,0.5,1,2.0,5.0,10.0],
        },
        "kernel_grid": {},
    }, 
    "krl": {
        "description": "Kernel expectation maximisation (KEM)",
        "config_overrides": {
            "do_krl": True,
            "do_rl": False,
            "do_drl": False,
            "rl_iterations_kernel": 100,
            "rl_iterations_standard": 100,
        },
        "kernel_overrides": {
            "hybrid": False,
            "num_neighbours": 7,
            "mask_k": 48,
            "sigma_dist": 10000,
            "distance_weighting": False,
            "normalize_features": True,
            "normalize_kernel": True,
            "use_mask": True,
        },
        "config_grid": {},
        "kernel_grid": {
            "sigma_anat": [0.1,0.2,0.5,1.0, 2.0, 5.0, 10.0, 20.0],
        },
    },
    "hkrl": {
        "description": "Hybrid kernel expectation maximisation (HKEM)",
        "config_overrides": {
            "do_krl": True,
            "do_rl": False,
            "do_drl": False,
            "rl_iterations_kernel": 100,
            "rl_iterations_standard": 100,
        },
        "kernel_overrides": {
            "hybrid": True,
            "use_mask": True,
            "num_neighbours": 7,
            "mask_k": 48,
            "sigma_dist": 10000,
            "distance_weighting": False,
            "normalize_features": True,
            "normalize_kernel": True,
        },
        "config_grid": {
            "freeze_iteration": [1, 2],
        },
        "kernel_grid": {
            "sigma_anat": [10.1,0.2,0.5,1.0, 2.0, 5.0, 10.0, 20.0],
            "sigma_emission": [0.1,0.2,0.5,1.0, 2.0, 5.0, 10.0, 20.0],
        },
    },
}


def build_runs(
    datasets: Iterable[str],
    pipelines: Iterable[str],
) -> List[Tuple[str, str, str, PipelineConfig, KernelParameters]]:
    runs: List[Tuple[str, str, str, PipelineConfig, KernelParameters]] = []

    for dataset in datasets:
        ds = DATASETS[dataset]
        base_config: PipelineConfig = ds["config"]
        base_kernel: KernelParameters = ds["kernel"]

        for pipeline in pipelines:
            spec = PIPELINES[pipeline]
            config_overrides: Dict[str, object] = spec["config_overrides"]  # type: ignore[assignment]
            kernel_overrides: Dict[str, object] = spec["kernel_overrides"]  # type: ignore[assignment]
            config_grid: Mapping[str, Iterable] = spec["config_grid"]  # type: ignore[assignment]
            kernel_grid: Mapping[str, Iterable] = spec["kernel_grid"]  # type: ignore[assignment]

            for cfg_updates in _iter_grid(config_grid):
                for kern_updates in _iter_grid(kernel_grid):
                    slug = _slugify(cfg_updates, kern_updates)

                    config = replace(base_config, **config_overrides)
                    if cfg_updates:
                        config = replace(config, **cfg_updates)
                    config = replace(
                        config, save_suffix=f"{dataset}_{pipeline}_{slug}"
                    )

                    kernel = replace(base_kernel, **kernel_overrides)
                    if kern_updates:
                        kernel = replace(kernel, **kern_updates)

                    runs.append((dataset, pipeline, slug, config, kernel))

    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deconvolution hyperparameter sweeps."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Datasets to include in the sweep.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=list(PIPELINES.keys()),
        default=list(PIPELINES.keys()),
        help="Pipelines to include (rl, dtv, krl, hkrl).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional limit on the number of runs to execute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort sweeps immediately after a failure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = build_runs(args.datasets, args.pipelines)

    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    print(f"Prepared {len(runs)} runs.")
    failures: List[Tuple[str, str, str, Exception]] = []

    run_pipeline = None
    if not args.dry_run:
        from krl.cli.run_deconv import run_pipeline as _run_pipeline

        run_pipeline = _run_pipeline

    for index, (dataset, pipeline, slug, config, kernel) in enumerate(runs, start=1):
        header = f"[{index:03d}/{len(runs):03d}] dataset={dataset} pipeline={pipeline} variant={slug}"
        print(header)
        print(f"  emission: {config.emission_path()}")
        print(f"  guidance: {config.guidance_path()}")
        print(f"  output dir: {config.output_directory(kernel)}")

        if args.dry_run:
            continue

        try:
            assert run_pipeline is not None  # for type checkers
            run_pipeline(config, kernel)
        except Exception as exc:  # pragma: no cover - diagnostics
            failures.append((dataset, pipeline, slug, exc))
            print(f"  FAILED: {exc}")
            if args.stop_on_error:
                break

    if args.dry_run:
        print("Dry-run complete; no pipelines executed.")
        return

    completed = len(runs) - len(failures)
    print(f"Executed {completed}/{len(runs)} runs.")
    if failures:
        print("\nFailures encountered:")
        for dataset, pipeline, slug, exc in failures:
            print(f"  {dataset}/{pipeline}/{slug}: {exc}")


if __name__ == "__main__":
    main()
