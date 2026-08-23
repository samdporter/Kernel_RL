"""Execute one RunSpec: build inputs, stream method iterates, record metrics."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from krl.utils import load_nifti_as_imagedata, save_image

from krl_studies.config import RunSpec
from krl_studies.datasets.lesions import (
    DEFAULT_CONTRAST,
    DEFAULT_TUMOUR_DIAMETERS_MM,
    default_tumour_specs,
    place_tumours,
)
from krl_studies.datasets.spheres import SphereDataset, quick_sim
from krl_studies.methods import METHOD_REGISTRY
from krl_studies.metrics import (
    background_variability,
    background_vois,
    crc_percent,
    derive_lesion_rois,
    nrmse,
    write_metrics_csv,
)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

_GUIDED_METHODS = frozenset({"krl", "hkrl", "dtv"})
_CIL_METHODS = frozenset({"rl", "krl", "hkrl", "dtv"})


@contextmanager
def tempfile_dir():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        yield td


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _wrap(arr: np.ndarray, voxel_mm) -> Any:
    """Wrap a (z,y,x) array as CIL ImageData with the given mm voxel sizes."""
    from cil.framework import ImageGeometry

    geom = ImageGeometry(
        voxel_num_x=arr.shape[2],
        voxel_num_y=arr.shape[1],
        voxel_num_z=arr.shape[0],
        voxel_size_x=voxel_mm[2],
        voxel_size_y=voxel_mm[1],
        voxel_size_z=voxel_mm[0],
    )
    img = geom.allocate()
    img.fill(arr.astype(np.float32))
    return img


def _build_observed(run: RunSpec, ds: SphereDataset, gt: np.ndarray) -> np.ndarray:
    if run.input_kind == "reference":
        return ds.reference_pet
    if run.input_kind == "quick_sim":
        return quick_sim(
            gt,
            fwhm_mm=float(run.input_params["fwhm_mm"]),
            counts=float(run.input_params["counts"]),
            realisation=int(run.input_params.get("realisation", 0)),
            voxel_mm=ds.voxel_mm,
        )
    raise ValueError(f"unknown input kind: {run.input_kind}")


def _iy_region_defaults(gt: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Two-compartment split (hot vs background) inside the support mask."""
    brain = gt > 0
    hot = brain & (gt > 0.25 * float(gt.max()))
    return [hot, brain & ~hot], brain


def execute_run(run: RunSpec, force: bool = False) -> Path:
    out_dir = Path(run.out_root) / run.run_id
    marker = out_dir / ".done"
    if marker.exists() and not force:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if run.study != "spheres":
        raise NotImplementedError("runner phase 1 supports study='spheres'")

    ds = SphereDataset(root=run.dataset["root"])
    gt = ds.ground_truth
    guidance_arr = ds.guidance
    observed_arr = _build_observed(run, ds, gt)

    lesion_masks: list[np.ndarray] = []
    if run.sim.get("add_tumours"):
        specs = default_tumour_specs(
            gt.shape,
            ds.voxel_mm,
            diameters_mm=tuple(run.sim.get("tumour_diameters_mm", DEFAULT_TUMOUR_DIAMETERS_MM)),
        )
        gt, lesion_masks = place_tumours(
            gt,
            specs,
            contrast=float(run.sim.get("tumour_contrast", DEFAULT_CONTRAST)),
            voxel_mm=ds.voxel_mm,
        )
        if run.input_kind == "quick_sim":
            observed_arr = quick_sim(
                gt,
                fwhm_mm=float(run.input_params["fwhm_mm"]),
                counts=float(run.input_params["counts"]),
                realisation=int(run.input_params.get("realisation", 0)),
                voxel_mm=ds.voxel_mm,
            )

    lesion_rois = derive_lesion_rois(gt) if lesion_masks else []
    exclusion = (
        np.logical_or.reduce(lesion_rois or lesion_masks)
        if (lesion_rois or lesion_masks)
        else np.zeros_like(gt, dtype=bool)
    )
    vois = background_vois(gt.shape, exclude_mask=exclusion)

    method_cls = METHOD_REGISTRY[run.method_name]
    params = dict(run.method_params)
    n_iterations = int(params.pop("iterations", 1))
    if run.method_name == "iy":
        regions, brain = _iy_region_defaults(gt)
        params.setdefault("region_masks", regions)
        params.setdefault(
            "psf_sigma_vox",
            tuple(float(params.get("fwhm_mm", 5.0)) * FWHM_TO_SIGMA for _ in range(3)),
        )
        params.setdefault("brain_mask", brain)

    rows: list[dict[str, Any]] = []
    best_nrmse = float("inf")
    best_img: np.ndarray | None = None
    final_img: np.ndarray | None = None

    with tempfile_dir() as td:
        td_path = Path(td)
        obs_path = td_path / "observed.nii"
        guide_path = td_path / "guidance.nii"
        save_image(_wrap(observed_arr, ds.voxel_mm), obs_path)
        save_image(_wrap(guidance_arr, ds.voxel_mm), guide_path)
        observed_cil = load_nifti_as_imagedata(obs_path)
        guidance_cil = load_nifti_as_imagedata(guide_path)

        stream = method_cls().run(
            observed=observed_cil if run.method_name in _CIL_METHODS else observed_arr,
            guidance=guidance_cil if run.method_name in _GUIDED_METHODS else None,
            params=params,
            n_iterations=n_iterations,
        )
        for it in stream:
            row: dict[str, Any] = {"iteration": it.iteration}
            if it.objective is not None:
                row["objective"] = it.objective
            value = nrmse(it.image, gt)
            row["nrmse"] = value
            if value < best_nrmse:
                best_nrmse = value
                best_img = it.image.copy()
            for i, mask in enumerate(lesion_masks):
                d_mm = (
                    sorted(DEFAULT_TUMOUR_DIAMETERS_MM)[i]
                    if i < len(DEFAULT_TUMOUR_DIAMETERS_MM)
                    else i * 10
                )
                if vois:
                    row[f"crc_mm{int(d_mm)}"] = crc_percent(mask, it.image, gt, vois)
            if vois:
                row["bv_percent"] = background_variability(it.image, vois)
            rows.append(row)
            final_img = it.image

    write_metrics_csv(rows, out_dir / "metrics.csv")
    if final_img is not None:
        save_image(_wrap(final_img, ds.voxel_mm), out_dir / "final.nii.gz")
    if best_img is not None:
        save_image(_wrap(best_img, ds.voxel_mm), out_dir / "best_nrmse.nii.gz")

    manifest = {
        "run_id": run.run_id,
        "study": run.study,
        "input_kind": run.input_kind,
        "input_params": run.input_params,
        "method": run.method_name,
        "method_params": run.method_params,
        "dataset": run.dataset,
        "sim": run.sim,
        "status": "complete",
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "krl_version": _pkg_version("cil-krl"),
        "krl_studies_version": _pkg_version("krl-studies"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    marker.write_text(dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    return out_dir
