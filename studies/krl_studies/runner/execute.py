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
from krl_studies.datasets.transforms import apply_guidance_condition
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


def _build_observed(run: RunSpec, ds: SphereDataset, gt: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if run.input_kind == "reference":
        return ds.reference_pet, {}
    if run.input_kind == "quick_sim":
        return quick_sim(
            gt,
            fwhm_mm=float(run.input_params["fwhm_mm"]),
            counts=float(run.input_params["counts"]),
            realisation=int(run.input_params.get("realisation", 0)),
            voxel_mm=ds.voxel_mm,
        ), {}
    if run.input_kind == "sirf_sim":
        from krl_studies.simulation.simulate import simulate_inputs  # noqa: I001,WPS433 - lazy so native dry-run doesn't need SIRF

        cfg = dict(run.input_params)
        if "seed" not in cfg:
            cfg["seed"] = int(run.sim.get("seed", 0))
        if "n_subits" not in cfg and "n_subiterations" not in cfg:
            if "n_subits" in run.sim:
                cfg["n_subits"] = int(run.sim["n_subits"])
            elif "n_subiterations" in run.sim:
                cfg["n_subiterations"] = int(run.sim["n_subiterations"])
        cfg.setdefault("scanner", run.input_params.get("scanner", "Siemens mMR"))
        cfg.setdefault("input_voxel_mm", ds.voxel_mm)
        recon, meta = simulate_inputs(gt, cfg)
        return recon, meta
    raise ValueError(f"unknown input kind: {run.input_kind}")


def _iy_region_defaults(gt: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Two-compartment split (hot vs background) inside the support mask."""
    brain = gt > 0
    hot = brain & (gt > 0.25 * float(gt.max()))
    return [hot, brain & ~hot], brain


def _apply_guidance_condition(
    guidance_arr: np.ndarray,
    condition: str,
    voxel_mm: tuple[float, float, float],
    ds,
) -> np.ndarray:
    """Apply guidance condition to the guidance array."""
    if condition == "exact":
        return guidance_arr
    if condition == "t2":
        if not hasattr(ds, "t2") or ds.t2 is None:
            raise FileNotFoundError(f"T2 not available for subject {getattr(ds, 'subject_id', 'unknown')}")
        return ds.t2
    # shift conditions
    return apply_guidance_condition(guidance_arr, condition, voxel_mm, order=1)


def execute_run(run: RunSpec, force: bool = False) -> Path:
    out_dir = Path(run.out_root) / run.run_id
    marker = out_dir / ".done"
    if marker.exists() and not force:
        return out_dir
    if force:
        marker.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt: np.ndarray | None = None
    observed_arr: np.ndarray
    guidance_arr: np.ndarray
    voxel_mm: tuple[float, float, float]
    vois: list[np.ndarray]
    lesion_masks: list[np.ndarray] = []
    lesion_labels: list[int] = []
    patient_ds = None
    simulation_meta: dict[str, Any] = {}

    guidance_condition = run.input_params.get("guidance_condition", "exact")

    if run.study == "spheres":
        from krl_studies.datasets.spheres import SphereDataset

        ds = SphereDataset(root=run.dataset["root"])
        gt = ds.ground_truth
        guidance_arr = ds.guidance
        voxel_mm = ds.voxel_mm

        specs = None
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
            lesion_labels = [round(2 * s["radius_mm"]) for s in specs]

        observed_arr, simulation_meta = _build_observed(run, ds, gt)
        guidance_arr = _apply_guidance_condition(guidance_arr, guidance_condition, voxel_mm, ds)

        lesion_rois = derive_lesion_rois(gt) if lesion_masks else []
        exclusion = (
            np.logical_or.reduce(lesion_rois or lesion_masks)
            if (lesion_rois or lesion_masks)
            else np.zeros_like(gt, dtype=bool)
        )
        vois = background_vois(gt.shape, exclude_mask=exclusion)

    elif run.study == "brainweb":
        from krl_studies.datasets.brainweb import BrainWebDataset

        subject_id = run.dataset.get("subject_id")
        if subject_id is None:
            subject_id = run.dataset.get("subject")
        if subject_id is None:
            raise KeyError("brainweb dataset requires 'subject_id' (or 'subject') key")
        root = run.dataset.get("root")
        if root is None:
            raise KeyError("brainweb dataset requires 'root' key")

        ds = BrainWebDataset(root=Path(root), subject_id=int(subject_id))
        gt = ds.ground_truth
        guidance_arr = ds.guidance
        voxel_mm = ds.voxel_mm

        # Persisted tumour masks for CRC
        lesion_masks_arr = ds.lesion_masks
        lesion_masks = lesion_masks_arr if isinstance(lesion_masks_arr, list) else []
        if len(lesion_masks) == 0 and lesion_masks_arr is not None and lesion_masks_arr.size > 0:
            lesion_masks = [lesion_masks_arr]
        lesion_labels = [int(d) for d in ds.lesion_diameters_mm] if ds.lesion_diameters_mm else []

        observed_arr, simulation_meta = _build_observed(run, ds, gt)
        guidance_arr = _apply_guidance_condition(guidance_arr, guidance_condition, voxel_mm, ds)

        lesion_rois = ds.lesion_masks if ds.lesion_masks.size > 0 else derive_lesion_rois(gt)
        exclusion = (
            np.logical_or.reduce(lesion_masks)
            if len(lesion_masks) > 0
            else np.zeros_like(gt, dtype=bool)
        )
        vois = background_vois(gt.shape, exclude_mask=exclusion)

    elif run.study == "patient":
        from krl_studies.datasets.patients import PatientDataset

        subject_id = run.dataset.get("subject_id")
        if subject_id is None:
            subject_id = run.dataset.get("subject")
        if subject_id is None:
            raise KeyError("patient dataset requires 'subject_id' (or 'subject') key")
        root = run.dataset.get("root")
        if root is None:
            raise KeyError("patient dataset requires 'root' key")

        if run.input_kind != "native":
            raise ValueError(
                f"patient study supports input_kind='native' only, got {run.input_kind!r} "
                "(patient native is pure CIL; sirf_sim belongs to spheres/brainweb)"
            )

        patient_ds = PatientDataset(subject_id=str(subject_id), root=Path(root))
        gt = None
        observed_arr = patient_ds.pet
        guidance_arr = patient_ds.guidance
        voxel_mm = patient_ds.voxel_mm
        guidance_arr = _apply_guidance_condition(guidance_arr, guidance_condition, voxel_mm, patient_ds)

        if patient_ds.rois is not None:
            try:
                exclude_mask = patient_ds.rois.astype(bool)
                vois = background_vois(observed_arr.shape, exclude_mask=exclude_mask)
            except ValueError:
                vois = []
        else:
            vois = []

    else:
        raise NotImplementedError(f"unknown study: {run.study!r} (expected 'spheres', 'brainweb', or 'patient')")

    method_cls = METHOD_REGISTRY[run.method_name]
    if run.method_name == "gtm":
        raise NotImplementedError("gtm via PETPVC is not wired into the runner yet")
    params = dict(run.method_params)
    n_iterations = int(params.pop("iterations", 1))
    if run.method_name == "iy":
        if gt is not None:
            regions, brain = _iy_region_defaults(gt)
            params.setdefault("region_masks", regions)
            params.setdefault(
                "psf_sigma_vox",
                tuple(float(params.get("fwhm_mm", 5.0)) * FWHM_TO_SIGMA for _ in range(3)),
            )
            params.setdefault("brain_mask", brain)
        else:
            assert patient_ds is not None
            if patient_ds.rois is None:
                raise NotImplementedError(
                    "iterative Yang requires ROI segmentation for patient study but no ROIs.nii.gz "
                    f"found for subject {patient_ds.subject_id!r} "
                    "(patient has no ground truth; provide ROIs.nii.gz or use non-PVC methods)"
                )
            unique_labels = np.unique(patient_ds.rois[patient_ds.rois > 0])
            if len(unique_labels) == 0:
                raise NotImplementedError(
                    f"iterative Yang requires non-empty ROI segmentation for patient study "
                    f"but ROIs.nii.gz for subject {patient_ds.subject_id!r} contains no labelled voxels"
                )
            regions = [patient_ds.rois == lbl for lbl in unique_labels]
            brain = observed_arr > 0
            if not np.any(brain):
                brain = np.ones_like(observed_arr, dtype=bool)
            background_region = brain & ~(patient_ds.rois > 0)
            if np.any(background_region):
                regions.append(background_region)
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
        save_image(_wrap(observed_arr, voxel_mm), obs_path)
        save_image(_wrap(guidance_arr, voxel_mm), guide_path)
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
            if gt is not None:
                value = nrmse(it.image, gt)
                row["nrmse"] = value
                if value < best_nrmse:
                    best_nrmse = value
                    best_img = it.image.copy()
                for i, mask in enumerate(lesion_masks):
                    d_mm = lesion_labels[i]
                    if vois:
                        row[f"crc_mm{d_mm}"] = crc_percent(mask, it.image, gt, vois)
                if vois:
                    row["bv_percent"] = background_variability(it.image, vois)
            else:
                if vois:
                    row["bv_percent"] = background_variability(it.image, vois)
            rows.append(row)
            final_img = it.image

    if not rows:
        raise RuntimeError(f"{run.run_id}: method produced no iterates")

    write_metrics_csv(rows, out_dir / "metrics.csv")
    if final_img is not None:
        save_image(_wrap(final_img, voxel_mm), out_dir / "final.nii.gz")
    if best_img is not None:
        save_image(_wrap(best_img, voxel_mm), out_dir / "best_nrmse.nii.gz")

    manifest = {
        "run_id": run.run_id,
        "study": run.study,
        "input_kind": run.input_kind,
        "input_params": run.input_params,
        "method": run.method_name,
        "method_params": run.method_params,
        "dataset": run.dataset,
        "sim": run.sim,
        "simulation": simulation_meta,
        "guidance_condition": guidance_condition,
        "status": "complete",
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "krl_version": _pkg_version("cil-krl"),
        "krl_studies_version": _pkg_version("krl-studies"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    marker = out_dir / ".done"
    marker.write_text(dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    return out_dir
