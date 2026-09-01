"""SIRF simulation pipeline: GT -> forward -> Poisson -> OSEM/RDP.

Ground-truth arrays are (z,y,x) on their native NIfTI grid. The volume is
resampled to the scanner's physical image grid before projection and the
reconstruction is resampled back to the original grid, so the returned array
stays aligned with the caller's GT/guidance while the simulation runs on real
scanner geometry.

Resolution modelling follows the Plan 3 truth/recon split (Task 4):
    L = S G P
where P_true (the truth-side blur) is fixed across all conditions and
P_recon (the in-model reconstruction-side blur) varies per condition. The
truth-side blur is applied as a Gaussian pre-blur before forward projection;
the recon-side blur is attached to the reconstruction acquisition model via
``make_acquisition_model(..., resolution_fwhm=...)``. Because P_true is
shared, the noisy prompts depend only on counts/seed -- the acquisition is
identical across conditions and only the reconstruction changes.

The only SIRF/STIR imports are through ``_api``.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from krl_studies.simulation import _api
from krl_studies.simulation.geometry import resample_from_fov_zyx, resample_to_fov_zyx
from krl_studies.simulation.presets import condition_spec

# Cache acquisition templates per scanner key to avoid repeated Interfile
# round-trips when simulate_inputs is called repeatedly (e.g. determinism
# test makes 3 calls). The cached AcquisitionData is treated as read-only.
_ACQ_CACHE: dict[str, object] = {}

# Reduced geometry used for emulation-friendly tests. Full scanner grids
# (4096 views etc.) are ~100× larger and time out under amd64 emulation;
# clinical full-grid runs belong on the cluster.
_REDUCED_KWARGS = {"span": 1, "max_ring_diff": 1, "num_views": 42, "num_tangential": 64}


def _get_acquisition(scanner_name: str):
    """Return (acq_template, scanner_used) for the exact requested scanner.

    Paper runs must not silently change scanner: if the requested template
    cannot be built, the original exception propagates. The cache is keyed by
    the exact scanner name (the reduced geometry is fixed in
    ``_REDUCED_KWARGS`` so changing it would require its own cache key).
    """
    cache_key = scanner_name
    if cache_key in _ACQ_CACHE:
        cached = _ACQ_CACHE[cache_key]
        if isinstance(cached, tuple):
            return cached
        raise RuntimeError("corrupt acquisition cache entry")

    acq = _api.acquisition_template(scanner_name, **_REDUCED_KWARGS)
    used = scanner_name
    _ACQ_CACHE[cache_key] = (acq, used)
    return acq, used


def _load_attenuation(
    attenuation_path: str | Path,
    acq_template,
    scanner_shape,
    scanner_voxel_mm,
):
    """Load a uMap NIfTI, resample it, and build its acquisition ASM."""
    nii = nib.load(str(attenuation_path))
    sizes = nib.affines.voxel_sizes(nii.affine)
    source_voxel_mm = (float(sizes[2]), float(sizes[1]), float(sizes[0]))
    umap_arr = np.transpose(nii.get_fdata().astype(np.float32), (2, 1, 0))
    umap_scanner, _ = resample_to_fov_zyx(umap_arr, source_voxel_mm, scanner_shape, scanner_voxel_mm)
    umap_image = _api.make_image(acq_template, umap_scanner)
    return _api.make_acquisition_sensitivity(umap_image, acq_template)


def simulate_inputs(gt_array, cfg_dict):
    """Forward-project a ground-truth volume and reconstruct a noisy input.

    Parameters
    ----------
    gt_array:
        3-D ndarray with shape (z, y, x). Values are emission activity on the
        input grid described by ``input_voxel_mm``.
    cfg_dict:
        Mapping with keys ``condition`` (psf-none / psf-undersized / psf-matched),
        ``beta`` (None or float for RDP prior), ``counts`` (target total prompts),
        ``realisation`` (int), ``seed`` (int), ``n_subits`` (or ``n_subiterations``),
        optional ``input_voxel_mm`` ((z,y,x) mm of gt_array; default 1 mm),
        optional ``attenuation_path`` (uMap NIfTI), and optional ``scanner``
        (``"Siemens mMR"`` or ``"Siemens VISION 600"``; defaults to mMR).

    Returns
    -------
    recon:
        3-D ndarray (z, y, x) on the original input grid.
    meta:
        Dict describing both grids, the resolution models, the scanner actually
        used, count/noise configuration, and the full derived seed.
    """
    gt = np.asarray(gt_array)
    if gt.ndim != 3:
        raise ValueError(f"gt_array must be 3-D (z,y,x), got shape {gt.shape}")

    spec = condition_spec(cfg_dict["condition"])

    beta = cfg_dict.get("beta", None)
    counts = float(cfg_dict["counts"])
    realisation = int(cfg_dict.get("realisation", 0))
    seed = int(cfg_dict.get("seed", 0))
    seed_full = int(seed + realisation * 7919)
    if "n_subits" in cfg_dict:
        n_subits = int(cfg_dict["n_subits"])
    elif "n_subiterations" in cfg_dict:
        n_subits = int(cfg_dict["n_subiterations"])
    else:
        n_subits = 4

    input_voxel_mm = tuple(float(v) for v in cfg_dict.get("input_voxel_mm", (1.0, 1.0, 1.0)))
    scanner_req = str(cfg_dict.get("scanner", "Siemens mMR"))

    acq, scanner_used = _get_acquisition(scanner_req)
    scanner_shape, scanner_voxel_mm = _api.scanner_grid(acq)

    attenuation = None
    attenuation_path = cfg_dict.get("attenuation_path")
    if attenuation_path is not None:
        attenuation = _load_attenuation(Path(attenuation_path), acq, scanner_shape, scanner_voxel_mm)

    # Resample GT onto the scanner's FULL image grid, centred like a clinical
    # reconstruction. Extent-preserving sub-FOV grids destabilise OSMAPOSL
    # (zero-sensitivity voxels blow up the update ratio); the native FOV grid
    # is well-conditioned.
    gt_scanner, _ = resample_to_fov_zyx(gt, input_voxel_mm, scanner_shape, scanner_voxel_mm)
    gt_image = _api.make_image(acq, gt_scanner)

    # Plan 3 truth/recon split (Task 4): the condition's forward_model_fwhm_xyz
    # is the truth-side pre-blur applied BEFORE forward projection -- it is
    # shared across all conditions so prompts depend only on counts/seed. The
    # condition's recon_model_fwhm_xyz (None for psf-none) is attached to the
    # reconstruction acquisition model via make_acquisition_model. Targets
    # remain provisional until calibration signs them off
    # (docs/reference/SIRF_API_NOTES.md).
    gt_blurred = _api.gaussian_smooth_image(gt_image, spec.forward_model_fwhm_xyz)
    forward_am = _api.make_acquisition_model(acq, gt_image, attenuation=attenuation)
    prompts = _api.forward_project(gt_blurred, forward_am)

    # Scale so sum(prompts) == counts, then Poisson-sample.
    arr = np.asarray(prompts.as_array(), dtype=np.float64)
    total = float(arr.sum())
    scale = counts / total if total > 0 else 0.0
    if scale <= 0:
        raise ValueError(
            f"prompt scale factor must be positive, got scale={scale} "
            f"(counts={counts}, sum(prompts)={total})"
        )
    scaled = prompts.clone()
    scaled.fill((arr * scale).astype(np.float32))

    noisy = _api.poisson_sample(scaled, seed=seed_full)

    prior = None
    if beta is not None:
        prior = _api.make_rdp_prior(float(beta))

    init = gt_image.clone()
    init.fill(1.0)
    recon_am = _api.make_acquisition_model(
        acq, init, attenuation=attenuation, resolution_fwhm=spec.recon_model_fwhm_xyz
    )

    recon_image = _api.reconstruct_osem(noisy, recon_am, init, n_subiterations=n_subits, prior=prior)

    recon_scanner = np.asarray(recon_image.as_array(), dtype=np.float32)
    recon = resample_from_fov_zyx(recon_scanner, scanner_voxel_mm, gt.shape, input_voxel_mm)
    recon = np.maximum(recon, 0.0, out=recon)

    # The OSEM reconstruction sees count-scaled prompts; invert the prompt
    # scaling so the returned image is in the same units as the input GT.
    recon = recon / scale

    meta = {
        "input_shape": tuple(int(v) for v in gt.shape),
        "input_voxel_mm": tuple(float(v) for v in input_voxel_mm),
        "scanner_shape": tuple(int(v) for v in scanner_shape),
        "scanner_voxel_mm": tuple(float(v) for v in scanner_voxel_mm),
        "forward_model_fwhm": tuple(float(v) for v in spec.forward_model_fwhm_xyz),
        "recon_model_fwhm": None if spec.recon_model_fwhm_xyz is None else tuple(
            float(v) for v in spec.recon_model_fwhm_xyz
        ),
        "target_residual_fwhm": tuple(float(v) for v in spec.target_residual_fwhm_xyz),
        "scanner": scanner_used,
        "beta": beta,
        "counts": counts,
        "realisation": realisation,
        "seed": seed_full,
        "n_subits": n_subits,
        "attenuation": attenuation is not None,
        "prompt_scale": scale,
        "activity_units": "ground_truth",
    }
    return recon, meta
