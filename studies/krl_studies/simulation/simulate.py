"""SIRF simulation pipeline: GT -> forward -> Poisson -> OSEM/RDP.

Ground-truth arrays are (z,y,x) at ~1 mm. Vision-grid resampling is deferred
to Plan 3; this module maps the GT onto the scanner's native image grid via
``_api.make_image`` which copies voxel sizes from the acquisition template's
uniform image (ring-spacing / integer) so ``AcquisitionModelUsingRayTracingMatrix``
accepts the geometry. The only SIRF/STIR imports are through ``_api``.
"""

from __future__ import annotations

import numpy as np

from krl_studies.simulation import _api
from krl_studies.simulation.presets import resolution_for_condition

# Cache acquisition templates per scanner key to avoid repeated Interfile
# round-trips when simulate_inputs is called repeatedly (e.g. determinism
# test makes 3 calls). The cached AcquisitionData is treated as read-only.
_ACQ_CACHE: dict[str, object] = {}

# Reduced geometry used for emulation-friendly tests. Full scanner grids
# (4096 views etc.) are ~100× larger and time out under amd64 emulation;
# clinical full-grid runs belong on the cluster (Plan 3).
_REDUCED_KWARGS = {"span": 1, "max_ring_diff": 1, "num_views": 42, "num_tangential": 64}


def _get_acquisition(scanner_name: str):
    """Return (acq_template, scanner_used) with mMR fallback.

    Tries the requested scanner with reduced kwargs; on any exception falls
    back to ``Siemens mMR`` reduced (Vision TOF=1 route is used inside
    ``_api.acquisition_template``). The returned scanner name is recorded in
    meta["scanner"].
    """
    # Check cache first for the requested name.
    cache_key = scanner_name
    if cache_key in _ACQ_CACHE:
        # We need to know which scanner was actually used for this key;
        # store tuple (acq, used) rather than just acq.
        cached = _ACQ_CACHE[cache_key]
        if isinstance(cached, tuple):
            return cached
        # Legacy entry (should not happen) – treat as mMR.
        return cached, scanner_name

    def _try(name: str):
        return _api.acquisition_template(name, **_REDUCED_KWARGS)

    try:
        acq = _try(scanner_name)
        used = scanner_name
    except Exception:
        # Fallback to mMR for stability if Vision route is still problematic.
        fallback = "Siemens mMR"
        if scanner_name == fallback:
            raise
        acq = _try(fallback)
        used = fallback

    _ACQ_CACHE[cache_key] = (acq, used)
    return acq, used


def simulate_inputs(gt_array, cfg_dict):
    """Forward-project a ground-truth volume and reconstruct a noisy input.

    Parameters
    ----------
    gt_array:
        3-D ndarray with shape (z, y, x). Values are emission activity.
    cfg_dict:
        Mapping with keys ``condition`` (psf-none / psf-undersized / psf-matched),
        ``beta`` (None or float for RDP prior), ``counts`` (target total prompts),
        ``realisation`` (int), ``seed`` (int), ``n_subits`` (or ``n_subiterations``),
        and optional ``scanner`` (``"Siemens mMR"`` or ``"Siemens VISION 600"``;
        defaults to mMR for stability).

    Returns
    -------
    recon:
        3-D ndarray (z, y, x) reconstructed from noisy prompts.
    meta:
        Dict with ``true_fwhm``, ``scanner``, ``beta``, ``counts``,
        ``realisation``, ``n_subits`` (and ``seed`` for reproducibility).
    """
    gt = np.asarray(gt_array)
    if gt.ndim != 3:
        raise ValueError(f"gt_array must be 3-D (z,y,x), got shape {gt.shape}")

    condition = cfg_dict["condition"]
    true_fwhm = resolution_for_condition(condition)

    beta = cfg_dict.get("beta", None)
    counts = float(cfg_dict["counts"])
    realisation = int(cfg_dict.get("realisation", 0))
    seed = int(cfg_dict.get("seed", 0))
    # Accept both n_subits and n_subiterations spellings.
    if "n_subits" in cfg_dict:
        n_subits = int(cfg_dict["n_subits"])
    elif "n_subiterations" in cfg_dict:
        n_subits = int(cfg_dict["n_subiterations"])
    else:
        n_subits = 4

    scanner_req = str(cfg_dict.get("scanner", "Siemens mMR"))

    acq, scanner_used = _get_acquisition(scanner_req)

    # Build ImageData with scanner-compatible voxel sizes; resampling to the
    # scanner grid is deferred to Plan 3 (phantom is already ~1 mm, we just
    # adopt the scanner's native spacing so the projector accepts the geometry).
    gt_image = _api.make_image(acq, gt)

    # Pre-blur to the condition's residual FWHM (Task 1 decided route).
    blurred = _api.gaussian_smooth_image(gt_image, true_fwhm)

    # Forward model.
    am = _api.make_acquisition_model(acq, gt_image)
    prompts = _api.forward_project(blurred, am)

    # Scale so sum(prompts) == counts, then Poisson-sample.
    arr = np.asarray(prompts.as_array(), dtype=np.float64)
    total = float(arr.sum())
    if total == 0:
        scale = 0.0
    else:
        scale = counts / total
    # Use a clone so the original forward prompts are not mutated.
    scaled = prompts.clone()
    scaled.fill((arr * scale).astype(np.float32))

    seed_full = int(seed + realisation * 7919)
    noisy = _api.poisson_sample(scaled, seed=seed_full)

    # Prior: None -> plain OSEM, else RDP(beta).
    prior = None
    if beta is not None:
        prior = _api.make_rdp_prior(float(beta))

    # Uniform initial estimate with the same geometry as gt_image.
    init = gt_image.clone()
    init.fill(1.0)

    recon_image = _api.reconstruct_osem(noisy, am, init, n_subiterations=n_subits, prior=prior)

    recon = np.asarray(recon_image.as_array())
    # Ensure non-negative output (STIR should already be non-negative, but clip
    # tiny negatives from prior handling).
    recon = np.asarray(recon, dtype=np.float32)
    # STIR images are (z,y,x); preserve that ordering.
    if recon.shape != gt.shape:
        # This should not happen when gt_image was built from gt_array, but
        # guard against geometry mismatches by cropping/padding? For now raise
        # to surface the issue rather than silently reshaping.
        raise RuntimeError(f"recon shape {recon.shape} != gt shape {gt.shape}")

    # Clip tiny negatives without affecting determinism for the test's min>=0.
    # Use maximum to avoid altering bitwise determinism for identical recons?
    # Clipping is deterministic and preserves bit-identical property for same
    # cfg (same recon array -> same clipped array).
    recon = np.maximum(recon, 0, out=recon)

    meta = {
        "true_fwhm": tuple(true_fwhm),
        "scanner": scanner_used,
        "beta": beta,
        "counts": counts,
        "realisation": realisation,
        "n_subits": n_subits,
        "seed": seed,
    }

    return recon, meta
