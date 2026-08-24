"""The ONLY module in krl_studies that imports sirf/stir directly.

Every wrapper matches the verified behaviour recorded in
docs/reference/SIRF_API_NOTES.md. If SIRF changes, fix here alone.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np


def _require_sirf():
    try:
        import sirf.STIR  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "SIRF is required for simulation; run inside the sirf container "
            "(make study-sirf-test) or a SIRF-enabled environment"
        ) from exc


def _sirf():
    _require_sirf()
    import sirf.STIR as st
    import stir

    return st, stir


# Subset counts STIR accepts for OSMAPOSL: must divide the (view-mashed) view
# count, so pick the largest candidate that divides it (SIRF_API_NOTES.md).
_SUBSET_CANDIDATES = (21, 14, 12, 8, 7, 6, 4, 3, 2, 1)

_VISION_NAMES = {"Siemens VISION 600", "VISION", "vision"}
_MMR_NAMES = {"Siemens mMR", "mMR"}


def acquisition_template(
    scanner_name: str,
    span: int = 1,
    max_ring_diff: int | None = None,
    num_views: int | None = None,
    num_tangential: int | None = None,
):
    """Acquisition template for 'Siemens mMR' or 'Siemens VISION 600'.

    Full-size templates are returned unless size reducers are given; the Vision
    route is always the span-1 TOF=1 ProjDataInfo construction because the
    stock Vision template raises a TOF odd-bin error in this build
    (docs/reference/SIRF_API_NOTES.md).
    """
    st, stir = _sirf()
    if scanner_name in _MMR_NAMES:
        stock = st.AcquisitionData("Siemens mMR")
        if max_ring_diff is None and num_views is None and num_tangential is None:
            return stock
        sc = stir.Scanner(stir.Scanner.Siemens_mMR)
    elif scanner_name in _VISION_NAMES:
        sc = stir.Scanner.get_scanner_from_name("Siemens VISION 600")
    else:
        raise ValueError(f"unsupported scanner_name: {scanner_name!r}")

    rings = sc.get_num_rings()
    mrd = rings - 1 if max_ring_diff is None else min(int(max_ring_diff), rings - 1)
    views = sc.get_max_num_views() if num_views is None else int(num_views)
    tangential = sc.get_max_num_non_arccorrected_bins() if num_tangential is None else int(num_tangential)
    pdi = stir.ProjDataInfo.construct_proj_data_info(sc, int(span), mrd, views, tangential)
    if pdi.get_num_tof_poss() != 1:
        raise RuntimeError(
            "expected TOF=1 template; see docs/reference/SIRF_API_NOTES.md"
        )
    pdm = stir.ProjDataInMemory(stir.ExamInfo(), pdi)
    workdir = tempfile.mkdtemp(prefix="krl_studies_acqtpl_")
    hs = os.path.join(workdir, "template.hs")
    pdm.write_to_file(hs)
    return st.AcquisitionData(hs)


def make_acquisition_model(acq_template, image):
    """Set-up ray-tracing acquisition model for the given geometry."""
    st, _ = _sirf()
    am = st.AcquisitionModelUsingRayTracingMatrix()
    am.set_up(acq_template, image)
    return am


def forward_project(image, acq_model):
    return acq_model.forward(image)


def gaussian_smooth_image(image, fwhm_mm):
    """SeparableGaussianImageFilter wrapper (post-filter / pre-blur).

    fwhm_mm is a scalar or an (fx, fy, fz) tuple in mm. STIR's set_fwhms takes
    its tuple in array-axis order (z, y, x), so tuples are reversed here
    (docs/reference/SIRF_API_NOTES.md). Returns a smoothed copy.
    """
    st, _ = _sirf()
    if np.isscalar(fwhm_mm):
        fwhm_zyx = (float(fwhm_mm),) * 3
    else:
        fx, fy, fz = (float(v) for v in fwhm_mm)
        fwhm_zyx = (fz, fy, fx)
    flt = st.SeparableGaussianImageFilter()
    flt.set_fwhms(fwhm_zyx)
    out = image.clone()
    flt.apply(out)
    return out


def poisson_sample(prompts, seed: int):
    """Count-domain Poisson noise on scaled prompt data.

    numpy-side sampling (STIR's PoissonNoiseGenerator binding is not
    reproducible same-seed): same seed => bitwise identical counts.
    prompts must already be scaled to the target count level by the caller.
    """
    _require_sirf()
    rng = np.random.default_rng(int(seed))
    lam = np.asarray(prompts.as_array(), dtype=np.float64)
    out = prompts.clone()
    out.fill(rng.poisson(lam).astype(np.float32))
    return out


def reconstruct_osem(prompts, acq_model, image, n_subiterations, prior=None):
    """OSMAPOSL reconstruction; prior=None gives plain OSEM, else RDP-beta etc.

    Deterministic given OMP_NUM_THREADS=1 (pinned in the compose command).
    """
    st, _ = _sirf()
    obj = st.make_Poisson_loglikelihood(prompts)
    obj.set_acquisition_model(acq_model)
    if prior is not None:
        obj.set_prior(prior)

    n_views = np.asarray(prompts.as_array()).shape[-2]
    subsets = next(n for n in _SUBSET_CANDIDATES if n_views % n == 0)

    rec = st.OSMAPOSLReconstructor()
    rec.set_objective_function(obj)
    rec.set_num_subsets(subsets)
    rec.set_num_subiterations(int(n_subiterations))
    rec.set_input(prompts)
    init = image.clone()
    rec.set_current_estimate(init)
    rec.set_up(init)
    rec.process()
    return rec.get_output()
