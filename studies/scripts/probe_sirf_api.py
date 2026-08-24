"""Probe the installed SIRF/STIR API surface and print a machine-checkable report.

Run inside the sirf container:

    docker compose -f studies/docker-compose.yaml run --rm sirf \
        python studies/scripts/probe_sirf_api.py [--only quick,geometry,ops,fwhm] [--full]

Stages:
    quick     constructors, signatures, dir() surfaces (cheap)
    geometry  custom/reduced geometries via ProjDataInfo -> Interfile round-trip
    ops       forward/backward, smoothing (axis mapping), Poisson noise routes,
              plain/RDP OSMAPOSL reconstruction, determinism checks
    fwhm      empirical effective-FWHM measurement of the decided simulation
              route (pre-blur -> ray-tracing forward -> plain OSEM)
    mmr-full  timed FULL-resolution Siemens mMR template operations (opt-in via
              --full; extremely slow under emulation)

Every check prints ``OK <name>: result`` or ``FAIL <name>: exception``; the
final verified signatures are recorded in docs/reference/SIRF_API_NOTES.md.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import traceback

import numpy as np
import sirf.STIR as st
import stir


def check(name, fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        dt = time.perf_counter() - t0
        print(f"OK   {name}: {out}  [{dt:.2f}s]")
        return out
    except Exception as exc:  # noqa: BLE001 - probe reports everything
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return None


def dump_doc(name, obj):
    try:
        doc = obj.__doc__
        if doc:
            head = "\n".join(doc.strip().splitlines()[:8])
            print(f"DOC  {name}: {head}")
    except Exception as exc:  # noqa: BLE001
        print(f"DOC  {name}: unavailable ({exc})")


# --------------------------------------------------------------- verified helpers


def make_proj_data_info(scanner, span, max_delta, num_views, num_tangential,
                        arc_correction=True, tof_bins=1):
    """Verified factory (STIR python exposes the CTI constructor here):

        ProjDataInfo.construct_proj_data_info(
            shared_ptr<Scanner>, span, max_delta, num_views,
            num_tangential_positions[, bool arc_correction[, int num_tof_bins]])

    The 7-argument overload is unreachable through SWIG; the 5-argument call
    defaults to arc_correction=True and num_tof_bins=1.
    """
    return stir.ProjDataInfo.construct_proj_data_info(
        scanner, int(span), int(max_delta), int(num_views), int(num_tangential))


def ad_from_pdi(pdi, workdir=None):
    """Materialise a sirf.STIR AcquisitionData from a raw stir ProjDataInfo.

    Verified route: ProjDataInfo -> ProjDataInMemory(ExamInfo, pdi) ->
    write_to_file(.hs) -> AcquisitionData('.hs'). SIRF's AcquisitionData
    constructor accepts only filenames, other AcquisitionData objects, or
    scanner names - never a raw ProjDataInfo.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="sirf_probe_")
    hs = os.path.join(workdir, f"ad_{next(ad_from_pdi._counter)}.hs")
    pdm = stir.ProjDataInMemory(stir.ExamInfo(), pdi)
    pdm.write_to_file(hs)
    return st.AcquisitionData(hs)


ad_from_pdi._counter = iter(range(100000))


def make_reconstructor(obj):
    """OSMAPOSLReconstructor: default ctor + set_objective_function (the ctor
    overload taking an objective is not exposed; it expects a filename)."""
    rec = st.OSMAPOSLReconstructor()
    rec.set_objective_function(obj)
    return rec


# ------------------------------------------------------------------------ quick


def quick():
    print("== quick ==")

    def scanner_enum_names():
        attrs = dir(stir.Scanner)
        out = {}
        for n in attrs:
            if any(k in n.lower() for k in ("siemens", "vision", "mmr", "test", "user")):
                v = getattr(stir.Scanner, n)
                out[n] = f"<int {v}>" if isinstance(v, int) else str(v)
        return out

    enums = check("scanner enums (VISION attr is Siemens_Vision_600)", scanner_enum_names)
    print(f"INFO scanner enums: {enums}")

    def scanner_facts():
        facts = {}
        makers = (
            ("mMR", lambda: stir.Scanner(stir.Scanner.Siemens_mMR)),
            ("VISION", lambda: stir.Scanner.get_scanner_from_name("Siemens VISION 600")),
            ("test_scanner", lambda: stir.Scanner(stir.Scanner.test_scanner)),
        )
        for label, maker in makers:
            try:
                sc = maker()
                facts[label] = {
                    "rings": sc.get_num_rings(),
                    "det_per_ring": sc.get_num_detectors_per_ring(),
                    "max_views": sc.get_max_num_views(),
                    "bin_size_mm": round(sc.get_default_bin_size(), 3),
                    "inner_radius_mm": sc.get_inner_ring_radius(),
                    "ring_spacing_mm": round(sc.get_ring_spacing(), 3),
                    "max_timing_poss": getattr(sc, "get_max_num_timing_poss", lambda: "n/a")(),
                    "tof_res_ps": sc.get_timing_resolution(),
                }
            except Exception as exc:  # noqa: BLE001
                facts[label] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return facts

    facts = check("scanner facts", scanner_facts)
    print(f"INFO scanner facts: {facts}")

    def mmr_template():
        t0 = time.perf_counter()
        acq = st.AcquisitionData("Siemens mMR")
        return {"dims": acq.dimensions(), "ctor_s": round(time.perf_counter() - t0, 2)}

    check("mMR stock acquisition template", mmr_template)

    def vision_direct():
        return st.AcquisitionData("Siemens VISION 600").dimensions()

    check("VISION direct template (expected FAIL: even TOF bins)", vision_direct)

    def vision_span11_tof1():
        sc = stir.Scanner.get_scanner_from_name("Siemens VISION 600")
        pdi = make_proj_data_info(sc, 1, sc.get_num_rings() - 1,
                                  sc.get_max_num_views(), 344)
        return {"views": pdi.get_num_views(),
                "tangential": pdi.get_num_tangential_poss(),
                "segments": pdi.get_num_segments(),
                "tof_bins": pdi.get_num_tof_poss(),
                "ad_dims": ad_from_pdi(pdi).dimensions()}

    check("VISION span-1 TOF=1 template via construct+interfile", vision_span11_tof1)

    def surfaces():
        out = {}
        r = st.OSMAPOSLReconstructor()
        out["osmaposl"] = sorted(n for n in dir(r) if not n.startswith("_"))
        p = st.RelativeDifferencePrior()
        out["rdp"] = [n for n in dir(p) if "set" in n and not n.startswith("_")]
        g = st.PoissonNoiseGenerator()
        out["poisson_gen"] = [n for n in dir(g) if not n.startswith("_")]
        am = st.AcquisitionModelUsingRayTracingMatrix()
        out["am"] = [n for n in dir(am) if ("resolution" in n.lower()
                                            or "processor" in n.lower())]
        f = st.SeparableGaussianImageFilter()
        out["sgif"] = [n for n in dir(f) if ("fwhm" in n.lower() or n in
                                             ("apply", "process", "set_up"))]
        obj = st.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
        out["logL"] = [n for n in dir(obj) if ("acquisition_model" in n or "prior" in n)]
        return out

    surf = check("class surfaces", surfaces)
    for k, v in surf.items():
        print(f"INFO {k}: {v}")
    dump_doc("make_Poisson_loglikelihood", st.make_Poisson_loglikelihood)


# --------------------------------------------------------------------- geometry


def build_tiny_scanner():
    """Small geometry for fast emulated probes: STIR's built-in test scanner."""
    return stir.Scanner(stir.Scanner.test_scanner)


def tiny_pdi(sc=None, span=1, max_delta=2, views=63, tangential=48):
    sc = sc or build_tiny_scanner()
    return make_proj_data_info(sc, span, max_delta, views, tangential)


def geometry():
    print("== geometry ==")

    def tiny():
        pdi = tiny_pdi()
        ad = ad_from_pdi(pdi)
        img = ad.create_uniform_image(1.0)
        return {"ad_dims": ad.dimensions(), "img_shape": np.asarray(img.as_array()).shape,
                "voxel_sizes_zyx_mm": [round(float(v), 3) for v in img.voxel_sizes()]}

    tinyinfo = check("tiny geometry (test_scanner, span1 d<=2, 63v, 48t)", tiny)
    print(f"INFO tiny: {tinyinfo}")

    def reduced_mmr():
        sc = stir.Scanner(stir.Scanner.Siemens_mMR)
        pdi = make_proj_data_info(sc, 1, 1, 42, 64)
        ad = ad_from_pdi(pdi)
        img = ad.create_uniform_image(1.0)
        am = st.AcquisitionModelUsingRayTracingMatrix()
        am.set_up(ad, img)
        t0 = time.perf_counter()
        proj = am.forward(img)
        return {"ad_dims": ad.dimensions(), "img_shape": np.asarray(img.as_array()).shape,
                "fwd_s": round(time.perf_counter() - t0, 2),
                "fwd_sum": float(np.asarray(proj.as_array()).sum())}

    rm = check("reduced-span mMR (span1 |d|<=1, 42v, 64t) end-to-end", reduced_mmr)
    print(f"INFO reduced mMR: {rm}")


# -------------------------------------------------------------------------- ops


def _tiny_setup(background=1.0, hotspot=None):
    ad = ad_from_pdi(tiny_pdi())
    img = ad.create_uniform_image(float(background))
    arr = np.asarray(img.as_array())
    if hotspot is not None:
        value, offset = hotspot
        idx = tuple(s // 2 + o for s, o in zip(arr.shape, offset))
        arr[idx] = value
        img.fill(arr.astype(np.float32))
    am = st.AcquisitionModelUsingRayTracingMatrix()
    am.set_up(ad, img)
    return ad, img, am


def _line_fwhm_bins(line):
    prof = np.asarray(line, dtype=np.float64)
    prof = prof - 0.5 * (prof[0] + prof[-1])
    pk = int(np.argmax(prof))
    half = prof[pk] / 2.0
    if half <= 0 or prof.max() <= 0:
        return float("nan")

    def cross(i_lo, i_hi):
        y0, y1 = prof[i_lo], prof[i_hi]
        if y1 == y0:
            return float(i_lo)
        return i_lo + (half - y0) / (y1 - y0) * (i_hi - i_lo)

    left = pk
    while left > 0 and prof[left] > half:
        left -= 1
    right = pk
    while right < len(prof) - 1 and prof[right] > half:
        right += 1
    lo = cross(left, left + 1) if prof[left] <= half else 0.0
    hi = cross(right, right - 1) if prof[right] <= half else float(len(prof) - 1)
    return hi - lo


def axis_fwhm_mm(arr, voxel_sizes_mm, peak_idx=None):
    arr = np.asarray(arr, dtype=np.float64)
    pk = peak_idx or tuple(int(i) for i in np.unravel_index(np.argmax(arr), arr.shape))
    out = []
    for ax in range(arr.ndim):
        sl = list(pk)
        sl[ax] = slice(None)
        out.append(round(_line_fwhm_bins(arr[tuple(sl)]) * float(voxel_sizes_mm[ax]), 2))
    return out


def ops():
    print("== ops ==")
    ad, img, am = _tiny_setup()

    def fwd_bwd():
        t0 = time.perf_counter()
        proj = am.forward(img)
        dt = time.perf_counter() - t0
        bp_sum = float(np.asarray(am.backward(proj).as_array()).sum())
        return {"dims": proj.dimensions(),
                "sum": float(np.asarray(proj.as_array()).sum()),
                "bp_sum": bp_sum, "fwd_s": round(dt, 3)}

    fb = check("forward/backward projection", fwd_bwd)
    print(f"INFO fwd/bwd: {fb}")

    def am_processor_hook():
        am2 = st.AcquisitionModelUsingRayTracingMatrix()
        f = st.SeparableGaussianImageFilter()
        f.set_fwhms((6.0, 6.0, 6.0))
        am2.set_image_data_processor(f)
        am2.set_up(ad, img)
        return "set_image_data_processor accepted SeparableGaussianImageFilter"

    check("AM in-forward-model blur hook", am_processor_hook)

    def smooth_mapping():
        arr = np.zeros(np.asarray(img.as_array()).shape, dtype=np.float32)
        arr[tuple(s // 2 for s in arr.shape)] = 1.0
        im2 = img.clone()
        im2.fill(arr)
        f = st.SeparableGaussianImageFilter()
        f.set_fwhms((10.0, 6.0, 3.0))
        if hasattr(f, "set_max_kernel_sizes"):
            f.set_max_kernel_sizes((25, 25, 25))
        f.apply(im2)
        meas = axis_fwhm_mm(im2.as_array(), img.voxel_sizes())
        vox = np.asarray(img.voxel_sizes(), dtype=np.float64)
        req = np.array([10.0, 6.0, 3.0])
        perms = [(0, 1, 2), (2, 1, 0), (2, 0, 1), (1, 2, 0)]
        scored = []
        for perm in perms:
            pred = req[list(perm)] / vox * 0 + req[list(perm)]
            scored.append((float(np.abs(np.array(meas) - pred).sum()), perm))
        scored.sort()
        return {"measured_fwhm_zyx_mm": meas, "best_request_perm_(req_idx->zyx)": scored[0][1]}

    sm = check("smooth fwhms=(10,6,3) axis mapping", smooth_mapping)
    print(f"INFO mapping result: {sm}  (array axes are z,y,x)")

    def numpy_poisson_determinism():
        lam = np.asarray(am.forward(img).as_array(), dtype=np.float64) * 50.0
        a = np.random.default_rng(123).poisson(lam)
        b = np.random.default_rng(123).poisson(lam)
        c = np.random.default_rng(124).poisson(lam)
        return {"same_seed_bitwise_equal": bool(np.array_equal(a, b)),
                "diff_seed_differs": not bool(np.array_equal(a, c))}

    check("numpy Poisson determinism (DECIDED noise route)", numpy_poisson_determinism)

    def stir_pg_attempts():
        proj = am.forward(img)
        scaled = proj.clone()
        arr = np.asarray(scaled.as_array()) * 50.0
        scaled.fill(arr.astype(np.float32))
        results = {}
        g = st.PoissonNoiseGenerator()
        g.set_seed(123)
        try:
            results["generate_noisy_data"] = repr(g.generate_noisy_data(scaled))
            out = g.get_output()
            results["gen_out_sum"] = float(np.asarray(out.as_array()).sum())
        except Exception as exc:  # noqa: BLE001
            results["generate_noisy_data"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        try:
            outs = []
            for _ in range(2):
                g2 = st.PoissonNoiseGenerator()
                g2.set_seed(123)
                clone = scaled.clone()
                g2.process(clone)
                outs.append(np.asarray(g2.get_output().as_array()))
            results["process_same_seed_bitwise_equal"] = bool(np.array_equal(*outs))
            results["process_int_valued"] = bool(np.all(outs[0] == np.round(outs[0])))
            results["process_changed_input_scale"] = not np.allclose(outs[0], arr.astype(np.float32))
        except Exception as exc:  # noqa: BLE001
            results["process"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return results

    pg = check("STIR PoissonNoiseGenerator invocation attempts", stir_pg_attempts)
    print(f"INFO STIR PG: {pg} (expected unusable -> DECIDED numpy-side sampling)")

    def osem_plain():
        proj = am.forward(img)
        obj = st.make_Poisson_loglikelihood(proj)
        obj.set_acquisition_model(am)
        rec = make_reconstructor(obj)
        rec.set_num_subsets(7)
        rec.set_num_subiterations(2)
        rec.set_input(proj)
        init = ad.create_uniform_image(1.0)
        rec.set_current_estimate(init)
        rec.set_up(init)
        rec.process()
        est = rec.get_output()
        return {"max": round(float(np.asarray(est.as_array()).max()), 4),
                "mean": round(float(np.asarray(est.as_array()).mean()), 4)}

    osemres = check("plain OSEM (7 subsets, 2 subits)", osem_plain)
    print(f"INFO OSEM: {osemres}")

    def osem_determinism():
        proj = am.forward(img)
        init = ad.create_uniform_image(1.0)
        runs = []
        for _ in range(2):
            obj = st.make_Poisson_loglikelihood(proj)
            obj.set_acquisition_model(am)
            rec = make_reconstructor(obj)
            rec.set_num_subsets(7)
            rec.set_num_subiterations(2)
            rec.set_input(proj)
            rec.set_current_estimate(init)
            rec.set_up(init)
            rec.process()
            runs.append(np.asarray(rec.get_output().as_array()).copy())
        diff = float(np.abs(runs[0] - runs[1]).max())
        return {"bitwise_equal": bool(diff == 0.0), "max_abs_diff": diff}

    check("OSEM bitwise determinism (no internal RNG)", osem_determinism)

    def osem_rdp():
        proj = am.forward(img)
        obj = st.make_Poisson_loglikelihood(proj)
        obj.set_acquisition_model(am)
        prior = st.RelativeDifferencePrior()
        prior.set_penalisation_factor(0.5)
        obj.set_prior(prior)
        rec = make_reconstructor(obj)
        rec.set_num_subsets(7)
        rec.set_num_subiterations(2)
        rec.set_input(proj)
        init = ad.create_uniform_image(1.0)
        rec.set_current_estimate(init)
        rec.set_up(init)
        rec.process()
        est = np.asarray(rec.get_output().as_array())
        return {"mean": round(float(est.mean()), 5),
                "penalisation_factor": prior.get_penalisation_factor()}

    rdpres = check("RDP(beta=0.5) OSMAPOSL via obj_fun.set_prior", osem_rdp)
    print(f"INFO RDP: {rdpres}")


# ------------------------------------------------------------------------- fwhm


PSF_NONE = (5.7, 5.7, 7.8)      # (x, y, z) target residual FWHM (mm)
PSF_MATCHED = (4.5, 4.5, 6.4)


def fwhm_experiment(condition_fwhm_xyz, n_subits=8, subsets=7):
    """Decided route for recon-PSF conditions: Gaussian-pre-blur the ground
    truth, forward project through the ray-tracing model, reconstruct with
    plain OSEM; measure the effective FWHM of the reconstructed point."""
    ad, img_hot, am = _tiny_setup(background=0.0, hotspot=(1.0e4, (0, 0, 0)))
    blurred = img_hot.clone()
    fx, fy, fz = condition_fwhm_xyz
    # set_fwhms takes its tuple in ARRAY-AXIS order (z, y, x) - verified by the
    # mapping probe above; presets are (x, y, z), so reverse.
    f = st.SeparableGaussianImageFilter()
    f.set_fwhms((fz, fy, fx))
    if hasattr(f, "set_max_kernel_sizes"):
        f.set_max_kernel_sizes((25, 25, 25))
    f.apply(blurred)

    proj = am.forward(blurred)
    obj = st.make_Poisson_loglikelihood(proj)
    obj.set_acquisition_model(am)
    rec = make_reconstructor(obj)
    rec.set_num_subsets(subsets)
    rec.set_num_subiterations(n_subits)
    rec.set_input(proj)
    init = ad.create_uniform_image(1.0)
    rec.set_current_estimate(init)
    rec.set_up(init)
    rec.process()
    est = np.asarray(rec.get_output().as_array(), dtype=np.float64)

    vox = np.asarray(img_hot.voxel_sizes(), dtype=np.float64)
    pk = tuple(int(i) for i in np.unravel_index(np.argmax(est), est.shape))
    widths_zyx = axis_fwhm_mm(est, vox, pk)
    return {"measured_fwhm_zyx_mm": widths_zyx,
            "target_fwhm_xyz_mm": condition_fwhm_xyz,
            "voxel_sizes_zyx_mm": [round(float(v), 2) for v in vox]}


def fwhm():
    print("== fwhm ==")
    for label, target in (("psf-none", PSF_NONE), ("psf-matched", PSF_MATCHED)):
        t0 = time.perf_counter()
        res = check(f"fwhm route {label} target(x,y,z)={target}",
                    lambda t=target: fwhm_experiment(t))
        print(f"TIME fwhm {label}: {time.perf_counter() - t0:.2f}s")
        print(f"INFO {label}: {res}")


# --------------------------------------------------------------------- mmr-full


def mmr_full():
    print("== mmr-full ==")

    def full_fwd():
        acq = st.AcquisitionData("Siemens mMR")
        img = acq.create_uniform_image(1.0)
        am = st.AcquisitionModelUsingRayTracingMatrix()
        t0 = time.perf_counter()
        am.set_up(acq, img)
        t_setup = time.perf_counter() - t0
        t0 = time.perf_counter()
        am.forward(img)
        t_fwd = time.perf_counter() - t0
        return {"img_dims": img.dimensions(), "set_up_s": round(t_setup, 1),
                "forward_s": round(t_fwd, 1)}

    check("FULL mMR set_up + forward (timed)", full_fwd)


# ------------------------------------------------------------------------ main


STAGES = {"quick": quick, "geometry": geometry, "ops": ops, "fwhm": fwhm,
          "mmr-full": mmr_full}


def main():
    import sirf
    print(f"INFO OMP threads before pin: {getattr(sirf, 'get_OMP_NUM_THREADS', lambda: 'n/a')()}")
    if hasattr(sirf, "set_OMP_NUM_THREADS"):
        sirf.set_OMP_NUM_THREADS(1)
        print("INFO pinned OMP threads to 1 via sirf.set_OMP_NUM_THREADS")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="quick,geometry,ops,fwhm",
                    help="comma-separated subset of: " + ",".join(STAGES))
    ap.add_argument("--full", action="store_true",
                    help="include the mmr-full stage (very slow under emulation)")
    args = ap.parse_args()
    wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    if args.full and "mmr-full" not in wanted:
        wanted.append("mmr-full")
    for stage in wanted:
        STAGES[stage]()
    print("PROBE DONE")


if __name__ == "__main__":
    main()
