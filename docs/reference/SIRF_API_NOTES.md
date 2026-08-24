# SIRF API notes (calibrated against synerbi/sirf, pinned by digest)

Recorded 2026-08-24 from `studies/scripts/probe_sirf_api.py` runs inside
`docker compose -f studies/docker-compose.yaml run --rm sirf` on
linux/amd64-under-emulation (Apple Silicon host). The compose service pins the
calibrated image digest
`sha256:643c7955717ac08c6f44c6d3fe2ef064ebb54167f1da68771ed3e6dc07caf58d`;
`studies/krl_studies/simulation/_api.py`
pins exactly these surfaces; fix deltas there alone and re-probe.

## Environment

- Image `synerbi/sirf:latest` ships SIRF + STIR + CIL (26.0.1.dev) Python bindings.
- `krl_studies`, `krl` and `sirf.STIR`/`cil` co-import cleanly after
  `pip install -e '.' -e './studies[dev]'` inside the container.
- The entrypoint (`start.sh`) sources hooks that swallow stdin and mangle
  quoted args: always run `python <mounted-file>`, never `python -`.
- `OMP_NUM_THREADS=1` must be exported **inside** the compose command (not via
  the `environment:` key): the image's `omp_num_threads.sh` hook runs
  `test -z "$OMP_NUM_THREADS" && export ...` under `set -e` and aborts the
  container when the variable is pre-set. With 8 OpenMP threads OSMAPOSL
  output differs between identical runs by float32 ulps (~6e-7); with 1 thread
  reconstruction is bit-identical across runs.

## Verified signatures

```python
import stir
import sirf.STIR as st

# Scanners: enum attrs are ints; VISION's attr is Siemens_Vision_600 (no "VISION").
sc_mmr = stir.Scanner(stir.Scanner.Siemens_mMR)
sc_vis = stir.Scanner.get_scanner_from_name("Siemens VISION 600")
# mMR: 64 rings, 504 det/ring, 252 views, bin 2.086mm, ring spacing 4.0625mm, non-TOF
# Vision 600: 80 rings, 798 det/ring, 399 views, bin 1.6mm, timing 214ps,
#             max_timing_poss=264 (even -> see template bug below)

# Stock templates
acq = st.AcquisitionData("Siemens mMR")   # dims (1, 4096, 252, 344), ~3s
st.AcquisitionData("Siemens VISION 600")  # RAISES: "Number of TOF bins should be
                                          # an odd number" (Vision has 264 timing bins)

# Custom ProjDataInfo (the CTI constructor lives here, NOT as ProjDataInfoCTI):
pdi = stir.ProjDataInfo.construct_proj_data_info(
    sc, span, max_delta, num_views, num_tangential_positions)
# SWIG exposes only the 5-arg overload; defaults arc_correction=True, num_tof_bins=1.
# (7-arg form documented in C++ is unreachable through the binding.)
pdi.get_num_tof_poss()   # -> 1 (verified)

# Materialising a sirf AcquisitionData from a raw PDI: st.AcquisitionData(pdi)
# RAISES 'Wrong source in AcquisitionData constructor'. Verified round-trip:
pdm = stir.ProjDataInMemory(stir.ExamInfo(), pdi)
hs = "<tmpdir>/name.hs"
pdm.write_to_file(hs)                    # interfile pair (.hs/.s)
ad = st.AcquisitionData(hs)              # OK

# Geometry helpers
img = ad.create_uniform_image(value)     # FOV-derived grid
img.voxel_sizes()                        # METHOD -> (z_mm, y_mm, x_mm)
np.asarray(img.as_array())               # image arrays are (z, y, x)

# Acquisition model
am = st.AcquisitionModelUsingRayTracingMatrix()
am.set_up(acq, img)
proj = am.forward(img)                   # ~0.01s for a 14x63x48 sinogram (emulated)
bp = am.backward(proj)
am.set_image_data_processor(gaussian_filter)  # in-forward-model blur hook EXISTS

# Gaussian filter: set_fwhms takes its tuple in ARRAY-AXIS order (z, y, x)!
flt = st.SeparableGaussianImageFilter()
flt.set_fwhms((fwhm_z_mm, fwhm_y_mm, fwhm_x_mm))
flt.apply(image)                         # in-place (process(image) also works)

# Objective + priors
obj = st.make_Poisson_loglikelihood(prompts)
obj.set_acquisition_model(am)
prior = st.RelativeDifferencePrior()
prior.set_penalisation_factor(beta)      # also set_gamma/set_epsilon/set_kappa/set_up
obj.set_prior(prior)

# Reconstruction: NO prior/objective ctor overload (ctor arg would be a filename),
# input setter is set_input (not set_input_data), and set_up takes the initial image.
rec = st.OSMAPOSLReconstructor()
rec.set_objective_function(obj)
rec.set_num_subsets(k)                   # k must divide views (after view mashing);
                                         # else STIR aborts with unbalanced-subsets error
rec.set_num_subiterations(n)
rec.set_input(prompts)
init = img.copy()
rec.set_current_estimate(init)
rec.set_up(init)                         # IterativeReconstructor.set_up(image)
rec.process()
out = rec.get_output()

# Poisson noise: STIR's generator exists but its binding is unusable here:
g = st.PoissonNoiseGenerator(); g.set_seed(123)
g.generate_noisy_data(ad)                # error: 'Noise generating not done'
g.process(ad); g.get_output()            # runs but NOT reproducible same-seed,
                                         # and silently returns unchanged-scale data
```

## DECIDED routes

### Vision template (span-1 TOF=1)

`AcquisitionData("Siemens VISION 600")` is broken in this build (even TOF-bin
count 264). Route: build the scanner via `get_scanner_from_name`, construct a
span-1 TOF=1 ProjDataInfo with `construct_proj_data_info(sc, 1, rings-1, views,
tangential)`, materialise through the Interfile round-trip above. Verified
full-size result: AD dims `(1, 6400, 399, 344)` with `tof_bins=1`, 159 segments
(~13 s including file I/O). `_api.acquisition_template("Siemens VISION 600")`
uses this route; reduced `num_views`/`max_ring_diff`/`num_tangential` kwargs are
supported for emulation-friendly tests.

### Resolution modelling (recon-PSF conditions)

No `set_resolution_model` on the ray-tracing AM, but `set_image_data_processor`
exists (a Gaussian filter applied inside the forward model was accepted).
Plan 3 update: `make_acquisition_model(acq, image, resolution_fwhm=...)` calls
`set_up` FIRST and attaches the processor afterwards — verified working in the
pinned digest build (blur changes forward projections as expected).

Attenuation (Plan 3 re-investigation, corrected): the documented route is
available — `AcquisitionSensitivityModel(mu_map_ImageData, acq_model)` with
mu in 1/cm and `asm.set_up(acq_data)`. SIRF issue #623 records a STIR
geometry limitation for using an image-backed ASM during reconstruction and
shows the supported workaround: apply the image-backed ASM to an all-ones
acquisition, then construct a new `AcquisitionSensitivityModel` from those
acquisition-data factors. The gateway follows that route and keeps the
tracing AM separate from emission AMs.

On the reduced mMR template, the factor ASM is finite and attenuating (a
10 cm water cylinder gives min 0.377 / mean 0.613, matching exp(-0.96)).
Attaching the factor ASM to a set-up ray-tracing AM matches manual `S G`
within numerical tolerance and one-subiteration OSEM produces finite output.
The earlier near-zero attachment result was not reproducible with the
documented route; controlled comparison showed that reusing the emission AM
to construct the attenuation ASM produces NaNs, while omitting `asm.set_up`
raises the explicit STIR setup error. `make_acquisition_model(attenuation=...)`
therefore attaches only a ready-built factor ASM, and `simulate_inputs`
converts a resampled NIfTI uMap through this helper.

Resolution processor (corrected understanding): the adjoint of the composed
model L = S G P is numerically consistent (<P-model inner-product error 5e-8),
and pairing data/model FWHM correctly still yields LOWER early-iteration
central recovery than unmodelled reconstruction of identically pre-blurred
data (2.34 vs 2.97 at 1e9 counts, 12 subits). At essentially noiseless counts
unmodelled MLEM deconvolves aggressively and accurately; the clinical benefit
of PSF modelling (noise suppression) does not manifest in this regime. The
psf conditions therefore remain realised by per-condition pre-blurring to the
Vision target residuals (measured agreement ~5% transverse); recon-side
processors stay available through the gateway but unused for conditions.
Revisit both behaviours when the image digest changes or full-grid cluster
runs land.

DECIDED route for Task 3: **pre-blur the ground truth with
`SeparableGaussianImageFilter` at the condition's residual FWHM before forward
projection** (+ optional matched post-filter during reconstruction later). This
keeps the AM clean so reconstruction matches data without an embedded processor.

Tuple-order trap: presets are (x, y, z); `set_fwhms` consumes (z, y, x).
`_api.gaussian_smooth_image(image, fwhm_mm)` accepts scalar or `(fx, fy, fz)`
and reverses before calling STIR.

Measured effective FWHM of the decided route (delta phantom, pre-blur ->
ray-tracing forward -> plain OSEM 7 subsets x 8 subits, test_scanner geometry,
voxels z 2.03 / y,x 2.09 mm, no noise):

| condition | target FWHM x,y,z mm | measured z,y,x mm | deviation |
|---|---|---|---|
| psf-none | 5.7, 5.7, 7.8 | 5.60, 5.60, 6.54 | xy -2%, z -16% |
| psf-matched | 4.5, 4.5, 6.4 | 4.26, 4.27, 5.97 | xy -5%, z -7% |

Transverse widths verify within ~5%. Axial widths undershoot because STIR's
separable kernel quantises to the coarse probe grid (mapping probe: requesting
10 mm on-axis yields 7.41 mm at 2 mm voxels; raising `set_max_kernel_sizes`
does not change it). Expect tighter agreement on finer grids (Vision 1.6 mm /
planned resampling). Clinical Hoffman-profile re-verification belongs to Task 3
on realistic grids.

### Plan 3 findings (scanner-grid round trip, 2026-08-24)

- Sub-FOV image grids destabilise OSMAPOSL on the reduced mMR template:
  extent-preserving resampling of a 48³ phantom to ~(24,23,23) voxels drives
  update ratios to float-max and the estimate to a plateau. Resampling onto
  the scanner's FULL uniform-image grid, centred like a clinical
  reconstruction, is well-conditioned. `simulate_inputs` therefore maps GT via
  `geometry.resample_to_fov_zyx` / `resample_from_fov_zyx`.
- Delta phantoms (near-zero background) diverge under MLEM regardless of
  grid; calibration tests must use a compact lesion on a uniform background.
- `scipy.ndimage.zoom` needs `grid_mode=True`: default corner alignment
  biases even-ratio resamples by half a voxel.

### Poisson noise

DECIDED: numpy-side sampling in `_api.poisson_sample(prompts, seed)` -
`np.random.default_rng(seed).poisson(lam)` on the scaled prompt array, filled
back into a clone of the AcquisitionData. Verified: same seed => bitwise
identical counts, different seed differs. STIR's `PoissonNoiseGenerator` is
recorded above as unusable (error path + non-reproducible `process`). Callers
(Task 3) scale prompts to the target count total *before* calling.

## Determinism summary

- Forward/backward projection: deterministic run-to-run at fixed thread count.
- OSMAPOSL: bit-identical across runs with `OMP_NUM_THREADS=1` (compose command
  exports it); NOT bit-identical with >1 threads (~6e-7 ulp noise).
- Noise: deterministic via numpy PCG64 seeded streams.
- Subset rule: `num_subsets` must divide the (view-mashed) view count;
  `_api.reconstruct_osem` picks the largest divisor from (21,14,12,8,7,6,4,3,2).

## Not verified in-container today

- Full-resolution mMR (4096x252x344 bins) forward projection cost under
  emulation: not executed (opt-in `--full` probe stage kept for cluster use).
  All container tests therefore use reduced-span/reduced-view templates built
  from the real mMR/Vision scanner objects, or STIR's tiny `test_scanner`.
- TOF modelling anywhere (route pins everything to 1 TOF bin).
