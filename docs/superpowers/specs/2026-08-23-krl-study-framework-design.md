# KRL Study Framework — Design Spec

Date: 2026-08-23
Status: Approved design, pending implementation plan
Branch: `study-framework`

## 1. Purpose

Build a reproducible experiment framework to benchmark Richardson–Lucy (RL)
deconvolution regularisation methods for PET, in support of a publication.
The framework must:

- Compare RL, KRL, HKRL, MAP-RL/DTV against PVC comparators (iterative Yang,
  GTM) and simple baselines.
- Quantify robustness to kernel mismatch: deconvolution kernel assumptions vs
  the true/effective blur of the input image, including recon-side resolution
  modelling errors and anatomical guidance mismatch.
- Scale to future patient cohorts without code changes (config-driven).
- Produce publication-quality figures and tables with one command.

## 2. Studies

| Study | Data | Ground truth | Role |
|---|---|---|---|
| `spheres` | Synthetic sphere phantom (files committed; pure simulation) | Exact | Controlled CRC/BV benchmark; continuity with MIC2025 results |
| `brainweb` | Generated from the `brainweb` package (regenerable; outputs gitignored) | Exact: known blur + Poisson noise + known tumours | Paper core: realistic OSEM/RDP inputs, tumour CRC, mismatch study |
| `patient` | MK-H001 now; more patients later. User-placed NIfTI pairs under `data/patients/<id>/` | None | Qualitative / ROI demonstration; proves cohort extensibility |

## 3. Repo layout

New branch off `master`. New importable sub-package `studies/` installed via a
`[studies]` optional extra. The existing `examples/` directory is untouched and
remains legacy reference material.

```
studies/
  krl_studies/
    data/          # spheres loader, brainweb prep, patient cohort adapter
    simulation/    # SIRF forward projection + Poisson + OSEM/RDP reconstruction
    methods/       # method wrappers with uniform per-iteration interface
    metrics/       # NRMSE, CRC per lesion size, background variability, bias/variance
    runner/        # scenario YAML parsing, sweep expansion, resumable execution
    analysis/      # tidy results store -> figures + LaTeX tables
    cli.py         # `python -m krl_studies ...`
  scenarios/       # committed YAML scenario definitions per study
  tests/           # unit tests (no SIRF) + slow SIRF smoke tests
data/              # runtime data tree (mostly gitignored; see section 4)
results/           # gitignored experiment output
```

SIRF/STIR itself is not pip-installable here; environment setup is documented
(conda env reusing the user's existing SIRF installation).

## 4. Data layer & conventions

- **Committed:** synthetic spheres phantom only (`phant_orig.nii`,
  `phant_mri.nii`; blurred `phant_pet.nii` is regenerable but also committed for
  exact MIC2025 continuity). Geometry: 200×200×100 @ 1 mm.
- **Not committed (gitignored):** everything derived from BrainWeb downloads,
  all patient data.
- **Patient ingestion:** `data/patients/<id>/PET.nii.gz` + `T1.nii.gz`
  (+ optional ROI/segmentation files). MK-H001 files live at
  `data/patients/MK-H001/`; geometry 181×217×181 MNI @ 1 mm. A committed
  `data/README.md` documents the expected layout and file sources; adding a
  patient = copying files + one YAML entry.
- **BrainWeb prep** extends the existing `brainweb_phantoms.py` approach:
  - Fixed standard tumour set per subject: diameters {8, 12, 16, 24} mm,
    contrast ≈ 4:1, placed at defined GM / WM / background locations,
    identical across subjects for per-size aggregation.
  - MR guidance stays lesion-free; tissue labels retained for PVC
    segmentation.

## 5. Simulation layer (SIRF/STIR)

Shared by `spheres` and `brainweb` studies:

1. Resample ground-truth emission (with tumours) to scanner grid.
2. Forward project with an STIR AcquisitionModel (PSF resolution model;
   attenuation from uMap optional/configurable).
3. Scale to configured count level; Poisson-sample with derived seeds.
   Default grid: {5e7, 1e8, 2e8} prompt counts (Vision 600 brain-scan order of
   magnitude: 98 MBq × 35 min), × 10 noise realisations at the mid level and
   3 elsewhere; exact values are scenario-configurable and calibrated against
   reconstructed-noise appearance during implementation.
4. Reconstruct from noisy prompts:
   - OSEM (plain),
   - RDP-β for a small β grid (defaults {10, 50, 100}; config-sweepable).
5. Save deconvolution input images (NIfTI) + `manifest.json` (counts, seed,
   β values, true FWHM, geometry, software versions).

**Resolution presets** (effective residual blur of the deconvolution *input*;
anchored to the Vision 600 Hoffman measurements in
`Vision_resolution.docx`, which estimate residual resolution on top of
recon-side PSF modelling):

| Preset | Residual FWHM (x,y,z) | Meaning |
|---|---|---|
| `psf-matched` | (4.5, 4.5, 6.4) mm | Recon PSF modelling matched to truth → matches clinical PSF-modelled recons (MK-H001 type) |
| `psf-undersized` | configurable (default: halfway) | Recon models PSF with too-small kernel → residual blur + sharper noise texture |
| `psf-none` | (5.7, 5.7, 7.8) mm | No resolution modelling in recon → full effective blur remains |

Simulation calibration ensures inputs actually exhibit the preset's effective
resolution. If the Siemens Vision 600 scanner model is unavailable in the
installed SIRF version, fall back to mMR and record it in the manifest.

## 6. Methods layer

Uniform interface: every method consumes (input image, guidance image,
parameters) and yields **per-iteration iterates**, so metric callbacks capture
convergence curves identically across methods.

| Method | Implementation |
|---|---|
| RL | `krl.RichardsonLucy`, no kernel operator |
| KRL | `krl.RichardsonLucy` + kernel operator (`sigma_anat` sweep) |
| HKRL | as KRL + hybrid weights + kernel freezing |
| MAP-RL / DTV | existing LBFGSB + directional prior path |
| Iterative Yang | new in-house implementation (numpy/CIL) |
| GTM | PETPVC CLI wrapper |
| Baselines | input image itself (OSEM / RDP-β); Gaussian post-smoothing |

PVC methods segment regions from BrainWeb tissue labels (exact in simulation).
Patient-study segmentation is deferred (out of scope for phase 1).

## 7. Metrics

Computed per iteration via callbacks:

- NRMSE vs ground truth.
- Contrast recovery coefficient (CRC) per tumour/sphere size.
- Background variability (COV over multiple background VOIs).
- Objective function value where defined.

Aggregated across noise realisations: bias and variance per
(method × parameter set), supporting recovery-vs-COV trade-off curves.

## 8. Analysis & reporting

- Per-run directory: `manifest.json`, per-iteration metric CSVs, selected
  iteration images (NIfTI).
- Aggregator builds a tidy store (CSV/parquet) across runs.
- Publication outputs:
  - Recovery-vs-COV trade-off curves
  - NRMSE-vs-iteration curves
  - CRC-vs-lesion-size curves
  - Mismatch sensitivity plots (metric vs assumed-FWHM error, per
    recon-PSF condition and anatomy condition)
  - Profiles through lesions/spheres
  - Best-iteration tables (oracle min-NRMSE and fixed-iteration policy),
    LaTeX-ready
- One command regenerates all figures/tables from the results store.

## 9. Runner & configs

- Scenario YAMLs define: dataset(s) + seeds; simulation parameters
  (count-level grid × realisations); recon conditions (`psf-none`,
  `psf-undersized`, `psf-matched`); input variants (OSEM, RDP-β…);
  method grids (product expansion of parameter values); mismatch axes
  (assumed-FWHM grid around each condition's actual residual; anatomy:
  exact, ±2 mm, ±5 mm shifts, T1↔T2 swap).
- Defaults exercise sensible subsets of the full cross-product; the full
  product is available by configuration.
- CLI: `python -m krl_studies.run <scenario.yaml> --out results/<study>/<tag>/`
- Resumable: each expanded run writes a completion marker; re-invocation skips
  completed runs (safe on flaky cluster nodes).
- HPC integration: generated SGE array-job submission script (one array task
  per run, UCL CS cluster conventions: `qsub`, `#$ -l gpu=true` when needed);
  docker-compose reuse for the local 3080 box. macOS development uses the
  numba CPU backend (CIL+torch OpenMP conflict documented in README).

## 10. Testing strategy

- Unit tests (no SIRF required): tumour placement masks, metric mathematics on
  toy arrays, iterative Yang against an analytic tiny case, YAML expansion,
  manifest round-trip, resumability markers.
- Slow integration test (SIRF env, opt-in marker): tiny-volume end-to-end
  sim → recon → deconvolve → metrics.
- Existing library CI untouched; studies tests run via dedicated make targets
  (`make study-test`, `make study-test-slow`).

## 11. Dependencies

- Core: existing `krl` package, CIL, numpy/scipy/nibabel, matplotlib, pandas,
  PyYAML (+pyarrow for parquet).
- Simulation env: SIRF/STIR (conda), `brainweb` package for BrainWeb prep.
- PVC comparator: PETPVC binary (cluster/docker), wrapped via subprocess.
- `[studies]` extra carries what is pip-installable; SIRF/PETPVC setup is
  documented in `studies/README.md`.

## 12. Out of scope (phase 1)

- Committing any patient-derived data.
- Automatic segmentation for patient PVC comparators.
- Scatter/randoms modelling beyond SIRF defaults.
- GPU/MPS backend work on macOS.

## 13. Reference facts

- Vision 600 brain protocol resolutions (Hoffman estimate): PSF+TOF recon
  residual (4.5, 4.5, 6.4) mm; plain OSEM3D+TOF (5.7, 5.7, 7.8) mm.
- MK-H001 PET range 0–731; T1 in HU-like units; both MNI 181×217×181 @ 1 mm.
- Spheres phantom intensity range 0–10 (GT), guidance MRI 0–2.6.
- Existing analysis artefacts worth preserving conceptually: recovery-vs-COV
  curves, NRMSE-vs-recovery curves (user's Aug 2026 exports).
