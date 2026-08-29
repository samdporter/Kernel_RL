# krl-studies

Benchmark framework for RL deconvolution regularisation methods (design spec:
`../docs/superpowers/specs/2026-08-23-krl-study-framework-design.md`, plan:
`../docs/superpowers/plans/2026-08-24-study-framework-analysis-orchestration.md`).

## Install

Requires the parent repo's environment (CIL + `cil-krl`; use the repo-root
`.venv`), then:

    uv pip install -e "./studies[dev]"     # or: make study-install

For analysis tools (figures, tables, LaTeX export):

    uv pip install -e "./studies[analysis,dev]"   # adds pyarrow>=14

## Quick start

    python -m krl_studies.run --scenario studies/scenarios/spheres_core.yaml --dry-run
    python -m krl_studies.run --scenario studies/scenarios/spheres_core.yaml --only reference

Results land under `results/<scenario>/<run_id>/`: `manifest.json`,
`metrics.csv` (per-iteration NRMSE / CRC-per-lesion / background
variability), `final.nii.gz`, `best_nrmse.nii.gz`. Re-invocation skips runs
with a `.done` marker (`skip (marker present)`); `--force` re-runs; `--only`
substring-filters run ids.

## Methods

Registry keys: `rl`, `krl`, `hkrl`, `dtv`, `iy`, `post_smoothing`, `gtm`.
All wrappers stream per-iteration results and fail fast on unknown params.
Notes:

- `dtv` may converge before the requested iteration count (L-BFGS-B gtol/ftol).
- `iy` needs region masks; the runner injects a crude 2-compartment split from
  ground truth for spheres — tissue-label regions arrive with the BrainWeb
  study. iY amplifies noise and is region-definition sensitive; keep iteration
  budgets small.
- `gtm` requires the PETPVC binary on PATH (cluster/docker) and is not yet
  wired into the runner (raises NotImplementedError).

## SIRF simulation

SIRF/STIR-based input simulation runs inside the `synerbi/sirf` docker image
(repo mounted at `/workspace`):

    make study-docker-pull      # fetch synerbi/sirf:latest
    make study-sirf-test        # sirf-marked test suite inside the container
    make study-docker-python    # interactive shell in the container

The image is linux/amd64; on Apple Silicon it runs under emulation — workable
but slow, so keep container volumes dev-scale only (≤64³ fixtures, few
subiterations). Heavy sweeps belong on the cluster (Plan 3).

All SIRF/STIR access is funnelled through `krl_studies.simulation._api`,
pinned to surfaces verified against the live container. Read
`../docs/reference/SIRF_API_NOTES.md` (verified signatures, Vision/mMR
template routes, resolution-modelling and Poisson-noise decisions,
determinism notes) before touching anything under `simulation/`.

### `sirf_sim` inputs

Input kind `sirf_sim` replaces image-space inputs with acquisitions simulated
by SIRF: ground truth blurred to the condition's residual FWHM, forward
projected, count-scaled, Poisson-sampled (seeded), then reconstructed by
OSMAPOSL (+RDP when β is set). Example fragment (full grid:
`scenarios/spheres_sirf.yaml`):

```yaml
inputs:
  - kind: sirf_sim
    params:
      condition: [psf-none, psf-undersized, psf-matched]  # recon-PSF presets
      beta: [null, 10.0, 50.0]                            # null = plain OSEM
      counts: [1.0e7]
      realisation: [0, 1]
```

Runs expand over condition × β × counts × realisation; the seed comes from
the scenario `sim:` block, so identical configs reproduce bit-identical recons.

### Attenuation / uMap

Input kind `sirf_sim` accepts an optional `attenuation_path` parameter
pointing to a NIfTI mu-map (units 1/cm). The map is resampled to the scanner
grid and converted to an acquisition-data-backed sensitivity model following
the SIRF documented workaround (issue #623). Example:

```yaml
inputs:
  - kind: sirf_sim
    params:
      condition: psf-matched
      beta: 10.0
      counts: 5.0e7
      realisation: 0
      attenuation_path: "data/brainweb/subject_04/mu_map.nii.gz"
```

BrainWeb preparation exports `mu_map.nii.gz` (1/cm, on the PET grid) for this
purpose. The attenuation model is materialised into bin efficiencies and
attached to the acquisition model after setup.

## Plan 3: Cluster Execution & Analysis

### Campaign execution order

```bash
# 1. Calibrate geometry/resolution and counts
python -m krl_studies.run --scenario studies/scenarios/resolution_calibration.yaml

# 2. Generate a plan instead of launching a large Cartesian product
python -m krl_studies.plan --scenario studies/scenarios/spheres_mismatch.yaml --out plans/spheres.jsonl --sge plans/spheres.sge.sh

# 3. Submit on UCL SGE
qsub plans/spheres.sge.sh

# 4. Aggregate completed runs
python -m krl_studies.report aggregate --results results/spheres_mismatch --out analysis/spheres_mismatch/aggregate

# 5. Generate figures and tables
python -m krl_studies.report all --results results/spheres_mismatch --out analysis/spheres_mismatch
```

### SIRF container execution

For container-based runs (required for `sirf_sim`):

```bash
# Run a scenario in the SIRF container
make study-docker-run SCENARIO=studies/scenarios/spheres_mismatch.yaml

# Or with custom arguments
make study-docker-run SCENARIO=studies/scenarios/spheres_mismatch.yaml ARGS="--force --only rl"
```

### Report CLI

```bash
# Aggregate results into canonical CSVs
python -m krl_studies.report aggregate --results results/spheres_mismatch --out analysis/spheres_mismatch/aggregate

# Generate figures
python -m krl_studies.report figures --analysis analysis/spheres_mismatch/aggregate --out figures/

# Generate LaTeX/CSV tables
python -m krl_studies.report tables --analysis analysis/spheres_mismatch/aggregate --out tables/

# All-in-one
python -m krl_studies.report all --results results/spheres_mismatch --out analysis/spheres_mismatch
```

### SGE array jobs

Generate and submit an SGE array job for a campaign:

```bash
python -m krl_studies.plan --scenario studies/scenarios/spheres_mismatch.yaml --out plans/spheres.jsonl --sge plans/spheres.sge.sh --gpu --slots 4
qsub plans/spheres.sge.sh
```

Each SGE array task executes one RunSpec (`--index "$SGE_TASK_ID"`). Failed tasks can be resubmitted independently.

## SIRF simulation

SIRF/STIR-based input simulation runs inside the `synerbi/sirf` docker image
(repo mounted at `/workspace`):

    make study-docker-pull      # fetch synerbi/sirf:latest
    make study-sirf-test        # sirf-marked test suite inside the container
    make study-docker-python    # interactive shell in the container

The image is linux/amd64; on Apple Silicon it runs under emulation — workable
but slow, so keep container volumes dev-scale only (≤64³ fixtures, few
subiterations). Heavy sweeps belong on the cluster (Plan 3).

All SIRF/STIR access is funnelled through `krl_studies.simulation._api`,
pinned to surfaces verified against the live container. Read
`../docs/reference/SIRF_API_NOTES.md` (verified signatures, Vision/mMR
template routes, resolution-modelling and Poisson-noise decisions,
determinism notes) before touching anything under `simulation/`.

### `sirf_sim` inputs

Input kind `sirf_sim` replaces image-space inputs with acquisitions simulated
by SIRF: ground truth blurred to the condition's residual FWHM, forward
projected, count-scaled, Poisson-sampled (seeded), then reconstructed by
OSMAPOSL (+RDP when β is set). Example fragment (full grid:
`scenarios/spheres_sirf.yaml`):

```yaml
inputs:
  - kind: sirf_sim
    params:
      condition: [psf-none, psf-undersized, psf-matched]  # recon-PSF presets
      beta: [null, 10.0, 50.0]                            # null = plain OSEM
      counts: [1.0e7]
      realisation: [0, 1]
      attenuation_path: "data/brainweb/subject_04/mu_map.nii.gz"  # optional
```

Runs expand over condition × β × counts × realisation; the seed comes from
the scenario `sim:` block, so identical configs reproduce bit-identical recons.

## BrainWeb phantoms

`krl_studies.datasets.brainweb` prepares BrainWeb subjects with tissue labels.
Requires the optional download stack (kept out of package deps):

    pip install brainweb nibabel scipy

Usage:

```python
from krl_studies.datasets.brainweb import prepare_subject, BrainWebDataset

# Prepare a subject (downloads + processes)
paths, labels = prepare_subject(4, out_dir="data/brainweb/subject_04")
# paths -> pet_gt.nii.gz, mr_t1.nii.gz, mr_t2.nii.gz, labels.nii.gz, mu_map.nii.gz, lesion_masks.npz, lesion_diameters_mm.json

# Load a prepared subject for campaigns
ds = BrainWebDataset("data/brainweb", 4)
ds.ground_truth       # PET ground truth
ds.guidance           # T1 MRI
ds.t2                 # T2 MRI (for guidance_condition: t2)
ds.labels             # tissue labels
ds.lesion_masks       # boolean tumour masks (n, z, y, x)
ds.lesion_diameters_mm  # list of diameters in mm
ds.voxel_mm           # voxel sizes (z, y, x) in mm
```

`labels.nii.gz` holds integer tissue regions (0 background, 1 CSF, 2 GM,
3 WM); `regions_from_labels(labels)` converts it to `[WM, GM, CSF]` boolean
masks for region-guided methods (`iy`) and PVC comparators. By default a
standard tumour set (8/12/16/24 mm spheres at 4× contrast) is placed into the
PET ground truth (`tumour=False` to skip). Outputs land under gitignored
`data/brainweb/` — never committed, regenerate any time.

## Patient studies

Study kind `patient` consumes native NIfTI data (pure CIL path, no SIRF):
observed = `PET.nii.gz`, guidance = `T1.nii.gz` from
`data/patients/<subject_id>/`. The optional `ROIs.nii.gz` label image enables
per-iteration `bv_percent` metrics and is required by `iy`; without it those
runs raise a clear error. Optional `T2.nii.gz` enables `guidance_condition: t2`.
`gtm` remains unwired (NotImplementedError). See `scenarios/patient_cohort.yaml`.

## Environment notes

- Heavy sweeps are intended for the GPU box (docker-compose under `examples/`)
  and the UCL CS SGE cluster (array-job templates arrive in Plan 3). macOS
  development uses the numba CPU backend only.

## Status / roadmap

- Done (Plan 1): data layer, methods, metrics, resumable runner, image-space
  inputs (`reference`, `quick_sim`).
- Done (Plan 2): SIRF container environment + calibrated `_api` layer, recon-PSF
  condition presets, `simulate_inputs` pipeline, `sirf_sim` input kind wired
  into the runner, BrainWeb preparation with tissue labels, patient study
  execution path.
- Done (Plan 3): aggregated tidy results store, publication figures/tables,
  SGE + docker orchestration, attenuation/uMap modelling.

## ⚠️ Pre-2026-08-27 artifacts are quarantined

Results, plans, and prepared data from before the Phase 0+2+3 readiness fixes are invalidated and quarantined in `../invalidated_2026-08-27/` with a README explaining the defects. Do not use them for any paper figures or analysis. Regenerate from a tagged revision after the fixes land.

## Analysis optional dependencies

For plotting, tables, and report generation:

    uv pip install -e "./studies[analysis,dev]"   # adds pyarrow>=14

This enables the `krl_studies.analysis` module and the `krl_studies.report` CLI.