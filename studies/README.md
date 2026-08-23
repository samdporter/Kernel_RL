# krl-studies

Benchmark framework for RL deconvolution regularisation methods (design spec:
`../docs/superpowers/specs/2026-08-23-krl-study-framework-design.md`, plan:
`../docs/superpowers/plans/2026-08-23-study-framework-core.md`).

## Install

Requires the parent repo's environment (CIL + `cil-krl`; use the repo-root
`.venv`), then:

    uv pip install -e "./studies[dev]"     # or: make study-install

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

## Environment notes

- SIRF/STIR (Plan 2 simulation layer): install into the same conda/venv env
  used for CIL; see https://github.com/SyneRBI/SIRF-SuperBuild.
- Heavy sweeps are intended for the GPU box (docker-compose under `examples/`)
  and the UCL CS SGE cluster (array-job templates arrive in Plan 3). macOS
  development uses the numba CPU backend only.

## Status / roadmap

- Plan 1 (this branch): data layer, methods, metrics, resumable runner,
  image-space inputs (`reference`, `quick_sim`).
- Plan 2: SIRF forward projection + Poisson + OSEM/RDP recon inputs,
  recon-PSF conditions, BrainWeb preparation, patient study enablement.
- Plan 3: aggregated tidy results store, publication figures/tables, SGE +
  docker orchestration.
