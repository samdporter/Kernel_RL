# Plan 3 Campaign Execution Guide

This document describes the exact commands to execute the full Plan 3 campaign in order.

## Campaign Order

### 1. Calibrate geometry/resolution and counts
```bash
python -m krl_studies.run --scenario studies/scenarios/resolution_calibration.yaml
```
This runs the resolution calibration scenario to establish measured PSF FWHM values for each condition.

### 2. Generate a plan instead of launching a large Cartesian product
```bash
python -m krl_studies.plan --scenario studies/scenarios/spheres_mismatch.yaml --out plans/spheres.jsonl --sge plans/spheres.sge.sh
```
This generates a JSONL run plan and an SGE array submission script instead of executing the full Cartesian product locally.

### 3. Submit on UCL SGE
```bash
qsub plans/spheres.sge.sh
```
Submit the SGE array job to the cluster. Each task executes one RunSpec independently.

### 4. Aggregate completed runs
```bash
python -m krl_studies.report aggregate --results results/spheres_mismatch --out analysis/spheres_mismatch/aggregate
```
This ingests all completed run directories, computes replicate summaries, and produces canonical CSV tables.

### 5. Generate figures and tables
```bash
python -m krl_studies.report all --results results/spheres_mismatch --out analysis/spheres_mismatch
```
This generates publication-ready figures and LaTeX/CSV tables from the aggregated data.

---

## Key Concepts

### True Forward-Model PSF
The physical point spread function of the scanner (scanner resolution + physics effects). This is fixed for a given scanner and acquisition protocol.

### Reconstruction-Side PSF Model
The PSF explicitly modeled during reconstruction:
- `none`: No resolution modeling in reconstruction
- `undersized`: PSF model smaller than true forward PSF
- `matched`: PSF model matches true forward PSF

### Deconvolution Method's Assumed FWHM
The FWHM value assumed by the deconvolution method (e.g., KRL's `sigma_anat` converted to FWHM). This is independent of the reconstruction-side PSF model.

### Anatomy Guidance Condition
The anatomical image used to guide the deconvolution:
- `exact`: Perfectly aligned T1 MRI
- `t2`: T2-weighted MRI (different contrast)
- `shift_p2`, `shift_m2`, `shift_p5`, `shift_m5`: Rigid spatial shifts of ±2mm/±5mm

### Count Level and Noise Realisation
- `counts`: Total coincidence counts in the simulated PET acquisition
- `realisation`: Poisson noise realisation index (different noise instances)

---

## Campaign Matrix (spheres_mismatch.yaml)

The mismatch campaign explicitly crosses:
- **Recon PSF condition**: `psf-none`, `psf-undersized`, `psf-matched` (3)
- **Deconvolution FWHM**: 4.0, 5.0, 5.7, 6.5, 7.5 mm (5)
- **Guidance perturbation**: `exact`, `shift_p2`, `shift_m2`, `shift_p5`, `shift_m5` (5)
- **Beta (prior strength)**: `null`, `10.0`, `50.0` (3)
- **Counts**: `5e7`, `1e8`, `2e8` (3)
- **Realisation**: `0`, `1`, `2` (3)

**Total combinations**: 3 × 5 × 5 × 3 × 3 × 3 = 2,025 runs per method.

With 7 methods (rl, krl, hkrl, dtv, iy, post_smoothing, rl+gtm): ~14,000 runs total.

---

## Other Campaigns

### resolution_calibration.yaml
Calibrates the effective resolution for each PSF condition. Single realisation, multiple count levels.

### brainweb_mismatch.yaml
Same mismatch design as spheres but with BrainWeb subject 04 anatomy. Adds `t2` guidance condition.

### patient_cohort.yaml
Demonstrates the no-ground-truth path on patient MK-H001. Uses `native` input kind (no simulation). No SGE submission.

---

## Execution Notes

- All random seeds are deterministic (`seed=1337` + `realisation * 7919`)
- SGE array tasks execute one RunSpec each (`--index "$SGE_TASK_ID"`)
- Failed SGE tasks can be resubmitted independently
- `OMP_NUM_THREADS=1` enforced for bitwise-deterministic reconstructions
- All outputs written to `results/`, `plans/`, `analysis/` directories (git-ignored)