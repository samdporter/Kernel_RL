# Paper Readiness Review

Date: 2026-08-27  
Reviewed revision: `a682552` on local branch `study-framework`  
Scope: scientific idea, novelty, implementation correctness, experimental validity, patient evidence, analysis, and reproducibility

## Executive assessment

The repository is a promising study platform, but it is not ready to support a paper submission or a full campaign run. The infrastructure is more mature than the scientific evidence. Several implementation and analysis defects would invalidate or materially mislabel the intended HKRL, BrainWeb, PSF-mismatch, and publication-reporting results.

The strongest defensible paper direction is:

> A reproducible, controlled comparison of post-reconstruction PET deconvolution and PVC methods under resolution, noise, and anatomical-guidance mismatch, with a locked selection policy and an illustrative patient cohort.

The current repository does not establish a new kernel reconstruction algorithm. KRL/KEM, hybrid PET-MR kernel EM, sensitivity to PET-MR inconsistency, DTV deconvolution, iterative Yang, GTM, and PETPVC all have prior literature. Novelty must therefore come from the rigor and scope of the comparison, a clearly distinct and validated freezing strategy, a practical parameter-selection rule, open CIL/SIRF software, or credible patient validation.

Overall status: **not ready for campaign execution or manuscript drafting beyond methods/background**.

## What is already strong

- The core package has a clean CIL-facing API, CPU/GPU backends, packaging metadata, and a meaningful core test suite.
- The changelog openly records and fixes a serious historical adjoint race and dtype problem (`CHANGELOG.md:19-37`).
- The study runner is config-driven, resumable, deterministic by run ID, and records per-run manifests (`studies/krl_studies/runner/execute.py:352-371`).
- Poisson seeds are deterministic and shared conditions can support paired comparisons (`studies/krl_studies/simulation/simulate.py:116-120`).
- The SIRF image is digest-pinned and deterministic thread settings are documented.
- Metric primitives, scenario expansion, ingestion, selection, reporting, SGE planning, and simulation adapters have substantial unit-test coverage.
- Synthetic sphere data are committed and BrainWeb generation is automated.
- The intended mismatch axes are scientifically useful: assumed deconvolution FWHM, input resolution, count level, reconstruction prior, and guidance perturbation.
- The framework is designed to add patients without changing source code.

## Blocking correctness findings

### 1. Campaign HKRL is not hybrid

Severity: **critical; invalidates all planned HKRL runs**

`HKRLMethod` changes only `freeze_iteration` (`studies/krl_studies/methods/richardson_lucy.py:148-154`). Hybrid weighting defaults to false, and neither mismatch scenario supplies `hybrid: true` (`studies/scenarios/spheres_mismatch.yaml:24-25`, `studies/scenarios/brainweb_mismatch.yaml:24-25`). The planned runs labelled HKRL are therefore static KRL variants; `sigma_emission` and freezing are inactive.

Required disposition: do not run or interpret the current HKRL plans. Add an end-to-end test that HKRL enables hybrid weighting, responds to `sigma_emission`, differs from KRL before freezing, and uses a fixed operator after freezing.

### 2. BrainWeb physical voxel size is wrong

Severity: **critical; invalidates BrainWeb geometry and may prevent execution**

`BrainWebDataset.voxel_mm` returns `self.pet_gt.shape` instead of NIfTI voxel spacing (`studies/krl_studies/datasets/brainweb.py:310-312`). The runner uses this value for simulation, millimetre shifts, CIL geometry, blur parameters, and saved images (`studies/krl_studies/runner/execute.py:202-220`, `306-313`). Nominal millimetre operations are consequently wrong.

The runner also leaves `lesion_masks` as a NumPy array and evaluates `if lesion_masks` at `studies/krl_studies/runner/execute.py:215-218`, which raises the ambiguous-array-truth-value error for a populated BrainWeb dataset.

Required disposition: all existing or future BrainWeb results from this revision are invalid. Verify affine, orientation, voxel spacing, lesion masks, physical shifts, and round-trip NIfTI geometry in an end-to-end test before rerunning.

### 3. The named reconstruction-PSF mismatch is not implemented

Severity: **critical for the proposed central claim**

The design and campaign guide describe a fixed physical forward PSF with absent, undersized, or matched reconstruction-side resolution modelling (`docs/superpowers/specs/2026-08-23-krl-study-framework-design.md:73-104`; `studies/README_PLAN3_CAMPAIGN.md:41-49`). In code, every preset sets `recon_model_fwhm_xyz=None` and varies a pre-blur applied to the ground truth (`studies/krl_studies/simulation/presets.py:27-43`; `studies/krl_studies/simulation/simulate.py:146-172`).

The current experiment can be described as sensitivity to different residual input blur, but not reconstruction-model mismatch. Either implement the stated experiment with fixed acquisition truth and varying reconstruction models, or rename and narrow the scientific claim.

### 4. SIRF activity scaling is not validated

Severity: **critical until disproved by a numerical recovery test**

Prompts are scaled to the requested count total before Poisson sampling, but the reconstructed image is not explicitly returned to the ground-truth activity scale (`studies/krl_studies/simulation/simulate.py:155-176`). The quick simulator does perform an inverse scale. If STIR does not internally provide the intended normalization, NRMSE and count-level comparisons use inconsistent activity units, and fixed `beta`, `alpha`, and emission-kernel parameters do not represent comparable regularization strengths.

Required disposition: add high-count, zero/noise-limited recovery tests that check global activity, regional activity, count-level invariance after normalization, and the expected scaling of every regularization parameter. Do not infer correctness from shape, positivity, or deterministic output tests.

### 5. Analysis pools distinct hyperparameters

Severity: **critical; publication tables can be numerically misleading**

Full `method_params_json` is retained only in the run table, not iteration or lesion tables (`studies/krl_studies/analysis/schema.py:9-24`). Replicate aggregation groups over the remaining columns (`studies/krl_studies/analysis/aggregate.py:11-17`). Distinct `sigma_anat`, `sigma_emission`, freeze iteration, DTV `alpha`, iY damping, and neighbourhood settings can therefore be averaged as if they were replicate noise realizations.

Required disposition: give every scientifically relevant parameter a canonical identity in all metric tables, or join a stable parameter-set ID before aggregation. Add tests proving that two hyperparameter settings never collapse into one summary row.

### 6. Publication plots connect incompatible experimental strata

Severity: **critical for reporting**

Plot functions loop only over method and then connect rows from multiple conditions, priors, guidance perturbations, FWHMs, iterations, and sometimes metrics (`studies/krl_studies/analysis/plots.py:59-86`, `116-138`, `174-192`, `220-247`). Labels are taken from the first row. Existing tests establish file creation, not correct grouping.

The CRC trade-off builder receives only the scalar iteration table, although CRC is stored in the separate lesion table (`studies/krl_studies/analysis/report.py:20-24`, `58-74`, `95-103`). It therefore cannot build the intended CRC-versus-BV data.

Required disposition: specify each figure's exact filters, grouping, selection policy, and uncertainty unit; then test the rows and plotted series, not only PNG existence.

## Major scientific and methodological risks

### Oracle tuning and stopping

`select_oracle` chooses minimum NRMSE separately for each run and noise realization (`studies/krl_studies/analysis/selection.py:6-16`). This uses test ground truth to select the stopping point. It is acceptable only as a clearly labelled upper bound. It is not a deployable policy and cannot be the main comparison.

Use separate development and held-out evaluation data. Lock method parameters and stopping rules before evaluating test subjects and test noise realizations. Report the oracle only as secondary context.

### Comparator fairness and incompleteness

The design promises RL, KRL, HKRL, DTV, iterative Yang, GTM, and simple baselines, but the mismatch scenarios contain five methods and GTM is explicitly unwired (`studies/krl_studies/runner/execute.py:259-261`). The unprocessed reconstruction and post-smoothing baseline are also absent from the main scenarios.

iY receives regions derived from ground truth rather than the documented BrainWeb tissue labels (`studies/krl_studies/runner/execute.py:111-115`, `264-272`). Its PSF sigma is expressed in millimetres but passed as voxel units without dividing by voxel spacing. This is both leakage and a unit error.

Comparator tuning budgets, inputs, segmentations, and stopping criteria must be symmetric and declared in advance.

### Metrics and ROIs

Global NRMSE includes the full padded volume and normalizes by the ground-truth maximum (`studies/krl_studies/metrics/nrmse.py:8-13`). It can be dominated by background rather than lesions or tissue.

Background VOIs are random voxel-space spheres constrained only by margins and lesion exclusion (`studies/krl_studies/metrics/rois.py:29-71`). They are not restricted to valid brain, phantom background, or NEMA locations and are not physically defined for anisotropic voxels. CRC/BV values can therefore sample air or irrelevant tissue.

Define a primary endpoint such as lesion CRC at matched background variability, plus masked NRMSE and secondary image-quality metrics. Validate metric code against analytic and NEMA-style examples.

### Statistics and sample size

The main scenarios use three noise realizations at all count levels, despite the design proposing ten at the central count level. Reporting provides mean, sample SD, and count only. There are no paired effects, confidence intervals, hierarchical or cluster-aware uncertainty, multiplicity handling, or power justification.

Noise realizations are technical replicates, not independent patients. Lesions within one image are also correlated. The analysis unit and hierarchy must be explicit.

### Simulation realism

The simulation always uses reduced acquisition geometry (`42` views, `64` tangential bins, span `1`, maximum ring difference `1`) and four subiterations in the main scenarios (`studies/krl_studies/simulation/simulate.py:33-36`; scenario lines 5-9). Scatter and randoms are out of scope. These choices can support a controlled technical study but not a claim that the data reproduce a clinical Vision or mMR acquisition.

The documented calibration scenario only runs zero post-smoothing and does not call the available FWHM measurement/writer. It does not currently establish measured residual resolution (`studies/scenarios/resolution_calibration.yaml:9-18`).

### Patient evidence

Only `MK-H001` is configured. Local data contain PET and T1 only, with no ROI labels. Existing patient result directories are empty. The patient path therefore supplies qualitative feasibility, not quantitative or cohort evidence.

Before using patient data, document ethics/consent or data-governance approval, inclusion criteria, tracer, injected activity, uptake time, scanner, acquisition duration, reconstruction settings, corrections, registration method, segmentation protocol, and de-identification. Add affine/orientation/coregistration QC rather than relying on matching array shape.

A small cohort can support a technical feasibility paper but not a clinical-benefit claim. Determine sample size from a preregistered paired endpoint and pilot variance; do not treat repeated lesions or ROIs as independent patients.

## Core algorithm concerns

- Adaptive HKRL is nonlinear before freezing but is exposed as a CIL `LinearOperator`; its `adjoint` is the transpose associated with current weights, not a linear-operator adjoint. Pre-freeze iterations should be described as a heuristic adaptive phase, with convergence claims restricted to the frozen operator.
- The study captures latent HKRL iterates, finishes the complete run, and then maps every stored iterate through the final kernel (`studies/krl_studies/methods/richardson_lucy.py:125-134`). Pre-freeze image curves will not represent the images produced at those iterations once hybrid mode is enabled.
- Freeze timing uses CIL's iteration counter inside `update` (`src/krl/algorithms/richardson_lucy.py:160-172`). Existing tests manually emulate iteration numbering rather than testing the real CIL lifecycle, so the documented freeze iteration may be off by one update.
- Even `num_neighbours` values are accepted although storage assumes `n^3` and loop bounds can visit `(n+1)^3` neighbours. Parameters need validation and backend-parity tests.
- CPU and Torch backends differ for nonpositive sigma handling, and blur backends use different boundary conditions. Results should not be compared across backends until semantics are aligned or explicitly treated as different methods.
- MAP-RL's Armijo fallback applies the last rejected candidate after line-search failure (`src/krl/algorithms/maprl.py:286-295`). L-BFGS-B has no direct correctness tests and does not reject unsuccessful optimizer termination. Any DTV/MAP-RL convergence claim needs finite-difference gradient checks, known-solution tests, status reporting, and objective monotonicity checks where expected.
- The historical adjoint race means results generated before the `0.2.0` fix should be treated as obsolete unless independently reproduced (`CHANGELOG.md:19-29`).

## Novelty assessment

### What is not new by itself

- Kernel PET reconstruction: Wang and Qi, *PET Image Reconstruction Using Kernel Method*, IEEE TMI 2015, DOI `10.1109/TMI.2014.2343916`.
- Hybrid PET-MR kernelized EM: Deidda et al., *Hybrid PET-MR list-mode kernelized expectation maximization reconstruction*, Inverse Problems 2019, DOI `10.1088/1361-6420/ab013f`.
- PET-MR inconsistency effects in kernel reconstruction: Deidda et al., IEEE TRPMS 2019, DOI `10.1109/TRPMS.2018.2884176`.
- Anatomically guided DTV deconvolution: Gillman et al., 2023 preprint, DOI `10.1101/2023.04.23.23289004`.
- Parallel-level-set deconvolution PVC: Zhu et al., PMB 2021, DOI `10.1088/1361-6560/ac0d8f`.
- PETPVC/GTM implementations: Thomas et al., PMB 2016, DOI `10.1088/0031-9155/61/22/7975`.

This is an initial targeted check, not a systematic literature review. A formal search should cover kernel EM/KEM/HKEM, MR-guidance mismatch, post-reconstruction kernel RL, DTV/PLS deconvolution, PVC benchmarks, stopping rules, and patient validation.

### Potentially publishable contribution

The most plausible contribution is a carefully controlled, open, post-reconstruction comparison that jointly varies residual resolution, deconvolution-kernel mismatch, anatomical mismatch, count level, and reconstruction prior while enforcing held-out tuning and paired statistics. This remains valuable if prior studies have not tested the same combination under one validated framework.

A stronger algorithmic claim is possible only if the freezing strategy is mathematically distinguished from prior HKEM, its behavior is correctly implemented, and ablations show a reproducible benefit over both static KRL and established hybrid formulations.

The CIL/SIRF software can be a secondary contribution after numerical validation, environment locking, complete provenance, and a public tagged release.

## Reproducibility and repository quality

- The local `study-framework` branch has no remote counterpart; `origin` exposes only `master`. A clean external clone cannot obtain the reviewed study framework.
- No complete mismatch campaign, aggregate analysis directory, publication figures, tables, or manuscript exists locally.
- Local results are partial and stale. The one SIRF run records revision `29ef16c`; sphere quick-simulation results record `a963c85`, not the reviewed revision.
- The generated plans contain 32,400 sphere runs and 25,920 BrainWeb runs, far beyond the approximately 14,000-run documentation estimate. Simulation is repeated for every method/hyperparameter run, making the design wasteful and difficult to audit.
- BrainWeb and patient inputs are ignored and have no archived checksums or DOI. Patient data have no acquisition or governance metadata in the repository.
- Python dependencies use broad lower bounds; `studies/uv.lock` is ignored. CIL and the BrainWeb package are not pinned. Run manifests omit input hashes, dirty-tree state, full dependency versions, container digest, command line, and hardware.
- Core CI does not execute the study suite. SIRF, BrainWeb, report-generation, and GPU checks are manual or optional.
- README and package metadata claim MIT licensing and link to `LICENSE`, but no `LICENSE` file is present.
- `CITATION.cff` lacks a software version, release date, DOI, and data citations.

## Verification performed

- `ruff check src tests studies/krl_studies studies/tests`: passed.
- Full root tests: collection blocked because local `.venv` does not contain CIL.
- Full study tests: collection blocked in six modules because local `.venv` does not contain CIL.
- Non-CIL study subset: passed, with expected skips for optional environments.
- Local ignored artifact inventory: BrainWeb subject 04, patient MK-H001 PET/T1, partial results, and generated plans are present; no `analysis/` directory exists.
- Current plans contain 32,400 sphere RunSpecs and 25,920 BrainWeb RunSpecs, plus one metadata/header line each.
- `git ls-remote --heads origin` exposes only `master`; the reviewed branch is local.

## Readiness gates

The project should not launch its large campaigns until all of the following are true:

- HKRL activation, freeze timing, per-iteration output, and nonlinear interpretation are tested end to end.
- BrainWeb physical geometry and lesion-mask execution pass a real-data smoke test.
- SIRF activity scaling and count-level parameter scaling are numerically validated.
- The PSF experiment either implements genuine reconstruction-model mismatch or is renamed.
- Comparator definitions, segmentations, physical units, tuning budgets, and baselines are complete.
- Analysis preserves parameter-set identity and produces correctly stratified figures from synthetic fixtures.
- A primary hypothesis, endpoint, locked selection policy, and statistical analysis plan are written before held-out runs.
- A small balanced pilot completes from simulation through final tables and is manually audited.
- Environment, input, run, and figure provenance are sufficient for replay from a clean checkout.

Only after these gates pass should existing affected outputs be deleted or quarantined and the definitive campaigns be generated from a tagged revision.
