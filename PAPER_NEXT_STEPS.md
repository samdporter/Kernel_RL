# Paper Next-Steps Plan

Date: 2026-08-27  
Starting point: `PAPER_READINESS_REVIEW.md`  
Objective: reach a defensible technical PET methods paper without spending cluster time on invalid or unidentifiable experiments

## Recommended paper scope

Use a robustness/comparison paper as the primary route:

> Evaluate anatomically guided post-reconstruction PET deconvolution and established PVC comparators under controlled residual-resolution, noise, reconstruction-prior, and PET-MR mismatch, using held-out parameter selection and an illustrative patient cohort.

Treat the software framework as a reproducibility contribution. Treat the freezing strategy as an algorithmic contribution only if a literature review and ablation study establish that it is distinct and beneficial.

Do not claim clinical benefit from a small convenience cohort. With limited patient numbers, frame the patient arm as feasibility, failure-mode analysis, and qualitative/ROI consistency.

## Phase 0: Stop invalid work and preserve provenance

- [ ] Do not submit `plans/spheres.jsonl` or `plans/brainweb.jsonl` in their current form.
- [ ] Record checksums and revision metadata for all local BrainWeb, patient, plan, and result artifacts before changing anything.
- [ ] Mark existing HKRL, BrainWeb, pre-`0.2.0`, and current PSF-labelled outputs as non-paper/invalidated artifacts rather than mixing them with future results.
- [ ] Create a small issue register from every critical and high finding in `PAPER_READINESS_REVIEW.md`.
- [ ] Decide the canonical repository and publish the study branch before collaborators generate results.

Exit criterion: no current invalidated result can enter a future aggregate unnoticed.

## Phase 1: Lock the scientific question

- [ ] Perform a systematic literature search for KEM/KRL, HKEM, PET-MR inconsistency, post-reconstruction kernel RL, DTV and parallel-level-set deconvolution, iterative Yang, GTM/PETPVC, mismatch robustness, and clinical PVC validation.
- [ ] Build a related-work table with algorithm, reconstruction versus post-processing, anatomy/emission features, mismatch experiments, datasets, patient count, metrics, comparators, and open code/data.
- [ ] Choose one primary claim. Recommended: robustness of methods to controlled input-resolution and anatomical mismatch under a deployable selection policy.
- [ ] Decide whether the experiment is about residual input blur or genuine reconstruction-side PSF mismatch. Do not use both descriptions interchangeably.
- [ ] Write one primary hypothesis and no more than three secondary hypotheses.
- [ ] Define the primary endpoint before seeing held-out results. Recommended candidate: lesion CRC at a prespecified background-variability level, with masked NRMSE secondary.
- [ ] Define a smallest scientifically meaningful paired difference for the primary endpoint.
- [ ] Specify which claims are simulation-only, phantom-supported, patient-feasibility, or clinical.

Exit criterion: a two-page protocol states the contribution, hypotheses, estimands, primary endpoint, and scope of claims without referring to whichever method wins.

## Phase 2: Correct and validate the algorithms

- [ ] Make HKRL explicitly enable hybrid emission weighting.
- [ ] Test that KRL and HKRL differ when expected, `sigma_emission` changes output, and freezing produces a fixed operator.
- [ ] Define freeze timing in terms of completed updates and test it through the real CIL `Algorithm.run()` lifecycle.
- [ ] Capture emission-domain HKRL images at each iteration using the kernel state from that iteration.
- [ ] Document the adaptive pre-freeze phase as nonlinear; restrict fixed-objective convergence claims to post-freeze iterations.
- [ ] Reject even or invalid neighbourhood sizes and invalid sigma values consistently across CPU and Torch.
- [ ] Align or explicitly document blur boundary conditions across backends.
- [ ] Add CPU/GPU parity tests for static, hybrid, frozen, masked, normalized, float32, and float64 configurations.
- [ ] Add RL/KRL tests for adjointness, sensitivity, non-unit PSFs, boundaries, zero observations, positivity, and objective behavior.
- [ ] Add MAP-RL/L-BFGS-B finite-difference gradient tests, known-solution tests, line-search failure tests, optimizer-status checks, and dtype checks.

Exit criterion: all core tests pass in a pinned CIL environment; GPU tests pass on a real CUDA runner; no known numerical issue is hidden by a paper scenario.

## Phase 3: Correct the data and simulation path

- [ ] Load BrainWeb voxel spacing from NIfTI metadata and preserve affine/orientation information.
- [ ] Fix NumPy lesion-mask truth handling and run one complete BrainWeb reconstruction/deconvolution smoke test.
- [ ] Add physical-geometry tests using anisotropic voxels for blur, shifts, lesion sizes, iY PSF conversion, and saved outputs.
- [ ] Validate PET, T1, T2, labels, mu-map, and lesion masks for shape, affine, orientation, voxel spacing, finite values, and expected physical overlap.
- [ ] Add a high-count SIRF activity-recovery experiment and quantify output scale relative to ground truth.
- [ ] Determine how count scaling changes OSEM/RDP activity units and the effective meaning of `beta`, DTV `alpha`, and `sigma_emission`.
- [ ] Replace the calibration placeholder with a workflow that measures reconstructed point/line/sphere resolution and writes auditable calibration records.
- [ ] Choose full or justified reduced acquisition geometry. If reduced geometry remains, limit claims to a controlled simulation and quantify its difference from a clinical protocol.
- [ ] Implement fixed-truth reconstruction-side PSF mismatch if that remains the primary question; otherwise rename conditions to residual input blur levels.
- [ ] Cache each simulated noisy reconstruction once per input condition and reuse it across methods to avoid recomputing identical SIRF work.

Exit criterion: a high-count/noise-free limiting case recovers expected activity and resolution; all physical quantities have explicit units; one BrainWeb case runs end to end.

## Phase 4: Make the comparison fair

- [ ] Include the observed OSEM/RDP input as a no-PVC baseline.
- [ ] Include Gaussian post-smoothing under a tuning budget comparable to other methods.
- [ ] Complete and validate iterative Yang and GTM/PETPVC, or remove them from the stated scope.
- [ ] Use BrainWeb tissue labels for region-based comparators instead of ground-truth intensity thresholding.
- [ ] Define exactly what segmentation information each method receives in simulation and patients.
- [ ] Use physical PSF units for every method and test anisotropic conversion.
- [ ] Give each method a comparable development-set tuning budget.
- [ ] Predefine failure handling, convergence criteria, excluded runs, and maximum iteration/compute budgets.
- [ ] Report runtime and memory as secondary implementation outcomes if software efficiency is part of the contribution.

Exit criterion: a method-comparison table shows equal input data, allowed side information, parameter-search budget, stopping policy, and failure policy for every method.

## Phase 5: Redesign metrics, selection, and statistics

- [ ] Define valid phantom and brain/tissue support masks for image metrics.
- [ ] Replace random whole-array background VOIs with physically defined, support-constrained VOIs; use NEMA-style layouts where applicable.
- [ ] Validate CRC, BV/COV, masked NRMSE, and any bias metric against hand-calculated fixtures.
- [ ] Preserve a stable parameter-set ID and all method parameters in run, iteration, lesion, summary, selection, and plot tables.
- [ ] Join lesion CRC data correctly into CRC-versus-BV trade-off tables.
- [ ] Make every figure function require explicit metric, iteration/selection, condition, parameter set, and grouping variables.
- [ ] Add fixture tests that assert exact plot series membership and prevent lines joining unrelated strata.
- [ ] Split development data from held-out evaluation by phantom/BrainWeb subject, not merely by iteration.
- [ ] Lock hyperparameters and a practical stopping rule using development data only.
- [ ] Retain per-run oracle NRMSE only as a labelled upper-bound analysis.
- [ ] Use paired method contrasts because methods share simulated acquisitions/noise seeds.
- [ ] Plan confidence intervals and effect sizes with a hierarchy that respects subject, noise realization, and lesions nested within subjects.
- [ ] Define multiplicity handling for methods, conditions, lesions, and secondary endpoints.
- [ ] Add completeness checks against the expected run plan before any aggregate is considered final.

Exit criterion: a frozen statistical analysis plan and synthetic end-to-end fixture produce correct tables and figures without manual filtering.

## Phase 6: Run a small acceptance pilot

- [ ] Select one sphere case and one BrainWeb subject.
- [ ] Use one count level, two paired noise realizations, one reconstruction/input-blur condition, exact guidance, and a minimal locked parameter set.
- [ ] Run every intended comparator through the same ingestion and reporting path.
- [ ] Inspect activity scale, objectives, final images, difference images, lesion profiles, CRC, BV, masked NRMSE, manifests, and run completeness.
- [ ] Confirm rerunning from a clean environment reproduces metrics within declared tolerances.
- [ ] Have a second researcher audit run IDs, parameter identities, figure grouping, and one metric by hand.

Exit criterion: the pilot passes a written acceptance checklist and produces no unexplained method, backend, or count-level discrepancy.

## Phase 7: Expand simulation evidence efficiently

- [ ] Use multiple BrainWeb subjects rather than subject 04 alone.
- [ ] Vary lesion location and contrast as well as diameter so results are not tied to one placement set.
- [ ] Use enough paired noise realizations to estimate uncertainty at the primary count level; determine the number from pilot variance rather than the current arbitrary three.
- [ ] Use a staged design: screen broad hyperparameters on development data, then evaluate only locked settings on the full held-out factorial design.
- [ ] Avoid the current 58,320-run brute-force product when a smaller identifiable design answers the hypotheses.
- [ ] Generate plans from a tagged, clean revision and record scenario hash, plan hash, input hashes, environment lock, container digest, command line, and hardware.
- [ ] Monitor balanced completion by method and condition; fail aggregation on missing expected runs.

Exit criterion: all held-out runs are complete and balanced, with no post-hoc parameter changes.

## Phase 8: Acquire and govern patient data

- [ ] Confirm ethics, consent, data-use permission, de-identification, and publication approval before adding subjects.
- [ ] Define prospective inclusion/exclusion criteria instead of selecting visually convenient cases.
- [ ] Record tracer, indication, injected activity, uptake time, scan duration, scanner, acquisition mode, reconstruction algorithm, subsets/iterations, PSF/TOF use, voxel size, corrections, and motion information.
- [ ] Add PET-T1 registration QC with affine/orientation checks and blinded visual review.
- [ ] Define patient ROIs or segmentations and document whether they are anatomical, functional, manual, automated, or consensus labels.
- [ ] Choose one patient endpoint that does not require unavailable ground truth, such as test-retest repeatability, regional stability, blinded image-quality scoring, or agreement with an external reference.
- [ ] Use a blinded pilot cohort to estimate paired endpoint variance, then perform a power/sample-size calculation.
- [ ] If only approximately 10-20 patients are feasible, frame the arm as exploratory feasibility and report uncertainty; do not claim clinical efficacy or generalizability.
- [ ] Reserve patients for development and held-out evaluation if tuning uses patient data.
- [ ] Include representative failure cases and registration mismatch, not only favorable examples.

Exit criterion: cohort metadata, governance, endpoint, analysis unit, and sample-size rationale are complete before final patient evaluation.

## Phase 9: Reproducible release and paper production

- [ ] Commit environment locks for core, studies, analysis, CIL, SIRF, BrainWeb, and PETPVC dependencies.
- [ ] Add study, report, and small SIRF smoke workflows to CI; schedule GPU validation where required.
- [ ] Extend manifests with dirty state, full dependency versions, input checksums, scenario/plan hashes, container digest, command line, and hardware.
- [ ] Add the missing MIT `LICENSE` file and complete `CITATION.cff` with version, release date, DOI, and data citations.
- [ ] Publish a tagged repository revision and archive code, inputs that can be shared, result tables, figure-source data, and environment metadata with persistent identifiers.
- [ ] Create a figure manifest mapping every manuscript figure/table to exact source tables, filters, selection policy, script, and command.
- [ ] Regenerate every figure and table from the archived result store in a clean environment.
- [ ] Write the manuscript in this order: protocol/claims, methods, locked results, limitations, discussion, abstract.
- [ ] State that simulation, phantom, and patient evidence support different levels of inference.

Exit criterion: a third party can reproduce every paper number and figure from the tagged release and archived artifacts.

## Suggested manuscript structure

1. Motivation: anatomical guidance can improve PVC but creates mismatch sensitivity and parameter-selection risks.
2. Contribution: validated open benchmark plus controlled joint mismatch study; optional freezing contribution only if established.
3. Methods: data hierarchy, simulation physics, algorithms, fair tuning, endpoints, and preregistered statistics.
4. Numerical validation: adjoints, physical units, activity scale, resolution calibration, backend parity, and limiting cases.
5. Controlled results: spheres and held-out multi-subject BrainWeb.
6. Patient feasibility: cohort, QC, predefined endpoint, uncertainty, and failure cases.
7. Discussion: comparison with prior KEM/HKEM/DTV/PVC work, limits of reduced simulation and no patient ground truth, and scope of clinical inference.
8. Reproducibility statement: code tag, data access, result archive, environment, and figure manifest.

## Immediate two-week priority order

- [ ] Day 1-2: write the scientific protocol and related-work table; choose residual-blur versus true recon-PSF mismatch.
- [ ] Day 2-5: fix and test HKRL, BrainWeb geometry/masks, and SIRF activity scaling.
- [ ] Day 5-7: fix parameter identity, CRC joins, plot stratification, and plan-completeness checks.
- [ ] Day 7-9: validate iY physical units/regions and decide whether GTM remains in scope.
- [ ] Day 9-11: run the small sphere plus BrainWeb acceptance pilot.
- [ ] Day 11-12: freeze selection/statistical policies based on development-only evidence.
- [ ] Day 12-14: revise the definitive campaign design and calculate resource/storage requirements before generating new plans.

The first cluster-scale run should occur only after the Phase 6 pilot exit criterion is signed off.
