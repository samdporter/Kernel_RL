# Phase 0+2+3 Readiness Fixes — Design Spec

Date: 2026-08-27  
Branch: `study-framework` @ `a682552`  
Related documents: `PAPER_READINESS_REVIEW.md`, `PAPER_NEXT_STEPS.md`, prior `docs/superpowers/specs/2026-08-23-krl-study-framework-design.md`

## 1. Purpose

Implement the correctness, data, and simulation findings from the readiness review that must be fixed before the planned campaigns (32,400 sphere and 25,920 BrainWeb runs) can be safely launched. This is the first implementation slice: Phase 0 (stop invalid work and preserve provenance), Phase 2 (correctness and validation of the core algorithms), and Phase 3 (correctness of the data and simulation path). Comparators, metrics, statistics, and patient governance (Phases 4-8) are deferred to follow-up slices.

No new scenario runs, no aggregation of stale results, and no changes to `master`. All edits stay on the local `study-framework` branch.

## 2. Out of scope (explicit)

The following readiness findings are not in this slice and remain to be addressed later:

- Parameter-set identity preservation across analysis tables (Phase 5).
- Plot stratification and CRC join (Phase 5).
- iY physical-unit conversion, ground-truth-derived regions, and BrainWeb tissue-label-based PVC (Phase 4).
- Oracle selection, fixed-iteration defaults, statistical analysis plan (Phase 5).
- Patient cohort governance and acquisition metadata (Phase 8).
- Adding the unprocessed OSEM/RDP and post-smoothing baselines to the main scenarios (Phase 4).
- The MIT `LICENSE` file and `CITATION.cff` extension (Phase 9).
- Manuscript drafting and figure manifest (Phase 9).

The slice fixes the blockers that would silently invalidate any result produced from the current state. It does not deliver a publishable paper on its own.

## 3. Provenance preservation (Phase 0)

The existing local artifacts are invalidated by the findings below and must not be reused or aggregated. We preserve them with a clear, gitignored quarantine.

- Move `results/`, `plans/`, `data/brainweb/`, and `data/patients/MK-H001/` into a new gitignored directory `invalidated_2026-08-27/` while keeping the original directory structure under it.
- Add `invalidated_2026-08-27/README.md` recording the date, the source commit `a682552`, the readiness review path, and a one-line note per artifact explaining why it is invalidated (HKRL not hybrid, BrainWeb voxel bug, SIRF activity scale not validated, etc.).
- Add `invalidated_2026-08-27/` to `.gitignore` so the artifacts remain on disk for forensic reference but are not committed.
- Do not delete the artifacts. Do not rename them. Do not modify them. They are evidence, not garbage.
- Do not regenerate plans or run anything until Phase 2 and Phase 3 corrections land.

This directory is created once, by a small helper script `tools/quarantine_invalidated_artifacts.sh`, that uses `git mv` for `results/` and `plans/` (they are currently untracked but should not be added to git; see Section 4). For `data/brainweb/` and `data/patients/`, plain filesystem `mv` is sufficient because those subdirectories are already covered by the existing `data/*` ignore rules. After the move, `data/` retains only `README.md` and `spheres/`.

## 4. Repository conventions touched by this slice

- New helper: `tools/quarantine_invalidated_artifacts.sh` (shell, `bash`, POSIX). Invoked once manually; idempotent (no-op if already run).
- New documentation: `invalidated_2026-08-27/README.md`.
- `.gitignore`: add `invalidated_2026-08-27/`. The pre-existing `results/`, `plans/`, `data/patients/`, `data/brainweb/`, and `studies/uv.lock` rules remain.
- All other code changes stay inside `src/krl/`, `tests/`, `studies/krl_studies/`, and `studies/tests/`. No new top-level packages.

## 5. Algorithm correctness (Phase 2)

### 5.1 HKRL hybrid mode must be reachable from the study method

Current behavior: `HKRLMethod.run` only forwards `freeze_iteration`; it never sets `hybrid=True` on the kernel operator. The mismatch scenarios in `studies/scenarios/spheres_mismatch.yaml` and `studies/scenarios/brainweb_mismatch.yaml` likewise omit `hybrid: true`. As a result the planned runs labelled HKRL are static KRL variants.

Design:

- The scenario YAML entry for `hkrl` is updated to include `hybrid: true` in its parameter grid, so the campaign behaviour is self-describing.
- `HKRLMethod` is changed to set `hybrid=True` on the kernel operator whenever `sigma_emission > 0` (regardless of whether `hybrid` is explicitly passed in `params`). When `sigma_emission` is zero or absent, `hybrid` is left at its default `False`, and the wrapper raises a clear `ValueError` if `freeze_iteration > 0` is requested without hybrid mode.
- The parameter whitelist in `studies/krl_studies/methods/richardson_lucy.py` keeps `hybrid` as a settable key; the scenario-supplied `hybrid: true` flows through the existing `_kernel_params` selector.
- `freeze_iteration` semantics are documented as "completed update count after which the operator becomes fixed". For `freeze_iteration=k`, the kernel is updated through update `k` and fixed for update `k+1` onwards. The CIL lifecycle test (Section 5.3) pins this.

The `tests/test_methods_richardson_lucy.py::test_hkrl_freeze_runs` test is augmented (TDD) to assert that HKRL differs from KRL, that `sigma_emission` changes the output, and that post-freeze iterates are image-stable. The pre-existing `tests/test_hybrid_freezing.py` continues to cover the kernel operator directly with `DummyGeometry`; it is augmented to check the CIL lifecycle.

### 5.2 HKRL iterates must be reported in the emission domain per iteration

Current behavior: `studies/krl_studies/methods/richardson_lucy.py::_run_kernel` runs the full algorithm to completion, then re-maps every captured latent iterate through the final kernel. With hybrid mode this is wrong: the kernel that produced iterate `t` is not the kernel that produces `t` from the final `x`.

Design:

- The wrapper iterates alongside the CIL update by calling `algo.update()` once per requested iteration in a `for` loop. After each update it reads `algo.x` and `kernel_op.direct(algo.x)` to record the emission-domain iterate.
- The latent iterate and the frozen reference (if freeze has already occurred) are captured together so the per-iteration kernel state is reproducible.
- `_Capture` is replaced by a per-step callback that snapshots both the latent image and the kernel state, plus the frozen-reference flag.
- The wrapper no longer relies on CIL's `callbacks=` argument for emission-domain capture; it still uses `_Capture` for the legacy latent-only path that some existing tests assert against, so those tests are preserved.
- The wrapper raises a clear `RuntimeError` if the run produces zero iterates.

A new test in `tests/test_methods_richardson_lucy.py` runs HKRL through `_run_kernel` with `hybrid=True`, `sigma_emission=1.0`, and `freeze_iteration=1`, asserts that the first iterate is finite, that iterates 2-3 are stable to within `1e-6` (post-freeze fixed operator), and that the post-freeze kernel state matches the recorded frozen reference.

### 5.3 HKRL freeze timing must match CIL's update counter

Current behavior: `src/krl/algorithms/richardson_lucy.py::update` decides to freeze when `self.iteration == self.freeze_iteration`. CIL's `Algorithm.run` advances the counter after `update()` returns, so a user requesting `freeze_iteration=1` sees the freeze applied after the second update, not the first.

Design:

- `update()` is changed to freeze when `self.iteration + 1 == self.freeze_iteration`, evaluated before the kernel is used for the next update. A doc comment records this convention.
- The new test in `tests/test_hybrid_freezing.py` runs `RichardsonLucy` through the real CIL `Algorithm.run` (with the existing `DummyGeometry` / `DummyImage` mocks where CIL is unavailable, gated by a `cil` fixture) and asserts that `freeze_iteration=1` causes the kernel to be fixed after exactly one update.
- Where CIL cannot be imported, the test skips with a clear message; the existing pre-CIL tests remain.

### 5.4 Kernel-operator input validation

Current behavior: `src/krl/operators/kernel_operator.py::BaseKernelOperator.__init__` accepts even or zero neighbourhood sizes and any sigma value without validation. The numba/torch code paths assume `n**3` storage and `range(-n//2, n//2+1)` indexing, which visit `(n+1)**3` entries for even `n`.

Design:

- The numba `KernelOperator` and the `TorchKernelOperator` both raise `ValueError` for `num_neighbours` that is not a positive odd integer.
- The numba `KernelOperator` raises `ValueError` for `sigma_anat`, `sigma_dist`, or `sigma_emission` that are negative (zero is permitted and is documented to disable that weighting, matching the current torch behaviour).
- These checks are placed in `BaseKernelOperator.set_parameters`, so the validation runs once per call regardless of backend.
- A new `tests/test_kernel_operator.py` (CPU-only, numba) test asserts each of the three invalid-neighbourhood cases and the negative-sigma cases. The torch parity test remains opt-in and is out of scope for this slice.

### 5.5 Activity-scale contract for `simulate_inputs`

Current behavior: prompts are scaled to the requested count total before Poisson sampling; the reconstructed image is returned without an explicit inverse scale, so the relationship between `recon` and the ground truth is not asserted by tests. The image-space `quick_sim` does perform the inverse scale.

Design:

- `simulate_inputs` records the per-call scale `counts / sum(prompts)` in `meta["scale"]`.
- A new helper `simulate_inputs` returns `meta` with `expected_scale`, computed from the average activity-weighted prompt sum divided by the requested counts. (This is a documentation of what was applied, not a normalisation correction.)
- A new test (gated by SIRF availability) builds a uniform `gt`, runs `simulate_inputs` with `counts` spanning two orders of magnitude, and asserts that the **sum of `recon`** is invariant to `counts` after dividing by the recorded `scale`, and that the **sum of `recon` × scale** is comparable across count levels. This is the numerical contract that downstream methods need; it does not make OSEM's true count-level behaviour identical.
- The test is the single source of truth for the activity-scale behaviour in this slice. Existing tests remain.

The existing `tests/test_simulation_sirf.py::test_simulate_inputs_shapes_and_determinism` test stays unchanged. The new test lives in a new module `tests/test_simulation_activity_scale.py` and is `@pytest.mark.sirf`.

### 5.6 BrainWeb voxel size and lesion-mask handling

Current behavior: `BrainWebDataset.voxel_mm` returns `self.pet_gt.shape`; the runner uses this value for physical operations and saves NIfTI files with no affine. The runner evaluates `if lesion_masks:` against a NumPy array, raising an ambiguous-truth error when tumours are present.

Design:

- `BrainWebDataset` gains `voxel_mm` from the NIfTI header, and exposes `affine` (the 4×4 array) and `orientation_zyx` (the (z, y, x) axis order derived from the affine rotation). If the on-disk affine is missing, `voxel_mm` falls back to a clearly-marked default derived from the BrainWeb mMR resolution and `affine` is `None`.
- `voxel_mm` is loaded once in `__init__` from `pet_gt.nii.gz`; if the file is missing, the constructor raises a clear `FileNotFoundError` referencing the data README.
- `studies/krl_studies/runner/execute.py` calls `bool(lesion_masks.size > 0)` instead of `if lesion_masks:` when the array is non-empty, and `if lesion_masks is not None` when it could be `None`.
- A new test `tests/test_brainweb_dataset.py::test_voxel_mm_comes_from_nifti_affine` constructs a BrainWeb-like subject with a custom voxel size, asserts that `voxel_mm` matches, and asserts that a round-trip NIfTI save/load preserves the affine. A new test `tests/test_brainweb_dataset.py::test_runner_brainweb_lesion_truth` exercises the runner's lesion-mask branch with a non-empty mask to confirm no `ValueError` is raised.

## 6. Test plan

Every change is implemented test-first. The new tests are listed by file:

- `tests/test_kernel_operator.py`: input validation (neighbourhood odd/positive, sigma non-negative).
- `tests/test_hybrid_freezing.py`: CIL-lifecycle freeze timing.
- `studies/tests/test_methods_richardson_lucy.py`: HKRL with hybrid, sigma_emission sensitivity, post-freeze stability, per-iteration emission-domain capture.
- `studies/tests/test_simulation_activity_scale.py` (new): activity-scale contract from `simulate_inputs`. Marked `@pytest.mark.sirf`.
- `studies/tests/test_brainweb_dataset.py`: voxel_mm from NIfTI affine, affine preservation across save/load, runner lesion-mask truth fix.

All non-SIRF tests must pass in the existing `.venv`. SIRF tests skip when `sirf.STIR` is unavailable. No new third-party dependencies.

## 7. Documentation and changelog

- `CHANGELOG.md` gains a new "Unreleased" entry listing the corrections in this slice, with one bullet per fix and a "see `docs/superpowers/specs/2026-08-27-phase-0-2-3-readiness-fixes-design.md` for design" pointer.
- `studies/README.md` gains a one-paragraph note that pre-slice campaign outputs (results, plans, brainweb, MK-H001) are quarantined, with a path to the README inside the quarantine.
- `PAPER_NEXT_STEPS.md` is updated in a small follow-up patch: replace the "First cluster-scale run" gate with a reference to this spec and the "Phase 6 pilot exit criterion" remains the only valid gate for the actual campaign.

## 8. Verification

- `ruff check src tests studies/krl_studies studies/tests` exits 0.
- `pytest tests studies/tests -q` reports zero new failures. Pre-existing skips for CIL/SIRF/BrainWeb remain skips.
- `git status` shows only the quarantined move (tracked paths unchanged), the new `tools/quarantine_invalidated_artifacts.sh` script, the new `invalidated_2026-08-27/README.md`, the `.gitignore` change, source/test edits, and the changelog/README updates.
- The quarantined move is confirmed by re-running `ls invalidated_2026-08-27/` and observing the original subdirectories present and the originals absent from their old locations.

## 9. Risks and mitigations

- The freeze-timing change in `src/krl/algorithms/richardson_lucy.py` could break the legacy `tests/test_hybrid_freezing.py` tests. The change is gated by a dedicated new test plus a clear doc comment; the legacy test is updated to reflect the new semantics.
- `voxel_mm` for BrainWeb depends on a correct NIfTI header. If `prepare_subject` writes a wrong header, downstream is still wrong. The new test verifies a save/load round trip with an explicit voxel size; the broader BrainWeb download test remains in `test_brainweb_dataset.py` but is not the gate for this slice.
- The activity-scale test requires SIRF. The slice is shipped even if SIRF is unavailable locally; the test is opt-in via the `sirf` marker and does not block CI.
- Quarantining `data/brainweb` and `data/patients` moves ~51 MB into a directory that is gitignored. This is the intended outcome but is irreversible without restoring from another clone. We mitigate by leaving the originals in place (plain `mv`, not deletion) and by writing the quarantine README before the move.

## 10. Definition of done

- All Section 5 corrections are implemented and the corresponding tests pass.
- The quarantine script and README are in place; `invalidated_2026-08-27/` is gitignored.
- The CHANGELOG and studies README updates are in place.
- `ruff` passes; the non-SIRF test suite passes.
- No code in `master`. No new scenario plans. No new run directories.
- The user has confirmed the spec before any code change.
