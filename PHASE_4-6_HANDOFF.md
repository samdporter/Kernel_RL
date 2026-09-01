# Paper Readiness Phases 4-6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not mark a box complete without running its stated check.

**Goal:** Make the comparison, analysis, and pilot paths defensible enough to generate a small acceptance dataset before any paper-scale campaign is submitted.

**Architecture:** Keep SIRF/STIR calls behind `studies/krl_studies/simulation/_api.py`, keep method wrappers under `studies/krl_studies/methods/`, and carry immutable run/parameter identity from manifests through every analysis table. Implement the work in dependency order: unresolved simulation contracts first, fair comparators second, analysis correctness third, and pilot execution last.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pandas, nibabel, CIL 26, SIRF/STIR, pytest, Ruff, Docker Compose.

**Audited:** 2026-08-30 against branch `study-framework` at `f9665ea` and the digest-pinned SIRF image in `studies/docker-compose.yaml`.

---

## Rules for the Implementing Agent

- Work from the repository root.
- Do one numbered task at a time, in order.
- Write or update the named test first, run it and observe the expected failure, then make the smallest implementation change.
- Do not run `studies/scenarios/spheres_mismatch.yaml` or `studies/scenarios/brainweb_mismatch.yaml` until every gate before Phase 6 is checked.
- Do not tune parameters using Phase 6 held-out results. The pilot validates plumbing, not scientific superiority.
- Do not add GTM/PETPVC to the pilot. It remains deferred because the runner and image do not support it.
- Do not silently fall back to another scanner in paper runs. A requested scanner failure must fail the run.
- Do not change the pinned SIRF image digest while executing this plan.
- Do not commit unless the supervising user asks. If commits are requested, stage only files named by the completed task.
- Stop and report the exact command and error if a required acceptance check fails.

## Status Key

- `[x]` verified complete in the current repository.
- `[ ]` not complete or not adequately verified.
- A checked implementation item can still have an unchecked paper-readiness validation item.

---

## Audited Starting State

### Completed and retained

- [x] Pre-fix artifacts are quarantined and ignored.
- [x] HKRL enables hybrid emission weighting when configured.
- [x] HKRL per-iteration images use the corresponding kernel state.
- [x] Freeze timing is defined after completed CIL updates.
- [x] Kernel neighbourhood and sigma validation is present.
- [x] BrainWeb voxel spacing and affine are read from NIfTI metadata.
- [x] The BrainWeb lesion-mask NumPy truth-value bug is fixed.
- [x] SIRF simulation metadata records the prompt scale.
- [x] `make_acquisition_model(..., resolution_fwhm=...)` can attach a SIRF Gaussian image-data processor.
- [x] Post-smoothing and iterative Yang method classes exist.
- [x] Oracle and fixed-iteration selection functions exist in basic form.
- [x] Result ingestion separates scalar iteration metrics from lesion CRC metrics.

### Not complete despite the old handoff

- [ ] Activity normalization back to GT units is scientifically verified.
- [ ] Identical SIRF inputs are cached and reused across methods.
- [ ] Scenario scanner and seed values are consistently propagated.
- [ ] The three PSF labels represent true reconstruction-side model mismatch.
- [ ] A no-PVC input baseline exists.
- [ ] Post-smoothing uses unambiguous physical units and real voxel spacing.
- [ ] BrainWeb iY uses tissue labels and anisotropic PSF conversion.
- [ ] Parameter identity survives every analysis table.
- [ ] Lesion CRC survives trade-off construction and selection.
- [ ] Plots cannot connect unrelated strata.
- [ ] Masked NRMSE and support-constrained physical VOIs exist.
- [ ] Development locking, paired contrasts, and completeness checks exist.
- [ ] Phase 6 pilot scenarios and acceptance artifacts exist.

### Repository facts that invalidate the old handoff

- The branch has 10 commits after `a682552`, not 9; `f9665ea` was omitted.
- `PHASE_4-6_HANDOFF.md` and `uv.lock` are also untracked in the audited worktree.
- `data/brainweb/` and `data/patients/` contain no active subject data.
- `make study-sirf-test` runs only `studies/tests -m sirf`; it does not run core `tests/` or non-SIRF study tests.
- `docs/reference/SIRF_API_NOTES.md:150-166` documents the current pre-blur route, while `simulate.py:9-12` describes true model mismatch. Code and prose currently disagree.
- The current activity-scale test multiplies reconstruction by `scale`; this direction must be re-derived and verified before paper metrics are trusted.

---

## Authoritative API Contract

The digest-pinned in-container probe in `docs/reference/SIRF_API_NOTES.md` is authoritative for this repository. Released documentation is a cross-check, not a replacement for probing the pinned image.

### SIRF/STIR syntax to use

```python
import sirf.STIR as st

acq_model = st.AcquisitionModelUsingRayTracingMatrix()
acq_model.set_up(acq_data, image)

processor = st.SeparableGaussianImageFilter()
processor.set_fwhms((fwhm_z_mm, fwhm_y_mm, fwhm_x_mm))
acq_model.set_image_data_processor(processor)

objective = st.make_Poisson_loglikelihood(prompts)
objective.set_acquisition_model(acq_model)

reconstructor = st.OSMAPOSLReconstructor()
reconstructor.set_objective_function(objective)
reconstructor.set_num_subsets(num_subsets)
reconstructor.set_num_subiterations(num_subiterations)
reconstructor.set_input(prompts)
reconstructor.set_current_estimate(initial)
reconstructor.set_up(initial)
reconstructor.process()
result = reconstructor.get_output()
```

Repository-specific constraints:

- Use `set_input`, not `set_input_data`.
- Do not pass the objective to `OSMAPOSLReconstructor(...)`; its constructor argument is a parameter filename.
- SIRF image arrays and filter tuples are `(z, y, x)`.
- Repository configuration tuples are `(x, y, z)` and must be reversed at the `_api.py` boundary.
- Prefer `asarray()` in new code when the pinned image supports it; existing `as_array()` remains valid in this repository.
- Keep NumPy Poisson sampling. The pinned STIR `PoissonNoiseGenerator` route is not reproducible.
- Set acquisition storage before creating intermediates if storage is changed: `st.AcquisitionData.set_storage_scheme("memory")` or `"file"`.
- A subset count must divide the number of views. Retain `_api.reconstruct_osem` divisor selection.

### CIL syntax to use

```python
from cil.framework import ImageGeometry

geometry = ImageGeometry(
    voxel_num_x=array.shape[2],
    voxel_num_y=array.shape[1],
    voxel_num_z=array.shape[0],
    voxel_size_x=voxel_mm[2],
    voxel_size_y=voxel_mm[1],
    voxel_size_z=voxel_mm[0],
)
image = geometry.allocate(0)
image.fill(array.astype("float32"))
```

Current CIL algorithm conventions:

```python
algorithm = SomeAlgorithm(initial=initial, ...)
algorithm.run(iterations=100, callbacks=[...])
result = algorithm.get_output()
```

- Use `initial`, not `x_init`.
- Use `run(iterations=..., callbacks=[...])`, not `max_iteration`, `print_interval`, or a singular `callback=` argument.
- Use `IdentityOperator` and `GradientOperator`, not legacy `Identity` or `Gradient` names.
- Use named `get_slice(vertical="centre")` arguments.
- Use `sapyb`, not removed `axpby`.
- Allocate from geometry and call `fill`; avoid array round-trips for ordinary container arithmetic.

### Sources checked

- SIRF v3.10.1 PET examples: <https://github.com/SyneRBI/SIRF/tree/v3.10.1/examples/Python/PET>
- SIRF OSEM example: <https://github.com/SyneRBI/SIRF/blob/v3.10.1/examples/Python/PET/osem_reconstruction.py>
- SIRF STIR API source: <https://github.com/SyneRBI/SIRF/blob/v3.10.1/src/xSTIR/pSTIR/STIR.py>
- SIRF attenuation issue/workaround: <https://github.com/SyneRBI/SIRF/issues/623>
- CIL 26 optimisation API: <https://tomographicimaging.github.io/CIL/nightly/optimisation/>
- CIL framework API: <https://tomographicimaging.github.io/CIL/nightly/framework/>
- CIL v26.0.0 source: <https://github.com/TomographicImaging/CIL/tree/v26.0.0/Wrappers/Python/cil>

---

## Gate 0: Establish a Reproducible Baseline

**Files:**

- Verify: `studies/docker-compose.yaml`
- Verify: `docs/reference/SIRF_API_NOTES.md`
- Verify: `studies/scripts/probe_sirf_api.py`
- Modify only if needed: `Makefile`

- [ ] **Step 0.1: Record the worktree without changing it**

Run:

```bash
git status --short
```

Expected: HEAD is `f9665ea` unless later work was intentionally committed. Preserve all unrelated/untracked files.

- [ ] **Step 0.2: Run native lint and non-SIRF study tests**

Run:

```bash
ruff check src tests studies
PYTHONPATH=src:studies uv run --no-project \
  --with numpy --with scipy --with nibabel --with pytest \
  --with pyyaml --with matplotlib --with pandas \
  python -m pytest studies/tests -m "not sirf and not brainweb" -v
```

Expected: both commands pass. Record any environment-dependent skip.

- [ ] **Step 0.3: Run the pinned SIRF tests**

Run:

```bash
make study-sirf-test
```

Expected: all `sirf`-marked tests pass inside the digest-pinned image.

- [ ] **Step 0.4: Re-probe before changing SIRF wrappers**

Run the existing probe as a mounted file, never through `python -`:

```bash
docker compose -f studies/docker-compose.yaml run --rm sirf \
  bash -lc "export OMP_NUM_THREADS=1 && python studies/scripts/probe_sirf_api.py"
```

Expected: signatures used in `_api.py` match `docs/reference/SIRF_API_NOTES.md`.

**Gate 0 exit:** Baseline commands and outputs are recorded. No Phase 4 code starts with unexplained failures.

---

## Gate 1: Resolve Simulation Blockers

### Task 1: Fix and validate activity units

**Files:**

- Modify: `studies/krl_studies/simulation/simulate.py:155-196`
- Modify: `studies/tests/test_simulation_activity_scale.py`
- Modify: `docs/reference/SIRF_API_NOTES.md`
- Test: `studies/tests/test_simulation_sirf.py`

- [ ] **Step 1.1: Replace the current scale-direction test with a physical-unit contract**

Use a non-uniform positive phantom. At high counts and without a prior, compare the reconstructed activity after normalization against the same GT over a support mask. Test at two count totals separated by at least 100x.

Required assertions:

```python
assert meta_low["activity_units"] == "ground_truth"
assert meta_high["activity_units"] == "ground_truth"
np.testing.assert_allclose(recon_low[support].mean(), recon_high[support].mean(), rtol=0.05)
np.testing.assert_allclose(recon_high[support].mean(), gt[support].mean(), rtol=0.10)
```

Run:

```bash
make study-sirf-test
```

Expected before implementation: the new test fails because returned reconstruction values remain count-scaled and `activity_units` is absent.

- [ ] **Step 1.2: Normalize once in `simulate_inputs`**

The expected linear model convention is that scaling prompts by `scale` scales the reconstructed activity by approximately `scale`; convert the reconstruction back with division by `scale` before resampling. Raise a clear error if prompt total or scale is non-positive. Record both `prompt_scale` and `activity_units: ground_truth` in metadata. Do not multiply by `scale`.

- [ ] **Step 1.3: Verify count and prior behavior**

Run the activity test at `beta=None`. Add a separate assertion documenting that fixed RDP `beta` is not assumed count-invariant until calibrated. Do not use this task to invent a beta rescaling law.

**Task 1 exit:** High-count output is in declared GT units and stable over count scaling within test tolerances.

### Task 2: Propagate scanner and seeds; remove silent scanner fallback

**Files:**

- Modify: `studies/krl_studies/runner/execute.py:82-107`
- Modify: `studies/krl_studies/simulation/simulate.py:39-67`
- Test: `studies/tests/test_runner_sirf.py`
- Test: `studies/tests/test_runner.py`

- [ ] **Step 2.1: Add failing propagation tests**

Assert that `run.sim["scanner"]`, `run.sim["seed"]`, and `run.sim["n_subits"]` reach `simulate_inputs` unless the input explicitly overrides them. Assert `quick_sim` receives `run.sim["seed"]`.

- [ ] **Step 2.2: Implement precedence explicitly**

Use this precedence: input parameter overrides scenario `sim`, which overrides the documented default. For scanner, seed, and subiterations, construct `cfg` with that rule before calling `simulate_inputs`.

- [ ] **Step 2.3: Fail requested scanner errors**

Remove `_get_acquisition` fallback from a requested scanner to mMR. A paper run must not silently change scanner. Keep cache behavior keyed by the exact scanner and geometry configuration.

**Task 2 exit:** A Vision request either uses Vision and records it or fails; it never records mMR as a successful substitute.

### Task 3: Cache each simulated input once

**Files:**

- Modify: `studies/krl_studies/runner/execute.py`
- Create: `studies/krl_studies/runner/cache.py`
- Test: `studies/tests/test_runner.py`
- Test: `studies/tests/test_runner_sirf.py`

- [ ] **Step 3.1: Define canonical input identity**

Hash canonical JSON containing `study`, dataset identity, input kind/params, simulation params, source checksums, and code/schema version. Exclude method name and method parameters. Use sorted keys and compact separators.

- [ ] **Step 3.2: Add a failing reuse test**

Execute two method runs with identical input identity. Spy on `simulate_inputs`; expect one call. Assert both manifests contain the same `input_id` and `observed_sha256`.

- [ ] **Step 3.3: Implement atomic cache writes**

Store the observed NIfTI and simulation metadata under the result root's cache directory. Write to a temporary sibling and rename only after data and metadata are complete. Verify a checksum before reuse.

- [ ] **Step 3.4: Reject identity collisions and corrupt entries**

If metadata does not match the canonical identity or the checksum fails, raise an error instead of reusing the entry.

**Task 3 exit:** Every comparator for a paired acquisition reads the same cached observed image checksum.

### Task 4: Implement true reconstruction-model PSF conditions

**Files:**

- Modify: `studies/krl_studies/simulation/presets.py`
- Modify: `studies/krl_studies/simulation/simulate.py:139-187`
- Modify: `studies/krl_studies/simulation/_api.py:117-140`
- Modify: `docs/reference/SIRF_API_NOTES.md:115-186`
- Modify: `studies/README.md`
- Test: `studies/tests/test_simulation_presets.py`
- Test: `studies/tests/test_simulation_sirf.py`
- Test: `studies/tests/test_calibration.py`

- [ ] **Step 4.1: Write the invariant tests before changing presets**

Required invariants:

```python
assert psf_none.forward_model_fwhm_xyz == psf_undersized.forward_model_fwhm_xyz
assert psf_none.forward_model_fwhm_xyz == psf_matched.forward_model_fwhm_xyz
assert psf_none.recon_model_fwhm_xyz is None
assert psf_undersized.recon_model_fwhm_xyz is not None
assert psf_matched.recon_model_fwhm_xyz is not None
```

The SIRF test must assert identical noiseless projections and identical noisy prompts for all conditions at the same seed. Reconstructed outputs must differ only because reconstruction models differ.

- [ ] **Step 4.2: Separate true-system blur from reconstruction-model blur**

Apply one fixed true-system Gaussian before forward projection for every condition. Pass `spec.recon_model_fwhm_xyz` as `resolution_fwhm=` only when constructing `recon_am`. Do not vary truth, prompts, counts, or seed by condition.

- [ ] **Step 4.3: Treat current widths as provisional until calibration passes**

Do not claim that `(5.7, 5.7, 7.8)` and `(4.5, 4.5, 6.4)` are interchangeable true-system, reconstruction-kernel, and residual-image widths. Record each quantity under a distinct metadata key with units. The paper/pilot configuration must use values signed off in `studies/scenarios/resolution_calibration.yaml` output.

- [ ] **Step 4.4: Re-run calibration on a stable phantom**

Use a compact lesion on non-zero uniform background, not a delta phantom. Persist requested true-system FWHM, reconstruction-model FWHM, measured residual FWHM, voxel sizes, scanner, subsets/subiterations, counts, and image digest.

- [ ] **Step 4.5: Reconcile documentation**

Remove all prose saying conditions are implemented by varying pre-blur. State that `L = S G P`, with fixed truth-side `P_true` and condition-specific reconstruction-side `P_recon`.

**Task 4 exit:** The same acquisition is reconstructed under none/undersized/matched models, and calibration records distinguish model PSF from measured residual resolution.

### Task 5: Add a complete synthetic BrainWeb runner smoke test

**Files:**

- Modify: `studies/tests/test_runner.py`
- Test helper may be added to: `studies/tests/conftest.py`

- [ ] **Step 5.1: Build a temporary `subject_99` fixture**

Write all files required by `BrainWebDataset`: PET GT, T1, T2, labels, mu-map, and lesion masks/metadata. Use an anisotropic affine.

- [ ] **Step 5.2: Run `execute_run` against the fixture**

Mock only the expensive SIRF reconstruction, not `BrainWebDataset`. Assert final image, metrics, manifest, affine-derived `(z, y, x)` voxel spacing, CRC rows, BV rows, and `.done` exist.

**Gate 1 exit:** Tasks 1-5 pass. Simulation units, acquisition identity, scanner identity, PSF semantics, and BrainWeb runner plumbing are no longer ambiguous.

---

## Phase 4: Fair Comparators

### Task 6: Add the no-PVC input baseline

**Files:**

- Modify: `studies/krl_studies/methods/baselines.py`
- Modify: `studies/krl_studies/methods/__init__.py`
- Modify: `studies/tests/test_baselines_petpvc.py`
- Modify later: pilot scenario files from Task 15

- [ ] **Step 6.1: Write the failing method test**

Instantiate the registry key `input`, run one iteration, and assert exactly one `Iterate` with `iteration == 1` and an image bitwise equal to observed. Assert `n_iterations != 1` raises `ValueError`.

- [ ] **Step 6.2: Implement `InputMethod`**

The method receives observed PET only, ignores guidance, and yields a defensive copy. It does not smooth, normalize, or clip the image.

- [ ] **Step 6.3: Preserve the input reconstruction identity**

The manifest and analysis rows must retain input `beta`, scanner, condition, counts, realization, and observed checksum. Label it `input`; do not label it OSEM if `beta` is non-null.

**Task 6 exit:** The registry exposes `input`, and fixed-selection logic can retain its sole iteration.

### Task 7: Correct the post-smoothing baseline

**Files:**

- Modify: `studies/krl_studies/methods/baselines.py:16-28`
- Modify: `studies/krl_studies/runner/execute.py:262-323`
- Modify: `studies/tests/test_baselines_petpvc.py`

- [ ] **Step 7.1: Rename the parameter contract to `fwhm_mm`**

The existing `sigma_mm` name is incorrect because the implementation converts it from FWHM to sigma. Use `fwhm_mm` in new scenarios and code. No compatibility alias is required because no valid paper artifacts depend on the old key.

- [ ] **Step 7.2: Inject real voxel spacing**

Set `params["voxel_mm"] = voxel_mm` in the runner for post-smoothing. Keep tuple order `(z, y, x)` for SciPy.

- [ ] **Step 7.3: Add an anisotropic impulse test**

For anisotropic spacing, verify the passed SciPy sigma tuple equals `fwhm_mm * FWHM_TO_SIGMA / voxel_mm` axis by axis.

- [ ] **Step 7.4: Define fair tuning budgets**

Development scenarios may sweep a prespecified FWHM grid. Held-out and pilot scenarios must contain one locked FWHM selected from development only.

**Task 7 exit:** Post-smoothing has one physical interpretation and uses the dataset's spacing.

### Task 8: Correct iterative Yang inputs and physical units

**Files:**

- Modify: `studies/krl_studies/runner/execute.py:111-115,267-302`
- Modify: `studies/krl_studies/methods/iterative_yang.py`
- Verify: `studies/krl_studies/datasets/brainweb.py:233-269`
- Modify: `studies/tests/test_iterative_yang.py`
- Modify: `studies/tests/test_runner.py`

- [ ] **Step 8.1: Add a BrainWeb label-source test**

Assert BrainWeb uses `regions_from_labels(ds.labels)`, sets `brain_mask = ds.labels != LABEL_BG`, and never calls `_iy_region_defaults(gt)`.

- [ ] **Step 8.2: Convert FWHM mm to anisotropic voxel sigma**

Use:

```python
sigma_vox = tuple(
    float(fwhm_mm) * FWHM_TO_SIGMA / float(axis_voxel_mm)
    for axis_voxel_mm in voxel_mm
)
```

Assert the exact tuple in a test with unequal z/y/x spacing.

- [ ] **Step 8.3: Record segmentation provenance**

Add a stable manifest field such as `segmentation_source: brainweb_tissue_labels_v1`; for patient data record the ROI path and checksum.

- [ ] **Step 8.4: Make sphere iY policy explicit**

Do not use PET GT intensity thresholds as a fair comparator. Either supply known phantom compartment labels independent of activity or exclude iY from the sphere pilot and record the exclusion. The current plan excludes sphere iY unless independent labels are added and tested.

- [ ] **Step 8.5: Harden empty-region behavior**

Reject empty region masks with a clear error before calculating a mean. Require masks to match observed shape and not overlap unless the policy explicitly permits overlap.

**Task 8 exit:** BrainWeb iY receives only allowed segmentation information and all PSF quantities have explicit units.

### Task 9: Publish the method parity and failure policy

**Files:**

- Create: `docs/METHOD_COMPARISON_PROTOCOL.md`
- Modify: `studies/README.md`
- Test: `studies/tests/test_config.py`

- [ ] **Step 9.1: Add the parity table**

The table must list `input`, `post_smoothing`, `rl`, `krl`, `hkrl`, `dtv`, and `iy`. For each, record observed input, anatomical guidance, segmentation, emission features, assumed PSF, iteration/stopping policy, development grid size, maximum compute budget, and failure/exclusion policy.

- [ ] **Step 9.2: State GTM/PETPVC status**

Mark it deferred/future work and remove it from claimed Phase 4-6 comparisons. Keep the wrapper only as non-runner experimental code.

- [ ] **Step 9.3: Test scenario expansion counts**

Create fixture scenarios matching the documented development grids. Assert each method receives exactly its declared number of parameter combinations.

**Phase 4 exit:** A reader can see exactly which side information, tuning opportunity, and stopping budget each method receives.

---

## Phase 5: Correct Analysis, Selection, and Statistics

### Task 10: Carry stable parameter and split identity everywhere

**Files:**

- Modify: `studies/krl_studies/analysis/schema.py`
- Modify: `studies/krl_studies/analysis/aggregate.py`
- Modify: `studies/krl_studies/analysis/selection.py`
- Modify: `studies/krl_studies/analysis/tables.py`
- Modify: `studies/tests/test_analysis_schema.py`
- Modify: `studies/tests/test_analysis_ingest.py`
- Modify: `studies/tests/test_analysis_selection.py`
- Modify: `studies/tests/test_analysis_tables.py`

- [ ] **Step 10.1: Define canonical `param_set_id`**

Hash method name plus sorted compact JSON method parameters. The same method/parameters across realizations must share an ID; changing any method parameter must change it.

- [ ] **Step 10.2: Add identity columns to every schema**

Add `param_set_id`, `method_params_json`, and `data_split` to run, iteration, and lesion tables. Preserve them in summaries, selections, trade-offs, tables, and figure-source data.

- [ ] **Step 10.3: Prove parameters cannot be pooled**

Use a fixture with two `sigma_anat` values. Assert aggregation returns two groups and never averages them together.

- [ ] **Step 10.4: Expose development versus held-out split**

Read `dataset.split` into `data_split`. Accept only `development` or `held_out` for paper scenarios.

**Task 10 exit:** Every numerical row can be traced to an exact canonical method configuration and split.

### Task 11: Rebuild trade-offs and selection around canonical tables

**Files:**

- Modify: `studies/krl_studies/analysis/report.py:20-120`
- Modify: `studies/krl_studies/analysis/selection.py`
- Modify: `studies/tests/test_analysis_tables.py`
- Modify: `studies/tests/test_analysis_selection.py`

- [ ] **Step 11.1: Make `_build_tradeoff` accept iterations and lesions**

Join scalar BV/NRMSE to lesion CRC on unique `run_id`, `param_set_id`, and `iteration`. Retain `lesion_diameter_mm` from the lesion table; it is not a scalar join key.

- [ ] **Step 11.2: Add the cardinality fixture**

For 2 runs x 2 iterations x 2 lesion sizes, assert exactly 8 trade-off rows with correct CRC values. Fail on duplicate scalar keys instead of using `aggfunc="first"`.

- [ ] **Step 11.3: Select keys, then apply them to both tables**

Selection returns `(run_id, param_set_id, iteration, policy)` keys. Apply those keys separately to iteration and lesion tables so selected CRC reaches final tables.

- [ ] **Step 11.4: Handle single-step and early-stopped methods**

Define fixed policy as the last available iteration at or before each method's locked stopping iteration. This retains input/post-smoothing and does not discard a DTV run that stopped early.

- [ ] **Step 11.5: Label oracle outputs as upper bounds**

Rename outputs and labels from generic `oracle` to `oracle_upper_bound`. Never use oracle results for parameter locking or primary claims.

**Task 11 exit:** CRC is present in selected trade-offs and final tables, with no silent row multiplication or loss.

### Task 12: Implement valid masks and physical background VOIs

**Files:**

- Modify: `studies/krl_studies/metrics/nrmse.py`
- Modify: `studies/krl_studies/metrics/rois.py`
- Modify: `studies/krl_studies/runner/execute.py:156-224,328-343`
- Modify: `studies/tests/test_metrics_nrmse_rois.py`
- Modify: `studies/tests/test_metrics_recovery.py`

- [ ] **Step 12.1: Add optional masked NRMSE**

Require mask shape equality and at least one included voxel. Compute both numerator mean and GT maximum over the mask. Add a hand-calculated test proving errors outside support do not affect the result.

- [ ] **Step 12.2: Define support by dataset**

For BrainWeb use `labels != LABEL_BG`. For spheres use the documented phantom support, not `gt > 0` if that excludes valid background material. Persist support provenance in the manifest.

- [ ] **Step 12.3: Express VOIs in millimetres**

Replace `radius_vox` with `radius_mm` plus anisotropic `voxel_mm`. A VOI is valid only if every voxel lies inside support and outside lesion exclusions.

- [ ] **Step 12.4: Use fixed sphere layout and persisted BrainWeb coordinates**

Use a documented NEMA-style/fixed phantom layout where geometry permits. For BrainWeb, deterministic support-constrained placement is acceptable, but persist centers, radius, and masks or coordinates for audit.

- [ ] **Step 12.5: Validate CRC and BV by hand**

Add small fixtures with analytically known lesion mean, background mean, and background standard deviation. Assert exact expected CRC and BV/COV.

**Task 12 exit:** Air/padding cannot influence primary image metrics, and every background VOI is physically auditable.

### Task 13: Make figure specifications explicit

**Files:**

- Modify: `studies/krl_studies/analysis/plots.py`
- Modify: `studies/krl_studies/analysis/report.py:123-155`
- Modify: `studies/tests/test_analysis_plots.py`
- Create: `studies/configs/figures/pilot.yaml`

- [ ] **Step 13.1: Require a figure specification**

Each publication plot must receive one metric, one selection policy or iteration, explicit filters for condition/count/beta/guidance/parameter set, x-axis, and line-grouping columns. Reject ambiguous input containing extra strata.

- [ ] **Step 13.2: Test series membership, not PNG existence**

Return or expose the plotted series specification. Assert exact line count, labels, run IDs, parameter IDs, and row membership. Assert no line spans multiple conditions, betas, counts, guidance states, or parameter sets unless that field is the declared x-axis.

- [ ] **Step 13.3: Save figure-source CSVs**

For each PNG, save the fully filtered source rows and the figure spec. This is required for paper traceability.

**Task 13 exit:** Plotting all summaries without filters is impossible through the publication API.

### Task 14: Add development locking, paired contrasts, and completeness

**Files:**

- Create: `studies/krl_studies/analysis/contrasts.py`
- Create: `studies/krl_studies/analysis/completeness.py`
- Modify: `studies/krl_studies/analysis/report.py`
- Modify: `studies/krl_studies/report.py`
- Create: `studies/tests/test_analysis_contrasts.py`
- Create: `studies/tests/test_analysis_completeness.py`
- Modify: `studies/tests/test_analysis_selection.py`

- [ ] **Step 14.1: Lock only from development rows**

Implement a locking function that rejects mixed or held-out-only input. Emit method, `param_set_id`, stopping policy, endpoint, and development scenario hash. Evaluation must reject held-out rows not matching the lock file.

- [ ] **Step 14.2: Add a different-optimum fixture**

Construct development and held-out data with different optima. Assert the lock follows development and remains unchanged when held-out values change.

- [ ] **Step 14.3: Compute paired method contrasts**

Pair on subject/case, input ID or observed checksum, condition, beta, count, realization, and guidance condition. Method and parameter set identify treatments and are not pair keys. Report missing pairs explicitly.

- [ ] **Step 14.4: Respect the independent unit**

For pilot output, report descriptive paired differences only. Do not claim population-level hierarchical inference from one subject and two realizations. Add hierarchical intervals only after a multi-subject statistical protocol defines the resampling/model unit.

- [ ] **Step 14.5: Compare completed runs to the expected plan**

Fail final aggregation on missing, duplicate, unexpected, malformed, or wrong-parameter runs. Emit machine-readable counts by method, condition, realization, and parameter set.

**Phase 5 exit:** A synthetic end-to-end fixture produces complete, traceable, correctly selected tables and figures without manual CSV filtering.

---

## Phase 6: Minimal Acceptance Pilot

### Task 15: Create two locked pilot scenarios

**Files:**

- Create: `studies/scenarios/pilot_spheres.yaml`
- Create: `studies/scenarios/pilot_brainweb.yaml`
- Create: `docs/PILOT_ACCEPTANCE.md`
- Modify: `studies/tests/test_config.py`
- Modify: `studies/tests/test_run_plan.py`

- [ ] **Step 15.1: Define the sphere pilot**

Use `quick_sim`, one count level, realizations `[0, 1]`, exact guidance, and one locked parameter set per method. Include `input`, `post_smoothing`, `rl`, `krl`, `hkrl`, and `dtv`. Include iY only if Task 8 added independent phantom labels.

- [ ] **Step 15.2: Define the BrainWeb pilot**

Use `sirf_sim`, one count level, realizations `[0, 1]`, one calibrated reconstruction-PSF condition, exact guidance, attenuation, and one locked parameter set per method. Include BrainWeb iY after Task 8 passes.

- [ ] **Step 15.3: Mark pilot data as held out**

Set `dataset.split: held_out`. Reference a development-generated lock file. Do not choose locked values from the pilot itself.

- [ ] **Step 15.4: Assert exact expansion counts**

With six methods, each scenario has 12 runs. BrainWeb has 14 if iY is included. Sphere has 14 only if independent labels justify iY. Assert exact run IDs and method sets in tests.

- [ ] **Step 15.5: Dry-run before computation**

Run:

```bash
PYTHONPATH=src:studies python -m krl_studies.run --scenario studies/scenarios/pilot_spheres.yaml --dry-run
PYTHONPATH=src:studies python -m krl_studies.run --scenario studies/scenarios/pilot_brainweb.yaml --dry-run
```

Expected: counts match Step 15.4 and every parameter ID matches the lock file.

### Task 16: Run and audit the pilot end to end

**Files produced:**

- `results/pilot_spheres/`
- `results/pilot_brainweb/`
- `analysis/pilot/`
- `docs/PILOT_ACCEPTANCE.md`

- [ ] **Step 16.1: Verify active BrainWeb data before running**

Require the subject directory, all expected NIfTIs, finite arrays, matching shape/affine/orientation, physical overlap, lesion metadata, and checksums. Do not use files under `invalidated_2026-08-27/`.

- [ ] **Step 16.2: Run the sphere pilot**

Run:

```bash
make study-docker-run SCENARIO=studies/scenarios/pilot_spheres.yaml
```

- [ ] **Step 16.3: Run the BrainWeb pilot**

Run:

```bash
make study-docker-run SCENARIO=studies/scenarios/pilot_brainweb.yaml
```

- [ ] **Step 16.4: Require completeness before reporting**

Run the completeness command implemented in Task 14. Expected: zero missing, duplicate, unexpected, malformed, or unlocked runs.

- [ ] **Step 16.5: Generate aggregate tables and figures**

Run:

```bash
PYTHONPATH=src:studies python -m krl_studies.report all \
  --results results \
  --out analysis/pilot
```

Expected: ingestion, selection, lesion joins, contrasts, tables, figure-source CSVs, and PNGs complete without manual filtering.

- [ ] **Step 16.6: Complete every acceptance check**

Record evidence in `docs/PILOT_ACCEPTANCE.md` for all items:

- [ ] Expected run count and method set match the dry-run.
- [ ] Every method sharing an acquisition has the same `input_id` and observed checksum.
- [ ] Realizations 0 and 1 differ; rerunning one realization reproduces it within declared tolerance.
- [ ] Input baseline output is bitwise equal to observed input.
- [ ] All outputs and metrics are finite; positivity-enforcing methods are non-negative.
- [ ] Activity-unit convention and prompt scale are recorded and verified.
- [ ] BrainWeb iY records tissue-label provenance and correct anisotropic sigma.
- [ ] Masked NRMSE excludes air/padding.
- [ ] Every expected lesion has one CRC per selected iterate.
- [ ] Every BV VOI lies inside declared support and has persisted coordinates.
- [ ] CRC survives ingestion, selection, trade-off generation, and final tables.
- [ ] Figure-source tests show no line joins unrelated strata.
- [ ] Oracle output is labeled as a GT upper bound.
- [ ] Fixed/practical selection retains single-step and early-stopped methods.
- [ ] Paired contrast output reports all expected pairs and missing-pair count is zero.
- [ ] Final report generation required no manual CSV filtering.
- [ ] A clean rerun reproduces metrics within the declared SIRF/CIL tolerance.
- [ ] One masked NRMSE, one CRC, and one BV value are recalculated by hand from saved data.
- [ ] A second researcher can map each plotted series to run IDs and parameter IDs.

**Phase 6 exit:** Every acceptance checkbox is signed with evidence. Pilot outputs remain engineering acceptance data unless the protocol explicitly designates them otherwise.

---

## Final Verification

- [ ] Run all native tests:

```bash
ruff check src tests studies
python -m pytest tests -v
python -m pytest studies/tests -m "not sirf and not brainweb" -v
```

- [ ] Run all SIRF-marked tests:

```bash
make study-sirf-test
```

- [ ] Run the combined suite inside the pinned image so CIL-dependent core tests are not skipped:

```bash
docker compose -f studies/docker-compose.yaml run --rm sirf \
  bash -lc "export OMP_NUM_THREADS=1 && \
  pip install -q -e '.' -e './studies[dev]' && \
  python -m pytest tests studies/tests -v"
```

- [ ] Inspect `git diff --check` and `git status --short`.
- [ ] Confirm no invalidated data or generated result artifacts are staged.
- [ ] Confirm both large mismatch scenarios remain unexecuted until pilot sign-off.

## Stop Condition Before Paper-Scale Runs

Do not generate or submit a full campaign plan until all of the following are true:

- [ ] Gate 0, Gate 1, Phase 4, Phase 5, and Phase 6 exit criteria are checked.
- [ ] The scientific protocol defines the primary endpoint and smallest meaningful paired difference.
- [ ] Resolution calibration and activity-unit validation are signed off.
- [ ] Development settings are frozen and held-out data have not influenced them.
- [ ] Pilot variance has informed the number of subjects and paired realizations.
- [ ] The expected plan records scenario hash, code revision, container digest, dependency versions, input checksums, and hardware.

Only after these checks should the large scenarios be redesigned; the current factorial files are not approved paper campaigns.
