# Study Framework SIRF Simulation Layer Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the SIRF/STIR simulation layer: ground-truth images → forward projection → count-scaled Poisson noise → OSEM/RDP reconstructions under three recon-PSF conditions, wired into the runner as new input kinds, plus BrainWeb phantom preparation and patient-study enablement.

**Architecture:** New `krl_studies.simulation` package wrapping SIRF behind thin, pinned helper functions (`_api.py`) calibrated against the live container in Task 1. A docker compose service (`synerbi/sirf:latest`, repo mounted at `/workspace`) is the development/runtime environment for everything SIRF-dependent; the existing pure-Python stack keeps working natively on macOS. The runner gains input kinds `sirf_sim` (condition × β × counts × realisation grid) and study kind `patient`.

**Tech Stack:** synerbi/sirf docker image (SIRF + STIR python bindings), existing krl-studies stack (numpy/scipy/nibabel/pandas/yaml), brainweb pip package, pytest markers `sirf` (skip when SIRF unavailable).

**Spec:** section 5 of `docs/superpowers/specs/2026-08-23-krl-study-framework-design.md`
**Working directory:** repo root, branch `study-framework`. All `make study-*` commands run from repo root with the repo-root `.venv` active EXCEPT docker-invoked ones.

**Environment facts verified by controller probe (2026-08-24, synerbi/sirf:latest digest 643c795…):**

- Image is linux/amd64; on Apple Silicon it runs under emulation (slow but functional). Heavy sweeps belong on the cluster (Plan 3).
- Container entrypoint (`start.sh`) swallows stdin scripts and emits hook noise; always run `python <path>` on a mounted file, never `python -`.
- Available in `sirf.STIR`: `RelativeDifferencePrior`, `QuadraticPrior`, `OSMAPOSLReconstructor`, `OSSPSReconstructor`, `AcquisitionModelUsingRayTracingMatrix`, `SeparableGaussianImageFilter`, `PoissonNoiseGenerator`, `TruncateToCylinderProcessor`.
- NOT available under naive names: `OSEMImageReconstructor` (use `OSMAPOSLReconstructor`), `AcquisitionData.get_scanner()` (use `stir.Scanner` directly).
- Scanner enums exist: `stir.Scanner.Siemens_VISION_600`, `stir.Scanner.Siemens_mMR`.
- `AcquisitionData("Siemens mMR")` template constructs; `AcquisitionData("Siemens VISION 600")` currently raises "Number of TOF bins should be an odd number" — Task 1 resolves (TOF mashing to 1 bin via explicit `ProjDataInfo`, else fall back to mMR geometry documented in manifest).
- Unknown until Task 1: exact constructor signatures for `OSMAPOSLReconstructor.set_prior`, `PoissonNoiseGenerator`, resolution-modelling route (no `set_resolution_model` found on `AcquisitionModelUsingRayTracingMatrix`; likely route is blurring inside the forward model via image-data processor or pre-blurred forward projection + matched post-filter — Task 1 decides and records).

**Plan-level risk control:** Task 1 is an API-calibration spike whose ONLY deliverable is `studies/krl_studies/simulation/_api.py` + a probe record. Every later task imports through `_api`, never importing `sirf` directly. If Task 1 discovers an interface different from what Tasks 2–6 assume, the implementer adapts `_api.py` signatures FIRST and notes the delta in their report; downstream task code changes only call sites if truly unavoidable.

---

## File map

```
studies/
  docker-compose.yaml             # sirf service: repo mount + pip install -e ./studies
  Makefile targets                # study-docker-build/run/sirf-test (root Makefile)
  scenarios/
    brainweb_core.yaml            # Task 6
    spheres_sirf.yaml             # Task 5
  krl_studies/
    simulation/
      __init__.py                 # exports simulate_inputs, RESOLUTION_PRESETS
      _api.py                     # Task 1: pinned SIRF wrappers (only sirf import site)
      presets.py                  # Task 2: Vision resolution presets + geometry
      simulate.py                 # Task 3: forward/noise/recon pipeline
    datasets/
      brainweb.py                 # Task 4: brainweb prep w/ labels + tumour set
    runner/
      execute.py                  # Task 5/7: sirf_sim inputs + patient study
  tests/
    test_simulation_presets.py    # pure unit tests (no SIRF)
    test_simulation_sirf.py       # marked slow/sirf; runs inside docker
    test_brainweb_dataset.py      # marked sirf (needs brainweb pkg data download)
```

---

### Task 1: Docker env + SIRF API calibration spike

**Files:**
- Create: `studies/docker-compose.yaml`
- Modify: root `Makefile` (add study-docker-build, study-docker-python, study-sirf-test)
- Create: `studies/krl_studies/simulation/__init__.py`, `_api.py`
- Create: `studies/tests/test_simulation_sirf.py`
- Create: `docs/reference/SIRF_API_NOTES.md`

- [ ] **Step 1: Compose service**

`studies/docker-compose.yaml`:

```yaml
services:
  sirf:
    image: synerbi/sirf:latest
    working_dir: /workspace
    volumes:
      - ..:/workspace
    command: >
      bash -lc "pip install -q -e './studies[dev]' &&
                python -m pytest studies/tests -m sirf -v"
```

Root `Makefile` additions (also extend `.PHONY`):

```makefile
study-docker-pull:
	docker pull synerbi/sirf:latest

study-sirf-test:
	docker compose -f studies/docker-compose.yaml run --rm sirf

study-docker-python:
	docker compose -f studies/docker-compose.yaml run --rm sirf bash
```

- [ ] **Step 2: Verify container can install krl-studies and import krl**

```bash
source .venv/bin/activate && make study-docker-pull
docker compose -f studies/docker-compose.yaml run --rm sirf python -c "
import krl_studies, krl, sirf.STIR
print('container imports ok', krl_studies.__version__)
"
```

Expected: prints `container imports ok 0.1.0` after hook noise. If CIL+krl fail to co-import inside the container (OpenMP clash is a macOS-only note, Linux should be fine), STOP and report BLOCKED — the whole plan depends on running krl methods inside this container eventually (Task 5 smoke).

Note: `pip install -e ./studies[dev]` requires numpy/scipy already present in image (they are, via SIRF deps); pandas/pyyaml/matplotlib get installed fresh each `run` (~30 s). Acceptable for Plan 2; bake a derived Dockerfile only if painful.

- [ ] **Step 3: Write the calibration probe script**

`/tmp`-style throwaway is not allowed here — commit it as documentation. Create `studies/scripts/probe_sirf_api.py`:

```python
"""Probe the installed SIRF/STIR API surface and print a machine-checkable report.

Run inside the sirf container:
    docker compose -f studies/docker-compose.yaml run --rm sirf \
        python studies/scripts/probe_sirf_api.py
"""

from __future__ import annotations

import traceback

import sirf.STIR as st
import stir


def check(name, fn):
    try:
        out = fn()
        print(f"OK   {name}: {out}")
        return out
    except Exception as exc:  # noqa: BLE001 - probe reports everything
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        return None


def main():
    check("stir scanner enum", lambda: stir.Scanner.Siemens_VISION_600.get_name())

    def mmr_template():
        acq = st.AcquisitionData("Siemens mMR")
        return acq.dimensions()

    check("mMR acquisition template", mmr_template)

    def vision_direct():
        return st.AcquisitionData("Siemens VISION 600").dimensions()

    check("VISION direct template", vision_direct)

    def vision_span11_tof1():
        sc = stir.Scanner(stir.Scanner.Siemens_VISION_600)
        sc.set_num_tof_bins(1)
        pdi = st.ProjDataInfo.ProjDataInfoCTI(sc, 1, sc.get_num_rings() - 1,
                                              sc.get_num_detectors_per_ring() // 2,
                                              sc.get_max_num_views(), 1)
        return st.AcquisitionData(pdi).dimensions()

    check("VISION span-1 TOF=1 template", vision_span11_tof1)

    def osmaposl():
        r = st.OSMAPOSLReconstructor()
        return [n for n in dir(r) if "prior" in n.lower() or "objective" in n.lower()]

    check("OSMAPOSL surface", osmaposl)

    def rdp():
        p = st.RelativeDifferencePrior()
        return hasattr(p, "set_penalisation_factor")

    check("RDP penalisable", rdp)

    def poisson():
        n = st.PoissonNoiseGenerator()
        return type(n).__name__

    check("PoissonNoiseGenerator ctor", poisson)

    def am():
        am = st.AcquisitionModelUsingRayTracingMatrix()
        return [n for n in dir(am) if "resolution" in n.lower() or "processor" in n.lower()
                or "image_data" in n.lower()]

    check("AM resolution hooks", am)

    def fwd_project():
        img = st.ImageData()
        am = st.AcquisitionModelUsingRayTracingMatrix()
        acq_t = st.AcquisitionData("Siemens mMR")
        geom = acq_t.create_uniform_image(1.0)
        am.set_up(acq_t, geom)
        proj = am.forward(geom)
        return proj.dimensions()

    check("tiny forward projection", fwd_project)

    def osem_run():
        acq_t = st.AcquisitionData("Siemens mMR")
        geom = acq_t.create_uniform_image(1.0)
        am = st.AcquisitionModelUsingRayTracingMatrix()
        am.set_up(acq_t, geom)
        proj = am.forward(geom)
        obj = st.make_Poisson_loglikelihood(acq_t)
        obj.set_acquisition_model(am)
        rec = st.OSMAPOSLReconstructor(obj)
        rec.set_num_subsets(1)
        rec.set_num_subiterations(2)
        rec.set_current_estimate(geom)
        rec.process()
        est = rec.get_output()
        return float(est.as_array().max())

    check("tiny OSEM 2 iterations", osem_run)


if __name__ == "__main__":
    main()
```

If any constructor signature above differs (very likely somewhere), adapt the PROBE until it passes all checks it can, recording which are impossible.

- [ ] **Step 4: Run the probe and write SIRF_API_NOTES.md**

```bash
source .venv/bin/activate
docker compose -f studies/docker-compose.yaml run --rm sirf \
  python studies/scripts/probe_sirf_api.py 2>&1 | grep -E "^OK|^FAIL"
```

Then write `docs/reference/SIRF_API_NOTES.md` capturing: which probes pass verbatim, which needed signature changes (with final working snippets), the chosen VISION template route (span-1 TOF=1 vs mMR fallback), and the chosen resolution-modelling route for Task 3:

- Preferred: pre-blur inside forward model is NOT available → simulate recon-PSF conditions as: forward project the (possibly Gaussian-pre-blurred) GT, then during RECONSTRUCTION apply `SeparableGaussianImageFilter` post-processing only where the condition demands, OR reconstruct plain and characterise residual blur empirically. The DECIDED route must satisfy: `psf-none` yields ~5.7/7.8 mm effective FWHM, `psf-matched` yields ~4.5/6.4 mm, measured with the Hoffman-style profile check in Task 3's test. Record actual numbers achieved.

- [ ] **Step 5: Implement `_api.py` pinning ONLY the verified surfaces**

`studies/krl_studies/simulation/_api.py` — skeleton with the functions Task 1 proved; fill bodies from the working probe snippets:

```python
"""The ONLY module in krl_studies that imports sirf/stir directly.

Every wrapper must match the verified behaviour recorded in
docs/reference/SIRF_API_NOTES.md. If SIRF changes, fix here alone.
"""

from __future__ import annotations


def _require_sirf():
    try:
        import sirf.STIR  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "SIRF is required for simulation; run inside the sirf container "
            "(make study-sirf-test) or a SIRF-enabled environment"
        ) from exc


def acquisition_template(scanner_name: str):
    """Verified acquisition template ('Siemens mMR' or Vision via TOF=1 route)."""
    raise NotImplementedError("Task 1 Step 5")


def make_acquisition_model(acq_template, image):
    """Set-up ray-tracing acquisition model for the given geometry."""
    raise NotImplementedError("Task 1 Step 5")


def forward_project(image, acq_model):
    raise NotImplementedError("Task 1 Step 5")


def poisson_sample(prompts, seed: int):
    """Count-domain Poisson noise on scaled prompt data."""
    raise NotImplementedError("Task 1 Step 5")


def reconstruct_osem(prompts, acq_model, image, n_subiterations, prior=None):
    """OSMAPOSL reconstruction; prior=None gives plain OSEM, else RDP-beta."""
    raise NotImplementedError("Task 1 Step 5")


def gaussian_smooth_image(image, fwhm_mm):
    """SeparableGaussianImageFilter wrapper (post-filter / pre-blur)."""
    raise NotImplementedError("Task 1 Step 5")
```

Implement each body using exactly the snippet proven in Step 4 (raise `RuntimeError` referencing SIRF_API_NOTES.md if a body cannot work yet).

Add `simulation/__init__.py`:

```python
from krl_studies.simulation._api import (  # noqa: F401
    acquisition_template,
    forward_project,
    gaussian_smooth_image,
    make_acquisition_model,
    poisson_sample,
    reconstruct_osem,
)
```

- [ ] **Step 6: Marker-guarded smoke test**

`studies/tests/test_simulation_sirf.py`:

```python
import numpy as np
import pytest

try:
    import sirf.STIR  # noqa: F401

    HAS_SIRF = True
except ImportError:
    HAS_SIRF = False

pytestmark = pytest.mark.skipif(not HAS_SIRF, reason="SIRF not available")


def test_forward_project_and_reconstruct_roundtrip():
    from krl_studies.simulation import (
        acquisition_template,
        forward_project,
        make_acquisition_model,
        reconstruct_osem,
    )

    acq = acquisition_template("Siemens mMR")
    image = acq.create_uniform_image(1000.0)
    am = make_acquisition_model(acq, image)
    prompts = forward_project(image, am)
    recon = reconstruct_osem(prompts, am, image, n_subiterations=2)
    arr = np.asarray(recon)
    assert arr.shape == np.asarray(image).shape
    assert arr.max() > 0
```

Register the marker in `studies/pyproject.toml` (`markers = ["sirf: tests requiring SIRF"]`) and ensure native suite still collects cleanly (`python -m pytest studies/tests -q` skips them on macOS).

- [ ] **Step 7: Run in container**

```bash
make study-sirf-test
```

Expected: 1 passed. Native `pytest studies/tests -q` still green (skipped marker).

- [ ] **Step 8: Commit**

```bash
git add studies Makefile docs/reference/SIRF_API_NOTES.md && git commit -m "ADD SIRF container environment and calibrated API layer"
```

---

### Task 2: Resolution presets (pure Python)

**Files:**
- Create: `studies/krl_studies/simulation/presets.py`
- Test: `studies/tests/test_simulation_presets.py`

- [ ] **Step 1: Failing tests**

```python
import pytest

from krl_studies.simulation.presets import (
    PRESET_NAMES,
    RECON_PSF_CONDITIONS,
    resolution_for_condition,
)


def test_preset_names_match_spec():
    assert set(PRESET_NAMES) == {"psf-none", "psf-undersized", "psf-matched"}
    assert set(RECON_PSF_CONDITIONS) == PRESET_NAMES


def test_matched_matches_vision_doc():
    assert resolution_for_condition("psf-matched") == pytest.approx((4.5, 4.5, 6.4))
    assert resolution_for_condition("psf-none") == pytest.approx((5.7, 5.7, 7.8))


def test_undersized_is_halfway_by_default():
    u = resolution_for_condition("psf-undersized")
    n = resolution_for_condition("psf-none")
    m = resolution_for_condition("psf-matched")
    assert u == pytest.approx(tuple((a + b) / 2 for a, b in zip(m, n)))


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        resolution_for_condition("psf-wat")
```

- [ ] **Step 2: RED run** → expect ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
"""Effective-residual-resolution presets anchored to the Vision 600 Hoffman
measurements (Vision_resolution.docx): values are the FWHM of the blur left ON
the reconstructed input image, on top of recon-side resolution modelling."""

from __future__ import annotations

PRESET_NAMES = ("psf-none", "psf-undersized", "psf-matched")

_PSF_MATCHED = (4.5, 4.5, 6.4)   # recon PSF modelled correctly -> clinical-like residual
_PSF_NONE = (5.7, 5.7, 7.8)      # no resolution modelling -> full effective blur


def _undersized() -> tuple[float, float, float]:
    return tuple((a + b) / 2 for a, b in zip(_PSF_MATCHED, _PSF_NONE))


def resolution_for_condition(condition: str) -> tuple[float, float, float]:
    if condition == "psf-matched":
        return _PSF_MATCHED
    if condition == "psf-none":
        return _PSF_NONE
    if condition == "psf-undersized":
        return _undersized()
    raise ValueError(f"unknown recon-PSF condition: {condition!r}")


RECON_PSF_CONDITIONS = PRESET_NAMES
```

- [ ] **Step 4: GREEN + full native suite + lint. Commit:**
`git commit -m "ADD recon-PSF condition resolution presets"`

---

### Task 3: `simulate_inputs` pipeline

**Files:**
- Create: `studies/krl_studies/simulation/simulate.py`
- Test: `studies/tests/test_simulation_sirf.py` (append; runs in container)

Semantics (spec §5): given a GT array (z,y,x) + config {condition ∈ presets, beta ∈ {None|float}, counts, realisation, seed}: resample-free (phantom already 1mm; Vision grid resampling deferred to Plan 3 — document), build AM on chosen scanner template, forward project the GT **blurred to the condition's residual FWHM** (pre-blur route decided in Task 1), scale so Σprompts = counts, Poisson-sample with `default_rng(seed + realisation*7919)`-derived integer, then reconstruct: plain OSEM when beta None, else OSMAPOSL+RDP(beta). Return `(z,y,x) ndarray` recon + metadata dict {true_fwhm, scanner, beta, counts, realisation, n_subits}.

- [ ] **Step 1: Failing test appended to test_simulation_sirf.py:**

```python
def test_simulate_inputs_shapes_and_determinism():
    import sirf.STIR as st

    from krl_studies.simulation import simulate_inputs

    gt = np.zeros((64, 64, 64), dtype=np.float32)
    gt[24:40, 24:40, 24:40] = 5.0
    gt += 1.0

    cfg = {"condition": "psf-matched", "beta": None, "counts": 5e6,
           "realisation": 0, "seed": 1337, "n_subits": 4}
    a, meta_a = simulate_inputs(gt, cfg)
    b, meta_b = simulate_inputs(gt, dict(cfg))
    assert np.array_equal(a, b)
    assert meta_a["true_fwhm"] == (4.5, 4.5, 6.4)
    assert a.shape == gt.shape
    assert a.min() >= 0

    c, _ = simulate_inputs(gt, dict(cfg, realisation=3))
    assert not np.array_equal(a, c)
```

- [ ] **Step 2:** RED (ImportError: simulate_inputs).
- [ ] **Step 3:** Implement `simulate.py` using only `_api` wrappers; deterministic seeding converts the rng draw to the integer count vector expected by `_api.poisson_sample` (decide exact route per Task 1 findings; keep determinism invariant sacred).
- [ ] **Step 4:** GREEN inside container (`make study-sirf-test` → 2 passed). Native suite unaffected. Export `simulate_inputs` from `simulation/__init__.py`.
- [ ] **Step 5:** Commit: `git commit -m "ADD SIRF simulate_inputs pipeline"`

---

### Task 4: BrainWeb dataset preparation

**Files:**
- Create: `studies/krl_studies/datasets/brainweb.py`
- Modify: root pyproject `[project.optional-dependencies] studies-extra = ["brainweb"]`? NO — keep brainweb OUT of krl-studies deps; document `pip install brainweb nibabel scipy` requirement and lazy-import in module.
- Test: `studies/tests/test_brainweb_dataset.py` (marker `sirf` too — brainweb downloads data)

Behaviour: `prepare_subject(subject_id, out_dir, tumour=True)` downloads/caches BrainWeb volume via the `brainweb` package, applies `default_tumour_specs`+`place_tumours` (reuse datasets.lesions), saves pet_gt.nii.gz, mr_t1.nii.gz, labels.nii.gz (GM/WM/CSF from brainweb tissue maps), returns paths + label array. Tissue-label iY regions builder: `regions_from_labels(labels)` → [WM, GM, CSF/background].

- [ ] Tests: placement respects labels (tumour centres land in GM/WM per fraction map sanity), files written, regions_from_labels partitions brain mask. RED→GREEN→ commit `git commit -m "ADD BrainWeb phantom preparation with tissue labels"`.

---

### Task 5: Runner integration for `sirf_sim` inputs

**Files:**
- Modify: `studies/krl_studies/runner/execute.py` (`_build_observed` gains branch), `studies/krl_studies/config.py` (nothing — sim dict carries new keys), scenario `studies/scenarios/spheres_sirf.yaml`

`sirf_sim` input params: `{condition, beta, counts, realisation}` (+ optional `scanner`). `_build_observed` calls `simulate_inputs(gt, {...run.input_params, seed from run.sim})`; manifest already stores input_params → reproducible. Scenario YAML:

```yaml
inputs:
  - kind: sirf_sim
    params:
      condition: [psf-none, psf-undersized, psf-matched]
      beta: [null, 10.0, 50.0]
      counts: [1.0e7]
      realisation: [0, 1]
```

- [ ] Unit-ish test in test_runner.py with monkeypatched `krl_studies.runner.execute._build_observed`? NO — instead add ONE container-marked end-to-end: tiny spheres fixture → sirf_sim run → metrics.csv has crc columns. Plus dry-run expansion-count test (native): 3 conditions × 3 betas × 2 realisations × N methods arithmetic.
- [ ] Commit: `git commit -m "ADD sirf_sim input kind wired into runner"`

---

### Task 6: Patient study enablement

**Files:**
- Modify: `studies/krl_studies/runner/execute.py`

Remove the phase-1 guard: study `patient` loads `PatientDataset(subject_id, root)`; observed = ds.pet; guidance = ds.guidance; gt None → rows carry objective/bv only when VOIs supplied via optional `ROIs.nii.gz` (build vois from labels>0 boundary band? KEEP SIMPLE: no VOIs/CRC without ROI file; nrmse column omitted). iy/gtm raise clear NotImplementedError without region source. Update `studies/scenarios/patient_mk_h001.yaml` comment; verify CLI executes rl+krl on MK-H001 (20 iterations, minutes-scale) producing metrics.csv with objective column and final.nii.gz.

- [ ] Test: tmp_path patient fixture (PET/T1 gz) → execute_run completes, manifest.status complete, no nrmse key. Commit: `git commit -m "ADD patient study execution path"`

---

### Task 7: Docs + final verification

- [ ] Update `studies/README.md` (SIRF section: docker workflow, brainweb prep command, sirf_sim example), `data/README.md` (brainweb/ layout), run FULL matrix: native suites + `make study-sirf-test` + one real `sirf_sim` reference run on spheres phantom; commit `git commit -m "ADD SIRF simulation documentation and verification"`.

## Out of scope (Plan 3)

Vision-grid resampling, attenuation/uMap, scatter/randoms, aggregated results store, figures/tables, SGE/docker orchestration, count-level calibration campaign.

## Risks / engineer notes

- amd64-under-emulation is SLOW: keep container volumes tiny (≤64³ fixtures, ≤5 subiterations) except Task 7's single real run.
- Never let `sirf` leak outside `_api.py`/tests' guarded imports; native macOS suite must stay green.
- Determinism invariant (same cfg ⇒ bit-identical recon) may break if OSMAPOSL uses RNG internally — it does not; Poisson sampling is ours. If STIR adds randomness anywhere, pin threads (`OMP_NUM_THREADS=1` in compose) and note it.
