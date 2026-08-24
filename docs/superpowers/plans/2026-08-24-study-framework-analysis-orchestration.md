# Study Framework Analysis and Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the paper-facing layer of `krl-studies`: calibrated scanner-grid simulation, canonical results aggregation, publication figures/tables, mismatch scenarios, and reproducible SGE/Docker execution.

**Architecture:** Keep the existing runner as the producer of immutable per-run artifacts. Add a separate `analysis` package that discovers completed runs, validates manifests, normalizes wide metric CSVs into long-form tables, computes replicate summaries and iteration selections, and generates every figure/table from those tables. Add a small `cluster` package and explicit container entrypoint for execution without changing the scientific method wrappers.

**Tech Stack:** Python 3.10+, numpy/scipy, pandas, matplotlib, pyarrow (optional parquet extra), nibabel, existing CIL/krl, SIRF/STIR in `synerbi/sirf`, POSIX shell, UCL SGE.

**Base:** `study-framework` branch, after Plan 2 (`32986d4` plus subsequent Plan 2 commits).

**Design inputs:**

- `docs/superpowers/specs/2026-08-23-krl-study-framework-design.md`
- `docs/superpowers/plans/2026-08-24-study-framework-sirf-simulation.md`
- `docs/reference/SIRF_API_NOTES.md`
- Vision resolution estimates: residual FWHM `(4.5, 4.5, 6.4)` mm for PSF+TOF and `(5.7, 5.7, 7.8)` mm for plain OSEM3D+TOF.

---

## File Map

```text
studies/krl_studies/
  analysis/
    __init__.py
    schema.py             # canonical run/iteration/lesion table columns
    ingest.py             # completed-run discovery and normalization
    aggregate.py          # replicate means/std/counts
    selection.py          # oracle and fixed-iteration selections
    plots.py              # publication figures, no filesystem discovery
    tables.py              # CSV and LaTeX summaries
    report.py              # aggregate/figure/table orchestration functions
  simulation/
    geometry.py           # physical voxel-grid resampling
    calibration.py        # measured resolution/count calibration
    presets.py             # condition model parameters added to existing presets
  datasets/
    transforms.py         # guidance shifts and modality selection
    brainweb.py            # prepared-subject loader, T2 and lesion metadata
  runner/
    plan.py               # JSONL run-plan serialization
    cli.py                 # --plan/--index/--out additions
  plan.py                 # scenario -> JSONL/SGE plan CLI
  cluster/
    __init__.py
    sge.py                # SGE array script generation
  scripts/
    container_entrypoint.sh
  report.py               # python -m krl_studies.report entrypoint
  scenarios/
    resolution_calibration.yaml
    spheres_mismatch.yaml
    brainweb_mismatch.yaml
    patient_cohort.yaml
studies/tests/
  test_analysis_schema.py
  test_analysis_ingest.py
  test_analysis_selection.py
  test_analysis_plots.py
  test_analysis_tables.py
  test_geometry.py
  test_guidance_transforms.py
  test_run_plan.py
  test_sge.py
  test_report_cli.py
studies/docker-compose.yaml   # add reproducible study service
Makefile                      # study plan and Docker execution targets
studies/README.md
data/README.md
```

The existing `examples/scripts/poster.py` remains untouched. Plan 3 replaces
its hardcoded paths and manual method selection for new studies; it does not
retrofit the old poster script.

---

### Task 1: Calibrated geometry, resolution modelling, and simulation provenance

**Files:**
- Create: `studies/krl_studies/simulation/geometry.py`
- Create: `studies/krl_studies/simulation/calibration.py`
- Modify: `studies/krl_studies/simulation/presets.py`
- Modify: `studies/krl_studies/simulation/_api.py`
- Modify: `studies/krl_studies/simulation/simulate.py`
- Modify: `studies/krl_studies/runner/execute.py`
- Modify: `studies/tests/test_simulation_sirf.py`
- Create: `studies/tests/test_geometry.py`
- Create: `studies/tests/test_calibration.py`
- Modify: `studies/docker-compose.yaml`
- Modify: `Makefile`
- Modify: `docs/reference/SIRF_API_NOTES.md`

#### Resolution model

Preserve the existing public `resolution_for_condition(name)` function and
add an explicit condition object. The values describe the true system blur
used in the forward model, the reconstruction-side resolution model, and the
target residual reported by the Vision measurement:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutionCondition:
    name: str
    target_residual_fwhm_xyz: tuple[float, float, float]
    forward_model_fwhm_xyz: tuple[float, float, float]
    recon_model_fwhm_xyz: tuple[float, float, float] | None


VISION_PLAIN_FWHM = (5.7, 5.7, 7.8)
VISION_PSF_FWHM = (4.5, 4.5, 6.4)

CONDITION_SPECS = {
    "psf-none": ResolutionCondition(
        "psf-none", VISION_PLAIN_FWHM, VISION_PLAIN_FWHM, None
    ),
    "psf-undersized": ResolutionCondition(
        "psf-undersized",
        tuple((a + b) / 2 for a, b in zip(VISION_PSF_FWHM, VISION_PLAIN_FWHM)),
        VISION_PLAIN_FWHM,
        tuple(v / 2 for v in VISION_PLAIN_FWHM),
    ),
    "psf-matched": ResolutionCondition(
        "psf-matched", VISION_PSF_FWHM, VISION_PLAIN_FWHM, VISION_PLAIN_FWHM
    ),
}


def condition_spec(condition: str) -> ResolutionCondition:
    try:
        return CONDITION_SPECS[condition]
    except KeyError as exc:
        raise ValueError(f"unknown recon-PSF condition: {condition!r}") from exc
```

The forward model always uses the same physical blur. Only the reconstruction
model changes: no resolution model, an undersized model at 50% of the true
FWHM, or a matched model. The measured residual is written to metadata rather
than assumed to equal the target.

- [ ] **Step 1: Add pure geometry tests first**

`studies/tests/test_geometry.py`:

```python
import numpy as np
import pytest

from krl_studies.simulation.geometry import resample_array_zyx


def test_constant_volume_is_preserved_when_resampled():
    source = np.full((10, 20, 30), 3.5, dtype=np.float32)
    result, target_shape = resample_array_zyx(
        source, source_voxel_mm=(2.0, 1.0, 1.0), target_voxel_mm=(1.0, 2.0, 2.0)
    )
    assert result.shape == target_shape == (20, 10, 15)
    assert np.allclose(result, 3.5, atol=1e-5)


def test_resampling_preserves_physical_extent():
    source = np.zeros((10, 20, 30), dtype=np.float32)
    source[5, 10, 15] = 1.0
    result, target_shape = resample_array_zyx(
        source, source_voxel_mm=(2.0, 1.0, 1.0), target_voxel_mm=(1.0, 2.0, 2.0)
    )
    assert result.shape == target_shape
    peak = np.array(np.unravel_index(np.argmax(result), result.shape))
    source_position_mm = np.array((5, 10, 15)) * np.array((2.0, 1.0, 1.0))
    target_position_mm = peak * np.array((1.0, 2.0, 2.0))
    assert np.all(np.abs(target_position_mm - source_position_mm) <= (1.0, 2.0, 2.0))


def test_invalid_voxel_sizes_raise():
    with pytest.raises(ValueError):
        resample_array_zyx(np.ones((2, 2, 2)), (0, 1, 1), (1, 1, 1))
```

- [ ] **Step 2: Run the new test to confirm RED**

Run: `source .venv/bin/activate && python -m pytest studies/tests/test_geometry.py -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement physical resampling**

`studies/krl_studies/simulation/geometry.py`:

```python
"""Physical voxel-grid utilities; all array axes are (z, y, x)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import zoom


def _validate_voxels(values: tuple[float, float, float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"{name} must contain three positive finite values")
    return arr


def resample_array_zyx(
    array: np.ndarray,
    source_voxel_mm: tuple[float, float, float],
    target_voxel_mm: tuple[float, float, float],
    *,
    output_shape: tuple[int, int, int] | None = None,
    order: int = 1,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Resample a volume while preserving its physical extent."""
    source = np.asarray(array)
    if source.ndim != 3:
        raise ValueError(f"array must be 3-D (z,y,x), got {source.shape}")
    source_mm = _validate_voxels(source_voxel_mm, "source_voxel_mm")
    target_mm = _validate_voxels(target_voxel_mm, "target_voxel_mm")
    if order not in (0, 1, 2, 3):
        raise ValueError("order must be one of 0, 1, 2, or 3")
    if output_shape is None:
        target = tuple(max(1, int(round(n * s / t))) for n, s, t in zip(source.shape, source_mm, target_mm))
    else:
        target = tuple(int(n) for n in output_shape)
        if len(target) != 3 or any(n < 1 for n in target):
            raise ValueError("output_shape must contain three positive integers")
    factors = tuple(t / n for t, n in zip(target, source.shape))
    result = zoom(source.astype(np.float32, copy=False), factors, order=order, mode="nearest", prefilter=order > 1)
    if result.shape != target:
        result = zoom(result, tuple(t / n for t, n in zip(target, result.shape)), order=order, mode="nearest")
    return result.astype(np.float32, copy=False), target
```

- [ ] **Step 4: Extend `_api.py` without leaking SIRF**

SIRF imports remain confined to `_api.py`. SIRF reports image voxel sizes in
array order `(z, y, x)`; keep that order at the gateway while resolution
presets remain physical `(x, y, z)` values.

```python
def image_voxel_sizes(image) -> tuple[float, float, float]:
    """Return SIRF image voxel sizes in (z, y, x) order."""
    _require_sirf()
    return tuple(float(v) for v in image.voxel_sizes())


def scanner_grid(acq_template) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Return the scanner image grid as (shape_zyx, voxel_mm_zyx)."""
    _require_sirf()
    image = acq_template.create_uniform_image(1.0)
    return tuple(int(v) for v in np.asarray(image.as_array()).shape), image_voxel_sizes(image)


def make_acquisition_model(acq_template, image, resolution_fwhm=None, attenuation=None):
    """Build a ray-tracing model with optional resolution and attenuation."""
    st, _ = _sirf()
    model = st.AcquisitionModelUsingRayTracingMatrix()
    model.set_up(acq_template, image)
    if resolution_fwhm is not None:
        processor = st.SeparableGaussianImageFilter()
        fx, fy, fz = (float(v) for v in resolution_fwhm)
        processor.set_fwhms((fz, fy, fx))
        model.set_image_data_processor(processor)
    if attenuation is not None:
        setter = getattr(model, "set_attenuation_image", None)
        if setter is None:
            raise RuntimeError("this SIRF build has no attenuation-image support")
        setter(attenuation)
    return model
```

Add a container-marked test that constructs a tiny model with and without an
attenuation image and asserts either successful forward projection or the
explicit unsupported-build error. Never silently ignore a requested uMap.

- [ ] **Step 5: Remove acquisition-template tempdir leakage**

Change the Vision Interfile round-trip in `_api.acquisition_template` to use a
context manager and construct `AcquisitionData` before the directory is
removed:

```python
with tempfile.TemporaryDirectory(prefix="krl_studies_acqtpl_") as workdir:
    hs = os.path.join(workdir, "template.hs")
    pdm.write_to_file(hs)
    return st.AcquisitionData(hs)
```

Add a container test that calls
`acquisition_template("Siemens VISION 600", span=1, max_ring_diff=1, num_views=42, num_tangential=64)`,
then performs a forward projection after the context has exited. This proves
SIRF has loaded the data rather than retaining a path to the deleted file.

- [ ] **Step 6: Make the Docker image reproducible**

Change `studies/docker-compose.yaml` from the floating tag to the calibrated
digest:

```yaml
services:
  sirf:
    platform: linux/amd64
    image: synerbi/sirf@sha256:643c7955717ac08c6f44c6d3fe2ef064ebb54167f1da68771ed3e6dc07caf58d
```

Keep the existing service command and OMP handling. Add the same
`platform: linux/amd64` to the future `study` service in Task 6.
Update the existing `study-docker-pull` target to pull the same digest and
update `SIRF_API_NOTES.md` so its environment heading records that digest rather
than the floating `latest` tag.

- [ ] **Step 7: Update `simulate_inputs` for scanner-grid round trips**

Use `cfg_dict.get("input_voxel_mm", (1.0, 1.0, 1.0))`, obtain the scanner
shape and voxel sizes through `_api.scanner_grid`, resample GT to that physical
grid before `_api.make_image`, and resample the reconstructed image back to the
original shape/voxel grid before returning it to the runner. Metadata must
include:

```python
{
    "input_shape": tuple(int(v) for v in gt.shape),
    "input_voxel_mm": tuple(float(v) for v in input_voxel_mm),
    "scanner_shape": tuple(int(v) for v in scanner_shape),
    "scanner_voxel_mm": tuple(float(v) for v in scanner_voxel_mm),
    "forward_model_fwhm": tuple(condition.forward_model_fwhm_xyz),
    "recon_model_fwhm": condition.recon_model_fwhm_xyz,
    "target_residual_fwhm": tuple(condition.target_residual_fwhm_xyz),
    "scanner": scanner_used,
    "beta": beta,
    "counts": counts,
    "realisation": realisation,
    "seed": seed_full,
}
```

Use two acquisition models: the forward model receives the true physical
resolution processor; the reconstruction model receives the condition's
recon-side processor (or none for `psf-none`). Pass the reconstruction model
to `_api.reconstruct_osem` instead of reusing the forward model. If
`attenuation_path` is configured, load and resample the uMap to the scanner
grid, wrap it with the gateway, and pass it to both models. A requested uMap
must either be used or raise the explicit unsupported-build error from Task 4.

Change `_build_observed` in `runner/execute.py` to return `(observed, meta)`;
image-space inputs return `{}` and `sirf_sim` returns the simulation metadata.
Set `cfg.setdefault("input_voxel_mm", ds.voxel_mm)` before calling
`simulate_inputs` so the physical round trip uses the NIfTI affine rather than
assuming one-millimetre voxels.
Write that metadata under the manifest's `simulation` key. The returned array
remains aligned with the original GT/guidance so existing runner metrics remain
valid.

- [ ] **Step 8: Implement resolution calibration helpers**

Create `simulation/calibration.py` with two pure functions:

```python
import json
from pathlib import Path

import numpy as np


def fwhm_from_profile(profile: np.ndarray, spacing_mm: float) -> float:
    """Return a sample-grid full width at half maximum."""
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1 or values.size < 3 or spacing_mm <= 0:
        raise ValueError("profile must be a 1-D sequence of at least three samples and spacing_mm must be positive")
    peak = float(values.max())
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("profile must contain a positive finite peak")
    above = np.flatnonzero(values >= peak / 2.0)
    if above.size < 2:
        raise ValueError("profile does not cross half maximum on both sides")
    return float((above[-1] - above[0]) * spacing_mm)


def write_resolution_calibration(records: dict[str, tuple[float, float, float]], path: str | Path) -> Path:
    """Write sorted condition -> measured (x,y,z) FWHM records as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({key: list(records[key]) for key in sorted(records)}, indent=2, sort_keys=True) + "\n")
    return output
```

The calibration test uses a delta phantom, extracts the three centre profiles
from each returned reconstruction, calls `fwhm_from_profile`, and writes a
temporary JSON file through `write_resolution_calibration`. It must record all
three conditions and assert the measured ordering is `psf-matched` sharper than
`psf-undersized`, which is sharper than `psf-none`; it must not assert that the
reduced geometry equals the Vision target exactly.

- [ ] **Step 9: Run tests and calibration**

Native: `python -m pytest studies/tests -q`.

Container: `make study-sirf-test`. Add a tiny resolution calibration test that
records measured FWHM for all three conditions in a JSON fixture or test output;
do not claim exact Vision values if the reduced geometry differs. The test must
assert that `psf-none`, `psf-undersized`, and `psf-matched` produce distinct
measured resolutions in the expected ordering.

- [ ] **Step 10: Commit**

```bash
git add studies/krl_studies/simulation studies/krl_studies/runner/execute.py studies/tests/test_geometry.py studies/tests/test_calibration.py studies/tests/test_simulation_sirf.py studies/docker-compose.yaml Makefile docs/reference/SIRF_API_NOTES.md
git commit -m "ADD scanner-grid resampling and explicit recon resolution models"
```

---

### Task 2: Canonical results schema and ingestion

**Files:**
- Create: `studies/krl_studies/analysis/__init__.py`
- Create: `studies/krl_studies/analysis/schema.py`
- Create: `studies/krl_studies/analysis/ingest.py`
- Create: `studies/tests/test_analysis_schema.py`
- Create: `studies/tests/test_analysis_ingest.py`
- Modify: `studies/pyproject.toml` (add optional `analysis` extra with `pyarrow>=14`)

The canonical store has three data tables plus an errors table. All tables are
written with stable column order and stable row sorting. Resolution vectors are
stored as compact JSON strings so CSV and parquet have identical semantics.

**runs table:** one row per completed run:

```text
run_id, run_path, study, subject_id, dataset_kind, input_kind, scanner,
condition, beta, counts, realisation, guidance_condition, method,
assumed_fwhm_mm, forward_model_fwhm_json, recon_model_fwhm_json,
target_residual_fwhm_json, method_params_json, sim_params_json, status,
git_rev, krl_version, krl_studies_version, finished_at
```

**iterations table:** one row per run and iteration:

```text
run_id, iteration, method, study, subject_id, dataset_kind, input_kind,
scanner, condition, beta, counts, realisation, guidance_condition,
assumed_fwhm_mm, forward_model_fwhm_json, recon_model_fwhm_json,
target_residual_fwhm_json, metric, value
```

`metric` is one of `nrmse`, `bv_percent`, or `objective`.

**lesions table:** one row per run, iteration, and lesion diameter:

```text
run_id, iteration, method, study, subject_id, dataset_kind, input_kind,
scanner, condition, beta, counts, realisation, guidance_condition,
assumed_fwhm_mm, forward_model_fwhm_json, recon_model_fwhm_json,
target_residual_fwhm_json, lesion_diameter_mm, metric, value
```

Here `metric` is `crc_percent`.

- [ ] **Step 1: Write schema tests**

```python
import json
import pytest

from krl_studies.analysis.schema import (
    ITERATION_COLUMNS,
    LESION_COLUMNS,
    RUN_COLUMNS,
    flatten_manifest,
    melt_metrics,
)


def test_flatten_manifest_normalises_nested_params():
    row = flatten_manifest(
        {
            "run_id": "r1",
            "study": "spheres",
            "dataset": {"kind": "spheres"},
            "input_kind": "sirf_sim",
            "input_params": {"condition": "psf-none", "beta": None, "counts": 1e7},
            "method": "krl",
            "method_params": {"sigma_anat": 0.2},
            "sim": {"seed": 1337},
            "simulation": {"scanner": "Siemens mMR"},
            "status": "complete",
        },
        "/tmp/r1",
    )
    assert row["condition"] == "psf-none"
    assert row["beta"] is None
    assert json.loads(row["method_params_json"]) == {"sigma_anat": 0.2}
    assert set(RUN_COLUMNS) <= row.keys()


def test_melt_metrics_separates_standard_metrics_and_crc():
    iterations, lesions = melt_metrics(
        "r1",
        {"iteration": [1], "nrmse": [0.2], "bv_percent": [4.0], "crc_mm8": [35.0]},
        {"study": "spheres", "method": "rl"},
    )
    assert set(iterations["metric"]) == {"nrmse", "bv_percent"}
    assert iterations["value"].tolist() == [0.2, 4.0]
    assert lesions["lesion_diameter_mm"].tolist() == [8.0]
    assert lesions["metric"].tolist() == ["crc_percent"]
    assert lesions["value"].tolist() == [35.0]
    assert set(ITERATION_COLUMNS) <= set(iterations.columns)
    assert set(LESION_COLUMNS) <= set(lesions.columns)


def test_invalid_metric_file_is_reported(tmp_path):
    (tmp_path / "metrics.csv").write_text("not_iteration,value\n1,2\n")
    with pytest.raises(ValueError, match="iteration"):
        melt_metrics("r1", {"not_iteration": [1], "value": [2]}, {})
```

- [ ] **Step 2: Implement schema normalization**

`schema.py` must define `RUN_COLUMNS`, `ITERATION_COLUMNS`,
`LESION_COLUMNS`, `flatten_manifest(manifest, run_path)`, and
`melt_metrics(run_id, frame_or_mapping, metadata)`, where the latter returns
`(iterations, lesions)` as two DataFrames.

Use `json.dumps(value, sort_keys=True, separators=(",", ":"))` for parameter
columns. Extract `subject_id` from `manifest["dataset"]`, `condition/beta/counts`
from `input_params`, and `guidance_condition` from either `input_params` or
`method_params`, defaulting to `"exact"`. Read `scanner` first from the
`simulation` metadata and then from `input_params` and `sim`. Read
`assumed_fwhm_mm` from `method_params["fwhm_mm"]` when present. Store the
simulation vectors from `simulation["forward_model_fwhm"]`,
`simulation["recon_model_fwhm"]`, and `simulation["target_residual_fwhm"]` as
JSON. Parse `crc_mm<number>` columns with the regex
`^crc_mm(?P<diameter>[-+0-9p.e]+)$`; convert `p` back to `.` before `float`.
Missing metrics become absent rows, not zeros.

Use this implementation shape so the table contract is executable without
guessing column names:

```python
import json
import re
from pathlib import Path

import pandas as pd

COMMON_COLUMNS = [
    "run_id", "iteration", "method", "study", "subject_id", "dataset_kind",
    "input_kind", "scanner", "condition", "beta", "counts", "realisation",
    "guidance_condition", "assumed_fwhm_mm", "forward_model_fwhm_json",
    "recon_model_fwhm_json", "target_residual_fwhm_json",
]
RUN_COLUMNS = [
    "run_id", "run_path", "study", "subject_id", "dataset_kind", "input_kind",
    "scanner", "condition", "beta", "counts", "realisation",
    "guidance_condition", "method", "assumed_fwhm_mm", "forward_model_fwhm_json",
    "recon_model_fwhm_json", "target_residual_fwhm_json", "method_params_json",
    "sim_params_json", "status", "git_rev", "krl_version", "krl_studies_version",
    "finished_at",
]
ITERATION_COLUMNS = COMMON_COLUMNS + ["metric", "value"]
LESION_COLUMNS = COMMON_COLUMNS + ["lesion_diameter_mm", "metric", "value"]

STANDARD_METRICS = ("nrmse", "bv_percent", "objective")
CRC_RE = re.compile(r"^crc_mm(?P<diameter>[-+0-9p.e]+)$")


def _compact(value):
    return None if value is None else json.dumps(value, sort_keys=True, separators=(",", ":"))


def flatten_manifest(manifest: dict, run_path: str | Path) -> dict:
    dataset = dict(manifest.get("dataset", {}))
    input_params = dict(manifest.get("input_params", {}))
    method_params = dict(manifest.get("method_params", {}))
    sim_params = dict(manifest.get("sim", {}))
    simulation = dict(manifest.get("simulation", {}))
    subject_id = dataset.get("subject_id", dataset.get("subject"))
    assumed = method_params.get("fwhm_mm")
    if isinstance(assumed, (list, tuple, dict)):
        assumed = None
    row = {
        "run_id": manifest.get("run_id"),
        "run_path": str(run_path),
        "study": manifest.get("study"),
        "subject_id": subject_id,
        "dataset_kind": dataset.get("kind"),
        "input_kind": manifest.get("input_kind"),
        "scanner": simulation.get("scanner", input_params.get("scanner", sim_params.get("scanner"))),
        "condition": input_params.get("condition"),
        "beta": input_params.get("beta"),
        "counts": input_params.get("counts"),
        "realisation": input_params.get("realisation"),
        "guidance_condition": input_params.get(
            "guidance_condition", method_params.get("guidance_condition", "exact")
        ),
        "method": manifest.get("method"),
        "assumed_fwhm_mm": assumed,
        "forward_model_fwhm_json": _compact(simulation.get("forward_model_fwhm")),
        "recon_model_fwhm_json": _compact(simulation.get("recon_model_fwhm")),
        "target_residual_fwhm_json": _compact(simulation.get("target_residual_fwhm")),
        "method_params_json": _compact(method_params),
        "sim_params_json": _compact(sim_params),
        "status": manifest.get("status"),
        "git_rev": manifest.get("git_rev"),
        "krl_version": manifest.get("krl_version"),
        "krl_studies_version": manifest.get("krl_studies_version"),
        "finished_at": manifest.get("finished_at"),
    }
    return {column: row.get(column) for column in RUN_COLUMNS}


def _frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = None
    return frame.reindex(columns=columns)


def melt_metrics(run_id: str, frame_or_mapping, metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame_or_mapping.copy() if isinstance(frame_or_mapping, pd.DataFrame) else pd.DataFrame(frame_or_mapping)
    if "iteration" not in frame.columns:
        raise ValueError("metrics require an iteration column")
    common = {column: metadata.get(column) for column in COMMON_COLUMNS}
    common["run_id"] = run_id
    standard = [column for column in STANDARD_METRICS if column in frame.columns]
    melted = frame.melt(
        id_vars=["iteration"], value_vars=standard, var_name="metric", value_name="value"
    )
    iteration_rows = []
    for record in melted.to_dict("records"):
        if pd.notna(record["value"]):
            iteration_rows.append({**common, "iteration": record["iteration"], "metric": record["metric"], "value": record["value"]})
    lesion_rows = []
    for column in frame.columns:
        match = CRC_RE.fullmatch(str(column))
        if match is None:
            continue
        diameter = float(match.group("diameter").replace("p", "."))
        for iteration, value in zip(frame["iteration"], frame[column]):
            if pd.notna(value):
                lesion_rows.append({
                    **common,
                    "iteration": iteration,
                    "lesion_diameter_mm": diameter,
                    "metric": "crc_percent",
                    "value": value,
                })
    return _frame(iteration_rows, ITERATION_COLUMNS), _frame(lesion_rows, LESION_COLUMNS)
```

`flatten_manifest` fills every `RUN_COLUMNS` key, using `None` for absent
optional values. `melt_metrics` first requires an `iteration` column, melts
only `STANDARD_METRICS` that are present, then emits one lesion row per
`crc_mm<number>` column with `metric="crc_percent"`. Both returned frames include
the metadata columns and are reindexed to the declared column lists.

- [ ] **Step 3: Ingest completed run directories**

`ingest.py`:

```python
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from krl_studies.analysis.schema import (
    ITERATION_COLUMNS,
    LESION_COLUMNS,
    RUN_COLUMNS,
    flatten_manifest,
    melt_metrics,
)


@dataclass(frozen=True)
class ResultsTables:
    runs: pd.DataFrame
    iterations: pd.DataFrame
    lesions: pd.DataFrame
    errors: pd.DataFrame


def discover_completed_runs(results_root: str | Path) -> list[Path]:
    root = Path(results_root)
    return sorted(
        marker.parent for marker in root.rglob(".done")
        if (marker.parent / "manifest.json").exists()
        and (marker.parent / "metrics.csv").exists()
    )


def ingest_results(results_root: str | Path) -> ResultsTables:
    """Read only complete runs; malformed runs are returned in errors."""
    run_rows, iteration_frames, lesion_frames, errors = [], [], [], []
    for run_path in discover_completed_runs(results_root):
        try:
            manifest = json.loads((run_path / "manifest.json").read_text())
            metrics = pd.read_csv(run_path / "metrics.csv")
            run_row = flatten_manifest(manifest, run_path)
            iterations, lesions = melt_metrics(run_row["run_id"], metrics, run_row)
        except (OSError, KeyError, ValueError, json.JSONDecodeError, pd.errors.ParserError,
                pd.errors.EmptyDataError) as exc:
            errors.append({"run_path": str(run_path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        run_rows.append(run_row)
        iteration_frames.append(iterations)
        lesion_frames.append(lesions)
    return ResultsTables(
        runs=pd.DataFrame(run_rows, columns=RUN_COLUMNS),
        iterations=pd.concat(iteration_frames, ignore_index=True) if iteration_frames else pd.DataFrame(columns=ITERATION_COLUMNS),
        lesions=pd.concat(lesion_frames, ignore_index=True) if lesion_frames else pd.DataFrame(columns=LESION_COLUMNS),
        errors=pd.DataFrame(errors, columns=["run_path", "error"]),
    )


def write_tables(tables: ResultsTables, out_dir: str | Path) -> dict[str, Path]:
    """Write CSV always and parquet when pyarrow is installed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "runs": tables.runs.reindex(columns=RUN_COLUMNS),
        "iterations": tables.iterations.reindex(columns=ITERATION_COLUMNS),
        "lesions": tables.lesions.reindex(columns=LESION_COLUMNS),
        "errors": tables.errors.reindex(columns=["run_path", "error"]),
    }
    paths = {}
    for name, frame in frames.items():
        sort_columns = [column for column in frame.columns if column in frame]
        ordered = frame.sort_values(sort_columns, kind="stable", na_position="last") if not frame.empty else frame
        csv_path = out_dir / f"{name}.csv"
        ordered.to_csv(csv_path, index=False)
        paths[name] = csv_path
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return paths
    for name, frame in frames.items():
        parquet_path = out_dir / f"{name}.parquet"
        frame.to_parquet(parquet_path, index=False)
        paths[f"{name}_parquet"] = parquet_path
    return paths
```

`ingest_results` reads only directories returned by `discover_completed_runs`,
so incomplete run directories are ignored. Malformed completed directories are
not fatal: they produce one row in `errors` with the path and exception text.
`write_tables` always writes CSV and writes parquet only when the optional
dependency is installed.

- [ ] **Step 4: Test ingestion and deterministic output**

Build two temporary completed-run directories with manifests/metrics and one
malformed directory. Assert both valid run IDs are returned, the malformed path
appears in `errors`, `discover_completed_runs` excludes a directory without
`.done`, and two calls to `write_tables` produce byte-identical CSV files.
Use `tmp_path` rather than `/tmp` and include one CRC column in one fixture.

- [ ] **Step 5: Commit**

```bash
git add studies/krl_studies/analysis studies/tests/test_analysis_schema.py studies/tests/test_analysis_ingest.py studies/pyproject.toml
git commit -m "ADD canonical results schema and completed-run ingestion"
```

---

### Task 3: Replicate summaries and iteration selection

**Files:**
- Create: `studies/krl_studies/analysis/aggregate.py`
- Create: `studies/krl_studies/analysis/selection.py`
- Create: `studies/tests/test_analysis_selection.py`

- [ ] **Step 1: Write tests**

```python
import pandas as pd

from krl_studies.analysis.aggregate import summarize_replicates
from krl_studies.analysis.selection import select_fixed_iteration, select_oracle


def frame():
    return pd.DataFrame(
        [
            {"run_id": "r0", "realisation": 0, "method": "rl", "iteration": 1, "metric": "nrmse", "value": 0.4},
            {"run_id": "r0", "realisation": 0, "method": "rl", "iteration": 2, "metric": "nrmse", "value": 0.3},
            {"run_id": "r1", "realisation": 1, "method": "rl", "iteration": 1, "metric": "nrmse", "value": 0.5},
            {"run_id": "r1", "realisation": 1, "method": "rl", "iteration": 2, "metric": "nrmse", "value": 0.35},
        ]
    )


def test_oracle_selects_lowest_nrmse_and_earliest_tie():
    result = select_oracle(frame())
    assert result.iloc[0]["iteration"] == 2
    assert result.iloc[0]["selection"] == "oracle_min_nrmse"
    assert len(result) == 2


def test_fixed_selection_returns_empty_for_missing_iteration():
    result = select_fixed_iteration(frame(), 3)
    assert result.empty


def test_oracle_retains_other_metrics_at_selected_iteration():
    iterations = pd.concat(
        [
            frame(),
            pd.DataFrame([
                {"run_id": "r0", "realisation": 0, "method": "rl", "iteration": 2, "metric": "bv_percent", "value": 3.0},
            ]),
        ],
        ignore_index=True,
    )
    result = select_oracle(iterations)
    assert set(result.loc[result["run_id"] == "r0", "metric"]) == {"nrmse", "bv_percent"}


def test_replicate_summary_has_mean_std_count():
    summary = summarize_replicates(frame())
    row = summary.iloc[0]
    assert row["n"] == 2
    assert row["value_mean"] == 0.45
    assert row["value_std"] > 0


def test_replicate_summary_excludes_missing_values():
    data = frame()
    data.loc[0, "value"] = None
    summary = summarize_replicates(data)
    assert summary.loc[summary["iteration"] == 1, "n"].iloc[0] == 1
```

- [ ] **Step 2: Implement selection**

`select_oracle(iterations)` filters `metric == "nrmse"` and non-null values,
sorts by `run_id`, `value` ascending, and `iteration` ascending, then keeps
the first row per `run_id`. It then joins those `(run_id, iteration)` pairs
back to the input frame, retaining every metric recorded at the selected
iteration. This selects one best iteration independently for each completed
realisation and therefore does not leak information between replicates. It
adds `selection="oracle_min_nrmse"` to every retained metric row.

`select_fixed_iteration(iterations, iteration)` filters exactly that iteration
and adds `selection=f"fixed_{iteration}"`; it returns an empty frame rather
than fabricating values when a method stopped early.

Implement `selection.py` as follows:

```python
import pandas as pd


def select_oracle(iterations: pd.DataFrame) -> pd.DataFrame:
    nrmse = iterations.loc[
        (iterations["metric"] == "nrmse") & iterations["value"].notna(),
        ["run_id", "iteration", "value"],
    ].sort_values(["run_id", "value", "iteration"], kind="stable")
    best = nrmse.drop_duplicates("run_id", keep="first")[["run_id", "iteration"]]
    selected = iterations.merge(best, on=["run_id", "iteration"], how="inner", validate="many_to_many")
    selected = selected.sort_values(["run_id", "iteration", "metric"], kind="stable").reset_index(drop=True)
    selected["selection"] = "oracle_min_nrmse"
    return selected


def select_fixed_iteration(iterations: pd.DataFrame, iteration: int) -> pd.DataFrame:
    selected = iterations.loc[iterations["iteration"] == int(iteration)].copy()
    selected["selection"] = f"fixed_{int(iteration)}"
    return selected.reset_index(drop=True)
```

- [ ] **Step 3: Implement replicate aggregation**

`summarize_replicates(iterations)` groups by all available metadata columns
except `run_id`, `realisation`, `iteration`, and `value`, then adds `iteration`
and `metric` back as grouping keys. The implementation must preserve null beta
and guidance values with `dropna=False`, sort the result by grouping columns,
and use:

```python
grouped["value_mean"] = grouped["value"].mean()
grouped["value_std"] = grouped["value"].std(ddof=1).fillna(0.0)
grouped["n"] = grouped["value"].count().astype(int)
```

Keep `iteration` and `metric` in the grouping keys. This is metric-level
bias/variance across noise realisations; do not claim voxelwise variance when
only scalar metric rows are stored. Implement `aggregate.py` with:

```python
def summarize_replicates(iterations: pd.DataFrame) -> pd.DataFrame:
    frame = iterations.copy()
    if frame.empty:
        return pd.DataFrame(columns=["value_mean", "value_std", "n"])
    group_columns = [
        column for column in frame.columns
        if column not in {"run_id", "realisation", "value", "value_mean", "value_std", "n"}
    ]
    grouped = frame.groupby(group_columns, dropna=False, sort=True)["value"].agg(
        value_mean="mean", value_std=lambda values: values.std(ddof=1), n="count"
    ).reset_index()
    grouped["value_std"] = grouped["value_std"].fillna(0.0)
    grouped["n"] = grouped["n"].astype(int)
    return grouped
```

The grouping columns retain `iteration` and `metric` while excluding the run
and realisation identifiers. Missing values are excluded from `n` rather than
treated as zero.

The selection tests must include a second metric at iteration 2 and assert that
the oracle output contains that metric, proving that table generation receives
the full selected iterate rather than only the NRMSE row.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest studies/tests/test_analysis_selection.py -v`

Expected: 5 passed. Then native full suite and lint.

```bash
git add studies/krl_studies/analysis studies/tests/test_analysis_selection.py
git commit -m "ADD replicate summaries and iteration selection policies"
```

---

### Task 4: Guidance mismatch transforms and campaign scenarios

**Files:**
- Create: `studies/krl_studies/datasets/transforms.py`
- Modify: `studies/krl_studies/datasets/brainweb.py` (save T2 and lesion metadata; load prepared subjects)
- Modify: `studies/krl_studies/datasets/patients.py` (optional `T2.nii.gz`)
- Modify: `studies/krl_studies/runner/execute.py` (brainweb study and guidance condition)
- Create: `studies/scenarios/resolution_calibration.yaml`
- Create: `studies/scenarios/spheres_mismatch.yaml`
- Create: `studies/scenarios/brainweb_mismatch.yaml`
- Create: `studies/scenarios/patient_cohort.yaml`
- Create: `studies/tests/test_guidance_transforms.py`
- Modify: `studies/tests/test_brainweb_dataset.py`
- Modify: `studies/tests/test_patients_dataset.py`
- Modify: `studies/tests/test_config.py`
- Modify: `data/README.md`

- [ ] **Step 1: Write transform tests**

```python
import numpy as np
import pytest

from krl_studies.datasets.transforms import apply_guidance_condition


def test_exact_guidance_is_unchanged():
    image = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    result = apply_guidance_condition(image, "exact", (2.0, 2.0, 2.0))
    assert np.array_equal(result, image)


def test_positive_shift_moves_an_impulse_in_physical_axes():
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, 4, 4] = 1.0
    result = apply_guidance_condition(image, "shift_p2", (1.0, 1.0, 1.0), order=0)
    assert np.unravel_index(np.argmax(result), result.shape) == (6, 6, 6)


def test_unknown_guidance_condition_raises():
    with pytest.raises(ValueError):
        apply_guidance_condition(np.zeros((3, 3, 3)), "shift_p9", (1, 1, 1))
```

- [ ] **Step 2: Implement physical guidance transforms**

`datasets/transforms.py`:

```python
import numpy as np

from scipy.ndimage import shift


def apply_guidance_condition(
    image: np.ndarray,
    condition: str,
    voxel_mm_zyx: tuple[float, float, float],
    *,
    order: int = 1,
) -> np.ndarray:
    """Apply exact or ±2/±5 mm rigid shifts; return a new float32 array."""
    offsets_mm = {
        "exact": (0.0, 0.0, 0.0),
        "shift_p2": (2.0, 2.0, 2.0),
        "shift_m2": (-2.0, -2.0, -2.0),
        "shift_p5": (5.0, 5.0, 5.0),
        "shift_m5": (-5.0, -5.0, -5.0),
    }
    if condition == "t2":
        raise ValueError("t2 guidance is loaded from the dataset, not shifted")
    if condition not in offsets_mm:
        raise ValueError(f"unknown guidance condition: {condition!r}")
    voxels = np.asarray(voxel_mm_zyx, dtype=float)
    if voxels.shape != (3,) or not np.all(np.isfinite(voxels)) or np.any(voxels <= 0):
        raise ValueError("voxel_mm_zyx must contain three positive finite values")
    if condition == "exact":
        return np.asarray(image, dtype=np.float32).copy()
    voxel_shift = tuple(mm / voxel for mm, voxel in zip(offsets_mm[condition], voxels))
    return shift(np.asarray(image, dtype=np.float32), voxel_shift, order=order, mode="nearest").astype(np.float32)
```

Validate that `voxel_mm_zyx` has three finite positive values before computing
the shift. For `t2`, the runner loads the optional T2 file rather than applying
a numeric shift. If `t2` is passed to this function, raise
`ValueError("t2 guidance is loaded from the dataset, not shifted")` so a caller
cannot accidentally treat a modality swap as a translation.

- [ ] **Step 3: Extend BrainWeb/patient guidance loading**

`prepare_subject` must save `mr_t2.nii.gz` from the already-loaded BrainWeb
`vol["T2"]` array, and must persist the exact tumour masks used to construct
`pet_gt.nii.gz`:

```python
specs = []
lesion_masks = []
if tumour:
    specs = default_tumour_specs(shape=pet.shape, voxel_mm=voxel_mm)
    pet_gt, lesion_masks = place_tumours(pet, specs, voxel_mm=voxel_mm)
else:
    pet_gt = pet
mask_array = np.asarray(lesion_masks, dtype=bool)
if not tumour:
    mask_array = np.empty((0, *pet.shape), dtype=bool)
np.savez_compressed(out_dir / "lesion_masks.npz", masks=mask_array)
(out_dir / "lesion_diameters_mm.json").write_text(json.dumps(
    [float(spec["radius_mm"]) * 2.0 for spec in specs]
))
```

When `tumour=False`, the mask array is explicitly `(0, *pet.shape)` and the
diameter list is empty. Add
`mr_t2`, `lesion_masks`, and `lesion_diameters_mm` to the returned `paths`
mapping. The existing tissue-label snapping logic remains in the tumour branch
before `place_tumours`.
Add a `BrainWebDataset(root, subject_id)` adapter in `brainweb.py` that loads
`pet_gt`, `mr_t1`, `labels`, the optional `mr_t2`, and the persisted lesion
masks/diameters from `data/brainweb/subject_<id>/`; it must expose
`ground_truth`, `guidance`, `t2`, `labels`, `lesion_masks`, `lesion_diameters_mm`,
and `voxel_mm` properties using `(z, y, x)` arrays.

Add `self.t2 = _load(self.dir / _FILES["T2"]) if (self.dir / _FILES["T2"]).exists() else None`
to `PatientDataset`,
extend `_FILES` with `"T2": "T2.nii.gz"`, and leave `discover_subjects`
compatible with subjects that do not have T2. Update `data/README.md` with the
optional `T2.nii.gz` convention and the BrainWeb output filenames.

- [ ] **Step 4: Apply guidance condition in the runner**

For `spheres`, `brainweb`, and `patient` paths, read
`guidance_condition = run.input_params.get("guidance_condition", "exact")`.

- `exact`: current guidance
- `shift_*`: `apply_guidance_condition(guidance_arr, condition, voxel_mm)`
- `t2`: BrainWeb `ds.t2` or patient `ds.t2`; raise
  `FileNotFoundError` naming the subject when absent

Add a `brainweb` branch in `execute_run` that constructs `BrainWebDataset` from
`dataset.root` and `dataset.subject_id`, uses its persisted tumour masks for
CRC, and uses its labels for the existing iY region builder. Apply the selected
guidance array immediately before wrapping it for CIL. Record
`guidance_condition` in the manifest even when it is `exact`; do not include
the full guidance volume in JSON metadata.

For BrainWeb iY runs, set `params["region_masks"] = regions_from_labels(ds.labels)`
and `params["brain_mask"] = ds.labels != 0` when those keys are absent. For
spheres retain `_iy_region_defaults`; for patient runs retain the existing ROI
path. A requested `t2` condition on spheres must raise `ValueError` because no
second modality is part of that dataset.

- [ ] **Step 5: Add campaign YAMLs**

`resolution_calibration.yaml`:

```yaml
study: spheres
dataset:
  kind: spheres
  root: data/spheres
sim:
  seed: 1337
  scanner: Siemens mMR
  n_subits: 2
inputs:
  - kind: sirf_sim
    params:
      condition: [psf-none, psf-undersized, psf-matched]
      beta: [null]
      counts: [5.0e7, 1.0e8, 2.0e8]
      realisation: [0]
methods:
  - name: post_smoothing
    params: {sigma_mm: [0.0]}
output: results/resolution_calibration
```

`spheres_mismatch.yaml`:

```yaml
study: spheres
dataset:
  kind: spheres
  root: data/spheres
sim:
  seed: 1337
  scanner: Siemens mMR
  n_subits: 4
  add_tumours: true
  tumour_contrast: 4.0
inputs:
  - kind: sirf_sim
    params:
      condition: [psf-none, psf-undersized, psf-matched]
      beta: [null, 10.0, 50.0]
      counts: [5.0e7, 1.0e8, 2.0e8]
      realisation: [0, 1, 2]
      guidance_condition: [exact, shift_p2, shift_m2, shift_p5, shift_m5]
methods:
  - name: rl
    params: {fwhm_mm: [4.0, 5.0, 5.7, 6.5, 7.5], iterations: 50}
  - name: krl
    params: {fwhm_mm: [4.0, 5.0, 5.7, 6.5, 7.5], sigma_anat: [0.2, 1.0], iterations: 50}
  - name: hkrl
    params: {fwhm_mm: [4.0, 5.0, 5.7, 6.5, 7.5], sigma_anat: [0.2, 1.0], sigma_emission: [0.5, 1.0], freeze_iteration: [5, 10], iterations: 50}
  - name: dtv
    params: {fwhm_mm: [4.0, 5.0, 5.7, 6.5, 7.5], alpha: [0.05, 0.1, 0.2], iterations: 100}
  - name: iy
    params: {fwhm_mm: [4.0, 5.0, 5.7, 6.5, 7.5], damping: [0.25, 0.5], iterations: 10}
output: results/spheres_mismatch
```

`brainweb_mismatch.yaml` must be a valid, explicit single-subject campaign
rather than a prose reference:

```yaml
study: brainweb
dataset:
  kind: brainweb
  root: data/brainweb
  subject_id: 04
sim:
  seed: 1337
  scanner: Siemens mMR
  n_subits: 4
inputs:
  - kind: sirf_sim
    params:
      condition: [psf-none, psf-undersized, psf-matched]
      beta: [null, 10.0, 50.0]
      counts: [5.0e7, 1.0e8, 2.0e8]
      realisation: [0, 1, 2]
      guidance_condition: [exact, t2, shift_p2, shift_m2]
methods:
  - name: rl
    params: {fwhm_mm: [4.0, 5.0, 5.7, 6.5, 7.5], iterations: 50}
  - name: krl
    params: {fwhm_mm: [4.0, 5.0, 5.7, 6.5, 7.5], sigma_anat: [0.2, 1.0], iterations: 50}
  - name: hkrl
    params: {fwhm_mm: [4.0, 5.0, 5.7, 6.5, 7.5], sigma_anat: [0.2, 1.0], sigma_emission: [0.5, 1.0], freeze_iteration: [5, 10], iterations: 50}
  - name: dtv
    params: {fwhm_mm: [4.0, 5.0, 5.7, 6.5, 7.5], alpha: [0.05, 0.1, 0.2], iterations: 100}
  - name: iy
    params: {fwhm_mm: [4.0, 5.0, 5.7, 6.5, 7.5], damping: [0.25, 0.5], iterations: 10}
output: results/brainweb_mismatch
```

`patient_cohort.yaml` must demonstrate the no-ground-truth path without adding
patient data to git:

```yaml
study: patient
dataset:
  kind: patient
  root: data/patients
  subject_id: MK-H001
inputs:
  - kind: native
methods:
  - name: rl
    params: {fwhm_mm: 4.5, iterations: 20}
  - name: krl
    params: {fwhm_mm: 4.5, num_neighbours: 9, sigma_anat: 0.2, iterations: 20}
output: results/patient_cohort
```

The scenario tests must parse all four YAML files and assert that the two
mismatch scenarios contain `psf-none`, `psf-undersized`, `psf-matched`, at
least two count levels, and at least two guidance conditions. They must also
assert that no scenario path points below `data/patients` as a committed file.

- [ ] **Step 6: Test and commit**

Run native transform tests and dry-run all non-patient campaigns; assert the
dry-run output is deterministic and the expanded count is non-zero:

```bash
python -m pytest studies/tests/test_guidance_transforms.py studies/tests/test_config.py -v
python -m krl_studies.run --scenario studies/scenarios/resolution_calibration.yaml --dry-run
python -m krl_studies.run --scenario studies/scenarios/spheres_mismatch.yaml --dry-run
python -m krl_studies.run --scenario studies/scenarios/brainweb_mismatch.yaml --dry-run
git add studies/krl_studies/datasets studies/krl_studies/runner/execute.py studies/scenarios data/README.md studies/tests/test_guidance_transforms.py studies/tests/test_brainweb_dataset.py studies/tests/test_patients_dataset.py studies/tests/test_config.py
git commit -m "ADD guidance mismatch transforms and campaign scenarios"
```

---

### Task 5: Aggregate analysis, selections, plots, and tables

**Files:**
- Create: `studies/krl_studies/analysis/plots.py`
- Create: `studies/krl_studies/analysis/tables.py`
- Create: `studies/krl_studies/analysis/report.py`
- Create: `studies/krl_studies/report.py`
- Create: `studies/tests/test_analysis_plots.py`
- Create: `studies/tests/test_analysis_tables.py`
- Create: `studies/tests/test_report_cli.py`
- Modify: `studies/krl_studies/analysis/__init__.py`

- [ ] **Step 1: Define plotting API and tests**

All plotting functions accept DataFrames and an output path; none discover
files, read global constants, or use absolute paths. Plotting consumes the
canonical summary frames, not raw result-directory globs:

```python
def plot_nrmse_convergence(summary: pd.DataFrame, output: Path, *, title: str) -> None:
    """Write mean +/- std NRMSE versus iteration, grouped by method."""


def plot_recovery_vs_cov(summary: pd.DataFrame, output: Path, *, title: str) -> None:
    """Write CRC/NRMSE versus background variability."""


def plot_crc_by_size(lesion_summary: pd.DataFrame, output: Path, *, title: str) -> None:
    """Write mean +/- std CRC versus lesion diameter."""


def plot_mismatch_sensitivity(summary: pd.DataFrame, output: Path, *, title: str) -> None:
    """Write the selected metric versus assumed deconvolution FWHM."""


def plot_profile(images: dict[str, np.ndarray], output: Path, *, axis: int, index: tuple[int, int]) -> None:
    """Write fixed-index one-dimensional profiles from supplied arrays."""
```

Use `matplotlib.use("Agg")` in the module, a fixed method colour map, and
`Path(output).parent.mkdir(parents=True, exist_ok=True)` in every writer. Each
writer calls `fig.savefig(output, dpi=200, bbox_inches="tight")` and closes the
figure in a `finally` block. Empty inputs produce an empty labelled figure and
do not raise.

The plotting tests construct small DataFrames containing two methods, two
realisations, the three resolution conditions, beta values, guidance
conditions, `assumed_fwhm_mm`, `value_mean`, `value_std`, `bv_percent`, and
`lesion_diameter_mm`, `crc_percent`, and `nrmse`. They call every writer with `tmp_path / "figure.png"`,
assert the file exists and has non-zero size, and assert that a second call
overwrites it deterministically.

- [ ] **Step 2: Implement the publication figures**

Required semantics:

- `plot_nrmse_convergence`: filter `metric == "nrmse"`, draw one line per
  method, use `value_mean` and `value_std`, and label condition and beta in the
  legend.
- `plot_recovery_vs_cov`: use the `tradeoff.csv`-shaped frame with
  `bv_percent` and a recovery value; prefer `crc_percent` when present and
  otherwise use NRMSE, connect iteration points, and label method.
- `plot_crc_by_size`: use `lesion_diameter_mm` on x and `value_mean` on y with
  `value_std` error bars; separate lines by condition and guidance condition.
- `plot_mismatch_sensitivity`: use `assumed_fwhm_mm` on x, separate lines by
  `condition`, `recon_model_fwhm_json`, and guidance condition, and plot the
  metric named by the caller's filtered input frame. The forward-model vector
  and target residual vector remain available in the frame for labels and
  auditability.
- `plot_profile`: extract a fixed line through a supplied image dictionary;
  validate `axis in {0, 1, 2}` and two valid fixed indices, and never hardcode
  sphere coordinates.

- [ ] **Step 3: Implement LaTeX/CSV tables**

`tables.py`:

```python
def best_results_table(selected: pd.DataFrame) -> pd.DataFrame:
    """Return one row per method/input condition with mean/std metric columns."""
    if selected.empty:
        return pd.DataFrame()
    group_columns = [
        column for column in (
            "study", "subject_id", "dataset_kind", "input_kind", "scanner",
            "condition", "beta", "counts", "guidance_condition", "method",
            "assumed_fwhm_mm", "selection",
        ) if column in selected.columns
    ]
    group_columns_with_metric = group_columns + ["metric"]
    result = selected.groupby(group_columns_with_metric, dropna=False, sort=True)["value"].agg(
        value_mean="mean", value_std=lambda values: values.std(ddof=1), n="count"
    ).reset_index()
    result["value_std"] = result["value_std"].fillna(0.0)
    result["n"] = result["n"].astype(int)
    return result.pivot(index=group_columns, columns="metric", values=["value_mean", "value_std", "n"]).reset_index()


def write_latex_table(frame: pd.DataFrame, output: Path, caption: str, label: str) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(frame.to_latex(index=False, escape=True, caption=caption, label=label))
```

Required outputs: `best_oracle.csv`, `best_oracle.tex`, `best_fixed.csv`, and
`best_fixed.tex`. Add a table test that verifies grouping keeps null beta values,
that metric columns are separate after the pivot, and that a one-row group has
`value_std == 0.0` for that metric.

- [ ] **Step 4: Report CLI**

`python -m krl_studies.report` must support:

```text
aggregate --results RESULTS --out ANALYSIS
figures --analysis ANALYSIS --out FIGURES
tables --analysis ANALYSIS --out TABLES
all --results RESULTS --out OUT
```

Implement the parser with four subcommands. `aggregate` and `all` require
`--results` and `--out`; `figures` and `tables` require `--analysis` and
`--out`. Dispatch directly to the three functions above and return `0` on
success. `--help` must not import SIRF or discover result files.

Add `--fixed-iteration INTEGER` to `aggregate` and `all`, defaulting to `10`;
the value is recorded in the generated analysis metadata. `analysis/report.py`
must define `aggregate_results(results_root, out_dir, fixed_iteration)`,
`generate_figures(analysis_dir, out_dir)`, and
`generate_tables(analysis_dir, out_dir)`. `aggregate_results` calls
`ingest_results`, `write_tables`, `summarize_replicates`,
`select_oracle`, and `select_fixed_iteration`, then writes:

```text
runs.csv
iterations.csv
lesions.csv
errors.csv
summary.csv
lesion_summary.csv
tradeoff.csv
oracle.csv
fixed.csv
analysis_metadata.json
```

`lesion_summary.csv` is produced by applying `summarize_replicates` to the
lesion table. `tradeoff.csv` is a deterministic wide join of background
variability rows and CRC rows. Build it by pivoting the scalar iteration rows
to `bv_percent`, `nrmse`, and `objective`, then left-joining those rows to the
CRC lesion rows on all shared identity columns plus `(run_id, iteration)`; it
contains `lesion_diameter_mm`, `bv_percent`,
`crc_percent`, and `nrmse` columns where available. `generate_figures` reads
only those canonical CSVs and writes
`nrmse_convergence.png`, `recovery_vs_cov.png`, `crc_by_size.png`, and
`mismatch_sensitivity.png`. `generate_tables` reads `oracle.csv` and
`fixed.csv` and writes the four best-result files. `all` runs ingestion →
aggregation → selections → figures → tables and uses an `out` directory with
`tables/`, `figures/`, and `aggregate/` subdirectories.

- [ ] **Step 5: CLI tests and commit**

Build a temporary two-run result tree with `.done`, `manifest.json`, and
`metrics.csv`, run `main(["all", "--results", str(root), "--out", str(out)])`,
and assert all ten aggregate files, four figure files, and four table files
exist. Test `aggregate`, `figures`, `tables`, and `all` independently, plus
`main(["--help"]) == 0`. Commit:

```bash
git add studies/krl_studies/analysis studies/krl_studies/report.py studies/tests/test_analysis_*.py studies/tests/test_report_cli.py
git commit -m "ADD aggregated analysis, publication figures, and LaTeX tables"
```

---

### Task 6: Reproducible run plans, SGE arrays, and Docker execution

**Files:**
- Create: `studies/krl_studies/runner/plan.py`
- Create: `studies/krl_studies/plan.py`
- Create: `studies/krl_studies/cluster/__init__.py`
- Create: `studies/krl_studies/cluster/sge.py`
- Create: `studies/scripts/container_entrypoint.sh`
- Modify: `studies/docker-compose.yaml`
- Modify: `studies/krl_studies/runner/cli.py`
- Modify: `Makefile`
- Create: `studies/tests/test_run_plan.py`
- Create: `studies/tests/test_sge.py`

- [ ] **Step 1: Add JSONL run-plan serialization**

`runner/plan.py`:

```python
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from krl_studies.config import RunSpec


PLAN_VERSION = 1


def run_to_dict(run: RunSpec) -> dict[str, Any]:
    data = asdict(run)
    data["out_root"] = str(run.out_root)
    return data


def write_run_plan(runs: Sequence[RunSpec], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"plan_version": PLAN_VERSION}) + "\n")
        for run in runs:
            f.write(json.dumps(run_to_dict(run), sort_keys=True) + "\n")
    return path


def read_run_plan(path: str | Path) -> list[RunSpec]:
    rows = Path(path).read_text().splitlines()
    if not rows:
        raise ValueError("run plan is empty")
    try:
        header = json.loads(rows[0])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON run-plan header") from exc
    if header != {"plan_version": PLAN_VERSION}:
        raise ValueError(f"unsupported run-plan header: {header!r}")
    runs = []
    for line_number, line in enumerate(rows[1:], start=2):
        if not line.strip():
            raise ValueError(f"blank run-plan row at line {line_number}")
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}") from exc
        try:
            runs.append(RunSpec(
                run_id=str(data["run_id"]),
                study=str(data["study"]),
                dataset=dict(data["dataset"]),
                input_kind=str(data["input_kind"]),
                input_params=dict(data["input_params"]),
                method_name=str(data["method_name"]),
                method_params=dict(data["method_params"]),
                sim=dict(data["sim"]),
                out_root=Path(data["out_root"]),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid run-plan row at line {line_number}") from exc
    return runs
```

`read_run_plan` validates the header exactly, rejects blank or malformed rows,
converts `out_root` back to `Path`, and raises a line-numbered `ValueError`
without silently skipping a task.

- [ ] **Step 2: Extend CLI for exact array-task execution**

Add mutually exclusive required `--scenario` and `--plan` arguments, optional
integer `--index`, and optional `--out`. Keep `--dry-run`, `--force`, and
`--only` for scenario mode. Reject `--index`, `--dry-run`, and `--only` in plan
mode with `parser.error`.

```python
if args.plan is not None:
    runs = read_run_plan(args.plan)
    if args.index is None or not 1 <= args.index <= len(runs):
        parser.error("--plan requires --index between 1 and the number of plan rows")
    run = runs[args.index - 1]
    if args.out is not None:
        run = replace(run, out_root=Path(args.out))
    return _execute_one(run, force=args.force)
```

`_execute_one` returns `0` after printing the current `done -> PATH` format and
returns `1` after printing `FAILED: ERROR`. Scenario mode continues to isolate
failures and execute every remaining run. Add tests for missing index, index 0,
index greater than the plan length, and successful one-row execution with
`execute_run` monkeypatched.

- [ ] **Step 3: Generate SGE scripts**

`cluster/sge.py`:

```python
def write_sge_array_script(
    plan_path: Path,
    script_path: Path,
    n_runs: int,
    *,
    gpu: bool = False,
    slots: int = 1,
    python_cmd: str = "python",
) -> Path:
    if n_runs < 1 or slots < 1:
        raise ValueError("n_runs and slots must be positive")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/bin/bash",
        "#$ -cwd",
        "#$ -V",
        f"#$ -t 1-{n_runs}",
        f"#$ -pe smp {slots}",
    ]
    if gpu:
        lines.append("#$ -l gpu=true")
    lines.extend([
        "set -euo pipefail",
        f'exec {shlex.quote(python_cmd)} -m krl_studies.run --plan {shlex.quote(str(plan_path))} --index "$SGE_TASK_ID"',
        "",
    ])
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)
    return script_path
```

Use an absolute `plan_path` when the plan CLI calls this function so an SGE
task is independent of the submitter's current directory. Tests assert
`#$ -t 1-N`, `#$ -l gpu=true` only when requested, a quoted plan path, and
executable permissions. Import `shlex` in the implementation.

- [ ] **Step 4: Add a container entrypoint**

`studies/scripts/container_entrypoint.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python -m pip install -q -e . -e './studies[analysis,dev]'
exec python -m krl_studies.run "$@"
```

Add a `study` service to `studies/docker-compose.yaml`:

```yaml
  study:
    platform: linux/amd64
    image: synerbi/sirf@sha256:643c7955717ac08c6f44c6d3fe2ef064ebb54167f1da68771ed3e6dc07caf58d
    working_dir: /workspace
    volumes:
      - ..:/workspace
    entrypoint: ["bash", "/workspace/studies/scripts/container_entrypoint.sh"]
```

Add the root Makefile target below and include it in `.PHONY`:

```makefile
study-docker-run:
	test -n "$(SCENARIO)"
	docker compose -f studies/docker-compose.yaml run --rm study --scenario "$(SCENARIO)" $(ARGS)
```

The target must run the `study` service, pass the scenario as an argument, and
allow optional `ARGS=--dry-run` or `ARGS=--only SUBSTRING` without embedding
host-specific paths.

- [ ] **Step 5: Test plan creation and Docker command rendering**

Add `krl_studies/plan.py` with a `main(argv=None)` that parses
`--scenario SCENARIO --out PLAN [--sge SCRIPT] [--gpu] [--slots N]`, expands
the scenario, writes the JSONL plan, optionally calls
`write_sge_array_script`, and prints the number of rows. Native tests cover a
JSONL round-trip, malformed header/rows, invalid CLI indices, SGE text, and the
Makefile command. The plan CLI must produce the SGE script used by the campaign
documentation rather than relying on a manually created file.

- [ ] **Step 6: Commit**

```bash
git add studies/krl_studies/runner/plan.py studies/krl_studies/plan.py studies/krl_studies/cluster studies/scripts/container_entrypoint.sh studies/docker-compose.yaml studies/krl_studies/runner/cli.py Makefile studies/tests/test_run_plan.py studies/tests/test_sge.py
git commit -m "ADD JSONL run plans and SGE Docker execution wrappers"
```

---

### Task 7: Campaign documentation and final verification

**Files:**
- Modify: `studies/README.md`
- Modify: `data/README.md`
- Create: `studies/README_PLAN3_CAMPAIGN.md`

- [ ] **Step 1: Document the complete campaign order**

`studies/README_PLAN3_CAMPAIGN.md` must give the exact commands in this order:

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

The document must explain the distinction between:

- true forward-model PSF,
- reconstruction-side PSF model (`none`, `undersized`, `matched`),
- deconvolution method's assumed FWHM,
- anatomy guidance condition,
- count level and noise realisation.

- [ ] **Step 2: Update README references**

Add Plan 3 commands to `studies/README.md`, document `analysis` optional
dependencies (`uv pip install -e './studies[analysis,dev]'`), and add the
`data/brainweb/subject_<id>/mr_t2.nii.gz`, `lesion_masks.npz`, and
`lesion_diameters_mm.json` optional/generated outputs to `data/README.md`.
Document that `patient_cohort.yaml` references ignored user-provided files and
is not a data commit.

- [ ] **Step 3: Run the complete verification matrix**

```bash
source .venv/bin/activate
python -m pytest tests -q
python -m pytest studies/tests -q
ruff check src/ tests/ studies/
make study-sirf-test
python -m krl_studies.run --scenario studies/scenarios/spheres_mismatch.yaml --dry-run
python -m krl_studies.report --help
git ls-files data/patients data/brainweb results plans
git status --short
```

Expected: library tests pass unchanged, native study tests pass with only
environment-marked skips, container SIRF tests pass, mismatch dry-run prints a
deterministic run count, report CLI help exits 0, and no patient data or result
files are tracked. `git ls-files` must print no paths for patient, BrainWeb,
result, or generated plan directories.

- [ ] **Step 4: Commit**

```bash
git add studies/README.md data/README.md studies/README_PLAN3_CAMPAIGN.md
git commit -m "ADD Plan 3 campaign documentation and final verification"
```

---

## Explicit Scope Decisions

- Metric-level replicate variance is reported; voxelwise variance requires
  storing every selected reconstruction and is not silently inferred.
- Publication figures consume canonical aggregated tables, never raw run
  directory glob patterns.
- Patient rows without ground truth are retained for qualitative/objective/BV
  analysis but excluded from oracle-NRMSE tables.
- The Docker image is pinned by digest and platform; native macOS remains a
  development environment only.
- SGE array tasks execute one serialized RunSpec each, so a failed task can be
  resubmitted without rerunning completed siblings.
- No patient-derived data, BrainWeb downloads, result images, or cluster plan
  outputs are committed.

## Final Acceptance Criteria

1. A completed results root can be converted into deterministic CSV tables in a
   clean directory with one command.
2. The same aggregate directory generates all required figures and LaTeX
   tables without absolute paths or manual parameter edits.
3. The mismatch scenarios explicitly cross recon-side PSF condition,
   deconvolution FWHM, guidance perturbation, beta, counts, and realisation.
4. A large plan can be submitted as an SGE array and each task is independently
   resumable.
5. The container and native test suites remain green, and git cannot stage
   `data/patients/**`.
