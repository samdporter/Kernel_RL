# Study Framework Core Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `krl-studies` experiment framework core: config-driven, resumable benchmark runner comparing RL/KRL/HKRL/DTV/iterative-Yang/GTM/baselines on the committed spheres phantom and user-placed patient data, producing per-run manifests and metric curves.

**Architecture:** New sibling installable package `studies/` (distribution `krl-studies`) importing the existing `krl` CIL plugin. Layers: `config` (YAML scenarios → expanded run specs), `datasets` (spheres/lesions/patients), `methods` (uniform per-iterate generators wrapping `krl` algorithms + new iterative Yang + PETPVC wrapper), `metrics` (NRMSE/CRC/background variability computed per iteration), `runner` (resumable execution with manifests + completion markers). Simulation via SIRF arrives in Plan 2 as additional input variants; analysis/reporting in Plan 3.

**Tech Stack:** Python 3.10+, CIL + `krl` (existing), numpy/scipy, nibabel, pandas, PyYAML, matplotlib, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-23-krl-study-framework-design.md`
**Spec amendment:** spec §3 says `[studies]` extra of the main package; setuptools cannot map two source roots in one distribution, so the framework ships as a sibling project `studies/pyproject.toml` (name `krl-studies`) instead. Behaviour otherwise identical.

**Working directory:** repo root `/Users/samd-work/Projects/Kernel_RL`, branch `study-framework` (already checked out).

**Environment facts the implementer must know:**

- `krl` public API (from `src/krl/__init__.py`): `RichardsonLucy(initial_estimate, blurring_operator, observed_data, kernel_operator=None, freeze_iteration=0, epsilon=1e-10)`, `.run(iterations, verbose, callbacks)`, `.x`, `.loss`; `LBFGSBOptimizer(initial_estimate, data_fidelity, prior, options)`, `.run(verbose, iterations, callbacks)`, `.solution`, `.objective`; `create_gaussian_blur(sigma_tuple, geometry, backend="numba")`; `get_kernel_operator(emission_imagedata, backend="auto")` then `.set_parameters(dict)` and `.set_anatomical_image(imagedata)`; `load_image(path)` / `save_image(image, path)` (CIL ImageData ↔ NIfTI, transposes z,y,x internally).
- CIL pieces used by the DTV path (copy of `examples/pipelines/run_deconv.py:462-534`): `fn.KullbackLeibler(b=observed, eta=geometry.allocate(1e-2))`, `fn.OperatorCompositionFunction`, `op.CompositionOperator(DirectionalOperator(grad_ref), grad)`, `GradientOperator(geometry, method="forward", bnd_cond="Neumann")`, `fn.SmoothMixedL21Norm(epsilon=observed.max()*1e-2)`, `LBFGSBOptions(max_linesearch=..., ftol=..., gtol=..., enforce_non_negativity=True)`.
- CIL `Callback` base: subclass, implement `__call__(self, algorithm)`; `algorithm.iteration`, `algorithm.solution`.
- Arrays are `(z, y, x)` inside CIL ImageData; NIfTI transposition handled by `krl.utils`.
- macOS dev box: numba CPU backend only (torch+CIL OpenMP conflict). Never require CUDA in these tasks.
- Existing pytest config (`pyproject.toml`) sets `testpaths = ["tests"]`; library tests are untouched. Studies tests live in `studies/tests` and run via their own make target.
- Run all commands from repo root. Activate the venv first: `source .venv/bin/activate`.

---

## File map (locked-in responsibilities)

```
studies/
  pyproject.toml                  # krl-studies distribution, deps, pytest+ruff config
  README.md                       # env setup, usage (Task 16)
  scenarios/
    spheres_core.yaml             # Task 15
    patient_mk_h001.yaml          # Task 15
  krl_studies/
    __init__.py                   # version only
    _compat.py                    # nothing needed yet (omit if unused)
    config.py                     # Scenario/RunSpec dataclasses, YAML load, sweep expansion, slugs
    datasets/
      __init__.py                 # re-export loaders
      lesions.py                  # fixed standard tumour set + sphere masks + placement
      spheres.py                  # spheres dataset loading + quick numpy gaussian+poisson sim
      patients.py                 # patient cohort adapter (YAML-declared NIfTI pairs)
    metrics/
      __init__.py
      nrmse.py
      rois.py                     # lesion ROI derivation + background VOI selection
      recovery.py                 # CRC + background variability
      curves.py                   # MetricRow helpers → pandas DataFrame + CSV
    methods/
      __init__.py                 # METHOD_REGISTRY
      base.py                     # Iterate dataclass + streaming contract
      richardson_lucy.py          # RL / KRL / HKRL wrappers
      dtv.py                      # MAP-RL/DTV wrapper
      iterative_yang.py           # in-house iY
      baselines.py                # Gaussian post-smoothing
      petpvc.py                   # GTM via PETPVC subprocess
    runner/
      __init__.py
      expand.py                   # scenario → list[RunSpec]
      execute.py                  # one run: manifest, streaming metrics, artifacts, marker
      cli.py                      # python -m krl_studies.run …
    __main__.py                   # delegates to runner.cli main
  tests/
    conftest.py
    test_config.py
    test_lesions.py
    test_spheres_dataset.py
    test_patients_dataset.py
    test_metrics_nrmse_rois.py
    test_metrics_recovery.py
    test_methods_richardson_lucy.py
    test_methods_dtv.py
    test_iterative_yang.py
    test_baselines_petpvc.py
    test_runner.py
data/
  README.md                       # Task 2
  spheres/*.nii.gz                # Task 2 (committed)
  patients/MK-H001/               # NOT committed (gitignored); user drops files
results/                          # gitignored output
```

---

### Task 1: Package scaffold + tooling

**Files:**
- Create: `studies/pyproject.toml`
- Create: `studies/krl_studies/__init__.py` (+ empty subpackage `__init__.py`s)
- Create: `studies/tests/conftest.py`
- Modify: `.gitignore`
- Modify: `Makefile`

- [ ] **Step 1: Create package skeleton**

```bash
mkdir -p studies/scenarios studies/krl_studies/{datasets,metrics,methods,runner} studies/tests
```

`studies/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "krl-studies"
version = "0.1.0"
description = "Benchmark study framework for krl PET deconvolution methods"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.23",
    "scipy>=1.7",
    "nibabel>=3.0",
    "pandas>=1.5",
    "pyyaml>=6.0",
    "matplotlib>=3.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.4",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["krl_studies*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-q --tb=short"

[tool.ruff]
line-length = 120
target-version = "py310"
```

`studies/krl_studies/__init__.py`:

```python
"""krl-studies: benchmark study framework for krl PET deconvolution."""

__version__ = "0.1.0"
```

Empty inits (repeat for each):

```bash
touch studies/krl_studies/{datasets,metrics,methods,runner}/__init__.py
```

`studies/tests/conftest.py`:

```python
import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


def write_test_nifti(path, array_zyx, voxel_mm=(1.0, 1.0, 1.0)):
    """Helper used by several test modules; transposes (z,y,x)->(x,y,z)."""
    import nibabel as nib

    data_xyz = np.transpose(array_zyx.astype(np.float32), (2, 1, 0))
    affine = np.diag([voxel_mm[2], voxel_mm[1], voxel_mm[0], 1.0])
    nib.save(nib.Nifti1Image(data_xyz, affine), str(path))
```

- [ ] **Step 2: Wire .gitignore and Makefile**

Append to `.gitignore`:

```gitignore
# study framework runtime data
data/patients/
data/brainweb/
results/
```

Append to `Makefile` (and add targets to `.PHONY` line):

```makefile
study-install:
	uv pip install -e "./studies[dev]"

study-test:
	python -m pytest studies/tests

study-lint:
	ruff check src/ tests/ studies/

STUDY_TARGETS = study-install study-test study-lint
```

Also change `.PHONY` first line to:

```makefile
.PHONY: help install test test-cov lint format gpu-test build study-install study-test study-lint
```

(Remove the stray `STUDY_TARGETS =` line — it is not valid make syntax; only the three targets plus `.PHONY` addition belong.)

- [ ] **Step 3: Install and verify imports**

```bash
source .venv/bin/activate && uv pip install -e "./studies[dev]" && python -c "import krl_studies; print(krl_studies.__version__)"
```

Expected: prints `0.1.0`.

- [ ] **Step 4: Commit**

```bash
git add studies .gitignore Makefile && git commit -m "ADD krl-studies package scaffold and make targets"
```

---

### Task 2: Commit spheres phantom + data conventions

**Files:**
- Create: `data/README.md`
- Add (binary): `data/spheres/phant_orig.nii`, `data/spheres/phant_mri.nii`, `data/spheres/phant_pet.nii`

The three NIfTI files are in `~/Downloads` (synthetic phantom, cleared for committing; patient data is NOT committed).

- [ ] **Step 1: Copy files and verify**

```bash
mkdir -p data/spheres
cp ~/Downloads/phant_orig.nii ~/Downloads/phant_mri.nii ~/Downloads/phant_pet.nii data/spheres/
source .venv/bin/activate && python - <<'EOF'
import nibabel as nib
for f in ["phant_orig", "phant_mri", "phant_pet"]:
    img = nib.load(f"data/spheres/{f}.nii")
    print(f, img.shape, tuple(round(v, 2) for v in img.header.get_zooms()))
EOF
```

Expected: all `(200, 200, 100)`, voxel `(1.0, 1.0, 1.0)`.

- [ ] **Step 2: Write data/README.md**

```markdown
# Data directory

Runtime data for `krl-studies`. Most content is NOT committed.

## Layout

- `spheres/` — COMMITTED synthetic sphere phantom (pure simulation, no patient
  data): `phant_orig.nii` (ground truth), `phant_mri.nii` (anatomical
  guidance), `phant_pet.nii` (reference blurred+noisy emission used by the
  MIC2025 experiments; regenerable via `datasets.spheres.quick_sim`).
  Geometry: 200×200×100, 1 mm isotropic.
- `patients/<subject_id>/` — NOT committed. Place one directory per subject:
  - `PET.nii.gz` — OSEM reconstruction to deconvolve
  - `T1.nii.gz` — co-registered anatomical guidance
  - optional `ROIs.nii.gz` — label image for ROI analyses
  Reference subject: `MK-H001` (181×217×181 MNI, 1 mm; original filenames
  `MK-H001_PET_MNI.nii`, `MK-H001_T1_MNI.nii` — rename to the convention
  above when copying).
- `brainweb/` — generated by BrainWeb preparation (Plan 2); never committed.

## Rules

Never commit anything under `patients/` or `brainweb/`. Adding a future
patient = copying files into `patients/<id>/` + adding an entry to a scenario
YAML. No code changes.
```

- [ ] **Step 3: Commit (data + README together)**

```bash
git add data/README.md data/spheres && git commit -m "ADD synthetic spheres phantom data and data directory conventions"
```

---

### Task 3: `config` — scenario loading + sweep expansion

**Files:**
- Create: `studies/krl_studies/config.py`
- Test: `studies/tests/test_config.py`

- [ ] **Step 1: Write failing tests**

`studies/tests/test_config.py`:

```python
import pytest

from krl_studies.config import RunSpec, expand_scenario, load_scenario_dict

SCENARIO = {
    "study": "spheres",
    "dataset": {"kind": "spheres", "root": "data/spheres"},
    "inputs": [{"kind": "preblurred"}, {"kind": "quick_sim", "params": {"fwhm_mm": [5.0], "counts": [1e5], "realisation": [0, 1]}}],
    "methods": [
        {"name": "post_smoothing", "params": {"sigma_mm": [2.0]}},
        {"name": "rl", "params": {"fwhm_mm": 5.0, "iterations": 10}},
    ],
    "output": "results/test",
}


def test_load_scenario_returns_defaults():
    sc = load_scenario_dict(SCENARIO)
    assert sc.study == "spheres"
    assert len(sc.inputs) == 2
    assert sc.methods[1].name == "rl"


def test_expand_produces_cartesian_product_with_slugs():
    sc = load_scenario_dict(SCENARIO)
    runs = expand_scenario(sc)
    kinds = {(r.input_kind, r.method_name) for r in runs}
    assert ("quick_sim", "rl") in kinds
    # quick_sim(2 realisations) x post_smoothing(1) + quick_sim x rl(1) + preblurred(2 methods)
    assert len(runs) == 2 + 2 + 2


def test_expand_respects_scalar_and_grid_params():
    sc = load_scenario_dict(SCENARIO)
    runs = expand_scenario(sc)
    rl_quick = [r for r in runs if r.method_name == "rl" and r.input_kind == "quick_sim"]
    assert all(r.method_params["fwhm_mm"] == 5.0 for r in rl_quick)
    assert {r.input_params["realisation"] for r in rl_quick} == {0, 1}


def test_slug_is_filesystem_safe_and_deterministic():
    sc = load_scenario_dict(SCENARIO)
    runs = expand_scenario(sc)
    slugs = [r.run_id for r in runs]
    assert len(slugs) == len(set(slugs))
    assert all(" " not in s and ".." not in s for s in slugs)
    again = expand_scenario(load_scenario_dict(SCENARIO))
    assert [r.run_id for r in again] == slugs


def test_missing_required_keys_raise():
    with pytest.raises(KeyError):
        load_scenario_dict({"study": "spheres"})
```

- [ ] **Step 2: Run to verify failure**

```bash
source .venv/bin/activate && python -m pytest studies/tests/test_config.py -v
```

Expected: FAIL, `ModuleNotFoundError: krl_studies.config`.

- [ ] **Step 3: Implement**

`studies/krl_studies/config.py`:

```python
"""Scenario configuration: YAML/dict parsing and sweep expansion."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_KEYS = ("study", "dataset", "inputs", "methods", "output")


@dataclass(frozen=True)
class InputSpec:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    study: str
    dataset: dict[str, Any]
    inputs: tuple[InputSpec, ...]
    methods: tuple[MethodSpec, ...]
    output: Path
    raw: dict[str, Any]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    scenario_name: str
    study: str
    dataset: dict[str, Any]
    input_kind: str
    input_params: dict[str, Any]
    method_name: str
    method_params: dict[str, Any]


def load_scenario_dict(raw: dict[str, Any]) -> Scenario:
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise KeyError(f"Scenario missing required keys: {missing}")
    inputs = tuple(InputSpec(kind=i["kind"], params=i.get("params", {})) for i in raw["inputs"])
    methods = tuple(MethodSpec(name=m["name"], params=m.get("params", {})) for m in raw["methods"])
    return Scenario(
        study=str(raw["study"]),
        dataset=dict(raw["dataset"]),
        inputs=inputs,
        methods=methods,
        output=Path(raw["output"]),
        raw=raw,
    )


def load_scenario(path: str | Path) -> Scenario:
    with open(path) as f:
        return load_scenario_dict(yaml.safe_load(f))


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.3g}".replace(".", "p")
    return str(value)


def _grid(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand scalar/list parameter values into the cartesian product."""
    keys = sorted(params)
    values = [v if isinstance(v, list) else [v] for v in (params[k] for k in keys)]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _input_slug(kind: str, params: dict[str, Any]) -> str:
    parts = [kind] + [f"{k}-{_format_value(params[k])}" for k in sorted(params)]
    return "_".join(parts)


def expand_scenario(scenario: Scenario) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for inp in scenario.inputs:
        for input_params in _grid(inp.params):
            input_slug = _input_slug(inp.kind, input_params)
            for method in scenario.methods:
                for method_params in _grid(method.params):
                    parts = [scenario.study, input_slug, method.name] + [
                        f"{k}-{_format_value(method_params[k])}" for k in sorted(method_params)
                    ]
                    run_id = "__".join(parts)
                    runs.append(
                        RunSpec(
                            run_id=run_id,
                            scenario_name=scenario.output.name,
                            study=scenario.study,
                            dataset=scenario.dataset,
                            input_kind=inp.kind,
                            input_params=input_params,
                            method_name=method.name,
                            method_params=method_params,
                        )
                    )
    return runs
```

- [ ] **Step 4: Run tests to green**

```bash
python -m pytest studies/tests/test_config.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add studies/krl_studies/config.py studies/tests/test_config.py && git commit -m "ADD scenario config parsing and sweep expansion"
```

---

### Task 4: `datasets.lesions` — fixed standard tumour set

**Files:**
- Create: `studies/krl_studies/datasets/lesions.py`
- Test: `studies/tests/test_lesions.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np

from krl_studies.datasets.lesions import (
    DEFAULT_TUMOUR_DIAMETERS_MM,
    default_tumour_specs,
    place_tumours,
    sphere_mask,
)


def test_default_specs_cover_four_sizes_at_distinct_positions():
    specs = default_tumour_specs(shape=(100, 200, 200), voxel_mm=(1.0, 1.0, 1.0))
    assert len(specs) == len(DEFAULT_TUMOUR_DIAMETERS_MM) == 4
    centres = [s["centre_zyx"] for s in specs]
    assert len({tuple(np.round(c, 2)) for c in centres}) == 4
    diameters = sorted(2 * s["radius_mm"] for s in specs)
    assert diameters == sorted(DEFAULT_TUMOUR_DIAMETERS_MM)


def test_sphere_mask_volume_close_to_analytic(rng):
    shape = (60, 60, 60)
    mask = sphere_mask(shape, centre_zyx=(30.0, 30.0, 30.0), radius_vox=8.0)
    analytic = 4.0 / 3.0 * np.pi * 8.0**3
    assert abs(mask.sum() - analytic) / analytic < 0.05


def test_place_tumours_multiplies_only_inside_masks():
    pet = np.full((80, 80, 80), 1.0, dtype=np.float32)
    specs = [{"centre_zyx": (40.0, 40.0, 40.0), "radius_mm": 6.0}]
    with_lesions, masks = place_tumours(pet, specs, contrast=4.0, voxel_mm=(1.0,) * 3)
    assert len(masks) == 1
    m = masks[0]
    assert np.allclose(with_lesions[m], 4.0)
    assert np.allclose(with_lesions[~m], 1.0)
    assert not np.allclose(pet, with_lesions)


def test_place_tumours_does_not_modify_input():
    pet = np.zeros((30, 30, 30), dtype=np.float32)
    place_tumours(pet, [{"centre_zyx": (15.0, 15.0, 15.0), "radius_mm": 3.0}], 2.0, (1.0,) * 3)
    assert pet.max() == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest studies/tests/test_lesions.py -v
```

Expected: FAIL, ModuleNotFoundError.

- [ ] **Step 3: Implement**

`studies/krl_studies/datasets/lesions.py`:

```python
"""Fixed standard tumour set for simulated PET studies.

Tumour positions are expressed as fractional offsets from the volume centre so
the same layout transfers across subjects and voxel sizes. Positions were
chosen (fraction of dimension) to sit in plausible GM / WM / background zones
of BrainWeb brains; validation against tissue labels happens at dataset level.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_TUMOUR_DIAMETERS_MM = (8.0, 12.0, 16.0, 24.0)
DEFAULT_CONTRAST = 4.0

# (dz, dy, dx) fraction-of-dimension offsets from centre, one per diameter
# (smallest tumour most central).
_POSITION_FRACTIONS = (
    (0.00, 0.00, -0.05),
    (0.12, 0.10, 0.10),
    (-0.10, -0.08, 0.12),
    (0.05, -0.15, -0.10),
)


def default_tumour_specs(
    shape: tuple[int, int, int],
    voxel_mm: tuple[float, float, float],
    diameters_mm: tuple[float, ...] = DEFAULT_TUMOUR_DIAMETERS_MM,
    contrast: float = DEFAULT_CONTRAST,
) -> list[dict[str, Any]]:
    if len(shape) != 3:
        raise ValueError("shape must be 3D (z, y, x)")
    centre = np.array(shape, dtype=float) / 2.0
    extent = np.array(shape, dtype=float) * np.array(voxel_mm, dtype=float)
    specs = []
    for diameter, frac in zip(sorted(diameters_mm), _POSITION_FRACTIONS):
        offset = np.array(frac, dtype=float) * extent
        specs.append(
            {
                "centre_zyx": tuple(centre + offset),
                "radius_mm": diameter / 2.0,
                "contrast": contrast,
            }
        )
    return specs


def sphere_mask(
    shape: tuple[int, int, int],
    centre_zyx: tuple[float, float, float],
    radius_vox: float,
) -> np.ndarray:
    z = np.arange(shape[0], dtype=np.float32)
    y = np.arange(shape[1], dtype=np.float32)
    x = np.arange(shape[2], dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    d2 = (zz - centre_zyx[0]) ** 2 + (yy - centre_zyx[1]) ** 2 + (xx - centre_zyx[2]) ** 2
    return d2 <= radius_vox**2


def place_tumours(
    pet: np.ndarray,
    specs: list[dict[str, Any]],
    contrast: float | None = None,
    voxel_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (pet with tumours, per-tumour boolean masks); input untouched."""
    out = pet.astype(np.float32, copy=True)
    masks = []
    for spec in specs:
        c = spec.get("contrast", contrast)
        if c is None:
            raise ValueError("each spec or the call must provide contrast")
        radius_vox = float(spec["radius_mm"]) / np.asarray(voxel_mm, dtype=float)
        if not np.ptp(radius_vox) < 1e-6:
            raise NotImplementedError("anisotropic lesion radii not supported")
        mask = sphere_mask(pet.shape, spec["centre_zyx"], float(radius_vox[0]))
        out[mask] *= float(c)
        masks.append(mask)
    return out, masks
```

- [ ] **Step 4: Run tests to green**

```bash
python -m pytest studies/tests/test_lesions.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add studies/krl_studies/datasets/lesions.py studies/tests/test_lesions.py && git commit -m "ADD fixed standard tumour set placement"
```

---

### Task 5: `datasets.spheres` — phantom loading + quick numpy simulation

**Files:**
- Create: `studies/krl_studies/datasets/spheres.py`
- Modify: `studies/krl_studies/datasets/__init__.py`
- Test: `studies/tests/test_spheres_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest

from krl_studies.datasets.spheres import SphereDataset, quick_sim


@pytest.fixture(scope="module")
def spheres_dir(tmp_path_factory, rng):
    """Tiny stand-in phantom with the three canonical files."""
    from conftest import write_test_nifti

    d = tmp_path_factory.mktemp("spheres")
    gt = np.zeros((40, 60, 60), dtype=np.float32)
    gt[10:30, 25:35, 25:35] = 8.0
    gt += 1.0
    mr = np.full((40, 60, 60), 0.5, dtype=np.float32)
    write_test_nifti(d / "phant_orig.nii", gt)
    write_test_nifti(d / "phant_mri.nii", mr)
    write_test_nifti(d / "phant_pet.nii", gt * 0.9)
    return d


def test_dataset_loads_images_and_geometry(spheres_dir):
    ds = SphereDataset(root=spheres_dir)
    assert ds.ground_truth.shape == (40, 60, 60)
    assert ds.guidance.shape == (40, 60, 60)
    assert ds.reference_pet.shape == (40, 60, 60)
    assert ds.voxel_mm == pytest.approx((1.0, 1.0, 1.0))


def test_dataset_requires_files(tmp_path):
    from pathlib import Path

    with pytest.raises(FileNotFoundError, match="phant_orig"):
        SphereDataset(root=tmp_path)


def test_quick_sim_is_deterministic_and_adds_noise(spheres_dir):
    ds = SphereDataset(root=spheres_dir)
    a = quick_sim(ds.ground_truth, fwhm_mm=3.0, counts=1e4, realisation=0, voxel_mm=ds.voxel_mm)
    b = quick_sim(ds.ground_truth, fwhm_mm=3.0, counts=1e4, realisation=0, voxel_mm=ds.voxel_mm)
    c = quick_sim(ds.ground_truth, fwhm_mm=3.0, counts=1e4, realisation=7, voxel_mm=ds.voxel_mm)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.min() >= 0.0
    # blur must reduce peak compared to sharp gt
    assert a.max() <= ds.ground_truth.max() + 1e-6
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest studies/tests/test_spheres_dataset.py -v
```

Expected: FAIL, ModuleNotFoundError.

- [ ] **Step 3: Implement**

`studies/krl_studies/datasets/spheres.py`:

```python
"""Synthetic spheres phantom dataset (files committed under data/spheres)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

_REQUIRED = {
    "ground_truth": "phant_orig.nii",
    "guidance": "phant_mri.nii",
    "reference_pet": "phant_pet.nii",
}

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass
class SphereDataset:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        missing = [name for name, fname in _REQUIRED.items() if not (self.root / fname).exists()]
        if missing:
            raise FileNotFoundError(
                f"{self.root} is missing spheres files for: {missing}; "
                "expected phant_orig.nii, phant_mri.nii, phant_pet.nii"
            )

    def _load(self, fname: str) -> np.ndarray:
        nii = nib.load(str(self.root / fname))
        return np.transpose(nii.get_fdata().astype(np.float32), (2, 1, 0))

    @property
    def ground_truth(self) -> np.ndarray:
        return self._load(_REQUIRED["ground_truth"])

    @property
    def guidance(self) -> np.ndarray:
        return self._load(_REQUIRED["guidance"])

    @property
    def reference_pet(self) -> np.ndarray:
        return self._load(_REQUIRED["reference_pet"])

    @property
    def voxel_mm(self) -> tuple[float, float, float]:
        nii = nib.load(str(self.root / _REQUIRED["ground_truth"]))
        sizes = nib.affines.voxel_sizes(nii.affine)
        return (float(sizes[2]), float(sizes[1]), float(sizes[0]))  # (z, y, x)


def quick_sim(
    gt: np.ndarray,
    fwhm_mm: float,
    counts: float,
    realisation: int,
    voxel_mm: tuple[float, float, float],
    seed: int = 1337,
) -> np.ndarray:
    """Deterministic image-space surrogate for the SIRF simulation (Plan 2).

    Gaussian blur with the given FWHM followed by Poisson noise scaled to
    `counts` total expected counts. Same (fwhm, counts, realisation, seed) =>
    identical output.
    """
    sigma_vox = [(fwhm_mm * FWHM_TO_SIGMA) / v for v in voxel_mm]
    blurred = gaussian_filter(gt.astype(np.float64), sigma=sigma_vox, mode="constant", cval=0.0)
    scale = counts / max(float(blurred.sum()), 1e-12)
    lam = np.clip(blurred * scale, 0.0, None)
    rng = np.random.default_rng(seed + int(realisation) * 7919)
    noisy = rng.poisson(lam).astype(np.float32) / scale
    return noisy
```

Modify `studies/krl_studies/datasets/__init__.py`:

```python
from krl_studies.datasets.spheres import SphereDataset, quick_sim

__all__ = ["SphereDataset", "quick_sim"]
```

Note: `from conftest import write_test_nifti` in the test works because pytest adds the test's rootdir to `sys.path` with the default importmode; if the import fails in your environment, replace it with a local copy of the helper.

- [ ] **Step 4: Run tests to green**

```bash
python -m pytest studies/tests/test_spheres_dataset.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add studies/krl_studies/datasets studies/tests/test_spheres_dataset.py && git commit -m "ADD spheres dataset loader and deterministic quick simulation"
```

---

### Task 6: `datasets.patients` — cohort adapter

**Files:**
- Create: `studies/krl_studies/datasets/patients.py`
- Modify: `studies/krl_studies/datasets/__init__.py`
- Test: `studies/tests/test_patients_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest

from krl_studies.datasets.patients import PatientDataset, discover_subjects


def _mk(root, sid, with_roi=False):
    from conftest import write_test_nifti

    d = root / sid
    d.mkdir(parents=True)
    arr = np.random.default_rng(abs(hash(sid)) % 2**31).random((24, 28, 28)).astype("float32")
    write_test_nifti(d / "PET.nii.gz", arr)
    write_test_nifti(d / "T1.nii.gz", arr * 2)
    if with_roi:
        write_test_nifti(d / "ROIs.nii.gz", (arr > 0.5).astype("float32"))


def test_discover_finds_only_complete_subjects(tmp_path):
    _mk(tmp_path, "A")
    _mk(tmp_path, "B", with_roi=True)
    (tmp_path / "incomplete").mkdir()
    found = discover_subjects(tmp_path)
    assert set(found) == {"A", "B"}
    assert set(found["B"].keys()) == {"PET", "T1", "ROIs"}


def test_patient_dataset_loads_optional_roi(tmp_path):
    _mk(tmp_path, "MK-H001", with_roi=True)
    ds = PatientDataset(subject_id="MK-H001", root=tmp_path)
    assert ds.pet.shape == (24, 28, 28)
    assert ds.guidance.shape == (24, 28, 28)
    assert ds.rois is not None
    assert ds.ground_truth is None


def test_missing_subject_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nobody"):
        PatientDataset(subject_id="nobody", root=tmp_path)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest studies/tests/test_patients_dataset.py -v
```

Expected: FAIL, ModuleNotFoundError.

- [ ] **Step 3: Implement**

`studies/krl_studies/datasets/patients.py`:

```python
"""Patient cohort adapter.

Convention (see data/README.md): one directory per subject under
`data/patients/` containing `PET.nii.gz` and `T1.nii.gz`, optionally
`ROIs.nii.gz`. Ground truth does not exist for patients by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

_FILES = {"PET": "PET.nii.gz", "T1": "T1.nii.gz", "ROIs": "ROIs.nii.gz"}


def _load(path: Path) -> np.ndarray:
    return np.transpose(nib.load(str(path)).get_fdata().astype(np.float32), (2, 1, 0))


def discover_subjects(patients_root: Path) -> dict[str, dict[str, Path]]:
    """Map subject id -> present files; only subjects with PET+T1 qualify."""
    patients_root = Path(patients_root)
    found: dict[str, dict[str, Path]] = {}
    if not patients_root.exists():
        return found
    for d in sorted(patients_root.iterdir()):
        if not d.is_dir():
            continue
        present = {key: d / fname for key, fname in _FILES.items() if (d / fname).exists()}
        if {"PET", "T1"} <= present.keys():
            found[d.name] = present
    return found


@dataclass
class PatientDataset:
    subject_id: str
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.dir = self.root / self.subject_id
        if not (self.dir / _FILES["PET"]).exists() or not (self.dir / _FILES["T1"]).exists():
            raise FileNotFoundError(
                f"{self.dir} must contain PET.nii.gz and T1.nii.gz (see data/README.md)"
            )
        self.pet = _load(self.dir / _FILES["PET"])
        self.guidance = _load(self.dir / _FILES["T1"])
        roi_path = self.dir / _FILES["ROIs"]
        self.rois = _load(roi_path) if roi_path.exists() else None
        self.ground_truth = None
```

Modify `studies/krl_studies/datasets/__init__.py`:

```python
from krl_studies.datasets.patients import PatientDataset, discover_subjects
from krl_studies.datasets.spheres import SphereDataset, quick_sim

__all__ = ["PatientDataset", "SphereDataset", "discover_subjects", "quick_sim"]
```

- [ ] **Step 4: Run tests to green**

```bash
python -m pytest studies/tests/test_patients_dataset.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Copy MK-H001 into place (uncommitted) and sanity-check**

```bash
mkdir -p data/patients/MK-H001
cp ~/Downloads/MK-H001_PET_MNI.nii data/patients/MK-H001/PET.nii.gz
cp ~/Downloads/MK-H001_T1_MNI.nii data/patients/MK-H001/T1.nii.gz
source .venv/bin/activate && python -c "
from krl_studies.datasets.patients import PatientDataset
ds = PatientDataset('MK-H001', 'data/patients')
print(ds.pet.shape, ds.pet.max())
"
```

Expected: `(181, 217, 181)` and a finite positive max. Confirm `git status` shows nothing under `data/patients/`.

- [ ] **Step 6: Commit**

```bash
git add studies/krl_studies/datasets studies/tests/test_patients_dataset.py && git commit -m "ADD patient cohort adapter with optional ROI support"
```

---

### Task 7: Metrics — NRMSE + ROI derivation

**Files:**
- Create: `studies/krl_studies/metrics/nrmse.py`
- Create: `studies/krl_studies/metrics/rois.py`
- Modify: `studies/krl_studies/metrics/__init__.py`
- Test: `studies/tests/test_metrics_nrmse_rois.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np

from krl_studies.metrics.nrmse import nrmse
from krl_studies.metrics.rois import background_vois, derive_lesion_rois


def test_nrmse_matches_definition():
    gt = np.full((10, 10, 10), 2.0, dtype=np.float32)
    img = gt + 0.5
    expected = np.sqrt(np.mean((0.5) ** 2)) / gt.max()
    assert float(nrmse(img, gt)) == float(expected)


def test_nrmse_zero_for_identical():
    gt = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
    assert nrmse(gt.copy(), gt) == 0.0


def _blob_gt():
    gt = np.zeros((50, 50, 50), dtype=np.float32)
    gt[25, 25, 25] = 10.0
    gt[10:14, 10:14, 10:14] = 6.0
    return gt


def test_derive_lesion_rois_finds_components_above_threshold():
    from scipy.ndimage import gaussian_filter

    gt = gaussian_filter(_blob_gt(), sigma=1.2)
    rois = derive_lesion_rois(gt, min_volume_vox=5)
    assert len(rois) >= 2
    volumes = sorted(int(r.sum()) for r in rois)
    assert volumes[0] >= 5


def test_background_vois_are_disjoint_from_lesions_and_deterministic():
    from scipy.ndimage import gaussian_filter

    gt = gaussian_filter(_blob_gt(), sigma=1.2)
    lesions = derive_lesion_rois(gt, min_volume_vox=5)
    exclusion = np.logical_or.reduce(lesions)
    v1 = background_vois(gt.shape, exclude_mask=exclusion, n_vois=4, radius_vox=3, seed=1)
    v2 = background_vois(gt.shape, exclude_mask=exclusion, n_vois=4, radius_vox=3, seed=1)
    assert len(v1) == 4
    assert all(np.array_equal(a, b) for a, b in zip(v1, v2))
    for voi in v1:
        assert not (voi & exclusion).any()
    union = np.logical_or.reduce(v1)
    assert union.sum() > 0
```


- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest studies/tests/test_metrics_nrmse_rois.py -v
```

Expected: FAIL, ModuleNotFoundError.

- [ ] **Step 3: Implement**

`studies/krl_studies/metrics/nrmse.py`:

```python
"""Normalised root-mean-square error, matching krl.callbacks.NRMSECallback."""

from __future__ import annotations

import numpy as np


def nrmse(image: np.ndarray, ground_truth: np.ndarray) -> float:
    diff = np.asarray(image, dtype=np.float64) - np.asarray(ground_truth, dtype=np.float64)
    gt_max = float(np.max(ground_truth))
    if gt_max == 0.0:
        raise ValueError("ground truth max is zero; NRMSE undefined")
    return float(np.sqrt(np.mean(diff**2)) / gt_max)
```

`studies/krl_studies/metrics/rois.py`:

```python
"""ROI construction: lesion ROIs from ground truth, background VOIs."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def derive_lesion_rois(
    ground_truth: np.ndarray,
    threshold_fraction: float = 0.5,
    min_volume_vox: int = 20,
) -> list[np.ndarray]:
    """Connected components above a fraction of GT max, largest first."""
    gt_max = float(np.max(ground_truth))
    mask = ground_truth > threshold_fraction * gt_max
    labels, n = label(mask)
    rois = []
    for i in range(1, n + 1):
        comp = labels == i
        if int(comp.sum()) >= min_volume_vox:
            rois.append(comp)
    rois.sort(key=lambda m: int(m.sum()), reverse=True)
    return rois


def background_vois(
    shape: tuple[int, int, int],
    exclude_mask: np.ndarray,
    n_vois: int = 8,
    radius_vox: float = 5.0,
    margin_vox: int = 5,
    n_candidates: int = 4000,
    seed: int = 2026,
) -> list[np.ndarray]:
    """Pick n well-separated spherical background VOIs avoiding exclude_mask.

    Candidates are sampled uniformly (seeded), filtered by exclusion + margin,
    then greedily selected to maximise pairwise separation.
    """
    from krl_studies.datasets.lesions import sphere_mask

    rng = np.random.default_rng(seed)
    lo = np.array([margin_vox] * 3, dtype=float)
    hi = np.array(shape, dtype=float) - margin_vox
    candidates = rng.uniform(lo, hi, size=(n_candidates, 3))

    allowed = np.ones(shape, dtype=bool)
    allowed[:margin_vox, :, :] = False
    allowed[-margin_vox:, :, :] = False
    allowed[:, :margin_vox, :] = False
    allowed[:, -margin_vox:, :] = False
    allowed[:, :, :margin_vox] = False
    allowed[:, :, -margin_vox:] = False
    allowed &= ~exclude_mask

    chosen: list[tuple[np.ndarray, np.ndarray]] = []  # (centre, mask)
    min_sep = 4.0 * radius_vox
    for c in candidates:
        if len(chosen) >= n_vois:
            break
        if any(float(np.linalg.norm(c - c0)) < min_sep for c0, _ in chosen):
            continue
        m = sphere_mask(shape, tuple(c), radius_vox)
        if allowed[m].all():
            chosen.append((c, m))
    if len(chosen) < n_vois:
        raise ValueError(
            f"could only place {len(chosen)}/{n_vois} background VOIs; "
            "reduce n_vois or radius_vox"
        )
    return [m for _, m in chosen]
```

Modify `studies/krl_studies/metrics/__init__.py`:

```python
from krl_studies.metrics.nrmse import nrmse
from krl_studies.metrics.rois import background_vois, derive_lesion_rois

__all__ = ["background_vois", "derive_lesion_rois", "nrmse"]
```

- [ ] **Step 4: Run tests to green**

```bash
python -m pytest studies/tests/test_metrics_nrmse_rois.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add studies/krl_studies/metrics studies/tests/test_metrics_nrmse_rois.py && git commit -m "ADD NRMSE metric and ROI/VOI construction"
```

---

### Task 8: Metrics — CRC + background variability

**Files:**
- Create: `studies/krl_studies/metrics/recovery.py`
- Create: `studies/krl_studies/metrics/curves.py`
- Modify: `studies/krl_studies/metrics/__init__.py`
- Test: `studies/tests/test_metrics_recovery.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np

from krl_studies.datasets.lesions import sphere_mask
from krl_studies.metrics.recovery import background_variability, crc_percent


def _world():
    gt = np.full((40, 40, 40), 1.0, dtype=np.float32)
    lesion = sphere_mask((40, 40, 40), (20.0, 20.0, 20.0), 4.0)
    gt[lesion] = 5.0
    return gt, lesion


def test_crc_perfect_for_ground_truth():
    gt, lesion = _world()
    vois = [sphere_mask((40, 40, 40), (c, 20.0, 20.0), 3.0) for c in (5.0, 35.0)]
    assert crc_percent(lesion, gt, gt, vois) == 100.0


def test_crc_zero_when_no_recovery():
    gt, lesion = _world()
    flat = np.full_like(gt, 2.0)  # constant image: measured ratio = 1
    vois = [sphere_mask((40, 40, 40), (c, 20.0, 20.0), 3.0) for c in (5.0, 35.0)]
    assert crc_percent(lesion, flat, gt, vois) == 0.0


def test_background_variability_zero_for_constant():
    const = np.full((20, 20, 20), 3.0, dtype=np.float32)
    vois = [sphere_mask((20, 20, 20), (c, 10.0, 10.0), 2.0) for c in (5.0, 10.0, 15.0)]
    assert background_variability(const, vois) == 0.0


def test_background_variability_positive_for_spread(rng):
    img = rng.normal(10.0, 1.0, size=(30, 30, 30)).astype(np.float32)
    vois = [sphere_mask((30, 30, 30), (c, 15.0, 15.0), 3.0) for c in (8.0, 15.0, 22.0)]
    bv = background_variability(img, vois)
    assert bv > 0.0


def test_curves_dataframe_roundtrip(tmp_path):
    import pandas as pd

    from krl_studies.metrics.curves import metrics_to_dataframe, write_metrics_csv

    rows = [
        {"iteration": 1, "nrmse": 0.5, "crc_mm8": 10.0},
        {"iteration": 2, "nrmse": 0.4, "crc_mm8": 20.0},
    ]
    df = metrics_to_dataframe(rows)
    assert list(df.columns) == ["iteration", "nrmse", "crc_mm8"]
    out = tmp_path / "m.csv"
    write_metrics_csv(rows, out)
    assert pd.read_csv(out).shape == (2, 3)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest studies/tests/test_metrics_recovery.py -v
```

Expected: FAIL, ModuleNotFoundError.

- [ ] **Step 3: Implement**

`studies/krl_studies/metrics/recovery.py`:

```python
"""Contrast recovery and noise variability metrics (NEMA-style definitions)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _voi_means(image: np.ndarray, vois: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([float(np.mean(image[m])) for m in vois], dtype=np.float64)


def crc_percent(
    lesion_mask: np.ndarray,
    image: np.ndarray,
    ground_truth: np.ndarray,
    background_vois: Sequence[np.ndarray],
) -> float:
    """CRC = (C_meas/B_meas - 1) / (C_true/B_true - 1) * 100.

    Background levels are measured as the mean over the background VOIs in the
    respective image, so bias common to lesion and background cancels.
    """
    b_meas = float(np.mean(_voi_means(image, background_vois)))
    b_true = float(np.mean(_voi_means(ground_truth, background_vois)))
    c_meas = float(np.mean(image[lesion_mask]))
    c_true = float(np.mean(ground_truth[lesion_mask]))
    denom = (c_true / b_true) - 1.0
    if denom == 0.0 or b_meas == 0.0:
        raise ValueError("degenerate CRC definition (zero contrast or background)")
    return float(100.0 * ((c_meas / b_meas) - 1.0) / denom)


def background_variability(
    image: np.ndarray, vois: Sequence[np.ndarray]
) -> float:
    """Percent coefficient of variation of VOI means (relative to their mean)."""
    means = _voi_means(image, vois)
    overall = float(np.mean(means))
    if overall == 0.0:
        raise ValueError("background mean is zero")
    return float(100.0 * np.std(means, ddof=1) / overall) if len(means) > 1 else 0.0
```

`studies/krl_studies/metrics/curves.py`:

```python
"""Per-iteration metric rows -> DataFrame/CSV."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


def metrics_to_dataframe(rows: Sequence[Mapping]) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).sort_values("iteration").reset_index(drop=True)


def write_metrics_csv(rows: Sequence[Mapping], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_to_dataframe(rows).to_csv(path, index=False)
```

Modify `studies/krl_studies/metrics/__init__.py`:

```python
from krl_studies.metrics.curves import metrics_to_dataframe, write_metrics_csv
from krl_studies.metrics.nrmse import nrmse
from krl_studies.metrics.recovery import background_variability, crc_percent
from krl_studies.metrics.rois import background_vois, derive_lesion_rois

__all__ = [
    "background_variability",
    "background_vois",
    "crc_percent",
    "derive_lesion_rois",
    "metrics_to_dataframe",
    "nrmse",
    "write_metrics_csv",
]
```

- [ ] **Step 4: Run tests to green**

```bash
python -m pytest studies/tests/test_metrics_recovery.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add studies/krl_studies/metrics studies/tests/test_metrics_recovery.py && git commit -m "ADD CRC and background variability metrics with curve IO"
```

---

### Task 9: `methods.base` + RL wrapper

**Files:**
- Create: `studies/krl_studies/methods/base.py`
- Create: `studies/krl_studies/methods/richardson_lucy.py`
- Modify: `studies/krl_studies/methods/__init__.py`
- Test: `studies/tests/test_methods_richardson_lucy.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest

from conftest import write_test_nifti
from krl_studies.datasets.spheres import quick_sim
from krl_studies.methods.base import Iterate
from krl_studies.methods.richardson_lucy import HKRLMethod, KRLMethod, RLMethod


@pytest.fixture(scope="module")
def observed_pair(tmp_path_factory, rng):
    """GT with a hot cube; blurred+noisy observation; MR-ish guidance."""
    from pathlib import Path

    d = tmp_path_factory.mktemp("imgs")
    gt = np.full((32, 40, 40), 1.0, dtype=np.float32)
    gt[12:20, 16:24, 16:24] = 6.0
    obs = quick_sim(gt, fwhm_mm=3.0, counts=5e4, realisation=0, voxel_mm=(1.0, 1.0, 1.0))
    guidance = np.where(gt > 3.0, 2.0, 0.4).astype(np.float32)
    write_test_nifti(d / "gt.nii", gt)
    write_test_nifti(d / "obs.nii", obs)
    write_test_nifti(d / "mr.nii", guidance)
    return Path(d)


def _load(d, name):
    import nibabel as nib

    return np.transpose(nib.load(str(d / name)).get_fdata().astype(np.float32), (2, 1, 0))


def _as_cil(arr):
    from krl.utils import load_nifti_as_imagedata
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        write_test_nifti(f.name, arr)
        return load_nifti_as_imagedata(f.name)


def test_base_iterate_fields():
    it = Iterate(iteration=1, image=np.zeros((2, 2, 2)), objective=None)
    assert it.iteration == 1 and it.image.shape == (2, 2, 2)


def test_rl_yields_stream_of_iterates_improving_nrmse(observed_pair):
    from krl_studies.metrics.nrmse import nrmse

    d = observed_pair
    gt, obs = _load(d, "gt.nii"), _load(d, "obs.nii")
    method = RLMethod()
    iters = list(
        method.run(
            observed=_as_cil(obs),
            guidance=None,
            params={"fwhm_mm": 3.0, "backend": "numba"},
            n_iterations=8,
        )
    )
    assert [it.iteration for it in iters] == list(range(1, 9))
    n_first = nrmse(iters[0].image, gt)
    n_min = min(nrmse(it.image, gt) for it in iters)
    assert n_min < n_first


def test_rl_lazy_generator_not_consumed_until_needed(observed_pair):
    d = observed_pair
    method = RLMethod()
    gen = method.run(observed=_as_cil(_load(d, "obs.nii")), guidance=None,
                     params={"fwhm_mm": 3.0}, n_iterations=3)
    assert hasattr(gen, "__next__")


def test_krl_guidance_changes_result_vs_rl(observed_pair):
    d = observed_pair
    obs, mr = _load(d, "obs.nii"), _load(d, "mr.nii")
    common = {"fwhm_mm": 3.0, "iterations": 5, "backend": "numba"}
    rl = list(RLMethod().run(_as_cil(obs), None, common, 5))
    krl = list(
        KRLMethod().run(
            _as_cil(obs), _as_cil(mr),
            {**common, "sigma_anat": 1.0, "num_neighbours": 5}, 5,
        )
    )
    assert not np.allclose(rl[-1].image, krl[-1].image)


def test_hkrl_freeze_runs(observed_pair):
    d = observed_pair
    obs, mr = _load(d, "obs.nii"), _load(d, "mr.nii")
    hkrl = list(
        HKRLMethod().run(
            _as_cil(obs), _as_cil(mr),
            {"fwhm_mm": 3.0, "sigma_anat": 1.0, "sigma_emission": 1.0,
             "freeze_iteration": 2, "num_neighbours": 5, "backend": "numba"},
            4,
        )
    )
    assert len(hkrl) == 4
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest studies/tests/test_methods_richardson_lucy.py -v
```

Expected: FAIL, ModuleNotFoundError.

- [ ] **Step 3: Implement base + wrappers**

`studies/krl_studies/methods/base.py`:

```python
"""Method contract: every deconvolution method streams per-iteration results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class Iterate:
    iteration: int
    image: np.ndarray  # (z, y, x), emission-domain estimate
    objective: float | None = None


class Method:
    """Subclasses return a lazy iterator of Iterates from `run`."""

    name: str = "method"

    def run(
        self,
        observed: Any,       # CIL ImageData
        guidance: Any | None,  # CIL ImageData or None
        params: dict[str, Any],
        n_iterations: int,
    ) -> Iterator[Iterate]:
        raise NotImplementedError
```

`studies/krl_studies/methods/richardson_lucy.py`:



```python
"""RL / KRL / HKRL wrappers around krl.RichardsonLucy."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from cil.optimisation.utilities.callbacks import Callback

from krl.algorithms.richardson_lucy import RichardsonLucy
from krl.operators.blurring import create_gaussian_blur
from krl.operators.kernel_operator import get_kernel_operator
from krl.utils import get_array

from krl_studies.methods.base import Iterate, Method

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _blur_op(observed, fwhm_mm: float, backend: str = "numba"):
    sigma = tuple(fwhm_mm * FWHM_TO_SIGMA for _ in range(3))
    return create_gaussian_blur(sigma, observed.geometry, backend=backend)


class _Capture(Callback):
    """Record solution (and objective when available) every iteration."""

    def __init__(self, sink: list):
        super().__init__()
        self._sink = sink

    def __call__(self, algorithm) -> None:
        obj = None
        loss = getattr(algorithm, "loss", None)
        if loss:
            try:
                obj = float(loss[-1])
            except (TypeError, ValueError):
                obj = None
        self._sink.append(
            Iterate(
                iteration=int(algorithm.iteration),
                image=get_array(algorithm.solution).astype(np.float32, copy=True),
                objective=obj,
            )
        )


class RLMethod(Method):
    name = "rl"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        blur = _blur_op(observed, float(params["fwhm_mm"]), params.get("backend", "numba"))
        captured: list[Iterate] = []
        algo = RichardsonLucy(
            initial_estimate=observed,
            blurring_operator=blur,
            observed_data=observed,
            epsilon=float(params.get("epsilon", 1e-10)),
            update_objective_interval=1,
        )

        def generate():
            algo.run(iterations=int(n_iterations), verbose=0, callbacks=[_Capture(captured)])
            yield from captured

        return generate()


def _kernel_params(params: dict[str, Any]) -> dict[str, Any]:
    """Whitelist of supported KernelOperator settings."""
    allowed = (
        "num_neighbours", "sigma_anat", "sigma_dist", "sigma_emission",
        "distance_weighting", "normalize_features", "normalize_kernel",
        "use_mask", "mask_k", "recalc_mask", "hybrid",
    )
    return {k: params[k] for k in allowed if k in params}


class _KernelMethod(Method):
    def _run_kernel(self, observed, guidance, params, n_iterations, freeze_iteration):
        blur = _blur_op(observed, float(params["fwhm_mm"]), params.get("backend", "numba"))
        kernel_op = get_kernel_operator(observed, backend=params.get("backend", "numba"))
        kernel_op.set_parameters(_kernel_params(params))
        kernel_op.set_anatomical_image(guidance)
        captured: list[Iterate] = []
        algo = RichardsonLucy(
            initial_estimate=observed,
            blurring_operator=blur,
            observed_data=observed,
            kernel_operator=kernel_op,
            freeze_iteration=int(freeze_iteration),
            epsilon=float(params.get("epsilon", 1e-10)),
            update_objective_interval=1,
        )

        def generate():
            algo.run(iterations=int(n_iterations), verbose=0, callbacks=[_Capture(captured)])
            # KRL operates on a latent image; map iterates to emission domain.
            for it in captured:
                latent = algo.x.geometry.allocate()
                latent.fill(it.image)
                deconv = kernel_op.direct(latent)
                arr = get_array(deconv).astype(np.float32)
                arr[arr < 0] = 0.0
                yield Iterate(it.iteration, arr, it.objective)

        return generate()


class KRLMethod(_KernelMethod):
    name = "krl"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        return self._run_kernel(observed, guidance, params, n_iterations,
                                freeze_iteration=params.get("freeze_iteration", 0))


class HKRLMethod(_KernelMethod):
    name = "hkrl"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        return self._run_kernel(observed, guidance, params, n_iterations,
                                freeze_iteration=params.get("freeze_iteration", 1))
```

Engineer note (Task 9): verify `get_kernel_operator(emission_imagedata, backend=...)` and `kernel_op.set_parameters(dict)` keyword names against `src/krl/operators/kernel_operator.py`; the whitelist mirrors `KernelParameters` fields from `examples/scripts/run_deconv_sweeps.py` (`DEFAULT_KERNEL`). If `set_parameters` rejects any key, raise `ValueError` listing the operator's accepted keys — never silently drop.

- [ ] **Step 4: Run tests to green**

```bash
python -m pytest studies/tests/test_methods_richardson_lucy.py -v
```

Expected: 5 passed (allow ~1–3 min; numba kernels compile once).

- [ ] **Step 5: Commit**

```bash
git add studies/krl_studies/methods studies/tests/test_methods_richardson_lucy.py && git commit -m "ADD RL/KRL/HKRL method wrappers with streaming iterates"
```

---

### Task 10: DTV / MAP-RL wrapper

**Files:**
- Create: `studies/krl_studies/methods/dtv.py`
- Modify: `studies/krl_studies/methods/__init__.py`
- Test: `studies/tests/test_methods_dtv.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pytest

from conftest import write_test_nifti
from krl_studies.methods.dtv import DTVMethod


def _as_cil(arr):
    from krl.utils import load_nifti_as_imagedata
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as f:
        write_test_nifti(f.name, arr)
        return load_nifti_as_imagedata(f.name)


def _load(path):
    import nibabel as nib

    return np.transpose(nib.load(str(path)).get_fdata().astype(np.float32), (2, 1, 0))


def test_dtv_streams_iterates(tmp_path):
    obs = np.full((24, 32, 32), 1.0, dtype=np.float32)
    obs[8:16, 12:20, 12:20] = 4.0
    guidance = np.where(obs > 2.0, 2.0, 0.5).astype(np.float32)
    write_test_nifti(tmp_path / "o.nii", obs)
    write_test_nifti(tmp_path / "g.nii", guidance)

    iters = list(
        DTVMethod().run(
            observed=_as_cil(_load(tmp_path / "o.nii")),
            guidance=_as_cil(_load(tmp_path / "g.nii")),
            params={"alpha": 0.05, "fwhm_mm": 3.0, "lbfgs_max_linesearch": 5,
                    "lbfgs_ftol": 1e-6, "lbfgs_gtol": 1e-6},
            n_iterations=3,
        )
    )
    assert [it.iteration for it in iters] == [1, 2, 3]
    assert all(it.image.min() >= 0 for it in iters)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest studies/tests/test_methods_dtv.py -v
```

Expected: FAIL, ModuleNotFoundError.

- [ ] **Step 3: Implement**

`studies/krl_studies/methods/dtv.py`:

```python
"""MAP-RL with directional TV, following examples/pipelines/run_deconv.py."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import cil.optimisation.functions as fn
import cil.optimisation.operators as op
from cil.optimisation.operators import BlurringOperator, GradientOperator
from cil.optimisation.utilities.callbacks import Callback

from krl.algorithms.lbfgsb import LBFGSBOptions, LBFGSBOptimizer
from krl.operators.directional import DirectionalOperator
from krl.utils import get_array

from krl_studies.methods.base import Iterate, Method
from krl_studies.methods.richardson_lucy import FWHM_TO_SIGMA


class _CaptureSolution(Callback):
    def __init__(self, sink: list):
        super().__init__()
        self._sink = sink

    def __call__(self, algorithm) -> None:
        arr = get_array(algorithm.solution).astype(np.float32, copy=True)
        arr[arr < 0] = 0.0
        obj = None
        objective = getattr(algorithm, "objective", None)
        if objective:
            try:
                obj = float(objective[-1])
            except (TypeError, ValueError, IndexError):
                obj = None
        self._sink.append(Iterate(int(algorithm.iteration), arr, obj))


class DTVMethod(Method):
    name = "dtv"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        fwhm = float(params["fwhm_mm"])
        sigma = tuple(fwhm * FWHM_TO_SIGMA for _ in range(3))
        try:
            from krl.operators.blurring import create_gaussian_blur

            blur = create_gaussian_blur(sigma, observed.geometry, backend="numba")
        except (ImportError, AttributeError):
            voxel = (
                observed.geometry.voxel_size_z,
                observed.geometry.voxel_size_y,
                observed.geometry.voxel_size_x,
            )
            blur = BlurringOperator(_psf_kernel(5, sigma, voxel), observed)

        fidelity = fn.KullbackLeibler(b=observed, eta=observed.geometry.allocate(value=1e-2))
        data_fidelity = fn.OperatorCompositionFunction(fidelity, blur)

        grad = GradientOperator(observed.geometry, method="forward", bnd_cond="Neumann")
        grad_ref = grad.direct(guidance)
        directional = op.CompositionOperator(DirectionalOperator(grad_ref), grad)
        prior_strength = float(params["alpha"])
        prior = prior_strength * fn.OperatorCompositionFunction(
            fn.SmoothMixedL21Norm(epsilon=float(observed.max()) * 1e-2), directional
        )

        options = LBFGSBOptions(
            max_linesearch=int(params.get("lbfgs_max_linesearch", 20)),
            ftol=float(params.get("lbfgs_ftol", 1e-6)),
            gtol=float(params.get("lbfgs_gtol", 1e-6)),
            enforce_non_negativity=True,
        )
        optimizer = LBFGSBOptimizer(
            initial_estimate=observed,
            data_fidelity=data_fidelity,
            prior=prior,
            options=options,
        )
        captured: list[Iterate] = []

        def generate():
            optimizer.run(
                verbose=0,
                iterations=int(n_iterations),
                callbacks=[_CaptureSolution(captured)],
            )
            yield from captured

        return generate()


def _psf_kernel(kernel_size: int, sigma: tuple[float, float, float], voxel: tuple[float, float, float]):
    axes = [np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size) for _ in range(3)]
    sig_vox = [sigma[i] / voxel[i] for i in range(3)]
    gauss = [np.exp(-0.5 * ax**2 / sv**2) for ax, sv in zip(axes, sig_vox)]
    k = (
        np.outer(gauss[0], gauss[1]).reshape(kernel_size, kernel_size, 1)
        * gauss[2].reshape(1, 1, kernel_size)
    )
    return (k / k.sum()).astype(np.float32)
```

Note: the module header shown in Step 3 must not import `cil.optimisation.algorithms` — nothing in this file uses it.

- [ ] **Step 4: Run tests to green**

```bash
python -m pytest studies/tests/test_methods_dtv.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add studies/krl_studies/methods/dtv.py studies/tests/test_methods_dtv.py && git commit -m "ADD MAP-RL directional TV method wrapper"
```

---

### Task 11: Iterative Yang (in-house PVC)

**Files:**
- Create: `studies/krl_studies/methods/iterative_yang.py`
- Modify: `studies/krl_studies/methods/__init__.py`
- Test: `studies/tests/test_iterative_yang.py`

Algorithm (Yang 1996 as reviewed in Erlandsson et al., PMB 2012; PETPVC `iy`):
given observation `y`, regions `R_k`, PSF `H`: initialise `x = y`; repeat
`m_k = mean_{R_k}(x)`; form step image `s` with `s|R_k = m_k`;
`x ← x + damping · mask · (y − H·s)`. Region means converge towards true
compartment uptake.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from krl_studies.methods.iterative_yang import IterativeYangMethod


def _two_box_world():
    """1D-like 3D phantom: two compartments, blurred observation."""
    shape = (8, 40, 40)
    truth = np.full(shape, 2.0, dtype=np.float32)
    truth[:, :, 5:15] = 8.0
    sigma = 2.0
    blurred = gaussian_filter(truth, sigma=sigma)
    return truth, blurred, sigma


def _regions(truth):
    return [truth[:, :, 5:15] > 4.0, truth[:, :, 5:15] <= 4.0]


def test_iy_recovers_region_means_better_than_observed():
    truth, blurred, sigma = _two_box_world()
    regions = _regions(truth)
    true_means = np.array([truth[r].mean() for r in regions])
    init_err = np.abs(np.array([blurred[r].mean() for r in regions]) - true_means).max()

    iters = list(
        IterativeYangMethod().run(
            observed=blurred,
            guidance=None,
            params={
                "region_masks": regions,
                "psf_sigma_vox": (sigma,) * 3,
                "brain_mask": np.ones_like(blurred, dtype=bool),
                "damping": 1.0,
            },
            n_iterations=30,
        )
    )
    final_err = np.abs(np.array([iters[-1].image[r].mean() for r in regions]) - true_means).max()
    assert final_err < init_err
    assert final_err < 0.25 * true_means.max()


def test_iy_streams_requested_iterations():
    truth, blurred, sigma = _two_box_world()
    iters = list(
        IterativeYangMethod().run(
            observed=blurred,
            guidance=None,
            params={
                "region_masks": _regions(truth),
                "psf_sigma_vox": (sigma,) * 3,
                "brain_mask": np.ones_like(blurred, dtype=bool),
            },
            n_iterations=5,
        )
    )
    assert [it.iteration for it in iters] == [1, 2, 3, 4, 5]


def test_iy_requires_regions():
    with pytest.raises(ValueError, match="region_masks"):
        IterativeYangMethod().run(
            observed=np.zeros((4, 4, 4)), guidance=None, params={}, n_iterations=1
        ).__next__()
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest studies/tests/test_iterative_yang.py -v
```

Expected: FAIL, ModuleNotFoundError.

- [ ] **Step 3: Implement**

`studies/krl_studies/methods/iterative_yang.py`:

```python
"""Iterative Yang partial volume correction (Yang 1996; Erlandsson PMB 2012 review).

Piecewise-constant anatomical model: regional means are re-estimated each
iteration and the residual between measured and model-simulated PET is added
back within the brain mask.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from scipy.ndimage import gaussian_filter

from krl_studies.methods.base import Iterate, Method


class IterativeYangMethod(Method):
    name = "iy"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        region_masks = params.get("region_masks")
        if not region_masks:
            raise ValueError("iterative Yang requires region_masks (from segmentation/GT)")
        sigma = tuple(float(s) for s in params["psf_sigma_vox"])
        mask = params.get("brain_mask")
        if mask is None:
            mask = np.ones(observed.shape, dtype=bool)
        damping = float(params.get("damping", 1.0))

        y = np.asarray(observed, dtype=np.float64)
        masked_y = np.where(mask, y, 0.0)
        x = masked_y.copy()

        def step_image(current: np.ndarray) -> np.ndarray:
            s = np.zeros_like(current)
            for m in region_masks:
                s[m] = current[m].mean()
            return s

        for iteration in range(1, int(n_iterations) + 1):
            model = gaussian_filter(step_image(x), sigma=sigma)
            residual = masked_y - model
            x = x + damping * residual * mask
            yield Iterate(iteration=iteration, image=x.astype(np.float32))
```

- [ ] **Step 4: Run tests to green**

```bash
python -m pytest studies/tests/test_iterative_yang.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add studies/krl_studies/methods/iterative_yang.py studies/tests/test_iterative_yang.py && git commit -m "ADD in-house iterative Yang PVC method"
```

---

### Task 12: Baselines + PETPVC GTM wrapper

**Files:**
- Create: `studies/krl_studies/methods/baselines.py`
- Create: `studies/krl_studies/methods/petpvc.py`
- Modify: `studies/krl_studies/methods/__init__.py`
- Test: `studies/tests/test_baselines_petpvc.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from krl_studies.methods.baselines import PostSmoothingMethod
from krl_studies.methods.petpvc import GTMMethod, build_petpvc_cmd


def test_post_smoothing_blurs_and_single_iterate():
    img = np.zeros((16, 20, 20), dtype=np.float32)
    img[8, 10, 10] = 100.0
    iters = list(
        PostSmoothingMethod().run(
            observed=img, guidance=None,
            params={"sigma_mm": 2.0, "voxel_mm": (1.0, 1.0, 1.0)}, n_iterations=1,
        )
    )
    assert len(iters) == 1 and iters[0].iteration == 1
    assert iters[0].image.max() < img.max()


def test_build_petpvc_cmd_shape():
    cmd = build_petpvc_cmd(
        petpvc_bin="petpvc",
        input_path="in.nii", output_path="out.nii",
        mode="GTM", pvc_fwhm=(5.0, 5.0, 5.0), mask_path="mask.nii",
        extra=["--reg", "rois.nii"],
    )
    assert cmd[0] == "petpvc"
    assert "-i" in cmd and "in.nii" in cmd
    assert "-o" in cmd and "out.nii" in cmd
    assert "-p" in cmd and "GTM" in cmd
    assert "-f" in cmd and "5.0" in " ".join(cmd)
    assert "--reg" in cmd


def test_gtm_missing_binary_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="PETPVC"):
        list(GTMMethod().run(
            observed=np.zeros((4, 4, 4)), guidance=None,
            params={"petpvc_bin": "definitely-not-a-binary", "input_path": str(tmp_path / "a.nii"),
                    "output_path": str(tmp_path / "b.nii")},
            n_iterations=1,
        ))
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest studies/tests/test_baselines_petpvc.py -v
```

Expected: FAIL, ModuleNotFoundError.

- [ ] **Step 3: Implement**

`studies/krl_studies/methods/baselines.py`:

```python
"""Simple comparison baselines."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from scipy.ndimage import gaussian_filter

from krl_studies.methods.base import Iterate, Method

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


class PostSmoothingMethod(Method):
    name = "post_smoothing"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        if n_iterations != 1:
            raise ValueError("post-smoothing is single-step; use n_iterations=1")
        voxel = tuple(float(v) for v in params.get("voxel_mm", (1.0, 1.0, 1.0)))
        fwhm = float(params["sigma_mm"])  # scenario-level smoothing width
        sigma = tuple(fwhm * FWHM_TO_SIGMA / v for v in voxel)
        smoothed = gaussian_filter(np.asarray(observed, dtype=np.float64), sigma=sigma)
        arr = smoothed.astype(np.float32)
        yield Iterate(iteration=1, image=arr)
```

`studies/krl_studies/methods/petpvc.py`:

```python
"""GTM PVC via the PETPVC command-line toolbox (Thomas et al., PMB 2016).

Requires the `petpvc` binary on PATH (cluster/docker environments). Runs as a
single-shot correction; exposed through the streaming interface with exactly
one iterate.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

from krl_studies.methods.base import Iterate, Method


def build_petpvc_cmd(
    petpvc_bin: str,
    input_path: str | Path,
    output_path: str | Path,
    mode: str,
    pvc_fwhm: tuple[float, float, float],
    mask_path: str | Path | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        petpvc_bin,
        "-i", str(input_path),
        "-o", str(output_path),
        "-p", mode,
        "-f", ",".join(str(v) for v in pvc_fwhm),
    ]
    if mask_path is not None:
        cmd += ["-m", str(mask_path)]
    if extra:
        cmd += list(extra)
    return cmd


class GTMMethod(Method):
    name = "gtm"

    def run(self, observed, guidance, params, n_iterations) -> Iterator[Iterate]:
        if n_iterations != 1:
            raise ValueError("GTM is single-step; use n_iterations=1")
        bin_name = str(params.get("petpvc_bin", "petpvc"))
        if shutil.which(bin_name) is None:
            raise FileNotFoundError(
                f"PETPVC binary '{bin_name}' not found on PATH; "
                "install PETPVC (cluster/docker) or skip GTM scenarios"
            )
        cmd = build_petpvc_cmd(
            petpvc_bin=bin_name,
            input_path=params["input_path"],
            output_path=params["output_path"],
            mode="GTM",
            pvc_fwhm=tuple(float(v) for v in params.get("pvc_fwhm", (5.0, 5.0, 5.0))),
            mask_path=params.get("mask_path"),
            extra=params.get("extra"),
        )
        subprocess.run(cmd, check=True)
        import nibabel as nib

        arr = np.transpose(
            nib.load(str(params["output_path"])).get_fdata().astype(np.float32), (2, 1, 0)
        )
        yield Iterate(iteration=1, image=arr)
```

Modify `studies/krl_studies/methods/__init__.py`:

```python
from krl_studies.methods.baselines import PostSmoothingMethod
from krl_studies.methods.dtv import DTVMethod
from krl_studies.methods.iterative_yang import IterativeYangMethod
from krl_studies.methods.petpvc import GTMMethod
from krl_studies.methods.richardson_lucy import HKRLMethod, KRLMethod, RLMethod

METHOD_REGISTRY = {
    cls.name: cls for cls in (
        RLMethod, KRLMethod, HKRLMethod, DTVMethod,
        IterativeYangMethod, PostSmoothingMethod, GTMMethod,
    )
}

__all__ = [
    "METHOD_REGISTRY",
    "DTVMethod",
    "GTMMethod",
    "HKRLMethod",
    "IterativeYangMethod",
    "KRLMethod",
    "PostSmoothingMethod",
    "RLMethod",
]
```

- [ ] **Step 4: Run tests to green**

```bash
python -m pytest studies/tests/test_baselines_petpvc.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add studies/krl_studies/methods studies/tests/test_baselines_petpvc.py && git commit -m "ADD post-smoothing baseline and PETPVC GTM wrapper"
```

---

### Task 13: Runner — expansion, execution, CLI

**Files:**
- Modify: `studies/krl_studies/config.py` (carry scenario-level `sim` config and output root on RunSpec)
- Create: `studies/krl_studies/runner/expand.py`
- Create: `studies/krl_studies/runner/execute.py`
- Create: `studies/krl_studies/runner/cli.py`
- Create: `studies/krl_studies/run.py` and `studies/krl_studies/__main__.py`
- Test: `studies/tests/test_runner.py`

- [ ] **Step 1: Amend `config.py` so runs carry `sim` and `out_root`**

Add `sim` to `Scenario` and both `sim` + `out_root` to `RunSpec`. Replace the two dataclasses in `studies/krl_studies/config.py` with:

```python
@dataclass(frozen=True)
class Scenario:
    study: str
    dataset: dict[str, Any]
    inputs: tuple[InputSpec, ...]
    methods: tuple[MethodSpec, ...]
    output: Path
    sim: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    study: str
    dataset: dict[str, Any]
    input_kind: str
    input_params: dict[str, Any]
    method_name: str
    method_params: dict[str, Any]
    sim: dict[str, Any] = field(default_factory=dict)
    out_root: Path = Path("results")
```

Update `load_scenario_dict` to pass `sim=dict(raw.get("sim", {}))`, and update the tail of `expand_scenario` so each RunSpec is built as:

```python
                    runs.append(
                        RunSpec(
                            run_id=run_id,
                            study=scenario.study,
                            dataset=scenario.dataset,
                            input_kind=inp.kind,
                            input_params=input_params,
                            method_name=method.name,
                            method_params=method_params,
                            sim=scenario.sim,
                            out_root=scenario.output,
                        )
                    )
```

(The `scenario_name` field from Task 3 is dropped; Task 3's tests do not reference it, so they still pass.)

- [ ] **Step 2: Write failing runner tests**

`studies/tests/test_runner.py`:

```python
import json

import numpy as np
import pytest
import yaml

from conftest import write_test_nifti
from krl_studies.config import load_scenario_dict
from krl_studies.runner.expand import expand_scenario
from krl_studies.runner.execute import execute_run
from krl_studies.runner.cli import main


def _mini_scenario(tmp_path):
    gt = np.full((24, 32, 32), 1.0, dtype=np.float32)
    gt[8:16, 12:20, 12:20] = 6.0
    d = tmp_path / "spheres"
    d.mkdir()
    write_test_nifti(d / "phant_orig.nii", gt)
    write_test_nifti(d / "phant_mri.nii", np.full_like(gt, 0.5))
    write_test_nifti(d / "phant_pet.nii", gt * 0.95)

    scenario = {
        "study": "spheres",
        "dataset": {"kind": "spheres", "root": str(d)},
        "inputs": [{"kind": "reference"}],
        "methods": [
            {"name": "post_smoothing", "params": {"sigma_mm": 2.0}},
        ],
        "sim": {"fwhm_mm": 3.0, "counts": 1e5},
        "output": str(tmp_path / "results"),
    }
    sp = tmp_path / "scen.yaml"
    sp.write_text(yaml.safe_dump(scenario))
    return sp


def test_execute_creates_manifest_metrics_marker(tmp_path):
    scenario = load_scenario_dict(yaml.safe_load(_mini_scenario(tmp_path).read_text()))
    run = expand_scenario(scenario)[0]
    assert run.sim == {"fwhm_mm": 3.0, "counts": 1e5}
    assert run.out_root == tmp_path / "results"

    out = execute_run(run)
    assert (out / ".done").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["method"] == "post_smoothing"
    assert manifest["status"] == "complete"
    csv_lines = (out / "metrics.csv").read_text().strip().splitlines()
    assert csv_lines[0].startswith("iteration")
    assert len(csv_lines) == 2  # header + single iterate
    assert (out / "final.nii.gz").exists()


def test_execute_skips_completed(tmp_path):
    scenario = load_scenario_dict(yaml.safe_load(_mini_scenario(tmp_path).read_text()))
    run = expand_scenario(scenario)[0]
    out1 = execute_run(run)
    marker1 = (out1 / ".done").read_text()
    out2 = execute_run(run, force=False)
    assert out1 == out2
    assert (out2 / ".done").read_text() == marker1


def test_cli_dry_run_lists_runs_without_executing(tmp_path, capsys):
    sp = _mini_scenario(tmp_path)
    rc = main(["--scenario", str(sp), "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "post_smoothing" in captured
    markers = list((tmp_path / "results").rglob(".done")) if (tmp_path / "results").exists() else []
    assert markers == []
```

Note: `execute_run` needs CIL ImageData for RL-family methods but `post_smoothing` consumes plain arrays, which keeps this test fast and SIRF-free.

- [ ] **Step 3: Run tests to verify failure**

```bash
python -m pytest studies/tests/test_runner.py -v
```

Expected: FAIL with `ImportError`/`ModuleNotFoundError` for `krl_studies.runner.*`.

- [ ] **Step 4: Implement**

`studies/krl_studies/runner/expand.py`:

```python
"""Stable re-export of expansion helpers for runner users."""

from krl_studies.config import RunSpec, expand_scenario

__all__ = ["RunSpec", "expand_scenario"]
```

`studies/krl_studies/runner/execute.py`:

```python
"""Execute one RunSpec: build inputs, stream method iterates, record metrics."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from krl.utils import load_nifti_as_imagedata, save_image

from krl_studies.config import RunSpec
from krl_studies.datasets.lesions import (
    DEFAULT_CONTRAST,
    DEFAULT_TUMOUR_DIAMETERS_MM,
    default_tumour_specs,
    place_tumours,
)
from krl_studies.datasets.spheres import SphereDataset, quick_sim
from krl_studies.metrics import (
    background_variability,
    background_vois,
    crc_percent,
    derive_lesion_rois,
    nrmse,
    write_metrics_csv,
)
from krl_studies.methods import METHOD_REGISTRY

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@contextmanager
def tempfile_dir():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        yield td


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "unknown"


def _wrap(arr: np.ndarray, voxel_mm) -> Any:
    """Wrap a (z,y,x) array as CIL ImageData with the given voxel sizes."""
    from cil.framework import ImageGeometry

    geom = ImageGeometry(
        voxel_num_x=arr.shape[2],
        voxel_num_y=arr.shape[1],
        voxel_num_z=arr.shape[0],
        voxel_size_x=voxel_mm[2],
        voxel_size_y=voxel_mm[1],
        voxel_size_z=voxel_mm[0],
    )
    img = geom.allocate()
    img.fill(arr.astype(np.float32))
    return img


def _build_observed(run: RunSpec, ds: SphereDataset, gt: np.ndarray) -> np.ndarray:
    if run.input_kind == "reference":
        return ds.reference_pet
    if run.input_kind == "quick_sim":
        return quick_sim(
            gt,
            fwhm_mm=float(run.input_params["fwhm_mm"]),
            counts=float(run.input_params["counts"]),
            realisation=int(run.input_params.get("realisation", 0)),
            voxel_mm=ds.voxel_mm,
        )
    raise ValueError(f"unknown input kind: {run.input_kind}")


def _iy_region_defaults(gt: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Two-compartment split (hot vs background) inside a brain mask."""
    brain = gt > 0
    hot = brain & (gt > 0.25 * float(gt.max()))
    return [hot, brain & ~hot], brain


def execute_run(run: RunSpec, force: bool = False) -> Path:
    out_dir = Path(run.out_root) / run.run_id
    marker = out_dir / ".done"
    if marker.exists() and not force:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if run.study != "spheres":
        raise NotImplementedError("runner phase 1 supports study='spheres'")

    ds = SphereDataset(root=run.dataset["root"])
    gt = ds.ground_truth
    guidance_arr = ds.guidance
    observed_arr = _build_observed(run, ds, gt)

    lesion_masks: list[np.ndarray] = []
    if run.sim.get("add_tumours"):
        specs = default_tumour_specs(
            gt.shape,
            ds.voxel_mm,
            diameters_mm=tuple(run.sim.get("tumour_diameters_mm", DEFAULT_TUMOUR_DIAMETERS_MM)),
        )
        gt, lesion_masks = place_tumours(
            gt,
            specs,
            contrast=float(run.sim.get("tumour_contrast", DEFAULT_CONTRAST)),
            voxel_mm=ds.voxel_mm,
        )
        if run.input_kind == "quick_sim":
            observed_arr = quick_sim(
                gt,
                fwhm_mm=float(run.input_params["fwhm_mm"]),
                counts=float(run.input_params["counts"]),
                realisation=int(run.input_params.get("realisation", 0)),
                voxel_mm=ds.voxel_mm,
            )

    lesion_rois = derive_lesion_rois(gt) if lesion_masks else []
    exclusion = np.logical_or.reduce(lesion_rois or lesion_masks) if (lesion_rois or lesion_masks) else np.zeros_like(gt, dtype=bool)
    vois = background_vois(gt.shape, exclude_mask=exclusion)

    method_cls = METHOD_REGISTRY[run.method_name]
    params = dict(run.method_params)
    if run.method_name == "iy":
        regions, brain = _iy_region_defaults(gt)
        params.setdefault("region_masks", regions)
        params.setdefault(
            "psf_sigma_vox",
            tuple(float(params.get("fwhm_mm", 5.0)) * FWHM_TO_SIGMA for _ in range(3)),
        )
        params.setdefault("brain_mask", brain)

    with tempfile_dir() as td:
        td_path = Path(td)
        obs_path = td_path / "observed.nii"
        guide_path = td_path / "guidance.nii"
        save_image(_wrap(observed_arr, ds.voxel_mm), obs_path)
        save_image(_wrap(guidance_arr, ds.voxel_mm), guide_path)
        observed_cil = load_nifti_as_imagedata(obs_path)
        guidance_cil = load_nifti_as_imagedata(guide_path)

        rows: list[dict[str, Any]] = []
        best_nrmse = float("inf")
        best_img: np.ndarray | None = None
        final_img: np.ndarray | None = None
        stream = method_cls().run(
            observed=observed_cil,
            guidance=guidance_cil if run.method_name in {"krl", "hkrl", "dtv"} else None,
            params=params,
            n_iterations=int(params.get("iterations", 1)),
        )
        for it in stream:
            row: dict[str, Any] = {"iteration": it.iteration}
            if it.objective is not None:
                row["objective"] = it.objective
            value = nrmse(it.image, gt)
            row["nrmse"] = value
            if value < best_nrmse:
                best_nrmse = value
                best_img = it.image.copy()
            for i, mask in enumerate(lesion_masks):
                d_mm = sorted(DEFAULT_TUMOUR_DIAMETERS_MM)[i] if i < len(DEFAULT_TUMOUR_DIAMETERS_MM) else i * 10
                if vois:
                    row[f"crc_mm{int(d_mm)}"] = crc_percent(mask, it.image, gt, vois)
            if vois:
                row["bv_percent"] = background_variability(it.image, vois)
            rows.append(row)
            final_img = it.image

    write_metrics_csv(rows, out_dir / "metrics.csv")
    if final_img is not None:
        save_image(_wrap(final_img, ds.voxel_mm), out_dir / "final.nii.gz")
    if best_img is not None:
        save_image(_wrap(best_img, ds.voxel_mm), out_dir / "best_nrmse.nii.gz")

    manifest = {
        "run_id": run.run_id,
        "study": run.study,
        "input_kind": run.input_kind,
        "input_params": run.input_params,
        "method": run.method_name,
        "method_params": run.method_params,
        "dataset": run.dataset,
        "sim": run.sim,
        "status": "complete",
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "krl_version": _pkg_version("cil-krl"),
        "krl_studies_version": _pkg_version("krl-studies"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    marker.write_text(dt.datetime.now().isoformat(timespec="seconds"))
    return out_dir
```

`studies/krl_studies/runner/cli.py`:

```python
"""CLI: python -m krl_studies.run --scenario FILE [--dry-run] [--force] [--only SUBSTR]"""

from __future__ import annotations

import argparse

from krl_studies.config import load_scenario
from krl_studies.runner.execute import execute_run
from krl_studies.runner.expand import expand_scenario


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="krl-studies.run", description="Run krl benchmark scenarios")
    parser.add_argument("--scenario", required=True, help="path to scenario YAML")
    parser.add_argument("--dry-run", action="store_true", help="list expanded runs and exit")
    parser.add_argument("--force", action="store_true", help="re-run even if completion marker exists")
    parser.add_argument("--only", default=None, help="substring filter on run ids")
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    runs = expand_scenario(scenario)
    if args.only:
        runs = [r for r in runs if args.only in r.run_id]

    if args.dry_run:
        for r in runs:
            print(r.run_id)
        print(f"-- {len(runs)} run(s)")
        return 0

    failures = []
    for i, run in enumerate(runs, start=1):
        print(f"[{i}/{len(runs)}] {run.run_id}")
        try:
            out = execute_run(run, force=args.force)
            print(f"    done -> {out}")
        except Exception as exc:  # runner continues remaining runs
            failures.append((run.run_id, exc))
            print(f"    FAILED: {exc}")
    for run_id, exc in failures:
        print(f"FAILURE {run_id}: {exc}")
    return 1 if failures else 0
```

`studies/krl_studies/run.py`:

```python
"""Entry point so `python -m krl_studies.run ...` works."""

from krl_studies.runner.cli import main

raise SystemExit(main())
```

`studies/krl_studies/__main__.py`:

```python
from krl_studies.runner.cli import main

raise SystemExit(main())
```

- [ ] **Step 5: Run runner tests to green**

```bash
python -m pytest studies/tests/test_runner.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Full suites**

```bash
python -m pytest studies/tests -q && python -m pytest tests -q
```

Expected: all pass; library suite unchanged.

- [ ] **Step 7: Commit**

```bash
git add studies && git commit -m "ADD resumable experiment runner with CLI and manifests"
```

---

### Task 14: Scenario YAMLs + real end-to-end smoke run

**Files:**
- Create: `studies/scenarios/spheres_core.yaml`
- Create: `studies/scenarios/patient_mk_h001.yaml`

- [ ] **Step 1: Write spheres scenario**

```yaml
# Core spheres benchmark: image-space inputs (SIRF variants arrive in Plan 2).
study: spheres
dataset:
  kind: spheres
  root: data/spheres
sim:
  fwhm_mm: 5.5
  counts: 1.0e5
  add_tumours: false
inputs:
  - kind: reference            # committed MIC2025 blurred image
  - kind: quick_sim
    params:
      fwhm_mm: [5.5]
      counts: [1.0e5]
      realisation: [0, 1, 2]
methods:
  - name: rl
    params: {fwhm_mm: 5.5, iterations: 30}
  - name: krl
    params:
      fwhm_mm: 5.5
      iterations: 30
      num_neighbours: 9
      sigma_anat: [0.2, 1.0]
  - name: iy
    params: {iterations: 30, fwhm_mm: 5.5}
  - name: post_smoothing
    params: {sigma_mm: [2.0]}
output: results/spheres_core
```

- [ ] **Step 2: Dry-run to inspect expansion**

```bash
source .venv/bin/activate && python -m krl_studies.run --scenario studies/scenarios/spheres_core.yaml --dry-run
```

Expected: per input variant the method grids give rl(1) + krl(2 σ values) + iy(1) + post_smoothing(1) = **5 runs**; with reference(1) + quick_sim(3 realisations) = **20 runs total**, printed ending `-- 20 run(s)`.

- [ ] **Step 3: Execute for real (short) and inspect outputs**

```bash
python -m krl_studies.run --scenario studies/scenarios/spheres_core.yaml --only reference
```

Expected: 5 reference runs complete; `ls results/spheres_core/*/` shows `manifest.json`, `metrics.csv`, `final.nii.gz` (and `best_nrmse.nii.gz` for iterative methods). Spot-check:

```bash
python - <<'EOF'
import pandas as pd, glob
for f in sorted(glob.glob("results/spheres_core/reference__*/metrics.csv")):
    df = pd.read_csv(f)
    cols = [c for c in ("iteration", "nrmse") if c in df]
    print(f.split("/")[-2], "->", df[cols].tail(1).to_dict("records"))
EOF
```

Expected: NRMSE decreasing or sensible finite values per method.

- [ ] **Step 4: Patient scenario (qualitative; no GT)**

```yaml
# MK-H001 demonstration: no ground truth; convergence + artefacts only.
study: patient
dataset:
  kind: patient
  subject_id: MK-H001
  root: data/patients
inputs:
  - kind: native
methods:
  - name: rl
    params: {fwhm_mm: 4.5, iterations: 20}
  - name: krl
    params: {fwhm_mm: 4.5, iterations: 20, num_neighbours: 9, sigma_anat: 0.2}
output: results/patient_demo
```

Patient support lands in the runner in Plan 2 alongside SIRF inputs; until then running this YAML exits with `NotImplementedError: runner phase 1 supports study='spheres'` — commit it as documentation of intent (Task 13 deliberately raised there). Verify the error message appears:

```bash
python -m krl_studies.run --scenario studies/scenarios/patient_mk_h001.yaml ; echo "rc=$?"
```

Expected: `FAILED ... NotImplementedError` and `rc=1`.

- [ ] **Step 5: Commit**

```bash
git add studies/scenarios && git commit -m "ADD spheres core and patient demo scenario definitions"
```

---

### Task 15: studies README + lint + final verification

**Files:**
- Create: `studies/README.md`
- Modify: root `README.md` (link only)

- [ ] **Step 1: Write studies/README.md**

```markdown
# krl-studies

Benchmark framework for RL deconvolution regularisation methods (design spec:
`../docs/superpowers/specs/2026-08-23-krl-study-framework-design.md`).

## Install

Requires the parent repo's environment (CIL + `cil-krl`), then:

    uv pip install -e "./studies[dev]"     # or: make study-install

## Quick start

    python -m krl_studies.run --scenario studies/scenarios/spheres_core.yaml --dry-run
    python -m krl_studies.run --scenario studies/scenarios/spheres_core.yaml

Results land under `results/<scenario>/<run_id>/`: `manifest.json`,
`metrics.csv`, `final.nii.gz`, `best_nrmse.nii.gz`. Re-invocation skips runs
with a `.done` marker; `--force` re-runs; `--only SUBSTR` filters.

## Environment notes

- SIRF/STIR (Plan 2 simulation layer): install in the same conda env you use
  for CIL; see https://github.com/SyneRBI/SIRF-SuperBuild.
- PETPVC (GTM comparator): provide the `petpvc` binary on PATH (cluster or
  docker). Scenarios including `gtm` fail fast with a clear error otherwise.
- Heavy sweeps are intended for the GPU box (docker-compose in `examples/`)
  and the UCL CS SGE cluster (array-job templates arrive in Plan 3).

## Status / roadmap

- Plan 1 (this): data layer, methods, metrics, resumable runner, image-space
  inputs (`reference`, `quick_sim`).
- Plan 2: SIRF forward projection + Poisson + OSEM/RDP recon inputs,
  recon-PSF conditions, patient study enablement, BrainWeb preparation.
- Plan 3: aggregated tidy results store, publication figures/tables, SGE +
  docker orchestration.
```

- [ ] **Step 2: Root README link**

In root `README.md`, in the Development section after the GPU-tests paragraph, append:

```markdown
Research benchmark studies live under [`studies/`](studies/README.md) (branch
`study-framework`): a config-driven, resumable framework for comparing RL
regularisation methods on simulated and patient data.
```

- [ ] **Step 3: Lint everything**

```bash
ruff check src/ tests/ studies/ --fix
ruff check src/ tests/ studies/
```

Expected: no errors remaining.

- [ ] **Step 4: Final verification**

```bash
make test && make study-test && make lint
git status --short
```

Expected: all suites pass; clean tree (nothing under `data/patients/` or `results/` staged).

- [ ] **Step 5: Commit**

```bash
git add studies/README.md README.md && git commit -m "ADD studies documentation and root README link"
```

---

## Out of scope for this plan (deliberate)

- SIRF simulation, recon-PSF conditions, RDP inputs, BrainWeb preparation, patient runner enablement → **Plan 2**
- Results aggregation store, publication figures/LaTeX tables, SGE/docker job generation → **Plan 3**
- Any change to `src/krl` or `examples/`

## Risks / engineer notes

- `get_kernel_operator` / `set_parameters` kwarg names: verify against `src/krl/operators/kernel_operator.py` during Task 9; the whitelist mirrors `KernelParameters` from the legacy sweep script.
- CIL `Callback.__call__` receives the algorithm *after* each iteration; `algorithm.iteration` starts at 1 — the tests assume exactly that.
- numba compile time dominates first test run of Task 9 (~minutes); subsequent runs are cached.
- All heavy paths stay on the numba backend; never import torch in `krl_studies` (macOS OpenMP conflict).
