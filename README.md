# cil-krl: Kernelised Richardson-Lucy Deconvolution for PET

[![CI](https://github.com/samdporter/Kernel_RL/actions/workflows/ci.yml/badge.svg)](https://github.com/samdporter/Kernel_RL/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**cil-krl** is a plugin for the [Core Imaging Library (CIL)](https://github.com/TomographicImaging/CIL)
implementing anatomically-guided Richardson-Lucy deconvolution for PET imaging:

- **KRL** — kernelised Richardson-Lucy: deconvolution steered by an anatomical image (e.g. MRI) via a kernel operator
- **HKRL** — hybrid KRL mixing emission and anatomical features, with optional kernel freezing
- **MAP-RL** — maximum-a-posteriori RL with Armijo line search and preconditioning
- **DTV** — directional total variation regularisation built on CIL's gradient operators
- **Backends** — numba CPU backend; optional PyTorch backend for CUDA GPUs and Apple MPS

Everything is built directly on CIL's optimisation framework: operators subclass
`cil.optimisation.operators.LinearOperator`, algorithms subclass
`cil.optimisation.algorithms.Algorithm`, so they compose with the rest of the CIL
ecosystem (callbacks, functions, block operators, ...).

## Installation

`cil-krl` is **not published on PyPI**, so `pip install cil-krl` does not work.
Install CIL first — it is distributed via conda (conda-forge + ccpi channels),
not PyPI — then install this package from a checkout or from a wheel you build
yourself.

Combinations tested in CI:

| Python | CIL |
|--------|-----|
| 3.10 | 25.0.0 |
| 3.11, 3.12, 3.13 | 26.0.0 |

The wheel-install gate in CI builds and installs the wheel on Python 3.11 /
CIL 26.0.0, which resolved to NumPy 2.4 and Numba 0.68 on CPU. The CUDA/torch
tests are opt-in and skip without a GPU, so they are not covered by that gate.

### Linux

```bash
# 1. Create an environment with CIL (conda or micromamba)
conda create -n krl -c conda-forge -c ccpi python=3.11 cil=26.0.0 pip
conda activate krl

# 2. Install cil-krl from a checkout
git clone https://github.com/samdporter/Kernel_RL.git && cd Kernel_RL
pip install -e ".[dev]"      # + pytest, ruff
# pip install -e ".[gpu]"    # optional: adds PyTorch for the torch backend
```

Alternatively build a wheel in the checkout with `make build` and install it with
`pip install dist/*.whl`.

### macOS

CIL has no `osx-arm64` build, so run the whole thing in a Linux (x86_64)
container, from the repository root:

```bash
docker run --rm --platform linux/amd64 -v "$PWD:/repo:ro" --workdir /tmp \
  --entrypoint /bin/bash mambaorg/micromamba:2.3.2 -lc '
    micromamba create -y -q -n krl -c conda-forge -c ccpi python=3.11 cil=26.0.0 pip &&
    cp -R /repo /tmp/krl &&
    micromamba run -n krl python -m pip install -e "/tmp/krl[dev]" &&
    micromamba run -n krl python -c "import krl; print(krl.__version__)"'
```

> **macOS / OpenMP caveat:** importing PyTorch into a process that also uses
> CIL's native acceleration libraries can abort due to duplicate OpenMP runtimes.
> On macOS keep to the numba backend — `backend="auto"` resolves to numba there
> *without importing torch* — or run torch-based work in a separate process.

A native Apple Silicon route also works: CIL built from source plus the torch
backend on MPS (with a documented OpenMP workaround). See
[macOS / ARM](docs/MACOS-ARM.md) for the verified setup and test commands.

## Quickstart

A complete, self-contained CPU example: a synthetic emission image, a synthetic
anatomical image, the observed blurred data and a callback. Every variable is
defined below, so the snippet runs as-is.

```python
import numpy as np
from cil.framework import ImageGeometry
from cil.optimisation.utilities.callbacks import Callback

from krl import RichardsonLucy, create_gaussian_blur, get_kernel_operator

# 1. Geometry and two aligned synthetic images
geometry = ImageGeometry(voxel_num_x=32, voxel_num_y=32, voxel_num_z=16)

z, y, x = np.indices((16, 32, 32), dtype=np.float32)
emission = geometry.allocate(0.0)
emission.fill(50.0 * np.exp(-((z - 8) ** 2 + (y - 12) ** 2 + (x - 14) ** 2) / 12.0))
mr_image = geometry.allocate(0.0)
mr_image.fill(np.exp(-((z - 6) ** 2 + (y - 20) ** 2 + (x - 18) ** 2) / 8.0))

# 2. Observed data: the emission image blurred by the PSF
blur_op = create_gaussian_blur(sigma=(1.5, 1.5, 1.5), geometry=geometry, backend="numba")
observed = blur_op.direct(emission)

# 3. Anatomical guidance operator
kernel_op = get_kernel_operator(
    geometry, backend="numba", num_neighbours=3, sigma_anat=0.5
)
kernel_op.set_anatomical_image(mr_image)


# 4. Callback: records and prints the objective after each iteration
class ObjectiveCallback(Callback):
    def __init__(self):
        super().__init__()
        self.values = []

    def __call__(self, algorithm):
        self.values.append(float(algorithm.loss[-1]))
        print(f"iteration {algorithm.iteration}: objective {self.values[-1]:.4f}")


# 5. Reconstruct (omit kernel_operator for standard RL)
callback = ObjectiveCallback()
algo = RichardsonLucy(
    initial_estimate=observed,
    blurring_operator=blur_op,
    observed_data=observed,
    kernel_operator=kernel_op,
)
algo.run(iterations=8, callbacks=[callback])

reconstruction = algo.get_output()
assert np.isfinite(reconstruction.as_array()).all()
assert (reconstruction.as_array() >= 0).all()
```

Callbacks receive the CIL `Algorithm` object after every iteration. `krl` also
provides `NRMSECallback` (NRMSE against a ground truth, appended to a CSV file)
and `SaveIterationCallback` (writes `.nii.gz` snapshots of the reconstruction).

Because `KernelOperator` is a plain CIL `LinearOperator`, you can also drop it
into your own CIL compositions (`CompositionOperator`, custom `Function`s, ...)
and drive it with any CIL algorithm.

## Backends

| Backend | Used by | Hardware | Notes |
|---------|---------|----------|-------|
| `numba` | kernel + blur | CPU | used by the quickstart above; kernel arithmetic in float64 |
| `torch` | kernel + blur | CUDA GPU / Apple MPS | optional (`gpu` extra); device falls back cuda → mps → cpu. The kernel operator's MPS path is verified end-to-end against the numba reference; the blur backend's MPS branch is not verified end-to-end |
| `scipy` | blur only | CPU | last-resort fallback for the blur operator |

- `backend="auto"` is the default for both `get_kernel_operator` and
  `create_gaussian_blur`:
  - on **macOS** it resolves to numba **without importing torch** (see the OpenMP
    caveat above);
  - elsewhere it probes for torch and uses it only when CUDA is available,
    otherwise it falls back to numba (blur: torch → numba → scipy).
- The torch kernel operator's `device="auto"` resolves cuda → mps → cpu
  (same order as the blur backend); when the device resolves to CPU it runs
  through the numba implementation, so installing torch never makes the
  package CUDA-only.

## Notes

### Boundary conditions

- The numba and torch blur backends **zero-pad** at the volume boundary; the
  scipy backend uses **reflection**.
- The kernel operator uses inclusive mirror reflection of its neighbourhood at
  the volume boundaries on both backends.

### Data type vs computational precision

- Operator results are written into a clone of the *input* container, so the
  storage dtype of a result follows the image you passed in (a float32 input
  gives a float32 output, even when the computation used float64).
- The numba backends accumulate in float64. The torch kernel backend computes in
  float32 by default (`dtype="float64"` selects double), and the torch blur
  backend always computes in float32.

### Aligned images

- The anatomical image must be on exactly the same grid as the emission image:
  same shape (validated, a mismatch raises `ValueError`) and voxel-by-voxel
  correspondence. No resampling is performed, so co-register beforehand.

### HKRL (adaptive hybrid) restrictions

- The kernel operator supports single-channel 3-D volumes only.
- With `hybrid=True` the emission reference is taken from the first `direct()`
  call, so `direct()` must run before `adjoint()` — the adjoint raises if the
  reference or the normalisation map has not been initialised.
- While unfrozen, the reference is refreshed on every forward call and
  `RichardsonLucy` recomputes the sensitivity `A^T 1` each iteration.
- `freeze_iteration=N` freezes the kernel after the N-th update: from then on
  forward and adjoint share the same frozen reference and the operator no longer
  changes between iterations. Freezing is a modelling choice; no convergence
  rate is claimed for it.

### L-BFGS-B

- `LBFGSBOptimizer` is a thin wrapper around `scipy.optimize.minimize(method=
  "L-BFGS-B")` operating on flat arrays. It is **not** a CIL `Algorithm`
  subclass: it has its own `run(iterations, callbacks, verbose)` and its
  callbacks receive the optimiser object rather than a CIL algorithm.

### Image I/O

- `load_image` / `save_image` accept only 3-D `.nii` / `.nii.gz` volumes
  (singleton axes are rejected on load). Saved files record voxel values and
  voxel spacing only — no original NIfTI header or affine is preserved.

## Development

```bash
export PYTHON=$(micromamba run -n krl which python)   # interpreter with CIL
make install   # editable install + dev tools (uv)
make test      # CPU test suite
make lint      # ruff check
make build     # sdist + wheel
```

GPU tests are opt-in via `KRL_RUN_GPU_TESTS=1`: `make gpu-test` runs the CUDA
modules, while the MPS module is run explicitly (see
[macOS / ARM](docs/MACOS-ARM.md)).

The research pipelines, benchmark scripts and BrainWeb data preparation used in
the original study live under [`examples/`](examples/README.md): they are
historical reference material, not part of the installed package and not
maintained.

## Documentation

- [Methods overview](docs/METHODS.md) — RL, KRL, HKRL and DTV via the Python API
- [Documentation index](docs/README.md)

## Citation

If you use cil-krl in your research, please cite it and CIL:

```bibtex
@software{krl2025,
  author = {Porter, Sam and Erlandsson, Kjell and Deidda, Daniel and Thielemans, Kris},
  title = {cil-krl: Kernelised Richardson-Lucy Deconvolution for PET},
  year = {2025},
  url = {https://github.com/samdporter/Kernel_RL}
}
```

See also the [CIL citation guidelines](https://github.com/TomographicImaging/CIL#citing-cil).

## License

cil-krl is MIT-licensed — see [LICENSE](LICENSE).
