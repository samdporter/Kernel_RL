# cil-krl: Kernelised Richardson-Lucy Deconvolution for PET

[![CI](https://github.com/KCL-BMEIS/KRL/actions/workflows/ci.yml/badge.svg)](https://github.com/KCL-BMEIS/KRL/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cil-krl)](https://pypi.org/project/cil-krl/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**cil-krl** is a plugin for the [Core Imaging Library (CIL)](https://github.com/TomographicImaging/CIL)
implementing anatomically-guided Richardson-Lucy deconvolution for PET imaging:

- **KRL** — kernelised Richardson-Lucy: deconvolution steered by an anatomical image (e.g. MRI) via a kernel operator
- **HKRL** — hybrid KRL mixing emission and anatomical features, with kernel freezing
- **MAP-RL** — maximum-a-posteriori RL with Armijo line search and preconditioning
- **DTV** — directional total variation regularisation built on CIL operators
- **GPU acceleration** — PyTorch CUDA backend for large volumes (256³ on consumer GPUs), numba CPU backend otherwise

Everything is built directly on CIL's optimisation framework: operators subclass
`cil.optimisation.operators.LinearOperator`, algorithms subclass
`cil.optimisation.algorithms.Algorithm`, so they compose with the rest of the CIL
ecosystem (callbacks, functions, block operators, ...).

## Installation

> CIL itself is currently distributed via conda rather than PyPI, so install it first.

```bash
# 1. Create an environment with CIL
conda create -n krl -c conda-forge -c ccpi python=3.11 cil

# 2. Activate and install cil-krl
conda activate krl
pip install cil-krl              # core (numba CPU backend)
pip install "cil-krl[gpu]"       # + PyTorch CUDA backend
```

From source:

```bash
git clone https://github.com/KCL-BMEIS/KRL.git && cd KRL
pip install -e ".[dev]"
```

## Quickstart

```python
from cil.framework import ImageGeometry
from cil.optimisation.utilities.callbacks import Callback

from krl import (
    get_kernel_operator,     # anatomical guidance operator (LinearOperator)
    create_gaussian_blur,    # PSF blurring operator (LinearOperator)
    RichardsonLucy,          # Algorithm subclass
    NRMSECallback,           # Callback subclass
)

geometry = ImageGeometry(voxel_num_x=144, voxel_num_y=144, voxel_num_z=127)

# Anatomical guidance: any CIL ImageData aligned with the emission image
kernel_op = get_kernel_operator(
    geometry,
    backend="auto",          # torch (CUDA) if available, else numba CPU
    num_neighbours=5,
    sigma_anat=0.1,
)
kernel_op.set_anatomical_image(mr_image)

blur_op = create_gaussian_blur(sigma=(1.0, 1.0, 1.0), geometry=geometry, backend="numba")

algo = RichardsonLucy(
    initial_estimate=observed,
    blurring_operator=blur_op,
    observed_data=observed,
    kernel_operator=kernel_op,   # omit for standard RL
)
algo.run(iterations=32, verbose=1)

reconstruction = algo.get_output()
```

Because `KernelOperator` is a plain CIL `LinearOperator`, you can also drop it into
your own CIL compositions (`CompositionOperator`, custom `Function`s, ...) and drive
it with any CIL algorithm.

## Backends & memory

| Backend | Hardware | Notes |
|---------|----------|-------|
| `numba` | CPU | default, float64 |
| `torch` | NVIDIA CUDA | float32 option halves memory; recommended for ≥128³ volumes |

`backend="auto"` picks torch when CUDA is available, else numba.

> **macOS note:** importing PyTorch into a process that also uses CIL's native
> acceleration libraries can abort due to duplicate OpenMP runtimes. On macOS keep
> to the numba backend (no CUDA anyway), or run torch-based work in a separate process.

## Development

```bash
make install   # editable install + dev tools (uses uv)
make test      # CPU test suite
make lint      # ruff
```

GPU tests need CUDA and are opt-in: `make gpu-test`.

The research pipelines, benchmark scripts and BrainWeb data preparation used in the
original study live under [`examples/`](examples/README.md) and are not part of the
installed package.

## Documentation

- [Methods overview](docs/METHODS.md) — RL, KRL, HKRL, DTV explained

## Citation

If you use cil-krl in your research, please cite it and CIL:

```bibtex
@software{krl2025,
  author = {Erlandsson, Kjell},
  title = {cil-krl: Kernelised Richardson-Lucy Deconvolution for PET},
  year = {2025},
  url = {https://github.com/KCL-BMEIS/KRL}
}
```

See also the [CIL citation guidelines](https://github.com/TomographicImaging/CIL#citing-cil).

## License

MIT License — see [LICENSE](LICENSE).
