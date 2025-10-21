# KRL: Kernelised Richardson-Lucy Deconvolution for PET

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-PyTorch%20CUDA-green.svg)](https://pytorch.org/)

Advanced PET image reconstruction using anatomically-guided Richardson-Lucy deconvolution with GPU acceleration.

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/KCL-BMEIS/KRL.git && cd KRL
make build && make start && make shell
python run_deconv.py --help
```

### Conda

```bash
git clone https://github.com/KCL-BMEIS/KRL.git && cd KRL
conda install -c conda-forge -c ccpi cil
pip install -e .
```

### VS Code Dev Container

Open in VS Code → Press `F1` → "Dev Containers: Reopen in Container"

## Features

- **GPU Acceleration**: Process 256³ volumes on consumer GPUs (RTX 3060+)
- **Memory Efficient**: float32 option uses 50% less memory
- **Auto Backend Selection**: Automatically uses GPU if available
- **Sparse Masking**: 3-7x faster than original implementation

## Usage

### Basic Deconvolution
```bash
python run_deconv.py \
  --data-path data/spheres \
  --emission-file OSEM.hv \
  --guidance-file T1.hv \
  --enable-krl
```

### GPU Acceleration (Automatic)
The kernel operator automatically uses GPU when PyTorch + CUDA are available:
```python
from krl.operators import get_kernel_operator

# Auto-select backend (GPU if available, else CPU)
kernel_op = get_kernel_operator(geometry, backend='auto')

# Force GPU with float32 for large volumes (256³)
kernel_op = get_kernel_operator(geometry, backend='torch', dtype='float32')

# Force CPU
kernel_op = get_kernel_operator(geometry, backend='numba')
```

## Docker Commands

```bash
make build    # Build
make start    # Start
make shell    # Access
make test     # Test
make stop     # Stop
```

## Documentation

- [Getting Started](docs/GETTING-STARTED.md) - Installation
- [Methods](docs/METHODS.md) - RL, KRL, HKRL, DTV explained
- [Docker Guide](docker/README.md) - Docker commands

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@software{krl2025,
  author = {Erlandsson, Kjell},
  title = {KRL: Kernelised Richardson-Lucy Deconvolution for PET},
  year = {2025},
  url = {https://github.com/KCL-BMEIS/KRL}
}
```
