# KRL: Kernelised Richardson-Lucy Deconvolution for PET

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advanced PET image reconstruction using anatomically-guided Richardson-Lucy deconvolution.

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

## Usage

```bash
python run_deconv.py \
  --data-path data/spheres \
  --emission-file OSEM.hv \
  --guidance-file T1.hv \
  --enable-krl
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
