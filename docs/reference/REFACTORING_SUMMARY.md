# KRL Package Refactoring Summary

## Overview

This repository has been refactored from a research script collection into a proper, publishable Python package following modern best practices.

## Key Changes

### 1. Package Structure

**Before:**
```
KRL/
├── src/
│   ├── kernel_operator.py
│   ├── gaussian_blurring.py
│   ├── gradient.py
│   ├── directional_operator.py
│   ├── map_rl.py
│   └── deconv_cli.py
├── run_deconv.py (root-level script)
└── tests/
```

**After:**
```
KRL/
├── src/krl/                      # Proper package structure
│   ├── __init__.py               # Version and exports
│   ├── operators/                # Modular operators
│   │   ├── kernel_operator.py
│   │   ├── blurring.py
│   │   ├── gradient.py
│   │   └── directional.py
│   ├── algorithms/
│   │   └── maprl.py
│   ├── cli/                      # CLI entry points
│   │   ├── config.py
│   │   └── run_deconv.py
│   └── utils.py                  # NIfTI loading utilities
├── pyproject.toml                # Modern packaging config
├── requirements.txt
└── tests/
```

### 2. Installation & Packaging

#### Added Files:
- **`pyproject.toml`** - Modern Python packaging configuration (PEP 518/517)
- **`requirements.txt`** - Core dependencies
- **`requirements-optional.txt`** - Optional accelerators (numba, torch, cupy)
- **`requirements-dev.txt`** - Development tools

#### Installation Methods:

```bash
# Editable install
pip install -e .

# With optional accelerators
pip install -e ".[accelerators]"

# With dev tools
pip install -e ".[dev]"
```

### 3. Command-Line Interface

**Before:** Manual Python script execution
```bash
python run_deconv.py --data-path data/spheres ...
```

**After:** Registered console scripts
```bash
krl-deconv --data-path data/spheres ...
krl-compare --data-path data/spheres ...
krl-sweep --dry-run
```

These are automatically available after `pip install`.

### 4. Dependency Management

#### Removed Hard Dependencies:
- **SIRF** - Now optional (only for .hv file support)
- **torch** - Now optional (falls back to scipy/numba)
- **numba** - Now optional (falls back to Python implementation)

#### Added Smart Fallbacks:
```python
# Example: Load images with automatic format detection
from krl import load_image

# Works with .nii, .nii.gz (via nibabel)
# Works with .hv (via SIRF, if available)
img = load_image("data/emission.nii.gz")
```

#### New Image I/O Utilities (`krl.utils`):
- `load_image(filepath)` - Auto-detect format and load
- `save_image(image, filepath)` - Save to NIfTI or .hv
- `load_nifti_as_imagedata(filepath)` - NIfTI → CIL ImageData
- Handles axis transposes between NIfTI (x,y,z) and CIL (z,y,x) conventions

### 5. Import Structure

**Before:**
```python
from src.kernel_operator import get_kernel_operator
from src.gaussian_blurring import create_gaussian_blur
from src.map_rl import MAPRL
```

**After:**
```python
from krl import (
    get_kernel_operator,
    create_gaussian_blur,
    MAPRL,
    load_image,
    save_image,
)
# Or import from submodules
from krl.operators import KernelOperator
from krl.algorithms import MAPRL
```

### 6. Documentation

- **README.md** - Completely rewritten with:
  - Clear installation instructions
  - CLI usage examples
  - Python API examples
  - Dependency information
  - Development guidelines

- **Inline docstrings** - Added to all public functions in `utils.py`

### 7. Version Control

- Added `__version__ = "0.1.0"` in `src/krl/__init__.py`
- Version also defined in `pyproject.toml`
- Accessible via `import krl; print(krl.__version__)`

### 8. Git Ignore

Updated `.gitignore` to properly exclude:
- Build artifacts (`dist/`, `*.egg-info/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Data directories (`data/`, `results/`)
- Virtual environments (`venv/`, `env/`)
- IDE files (`.vscode/`, `.idea/`)
- Jupyter notebooks (`.ipynb_checkpoints/`, `*.ipynb`)

---

## Migration Guide for Users

### For End Users

**Old workflow:**
```bash
git clone <repo>
cd KRL
python run_deconv.py --data-path data/spheres ...
```

**New workflow:**
```bash
git clone <repo>
cd KRL
pip install -e .
krl-deconv --data-path data/spheres ...
```

### For Python API Users

**Old:**
```python
import sys
sys.path.insert(0, 'src')
from kernel_operator import get_kernel_operator
```

**New:**
```python
from krl import get_kernel_operator
```

### For Developers

**Old:**
```bash
# No formal development setup
python -m pytest
```

**New:**
```bash
pip install -e ".[dev]"
black src/ tests/          # Format code
flake8 src/ tests/          # Lint
mypy src/                   # Type check
pytest --cov=krl            # Test with coverage
```

---

## Breaking Changes

### File Locations
- All source code moved from `src/<file>.py` to `src/krl/<module>/<file>.py`
- Tests remain in `tests/` but may need import updates

### CLI Entry Points
- `run_deconv.py` → `krl-deconv` command
- `run_deconv_kl_comparison.py` → `krl-compare` command
- `scripts/run_deconv_sweeps.py` → `krl-sweep` command

### Import Paths
- `from src.kernel_operator import ...` → `from krl.operators.kernel_operator import ...`
- `from src.map_rl import ...` → `from krl.algorithms.maprl import ...`

### Dependency Changes
- SIRF is now optional (users with .nii files don't need it)
- Added `nibabel` as required dependency for NIfTI support
- CIL remains required (install via conda)

---

## What Was NOT Changed

### Core Algorithms
- All operators (kernel, blurring, gradient, directional) are **unchanged**
- MAPRL algorithm implementation is **unchanged**
- Test logic and fixtures are **unchanged**

### Configuration System
- `PipelineConfig` and `KernelParameters` dataclasses are **unchanged**
- CLI argument parsing is **unchanged**

### Data Processing
- RL/KRL/DTV pipelines are **unchanged**
- Objective functions and callbacks are **unchanged**

---

## Next Steps

### Recommended Follow-Up Tasks

1. **Test the new package**
   ```bash
   pip install -e ".[dev]"
   pytest
   ```

2. **Try the CLI commands**
   ```bash
   krl-deconv --help
   krl-deconv --data-path data/spheres \
     --emission-file <your-file>.nii.gz \
     --guidance-file <your-file>.nii.gz
   ```

3. **Update existing scripts**
   - Update import statements
   - Use `krl.load_image()` instead of direct SIRF/CIL calls

4. **Consider publishing to PyPI** (future)
   ```bash
   python -m build
   twine upload dist/*
   ```

### Optional Improvements

- [ ] Add more comprehensive unit tests
- [ ] Add integration tests with real data
- [ ] Add CI/CD (GitHub Actions)
- [ ] Add pre-commit hooks
- [ ] Generate API documentation with Sphinx
- [ ] Add example notebooks
- [ ] Create Docker container for reproducibility

---

## Questions & Support

- **Package not installing?** Make sure you have CIL: `conda install -c conda-forge -c ccpi cil`
- **Import errors?** Make sure you ran `pip install -e .` from the repository root
- **Can't load .hv files?** Install SIRF or convert to NIfTI format
- **Tests failing?** Check that you have pytest: `pip install pytest`

For more help, see the updated [README.md](README.md) or open an issue on GitHub.
