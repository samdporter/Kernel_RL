# KRL Quick Start Guide

Get up and running with KRL in 5 minutes!

## 1. Installation

```bash
# Navigate to the repository
cd KRL

# Install CIL (required)
conda install -c conda-forge -c ccpi cil

# Install KRL package
pip install -e .

# Optional: Install accelerators for better performance
pip install -e ".[accelerators]"
```

## 2. Verify Installation

```bash
# Check that CLI commands are available
krl-deconv --help
krl-compare --help
krl-sweep --help

# Check installed version
python -c "import krl; print(krl.__version__)"
```

## 3. Run Example Script

The easiest way to get started is with the example script:

```bash
# Run the spheres phantom reconstruction
python example_spheres_reconstruction.py
```

This will:
- Load spheres phantom data from `data/spheres/`
- Run 4 reconstruction methods (RL, KRL, HKRL, DTV)
- Save results to `results/spheres_example/`
- Generate comparison plots

**Note:** Make sure you have the spheres data in `data/spheres/`. The script expects:
- `data/spheres/OSEM_b1337_n5.hv` (or `.nii.gz`) - PET emission
- `data/spheres/T1_b1337.hv` (or `.nii.gz`) - Anatomical guidance

## 4. Using the Python API

```python
from krl import (
    get_kernel_operator,
    create_gaussian_blur,
    load_image,
    save_image,
)

# Load images
emission = load_image("data/emission.nii.gz")
anatomy = load_image("data/anatomy.nii.gz")

# Create kernel operator with anatomical guidance
kernel_op = get_kernel_operator(emission, backend="numba")
kernel_op.set_anatomical_image(anatomy)
kernel_op.set_parameters({
    "num_neighbours": 9,
    "sigma_anat": 1.0,
    "use_mask": True,
    "hybrid": False,
})

# Apply kernel smoothing
smoothed = kernel_op.direct(emission)

# Save result
save_image(smoothed, "output/smoothed.nii.gz")
```

## 5. Using the CLI

### Basic Reconstruction

```bash
krl-deconv \
  --data-path data/spheres \
  --emission-file OSEM.nii.gz \
  --guidance-file T1.nii.gz \
  --enable-rl \
  --enable-krl \
  --fwhm 6.0 6.0 6.0
```

### Customize Parameters

```bash
krl-deconv \
  --data-path data/spheres \
  --emission-file OSEM.nii.gz \
  --guidance-file T1.nii.gz \
  --enable-krl \
  --enable-drl \
  --backend numba \
  --rl-iterations-kernel 100 \
  --dtv-iterations 100 \
  --kernel-num-neighbours 9 \
  --kernel-sigma-anat 1.0 \
  --kernel-sigma-dist 3.0 \
  --alpha 0.1 \
  --show-plots
```

### Run Hyperparameter Sweep

```bash
# Dry run to see what will be executed
krl-sweep --dry-run --datasets spheres

# Run sweep on spheres dataset with KRL pipeline
krl-sweep --datasets spheres --pipelines krl --max-runs 10
```

## 6. File Formats

KRL supports multiple image formats:

### NIfTI (Recommended)
- Extensions: `.nii`, `.nii.gz`
- No additional dependencies (uses nibabel)
- Standard medical imaging format

```python
from krl import load_image, save_image

# Load NIfTI
img = load_image("data/image.nii.gz")

# Save NIfTI
save_image(img, "output/result.nii.gz")
```

### SIRF Interfile (.hv)
- Requires SIRF installation
- Native STIR/SIRF format
- Better for PET-specific workflows

```python
# Load .hv (requires SIRF)
img = load_image("data/image.hv")

# Save .hv (requires SIRF ImageData)
save_image(img, "output/result.hv")
```

## 7. Common Issues

### "CIL not found"
```bash
conda install -c conda-forge -c ccpi cil
```

### "nibabel not found"
```bash
pip install nibabel
```

### "Backend not available"
If you see warnings about backends:
```bash
# For faster CPU performance
pip install numba

# For GPU acceleration
pip install torch
```

### "File not found" errors
Make sure your data is in the correct location:
```bash
ls data/spheres/
# Should show: OSEM_b1337_n5.hv, T1_b1337.hv (or .nii.gz versions)
```

## 8. Next Steps

- **Read the full README**: [README.md](README.md)
- **Explore the code**: Check out `src/krl/operators/` for operator implementations
- **Run tests**: `pytest` to ensure everything works
- **Customize parameters**: See [src/krl/cli/config.py](src/krl/cli/config.py) for all options
- **Try your own data**: Replace spheres data with your own PET/MRI images

## 9. Getting Help

- Check the documentation: [README.md](README.md)
- Look at examples: [example_spheres_reconstruction.py](example_spheres_reconstruction.py)
- Review the refactoring notes: [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
- Open an issue on GitHub

## 10. Quick Reference

### Key Parameters

**Kernel Operator (KRL/HKRL):**
- `num_neighbours`: Size of local neighbourhood (e.g., 9 = 9×9×9)
- `sigma_anat`: Anatomical similarity weight (lower = stricter)
- `sigma_dist`: Spatial distance weight
- `sigma_emission`: Emission similarity (hybrid mode only)
- `use_mask`: Enable k-NN masking for efficiency
- `hybrid`: Enable hybrid mode (mix emission + anatomy)

**Reconstruction:**
- `rl_iterations_standard`: Iterations for standard RL
- `rl_iterations_kernel`: Iterations for KRL/HKRL
- `dtv_iterations`: Iterations for DTV
- `alpha`: DTV regularization strength (higher = more smoothing)
- `lbfgs_max_linesearch`: Max line-search steps for DTV optimiser
- `lbfgs_ftol` / `lbfgs_gtol`: Function and gradient tolerances for DTV optimiser
- `fwhm`: PSF full-width at half-maximum (mm)

**Backends:**
- `auto`: Auto-select best available
- `numba`: Fast CPU (recommended if available)
- `python`: Pure Python (slowest, always available)
- `torch`: GPU acceleration (for blurring)

---

Happy reconstructing! 🎉
