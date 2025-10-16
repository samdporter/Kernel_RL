# SIRF vs CIL: Do You Need SIRF?

## Short Answer

**No, you do NOT need SIRF for most KRL functionality!**

The refactored KRL package now works with **CIL + NIfTI files** as the primary workflow. SIRF is **optional** and only needed for specific use cases.

---

## What Works WITHOUT SIRF

✅ **Full reconstruction pipeline with CIL + NIfTI:**

### Core Operators (All Work)
- ✅ **Kernel Operator** (`KernelOperator`) - Anatomical-guided kernel
- ✅ **Gaussian Blurring** (`GaussianBlurringOperator`) - PSF convolution
- ✅ **Gradient** (`Gradient`) - Finite-difference gradient
- ✅ **Directional Operator** (`DirectionalOperator`) - Directional TV

### Algorithms (All Work)
- ✅ **Richardson-Lucy (RL)** - Standard deconvolution
- ✅ **Kernelised RL (KRL)** - With anatomical guidance
- ✅ **Hybrid KRL (HKRL)** - Mixing emission and anatomy
- ✅ **MAP-RL with DTV** - Directional total variation

### File I/O
- ✅ Load NIfTI files (`.nii`, `.nii.gz`)
- ✅ Save NIfTI files (`.nii`, `.nii.gz`)
- ✅ Convert between formats

### Utilities
- ✅ All visualization and plotting
- ✅ Objective function tracking
- ✅ Parameter sweeps
- ✅ CLI commands (`krl-deconv`, `krl-compare`, `krl-sweep`)

---

## What REQUIRES SIRF

❌ **SIRF is only needed for:**

1. **Interfile/STIR format (.hv files)**
   - Loading `.hv` files
   - Saving `.hv` files
   - If you only have `.hv` files, you need SIRF OR convert to NIfTI

2. **SIRF-specific PET features** (advanced, not part of core KRL)
   - Raw sinogram data
   - Forward/backward projection operators
   - Scatter/attenuation correction
   - SIRF's OSEM/OSMAPOSL implementations
   - `TruncateToCylinderProcessor` (if you want to use it)

3. **Legacy scripts in root directory**
   - `run_deconv.py` (old version, use `krl-deconv` instead)
   - `run_deconv_kl_comparison.py` (old version)
   - Some scripts in `scripts/` folder

---

## Recommended Workflow (No SIRF Needed!)

### 1. Installation

```bash
# Install CIL (the main requirement)
conda install -c conda-forge -c ccpi cil

# Install KRL
pip install -e .

# Optional: accelerators for performance
pip install -e ".[accelerators]"
```

**That's it! No SIRF needed.**

### 2. Prepare Your Data

Convert any `.hv` files to NIfTI:

```python
# If you have SIRF installed temporarily, you can convert:
import sirf.STIR as pet
import nibabel as nib

# Load .hv
img = pet.ImageData("data/image.hv")
data = img.as_array()

# Save as NIfTI
nii = nib.Nifti1Image(data.transpose(2, 1, 0), affine=np.eye(4))
nib.save(nii, "data/image.nii.gz")
```

Or use STIR tools:
```bash
# Convert .hv to NIfTI using STIR utilities (if you have STIR installed)
stir_img_to_nifti input.hv output.nii.gz
```

### 3. Run Reconstructions

```bash
# Using CLI
krl-deconv \
  --data-path data/spheres \
  --emission-file OSEM.nii.gz \
  --guidance-file T1.nii.gz \
  --enable-krl \
  --enable-drl
```

```python
# Using Python API
from krl import load_image, get_kernel_operator, create_gaussian_blur

# Load NIfTI files
emission = load_image("data/emission.nii.gz")
anatomy = load_image("data/anatomy.nii.gz")

# Create operators (works without SIRF!)
kernel_op = get_kernel_operator(emission, backend="numba")
kernel_op.set_anatomical_image(anatomy)

# Run reconstruction...
```

---

## Architecture: How SIRF was Made Optional

### Before Refactoring
```
run_deconv.py
  ↓
import sirf.STIR as pet  ← REQUIRED!
  ↓
pet.ImageData(...)  ← Hard dependency
```

### After Refactoring
```
krl-deconv (CLI)
  ↓
krl/cli/run_deconv.py
  ↓
try:
    import sirf.STIR as pet  ← Optional
except ImportError:
    from cil.framework import ImageData as pet  ← Fallback
```

### Key Changes Made

1. **Removed hard SIRF import from MAPRL**
   - Was: `from sirf.STIR import TruncateToCylinderProcessor`
   - Now: CIL Algorithm base class only (SIRF import was unused anyway)

2. **Added NIfTI support**
   - `load_image()` and `save_image()` in `krl/utils.py`
   - Uses `nibabel` for NIfTI I/O
   - Handles axis transposes automatically

3. **Try/except for SIRF everywhere**
   - CLI entry points check for SIRF, fall back to CIL
   - Clear error messages if neither is available

4. **CIL as primary framework**
   - All operators work with CIL's `ImageData`
   - All algorithms use CIL's optimization framework
   - SIRF's `ImageData` is compatible (duck-typed)

---

## Comparison Table

| Feature | CIL + NIfTI | SIRF Required |
|---------|-------------|---------------|
| Load NIfTI files | ✅ Yes | ❌ No |
| Load .hv files | ❌ No | ✅ Yes |
| Kernel operator | ✅ Yes | ❌ No |
| Gaussian blurring | ✅ Yes | ❌ No |
| Gradient operator | ✅ Yes | ❌ No |
| Richardson-Lucy | ✅ Yes | ❌ No |
| KRL/HKRL | ✅ Yes | ❌ No |
| MAP-RL + DTV | ✅ Yes | ❌ No |
| Save results | ✅ NIfTI | ✅ .hv (needs SIRF) |
| CLI commands | ✅ Yes | ❌ No |
| Python API | ✅ Yes | ❌ No |
| Raw sinogram data | ❌ No | ✅ Yes |
| SIRF projectors | ❌ No | ✅ Yes |

---

## When Should You Install SIRF?

**You ONLY need SIRF if:**

1. ✅ You have data in `.hv` format and can't convert to NIfTI
2. ✅ You're working with raw PET sinogram data
3. ✅ You need SIRF's forward/backward projection operators
4. ✅ You want to use SIRF's built-in OSEM/OSMAPOSL
5. ✅ You're collaborating with others who use SIRF format

**You DON'T need SIRF if:**

1. ❌ You have NIfTI files (`.nii`, `.nii.gz`)
2. ❌ You just want to run KRL reconstruction
3. ❌ You're only using the core KRL operators
4. ❌ You want the simplest installation

---

## Testing Without SIRF

The test suite works WITHOUT SIRF:

```bash
# Run tests (no SIRF needed)
pytest

# Tests use mock SIRF objects (see tests/conftest.py)
# All core functionality is tested
```

The `tests/conftest.py` file creates lightweight stubs for SIRF when it's not available, so tests run on any machine.

---

## Migration Path

### If you currently use SIRF:

**Option 1: Convert data to NIfTI (recommended)**
```bash
# Convert your .hv files to .nii.gz
# Then use KRL with just CIL (simpler!)
```

**Option 2: Keep using SIRF**
```bash
# KRL still works with SIRF!
# Just make sure SIRF is installed
# Use .hv files as before
```

### If you're starting fresh:

**Just use NIfTI + CIL:**
```bash
conda install -c conda-forge -c ccpi cil
pip install -e .
# You're done!
```

---

## Conclusion

**The refactored KRL package is SIRF-optional!**

- ✅ **Core functionality**: Works with CIL + NIfTI
- ✅ **All algorithms**: RL, KRL, HKRL, DTV
- ✅ **All operators**: Kernel, blur, gradient, directional
- ✅ **CLI and Python API**: Fully functional
- ✅ **Easier installation**: No SIRF compilation needed
- ✅ **Better portability**: NIfTI is a standard format

SIRF is only needed for `.hv` file I/O and advanced PET-specific features. For most users, **CIL + NIfTI is sufficient**!

---

## Quick Start (No SIRF)

```bash
# 1. Install
conda install -c conda-forge -c ccpi cil
pip install -e .

# 2. Prepare data (NIfTI format)
# emission.nii.gz - Your PET emission data
# anatomy.nii.gz - Your anatomical guidance (T1 MRI)

# 3. Run reconstruction
krl-deconv \
  --data-path data \
  --emission-file emission.nii.gz \
  --guidance-file anatomy.nii.gz \
  --enable-krl

# Done! No SIRF needed.
```
