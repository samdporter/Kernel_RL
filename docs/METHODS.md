# Methods

## Richardson-Lucy (RL)

Classical expectation-maximization deconvolution for PET images.

**How it works:**
- Iteratively deconvolves the point spread function (PSF)
- No anatomical guidance
- Can amplify noise with too many iterations

**When to use:** Baseline comparison, or when no anatomical image available.

```bash
python run_deconv.py --enable-rl --rl-iterations-standard 50
```

---

## Kernelised RL (KRL)

RL with anatomically-weighted smoothing kernels.

**How it works:**
- Uses anatomical image (e.g., T1 MRI) to create adaptive kernels
- Smooths within similar tissue regions, not across boundaries
- Preserves edges that match anatomy

**When to use:** When you have co-registered anatomical images.

```bash
python run_deconv.py \
  --enable-krl \
  --guidance-file T1.hv \
  --kernel-num-neighbours 9 \
  --kernel-sigma-anat 1.0
```

**Key parameters:**
- `--kernel-num-neighbours`: Kernel size (default: 9)
- `--kernel-sigma-anat`: Anatomical sensitivity (lower = stricter boundaries)

---

## Hybrid KRL (HKRL)

KRL that combines emission and anatomical features with kernel freezing for convergence.

**How it works:**
- Weights kernels using both PET emission and anatomical image
- Adapts to cases where anatomy doesn't perfectly match emission
- More flexible than pure KRL
- **Freezes the emission reference after a specified iteration** to ensure convergence
  - Before freezing: Kernel adapts to current emission estimate
  - After freezing: Kernel becomes constant, enabling objective function convergence
  - Sensitivity is automatically recomputed each iteration until freezing

**When to use:** When emission and anatomy have partial mismatch.

```bash
python run_deconv.py \
  --enable-krl \
  --kernel-hybrid \
  --kernel-sigma-emission 1.0 \
  --freeze-iteration 2
```

**Key parameters:**
- `--kernel-sigma-emission`: Emission sensitivity (how much to weight emission features)
- `--freeze-iteration`: Iteration at which to freeze the emission kernel (default: 0, no freezing)
  - Setting to 1-2 is recommended for convergence
  - After this iteration, the kernel operator becomes constant

**Important notes:**
- HKRL requires `--freeze-iteration > 0` for proper convergence
- Without freezing, the kernel keeps changing, preventing the objective from converging
- The frozen reference is set at the END of the freeze iteration
- Both forward and adjoint operations use the same frozen reference for consistency

---

## Directional Total Variation (DTV)

MAP-RL with directional TV regularization.

**How it works:**
- Regularization term that penalizes gradients NOT aligned with anatomy
- Preserves edges parallel to anatomical boundaries
- Smooths orthogonal to anatomy

**When to use:** Strong regularization with anatomical edge preservation.

```bash
python run_deconv.py \
  --enable-drl \
  --dtv-iterations 100 \
  --alpha 0.5
```

**Key parameters:**
- `--alpha`: Regularization strength (higher = more smoothing)
- `--dtv-iterations`: Number of iterations

---

## Comparison

| Method | Anatomical Guidance | Regularization | Speed |
|--------|-------------------|----------------|-------|
| RL | No | None | Fast |
| KRL | Yes | Implicit (kernel) | Medium |
| HKRL | Yes (adaptive) | Implicit (kernel) | Medium |
| DTV | Yes | Explicit (TV prior) | Slow |

## Run All Methods

```bash
python run_deconv.py \
  --data-path data/spheres \
  --emission-file OSEM.hv \
  --guidance-file T1.hv \
  --enable-rl \
  --enable-krl \
  --enable-drl
```

Results saved to `results/` with comparison plots.
