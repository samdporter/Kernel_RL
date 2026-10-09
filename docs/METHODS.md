# Methods

All methods are used through the Python API (`import krl`); the package has no
command-line interface. The snippets below assume the variables built in the
[top-level README quickstart](../README.md): `geometry`, `emission`, `mr_image`,
`observed` and `blur_op`.

## Richardson-Lucy (RL)

Classical expectation-maximization deconvolution of the point spread function (PSF),
with no anatomical guidance.

```python
from krl import RichardsonLucy

rl = RichardsonLucy(
    initial_estimate=observed,
    blurring_operator=blur_op,
    observed_data=observed,
)
rl.run(iterations=50)
reconstruction = rl.get_output()
```

Each update is `x <- x * (A^T (y / A x)) / A^T 1` with `A` the blur operator,
followed by a clamp to non-negative values. `A` is the blur alone (with a kernel
operator it becomes `blur ∘ kernel`, which is KRL). The reported objective
(`rl.loss`) is the KL divergence between `A x` and the observed data `y`.

More iterations sharpen the reconstruction and also amplify noise; choose the
iteration count with a callback (for example `NRMSECallback` against a phantom).

**When to use:** baseline comparison, or when no anatomical image is available.

---

## Kernelised RL (KRL)

KRL replaces the forward operator with `blur ∘ kernel`: before blurring, each
voxel is replaced by a weighted average of the voxels in a `num_neighbours³`
neighbourhood. Weights decay with the anatomical-intensity difference between
the voxel and its neighbour, so emission is mixed within similar tissue and
preserved across anatomical boundaries.

```python
from krl import RichardsonLucy, get_kernel_operator

kernel_op = get_kernel_operator(geometry, backend="auto")
kernel_op.set_anatomical_image(mr_image)

rl = RichardsonLucy(
    initial_estimate=observed,
    blurring_operator=blur_op,
    observed_data=observed,
    kernel_operator=kernel_op,
)
rl.run(iterations=50)
reconstruction = rl.get_output()   # applies the kernel to the latent image
```

The anatomical image must be aligned with the emission image: identical grid and
voxel-by-voxel correspondence, with no resampling inside the operator.
Neighbourhoods are extended by mirror reflection at the volume boundary.

Kernel parameters and their defaults (from `krl.DEFAULT_PARAMETERS`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `num_neighbours` | `5` | Odd neighbourhood width per axis, giving `n³` candidate neighbours |
| `sigma_anat` | `0.1` | Gaussian width of the anatomical-intensity weight (larger = more mixing across anatomy) |
| `sigma_dist` | `10000` | Gaussian width of the distance weight, used only when `distance_weighting=True` |
| `sigma_emission` | `0.1` | Gaussian width of the emission weight, used only when `hybrid=True` |
| `normalize_features` | `True` | Divide the anatomical image by its standard deviation when it is set |
| `normalize_kernel` | `True` | Normalise each output voxel by the sum of its kernel weights |
| `use_mask` | `True` | Keep only the `mask_k` most similar neighbours per voxel |
| `mask_k` | `20` | Neighbourhood entries kept by the mask (capped at `n³`) |
| `recalc_mask` | `False` | Recompute the mask on every forward call |
| `distance_weighting` | `False` | Add the distance-decay weight to the anatomical weight |
| `hybrid` | `False` | Enable HKRL (emission-based weights) |

**When to use:** when you have a co-registered anatomical image (e.g. T1 MRI).

---

## Hybrid KRL (HKRL)

HKRL adds an emission term to the kernel weights: each neighbour is weighted by
anatomy *and* by how close its current estimate is to the centre voxel
(`hybrid=True`, `sigma_emission`), so the kernel can adapt when the anatomy does
not match the emission.

```python
from krl import RichardsonLucy, get_kernel_operator

kernel_op = get_kernel_operator(geometry, hybrid=True, sigma_emission=0.5)
kernel_op.set_anatomical_image(mr_image)

rl = RichardsonLucy(
    initial_estimate=observed,
    blurring_operator=blur_op,
    observed_data=observed,
    kernel_operator=kernel_op,
    freeze_iteration=10,
)
rl.run(iterations=50)
reconstruction = rl.get_output()
```

Restrictions:

- Single-channel 3-D volumes only.
- The emission reference is taken from the first `direct()` call, so `direct()`
  must run before `adjoint()`; the adjoint raises if the reference or the
  normalisation map has not been initialised yet.
- While unfrozen, the reference is refreshed on every forward call, and
  `RichardsonLucy` recomputes the sensitivity `A^T 1` each iteration because the
  operator changes.

Freezing semantics:

- `freeze_iteration=N` (passed to `RichardsonLucy`) freezes the kernel **after
  the N-th completed update**. The reference captured is the estimate seen by
  the first forward call after that update.
- From then on `freeze_emission_kernel` is set, forward and adjoint share the
  same frozen reference, the sensitivity is recomputed once more, and the
  operator stays constant for the remaining iterations.
- `freeze_iteration=0` (default) never freezes.
- Freezing is a modelling choice that stops the operator from changing under the
  iteration; no convergence rate is claimed for it.

**When to use:** when emission and anatomy only partially match.

---

## Directional Total Variation (DTV)

DTV is a regularisation term for `MAPRL` (or `LBFGSBOptimizer`), built from CIL's
gradient operators rather than from a `krl` operator:

1. `GradientOperator` (CIL) computes the image gradient.
2. `DirectionalOperator` derives a direction field `xi` from the *anatomical*
   gradient, scaled by `1 / sqrt(||grad anatomy||^2 + eta^2)` with `eta` a robust
   scale taken from percentiles of `||grad anatomy||`. Because of the positive
   `eta` term `xi` is generally shorter than unit length, and the operator
   applies a regularized (soft) projection that attenuates — rather than removes —
   the component of its input along `xi`: `v -> v - gamma * xi * (xi . v)`.
3. The resulting block is penalised with CIL's `SmoothMixedL21Norm`.

```python
from cil.optimisation.functions import (
    L2NormSquared,
    OperatorCompositionFunction,
    SmoothMixedL21Norm,
)
from cil.optimisation.operators import CompositionOperator, GradientOperator

from krl import MAPRL, DirectionalOperator

gradient = GradientOperator(geometry, method="forward", bnd_cond="Neumann")
directional = CompositionOperator(DirectionalOperator(gradient.direct(mr_image)), gradient)
prior = 0.01 * OperatorCompositionFunction(
    SmoothMixedL21Norm(epsilon=observed.max() * 1e-2), directional
)

maprl = MAPRL(
    initial_estimate=observed.clone(),
    data_fidelity=L2NormSquared(b=observed),
    prior=prior,
    step_size=1e-3,
    relaxation_eta=0.0,
    initial_line_search=False,
    armijo_iterations=0,
)
maprl.run(iterations=50)
reconstruction = maprl.x
```

Since `xi` points across anatomical boundaries, the soft projection attenuates
the component of `grad x` in that direction before the penalty is evaluated, so
gradient components across anatomical boundaries are penalised less than
components along them: intensity jumps that sit on anatomical boundaries survive
while variation along them is smoothed. The scalar in front of the prior
(`0.01` above) is the regularisation strength: larger values smooth more.

**When to use:** stronger, anatomy-aware regularisation than the implicit
smoothing of the kernel.

---

## MAP-RL and L-BFGS-B

- **`MAPRL`** is a CIL `Algorithm` minimising `data_fidelity + prior`. It exposes
  `loss` like any other CIL algorithm and accepts CIL callbacks in `run()`. Its
  step size comes from Armijo line searches scheduled by the `armijo_*` options
  and decays with `relaxation_eta`; an optional preconditioner can be refreshed
  on its own schedule:
  - `initial_line_search=True` (default) runs a search at construction to choose
    the starting `step_size`; pass `False` to keep the `step_size` you supply.
  - In-run searches cover a leading block of updates: `armijo_update_initial`
    updates when that is positive (capped at `armijo_iterations` when that is
    positive too), otherwise `armijo_iterations` updates.
  - With the defaults (`armijo_update_initial=0`, `armijo_iterations=25`,
    `armijo_update_interval=10`) that block is updates 1-25, one search per
    update, and no search runs after update 25: the periodic branch applies only
    once the leading block has ended *and* the update count is still
    `<= armijo_iterations`, which cannot both hold with these values.
  - `armijo_update_interval` therefore only has an effect when
    `armijo_update_initial` is positive and below `armijo_iterations`; searches
    then also run every `armijo_update_interval` updates, up to
    `armijo_iterations`.
  - Once the leading block is past, the step size decays with `relaxation_eta`.
- **`LBFGSBOptimizer`** minimises the same objective with SciPy's
  `L-BFGS-B`. It is a wrapper around `scipy.optimize.minimize`, **not** a CIL
  `Algorithm` subclass: it has its own `run(iterations, callbacks, verbose)`,
  enforces non-negativity by default (`LBFGSBOptions(enforce_non_negativity=True)`),
  and its callbacks receive the optimiser object rather than a CIL algorithm.

## Comparison

| Method | Anatomical guidance | Regularisation | Implemented by |
|--------|--------------------|----------------|----------------|
| RL | No | None | `RichardsonLucy` |
| KRL | Yes | Implicit (kernel) | `RichardsonLucy` + `get_kernel_operator` |
| HKRL | Yes (adaptive) | Implicit (kernel) | `RichardsonLucy` + `hybrid=True` kernel |
| DTV | Yes | Explicit (TV prior) | `MAPRL` or `LBFGSBOptimizer` + CIL gradient operators |
