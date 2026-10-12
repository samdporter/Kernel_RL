# Examples

Runnable examples for `cil-krl`. These are maintained, plugin-facing scripts —
not historical research code.

## Prerequisite

Install CIL first (it is distributed via conda or from source, not PyPI), then
install `cil-krl` into the same environment:

```bash
pip install -e .        # from a checkout
# or
pip install cil-krl     # from PyPI
```

See the [README](../README.md#installation) for the full installation routes.

## rl_krl_dtv.py

`rl_krl_dtv.py` is a self-contained CPU comparison of four deconvolution
variants on a synthetic 3-D two-blob phantom:

- **RL** — plain Richardson-Lucy deconvolution.
- **KRL** — kernelised Richardson-Lucy, using the synthetic anatomical image as
  kernel guidance.
- **RL-dTV** — `MAPRL` with a Poisson (Kullback-Leibler) data fidelity that
  includes the PSF and a directional total-variation prior steered by the
  anatomical gradient, stepped with a sensitivity-normalised EM preconditioner
  `P(x) = (x + 1e-6) / max(Bᵀ1, 1e-6)`, a suggested step of 1.0 and Armijo
  backtracking on every update.
- **L-BFGS-B** — `LBFGSBOptimizer` (SciPy) minimising the *same* Poisson
  fidelity and DTV prior as RL-dTV, with non-negativity enforced.

Run it from the repository root:

```bash
python examples/rl_krl_dtv.py
```

For a quick check on a tiny volume:

```bash
python examples/rl_krl_dtv.py --shape 6 8 8 --iterations 2
```

The script runs a default budget of 50 iterations per method. It prints the
objective history and final NRMSE for each variant together with the iteration
count actually taken, RL-dTV's final step size, and SciPy's L-BFGS-B
diagnostics (`nfev`, `njev`, `success`, `status`, `message`). The reported
iteration count can be lower than the budget: L-BFGS-B stops at SciPy's own
convergence test and reports that `nit`, not the budget it was given. It writes
no files and produces no plots.

The per-method objective histories are not cross-comparable between methods
that optimise different objectives (RL/KRL vs the RL-dTV/L-BFGS-B pair).
NRMSE is computed against the same ground truth with the same normalisation for
all methods, so the reported NRMSE values are directly comparable on this
phantom. Everything reported is illustrative only: a lower NRMSE for one
variant here is not a general performance ranking.
