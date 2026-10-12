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

`rl_krl_dtv.py` is a self-contained CPU comparison of three deconvolution
variants on a synthetic 3-D two-blob phantom:

- **RL** — plain Richardson-Lucy deconvolution.
- **KRL** — kernelised Richardson-Lucy, using the synthetic anatomical image as
  kernel guidance.
- **RL-dTV** — `MAPRL` with a Poisson (Kullback-Leibler) data fidelity that
  includes the PSF and a directional total-variation prior steered by the
  anatomical gradient.

Run it from the repository root:

```bash
python examples/rl_krl_dtv.py
```

For a quick check on a tiny volume:

```bash
python examples/rl_krl_dtv.py --shape 6 8 8 --iterations 2
```

The script prints the objective history and final NRMSE for each variant. It
writes no files and produces no plots. The NRMSE values are illustrative only —
the three objectives are not cross-comparable, so a lower NRMSE for one variant
on this phantom is not a general ranking.
