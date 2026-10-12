# Examples

Research code and data-preparation pipelines for the KRL PET deconvolution
experiments on this branch. They are **not part of the `cil-krl` package** and
are not installed with it — treat them as this branch's experiment tooling
rather than as runnable, turnkey examples. A few example entry points have
targeted smoke tests under `tests/test_examples_*.py`; the remaining scripts are
not covered by those tests. They also expect the original study's data layout,
configuration files and extra dependencies (matplotlib, PyTorch, BrainWeb, ...).

| Path | Purpose |
|------|---------|
| `pipelines/run_deconv.py` | Research CLI for the RL / KRL / HKRL / DTV experiments (the installed package itself has no CLI) |
| `pipelines/config.py` | Argument parsing and configuration helpers for the pipeline |
| `scripts/` | One-off research scripts (benchmarks, sweeps, figures) |
| `configs/` | YAML configurations used by the batch runner |
| `data/install_brainweb_helper.py` | Installs the optional `brainweb` dependency used by the phantom scripts |
| `data/gpu_utils.py` | GPU selection helpers used by the scripts |
| `requirements*.txt` | Dependency lists used by the legacy Docker environment |
| `docker-compose*.yml`, `docker/`, `Makefile.docker`, `docker-run.sh` | Legacy Docker research environment |

## Pipeline entry point

For reference, the entry point is:

```bash
python examples/pipelines/run_deconv.py --help
```

Expect to adapt paths, data files and options before anything runs. The scripts
import `krl` from your installed environment, so install the package first
(`pip install -e .` in a checkout) — see the top-level
[README](../README.md).

## BrainWeb known limitations

`data/brainweb_phantoms.py` (used by `scripts/brainweb_batch_runner.py` and the
configs in `configs/`) has two known limitations. These outputs are historical
examples, **not** validated for quantitative physical-geometry studies.

- **No resampling to `voxel_size_mm`.** The arrays stay on BrainWeb's native
  mMR grid (~2.03, 2.09, 2.09 mm). `voxel_size_mm` (default 2 mm) does not
  resample them; it only controls the output NIfTI spacing, the PSF voxel-size
  conversion for the added blur, and custom-lesion coordinates. Built-in
  BrainWeb lesion geometry therefore can be inconsistent with the exported
  spacing.
- **Seeding does not make a run reproducible end to end.** The local
  `default_rng(seed)` controls only the noise added by the helper, and the
  `brainweb.seed()` call controls built-in lesion randomness. Upstream PET/MR
  generation is not controlled by the local generator and may reuse previously
  generated noisy volumes, so identical arguments/seed do **not** guarantee
  identical complete datasets across cold caches or differing cache histories;
  upstream generation is not reproducible across caches.
