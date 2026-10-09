# Examples

Historical research code and data-preparation pipelines from the KRL PET
deconvolution study. They are **not part of the `cil-krl` package**, are not
installed with it, and are not maintained or tested — treat them as a reference
for how the library was used in the original experiments rather than as runnable,
turnkey examples. They also expect the original study's data layout,
configuration files and extra dependencies (matplotlib, PyTorch, BrainWeb, ...).

| Path | Purpose |
|------|---------|
| `pipelines/run_deconv.py` | Research CLI for the RL / KRL / HKRL / DTV experiments (the installed package itself has no CLI) |
| `pipelines/config.py`, `pipelines/cli_utils.py` | Argument parsing helpers for the pipeline |
| `scripts/` | One-off research scripts (benchmarks, sweeps, figures) |
| `configs/` | YAML configurations used by the batch runner |
| `data/install_brainweb_helper.py` | Installs the optional `brainweb` dependency used by the phantom scripts |
| `data/gpu_utils.py` | GPU selection helpers used by the scripts |
| `requirements*.txt` | Dependency lists used by the legacy Docker environment |
| `docker-compose*.yml`, `docker/`, `Makefile.docker`, `docker-run.sh` | Legacy Docker research environment |

## Pipeline entry point

For reference, the historical entry point was:

```bash
python examples/pipelines/run_deconv.py --help
```

Expect to adapt paths, data files and options before anything runs. The scripts
import `krl` from your installed environment, so install the package first
(`pip install -e .` in a checkout) — see the top-level
[README](../README.md).
