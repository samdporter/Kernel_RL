# Examples

Research code and data-preparation pipelines that are **not part of the
`cil-krl` package** and are not installed with it. They show how the library
was used in the KRL PET deconvolution study and serve as reference material.

| Path | Purpose |
|------|---------|
| `pipelines/run_deconv.py` | Full RL / KRL / HKRL / DTV deconvolution experiment CLI |
| `pipelines/config.py`, `pipelines/cli_utils.py` | Argument parsing helpers for the pipeline |
| `scripts/` | One-off research scripts (benchmarks, sweeps, figures) |
| `configs/` | YAML configurations used by the batch runner |
| `data/brainweb_phantoms.py` | BrainWeb phantom generation (needs `pip install brainweb`) |
| `docker-compose*.yml`, `docker/`, `Makefile.docker`, `docker-run.sh` | Legacy Docker research environment |

## Running the pipeline example

From the repository root:

```bash
python examples/pipelines/run_deconv.py --help
```

The scripts import `krl` from your installed environment, so install the
package first (`pip install -e .` in a checkout).
