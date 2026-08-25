#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python -m pip install -q -e . -e './studies[analysis,dev]'
exec python -m krl_studies.run "$@"