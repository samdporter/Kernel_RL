"""SGE array job script generation for Task 6."""

import shlex
from pathlib import Path


def write_sge_array_script(
    plan_path: Path,
    script_path: Path,
    n_runs: int,
    *,
    gpu: bool = False,
    slots: int = 1,
    python_cmd: str = "python",
) -> Path:
    if n_runs < 1 or slots < 1:
        raise ValueError("n_runs and slots must be positive")

    script_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#!/bin/bash",
        "#$ -cwd",
        "#$ -V",
        f"#$ -t 1-{n_runs}",
        f"#$ -pe smp {slots}",
    ]
    if gpu:
        lines.append("#$ -l gpu=true")

    lines.extend([
        "set -euo pipefail",
        "exec "
        f'{shlex.quote(python_cmd)} -m krl_studies.run --plan {shlex.quote(str(plan_path))} --index "$SGE_TASK_ID"',
        "",
    ])

    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)
    return script_path
