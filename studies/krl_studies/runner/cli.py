"""CLI: python -m krl_studies.run --scenario FILE [--dry-run] [--force] [--only SUBSTR]
       python -m krl_studies.run --plan FILE --index INT [--out PATH] [--force]"""

from __future__ import annotations

import argparse
from pathlib import Path

from krl_studies.config import load_scenario
from krl_studies.runner.execute import execute_run
from krl_studies.runner.expand import expand_scenario
from krl_studies.runner.plan import read_run_plan


def _execute_one(run, force: bool, out_root: Path | None = None) -> int:
    """Execute a single run, returning 0 on success, 1 on failure."""
    if out_root is not None:
        run = run.replace(out_root=Path(out_root))
    try:
        out = execute_run(run, force=force)
        print(f"    done -> {out}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"    FAILED: {exc}")
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="krl-studies.run", description="Run krl benchmark scenarios"
    )

    # Mutually exclusive: scenario vs plan
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", help="path to scenario YAML")
    group.add_argument("--plan", help="path to JSONL run plan")

    # Scenario mode options
    parser.add_argument("--dry-run", action="store_true", help="list expanded runs and exit (scenario mode)")
    parser.add_argument("--only", default=None, help="substring filter on run ids (scenario mode)")

    # Plan mode options
    parser.add_argument("--index", type=int, help="1-based task index (plan mode, required with --plan)")
    parser.add_argument("--out", type=Path, help="override output root (plan mode)")

    # Common options
    parser.add_argument("--force", action="store_true", help="re-run even if completion marker exists")

    args = parser.parse_args(argv)

    if args.plan is not None:
        # Plan mode
        if args.index is None:
            parser.error("--plan requires --index")
        runs = read_run_plan(args.plan)
        if not 1 <= args.index <= len(runs):
            parser.error(f"--plan requires --index between 1 and {len(runs)}")
        run = runs[args.index - 1]
        if args.out is not None:
            run = run.replace(out_root=Path(args.out))
        return _execute_one(run, force=args.force, out_root=None)

    # Scenario mode
    scenario = load_scenario(args.scenario)
    runs = expand_scenario(scenario)
    if args.only:
        runs = [r for r in runs if args.only in r.run_id]

    if args.dry_run:
        for r in runs:
            print(r.run_id)
        print(f"-- {len(runs)} run(s)")
        return 0

    failures = []
    for i, run in enumerate(runs, start=1):
        target = Path(run.out_root) / run.run_id / ".done"
        if target.exists() and not args.force:
            print(f"[{i}/{len(runs)}] {run.run_id}  skip (marker present)")
            continue
        print(f"[{i}/{len(runs)}] {run.run_id}")
        failures.append((run.run_id, 1)) if _execute_one(run, args.force) else None

    for run_id, _ in failures:
        print(f"FAILURE {run_id}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
