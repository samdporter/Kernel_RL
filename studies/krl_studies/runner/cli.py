"""CLI: python -m krl_studies.run --scenario FILE [--dry-run] [--force] [--only SUBSTR]"""

from __future__ import annotations

import argparse
from pathlib import Path

from krl_studies.config import load_scenario
from krl_studies.runner.execute import execute_run
from krl_studies.runner.expand import expand_scenario


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="krl-studies.run", description="Run krl benchmark scenarios"
    )
    parser.add_argument("--scenario", required=True, help="path to scenario YAML")
    parser.add_argument("--dry-run", action="store_true", help="list expanded runs and exit")
    parser.add_argument("--force", action="store_true", help="re-run even if completion marker exists")
    parser.add_argument("--only", default=None, help="substring filter on run ids")
    args = parser.parse_args(argv)

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
        try:
            out = execute_run(run, force=args.force)
            print(f"    done -> {out}")
        except Exception as exc:  # noqa: BLE001 - isolate the failure, keep running remaining runs
            failures.append((run.run_id, exc))
            print(f"    FAILED: {exc}")
    for run_id, exc in failures:
        print(f"FAILURE {run_id}: {exc}")
    return 1 if failures else 0
