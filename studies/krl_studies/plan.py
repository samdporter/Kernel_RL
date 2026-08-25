"""Plan CLI for Task 6: JSONL run-plan serialization and SGE generation."""

import argparse
import sys
from pathlib import Path

from krl_studies.cluster.sge import write_sge_array_script
from krl_studies.config import expand_scenario, load_scenario
from krl_studies.runner.plan import write_run_plan


def _plan_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="krl_studies.plan",
        description="Generate JSONL run plans from scenarios, optionally create SGE scripts.",
    )
    parser.add_argument("--scenario", type=Path, required=True, help="Path to scenario YAML")
    parser.add_argument("--out", type=Path, required=True, help="Output plan file (JSONL)")
    parser.add_argument("--sge", type=Path, help="Also write SGE array script")
    parser.add_argument("--gpu", action="store_true", help="Request GPU in SGE script")
    parser.add_argument("--slots", type=int, default=1, help="SMP slots per task (default: 1)")
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    runs = expand_scenario(scenario)

    # Write JSONL plan
    write_run_plan(runs, args.out)
    print(f"Wrote {len(runs)} runs to {args.out}")

    # Optionally write SGE script
    if args.sge is not None:
        write_sge_array_script(
            plan_path=args.out,
            script_path=args.sge,
            n_runs=len(runs),
            gpu=args.gpu,
            slots=args.slots,
        )
        print(f"Wrote SGE script to {args.sge}")

    return 0


if __name__ == "__main__":
    sys.exit(_plan_cli())
