"""Report CLI for Task 5: aggregate, figures, tables, and all."""

import argparse
import sys
from pathlib import Path

from krl_studies.analysis.report import (
    aggregate_results,
    generate_figures,
    generate_tables,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="krl_studies.report",
        description="Aggregate results, generate figures and tables.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Ingest results and compute aggregate CSVs")
    p_agg.add_argument("--results", type=Path, required=True, help="Results root directory")
    p_agg.add_argument("--out", type=Path, required=True, help="Output directory")
    p_agg.add_argument(
        "--fixed-iteration",
        type=int,
        default=10,
        help="Fixed iteration for fixed selection (default: 10)",
    )

    # figures
    p_fig = sub.add_parser("figures", help="Generate publication figures from analysis CSVs")
    p_fig.add_argument("--analysis", type=Path, required=True, help="Analysis directory (output of aggregate)")
    p_fig.add_argument("--out", type=Path, required=True, help="Output directory for figures")

    # tables
    p_tab = sub.add_parser("tables", help="Generate best-result tables from analysis CSVs")
    p_tab.add_argument("--analysis", type=Path, required=True, help="Analysis directory (output of aggregate)")
    p_tab.add_argument("--out", type=Path, required=True, help="Output directory for tables")

    # all
    p_all = sub.add_parser("all", help="Run aggregate, figures, and tables in sequence")
    p_all.add_argument("--results", type=Path, required=True, help="Results root directory")
    p_all.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output root directory (creates aggregate/, figures/, tables/)",
    )
    p_all.add_argument(
        "--fixed-iteration",
        type=int,
        default=10,
        help="Fixed iteration for fixed selection (default: 10)",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.command == "aggregate":
        aggregate_results(args.results, args.out, args.fixed_iteration)
    elif args.command == "figures":
        generate_figures(args.analysis, args.out)
    elif args.command == "tables":
        generate_tables(args.analysis, args.out)
    elif args.command == "all":
        # aggregate
        agg_dir = Path(args.out) / "aggregate"
        aggregate_results(args.results, agg_dir, args.fixed_iteration)
        # figures
        fig_dir = Path(args.out) / "figures"
        generate_figures(agg_dir, fig_dir)
        # tables
        tab_dir = Path(args.out) / "tables"
        generate_tables(agg_dir, tab_dir)
    else:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
