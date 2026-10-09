"""CLI: reproducible BrainWeb preparation.

``python -m krl_studies.prepare --out-root data/brainweb [--subjects ...]``

Freezes the installed BrainWeb subject inventory and writes the exact list plus
the incomplete-subject report to ``<out-root>/inventory.json`` and
``<out-root>/prep_report.json``. Incomplete subjects are reported, never dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="krl-studies.prepare",
        description="Prepare reproducible BrainWeb subjects and freeze the inventory.",
    )
    parser.add_argument("--out-root", type=Path, required=True, help="Dataset root to prepare into")
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated subject ids/filenames; defaults to the full frozen inventory",
    )
    parser.add_argument("--seed", type=int, default=None, help="Structural-texture seed")
    parser.add_argument("--inventory", type=Path, default=None, help="Inventory JSON output path")
    parser.add_argument("--report", type=Path, default=None, help="Preparation report JSON output path")
    args = parser.parse_args(argv)

    from krl_studies.datasets.brainweb import (
        DEFAULT_PREP_SEED,
        GUIDANCE_T1_ABSENT,
        GUIDANCE_T1_INJECTION_MULTIPLIER,
        GUIDANCE_T1_PRESENT,
        inventory_digest,
        prepare_subjects,
        subject_inventory,
    )

    inventory = subject_inventory()
    if args.subjects is None:
        subjects: list[str] = list(inventory)
    else:
        subjects = [item.strip() for item in args.subjects.split(",") if item.strip()]
    seed = DEFAULT_PREP_SEED if args.seed is None else args.seed

    args.out_root.mkdir(parents=True, exist_ok=True)
    # The campaign is tumour-injected only: every subject gets injected PET, the
    # four planned masks and both T1 variants. There is deliberately no way to
    # prepare a tumour-free subject into the campaign root.
    report = prepare_subjects(subjects, args.out_root, tumour=True, seed=seed)
    digest = inventory_digest(inventory)
    report["t1_variants"] = {
        "absent": GUIDANCE_T1_ABSENT,
        "present": GUIDANCE_T1_PRESENT,
        "injection_multiplier": float(GUIDANCE_T1_INJECTION_MULTIPLIER),
    }

    inventory_path = args.inventory or (args.out_root / "inventory.json")
    report_path = args.report or (args.out_root / "prep_report.json")
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps({"inventory": list(inventory), "count": len(inventory), "digest": digest}, indent=2)
    )
    report["inventory_digest"] = digest
    report_path.write_text(json.dumps(report, indent=2))

    print(f"inventory: {len(inventory)} subject(s) frozen -> {inventory_path}")
    print(
        f"prepared: {len(report['prepared'])}; incomplete: {len(report['incomplete'])} "
        f"-> {report_path}"
    )
    print(
        f"T1 variants per subject: {GUIDANCE_T1_ABSENT} + {GUIDANCE_T1_PRESENT} "
        f"(present scales the mask union by {GUIDANCE_T1_INJECTION_MULTIPLIER:g}x)"
    )
    for subject_id, error in report["incomplete"].items():
        print(f"  INCOMPLETE {subject_id}: {error}")
    return 1 if report["incomplete"] else 0


if __name__ == "__main__":
    sys.exit(main())
