#!/usr/bin/env bash
# Quarantine invalidated artifacts from 2026-08-27 readiness review
# Moves known-invalid local directories into a dated, gitignored quarantine folder.
# Idempotent: safe to run multiple times.

set -euo pipefail

QUARANTINE_DIR="invalidated_2026-08-27"
SOURCES=(
  "results"
  "plans"
  "data/brainweb/subject_04"
  "data/patients/MK-H001"
)

mkdir -p "$QUARANTINE_DIR"

for src in "${SOURCES[@]}"; do
  if [[ -e "$src" ]]; then
    echo "Moving $src -> $QUARANTINE_DIR/"
    mv "$src" "$QUARANTINE_DIR/"
  else
    echo "Skipping $src (not found)"
  fi
done

# Create README explaining why these are quarantined
cat > "$QUARANTINE_DIR/README.md" <<'EOF'
# Quarantined Artifacts (2026-08-27)

These directories were moved here during the Phase 0+2+3 readiness review because they contain invalidated artifacts:

- **results/** — Partial SIRF/quick-sim runs with incorrect configuration
- **plans/** — Generated from buggy scenarios (wrong voxel spacing, missing ROIs)
- **data/brainweb/subject_04/** — BrainWeb preparation with incorrect voxel spacing (2mm vs required 1mm)
- **data/patients/MK-H001/** — Patient data lacking ROI contours and governance approval

All of these were already gitignored. They are preserved here for audit trail only.
Do not use these artifacts for any analysis or publication figures.
EOF

echo "Done. Contents of $QUARANTINE_DIR:"
ls -la "$QUARANTINE_DIR"