#!/bin/bash
# Phase 2 anchor-manifest builder.
#
# This script materializes the canonical TotalSegmenter CT plus TotalSegmenterMRI
# Phase 2 anchor manifest namespace.
#
# Usage:
#   bash ./scripts/run_phase2_build_anchor_manifest.sh
#
# Examples:
#   PYTHON=python bash ./scripts/run_phase2_build_anchor_manifest.sh
#
#   PYTHON=python bash ./scripts/run_phase2_build_anchor_manifest.sh --min-voxels 500 --min-samples-per-modality 20

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON:-python}

CMD=(
    "$PYTHON_BIN"
    "$ROOT_DIR/build_phase2_anchor_manifest.py"
)

"${CMD[@]}" "$@"