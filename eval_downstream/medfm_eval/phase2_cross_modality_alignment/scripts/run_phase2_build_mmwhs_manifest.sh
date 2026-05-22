#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON:-python}

CMD=(
    "$PYTHON_BIN"
    "$ROOT_DIR/build_phase2_mmwhs_manifest.py"
)

"${CMD[@]}" "$@"