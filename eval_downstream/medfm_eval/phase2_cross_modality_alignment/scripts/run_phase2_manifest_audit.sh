#!/bin/bash
# Phase 2 manifest audit launcher.
#
# This script runs only the manifest validation and cohort summary stage.
#
# It intentionally does not run primary metrics. Per-checkpoint metric
# evaluation must go through run_phase2_extract_and_eval.sh so the resulting
# phase2_primary_metrics.json files stay in canonical per-checkpoint folders.
#
# Usage:
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core ./scripts/run_phase2_manifest_audit.sh
#
# Examples:
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core ./scripts/run_phase2_manifest_audit.sh
#
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/phase2_dataset_presets.sh"
phase2_init_namespace

PYTHON_BIN=${PYTHON:-python}
EXTRA_ARGS=("$@")
REQUIRED_MODALITIES_RAW="${PHASE2_REQUIRED_MODALITIES:-${PHASE2_REQUIRED_MODALITIES_DEFAULT:-ct mr}}"
REQUIRED_MODALITIES_RAW="${REQUIRED_MODALITIES_RAW//,/ }"
read -r -a REQUIRED_MODALITIES <<< "$REQUIRED_MODALITIES_RAW"

if [[ -n "${PHASE2_EMBEDDINGS_NPZ:-}" ]]; then
    echo "ERROR: run_phase2_manifest_audit.sh is audit-only and does not accept PHASE2_EMBEDDINGS_NPZ." >&2
    echo "Use bash ./scripts/run_phase2_extract_and_eval.sh for per-checkpoint metrics, or call phase2_evaluation_pipeline.py directly with --checkpoint-name." >&2
    exit 1
fi

CMD=(
    "$PYTHON_BIN"
    "$ROOT_DIR/phase2_evaluation_pipeline.py"
    -m "$PHASE2_MANIFEST"
    -a "$PHASE2_ANALYSIS_NAME"
    --required-modalities "${REQUIRED_MODALITIES[@]}"
)

"${CMD[@]}" "${EXTRA_ARGS[@]}"