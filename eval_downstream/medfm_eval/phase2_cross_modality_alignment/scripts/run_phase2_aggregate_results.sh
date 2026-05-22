#!/bin/bash
# Phase 2 aggregated-results refresher.
#
# This script rebuilds the root-level comparison CSV and JSON from the
# authoritative per-checkpoint metric files under results/<checkpoint>/.
#
# Usage:
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core bash ./scripts/run_phase2_aggregate_results.sh
#
# Examples:
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core \
#   bash ./scripts/run_phase2_aggregate_results.sh
#
#   PHASE2_RESULTS_DIR=./outputs_phase2/totalsegmenter_ct_mr_anchor/phase2/core/results \
#   bash ./scripts/run_phase2_aggregate_results.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON:-python}
PHASE2_FEATURE_TYPE=${PHASE2_FEATURE_TYPE:-cls}

if [[ -z "${PHASE2_RESULTS_DIR:-}" ]]; then
    source "$ROOT_DIR/scripts/phase2_dataset_presets.sh"
    phase2_init_namespace
    MANIFEST_VARIANT=$(phase2_manifest_variant_from_manifest "$PHASE2_MANIFEST")
    PHASE2_RESULTS_DIR="$(phase2_resolve_output_root "$ROOT_DIR" "$PHASE2_ANALYSIS_NAME" "$MANIFEST_VARIANT")/results"
fi

CMD=(
    "$PYTHON_BIN"
    "$ROOT_DIR/aggregate_phase2_checkpoint_results.py"
    --results-dir "$PHASE2_RESULTS_DIR"
    --feature-type "$PHASE2_FEATURE_TYPE"
)

"${CMD[@]}"