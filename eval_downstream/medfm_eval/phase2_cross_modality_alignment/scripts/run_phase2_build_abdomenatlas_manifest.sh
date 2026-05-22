#!/bin/bash
# AbdomenAtlas Phase 2 manifest builder.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON:-python}"
PHASE2_MANIFEST_VARIANT=${PHASE2_MANIFEST_VARIANT:-core}
PHASE2_SUBSET_SEED=${PHASE2_SUBSET_SEED:-42}
PHASE2_SUBSET_CASE_COUNT=${PHASE2_SUBSET_CASE_COUNT:-}
if [[ "$PHASE2_MANIFEST_VARIANT" == "subset_1k" && -z "$PHASE2_SUBSET_CASE_COUNT" ]]; then
    PHASE2_SUBSET_CASE_COUNT=1000
fi

CMD=(
    "$PYTHON_BIN"
    "$ROOT_DIR/build_phase2_abdomenatlas_manifest.py"
    --manifest-variant "$PHASE2_MANIFEST_VARIANT"
    --min-voxels "${PHASE2_MIN_MASK_VOXELS:-100}"
    --min-samples-per-organ "${PHASE2_MIN_SAMPLES_PER_ORGAN:-20}"
)

if [[ -n "${PHASE2_SOURCE_MANIFEST:-}" ]]; then
    CMD+=(--phase1-manifest "$PHASE2_SOURCE_MANIFEST")
fi

if [[ -n "${PHASE2_SEMANTIC_CACHE:-}" ]]; then
    CMD+=(--semantic-cache "$PHASE2_SEMANTIC_CACHE")
fi

if [[ -n "$PHASE2_SUBSET_CASE_COUNT" ]]; then
    CMD+=(--subset-case-count "$PHASE2_SUBSET_CASE_COUNT" --subset-seed "$PHASE2_SUBSET_SEED")
fi

echo "================================================================="
echo "Phase 2 AbdomenAtlas Manifest Builder"
echo "================================================================="
echo "Output namespace : abdomenatlas/$PHASE2_MANIFEST_VARIANT"
echo "Source manifest  : ${PHASE2_SOURCE_MANIFEST:-Phase 1 sampled AbdomenAtlas manifest}"
echo "Semantic cache   : ${PHASE2_SEMANTIC_CACHE:-Phase 1 sampled semantic-label cache}"
echo "Min voxels       : ${PHASE2_MIN_MASK_VOXELS:-100}"
echo "Min organ support: ${PHASE2_MIN_SAMPLES_PER_ORGAN:-20}"
if [[ -n "$PHASE2_SUBSET_CASE_COUNT" ]]; then
echo "Subset cases     : $PHASE2_SUBSET_CASE_COUNT"
echo "Subset seed      : $PHASE2_SUBSET_SEED"
fi
echo "================================================================="

"${CMD[@]}"

echo "================================================================="
echo "AbdomenAtlas Phase 2 manifest build completed at $(date)"
echo "================================================================="