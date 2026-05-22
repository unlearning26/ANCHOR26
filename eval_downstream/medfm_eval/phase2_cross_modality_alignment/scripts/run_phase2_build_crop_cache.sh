#!/bin/bash
# Phase 2 crop-cache builder.
#
# This script precomputes reusable organ crops for a single Phase 2 manifest and
# crop size. The resulting cache can be reused by every checkpoint that shares
# that crop size, which avoids repeated image loading, mask loading, bounding
# box computation, and resize work inside each checkpoint job.
#
# Usage:
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core PHASE2_CROP_SIZE=96 ./scripts/run_phase2_build_crop_cache.sh
#
# Examples:
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core \
#   PHASE2_CROP_SIZE=96 \
#   ./scripts/run_phase2_build_crop_cache.sh
#
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core \
#   PHASE2_CROP_SIZE=112 \
#   PHASE2_CROP_CACHE_WORKERS=8 \
#   ./scripts/run_phase2_build_crop_cache.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/phase2_dataset_presets.sh"
phase2_init_namespace

PYTHON_BIN=${PYTHON:-python}
PHASE2_CHECKPOINT=${PHASE2_CHECKPOINT:-Med3DINO_REL_c96}
PHASE2_CROP_SIZE=${PHASE2_CROP_SIZE:-}
PHASE2_BBOX_MARGIN=${PHASE2_BBOX_MARGIN:-5}
PHASE2_MIN_MASK_VOXELS=${PHASE2_MIN_MASK_VOXELS:-100}
PHASE2_CROP_CACHE_WORKERS=${PHASE2_CROP_CACHE_WORKERS:-4}
PHASE2_CROP_CACHE_DIR=${PHASE2_CROP_CACHE_DIR:-}
PHASE2_OVERWRITE_CROP_CACHE=${PHASE2_OVERWRITE_CROP_CACHE:-0}

if [[ -z "$PHASE2_CROP_SIZE" ]]; then
    PHASE2_CROP_SIZE=$(phase2_checkpoint_crop_size "$PHASE2_CHECKPOINT")
fi

MANIFEST_VARIANT=$(phase2_manifest_variant_from_manifest "$PHASE2_MANIFEST")
if [[ -z "$PHASE2_CROP_CACHE_DIR" ]]; then
    PHASE2_CROP_CACHE_DIR=$(phase2_resolve_crop_cache_dir "$ROOT_DIR" "$PHASE2_ANALYSIS_NAME" "$MANIFEST_VARIANT" "$PHASE2_CROP_SIZE")
fi

CMD=(
    "$PYTHON_BIN"
    "$ROOT_DIR/phase2_organ_feature_extractor.py"
    -m "$PHASE2_MANIFEST"
    -c "$PHASE2_CHECKPOINT"
    --crop-size "$PHASE2_CROP_SIZE"
    --bbox-margin "$PHASE2_BBOX_MARGIN"
    --min-mask-voxels "$PHASE2_MIN_MASK_VOXELS"
    --crop-cache-dir "$PHASE2_CROP_CACHE_DIR"
    --crop-cache-workers "$PHASE2_CROP_CACHE_WORKERS"
    --build-crop-cache-only
)

if [[ -n "${PHASE2_MAX_SAMPLES:-}" ]]; then
    CMD+=(--max-samples "$PHASE2_MAX_SAMPLES")
fi

if [[ "$PHASE2_OVERWRITE_CROP_CACHE" == "1" ]]; then
    CMD+=(--overwrite-crop-cache)
fi

echo "[Phase2] Analysis        : $PHASE2_ANALYSIS_NAME"
echo "[Phase2] Manifest        : $PHASE2_MANIFEST"
echo "[Phase2] Variant         : $MANIFEST_VARIANT"
echo "[Phase2] Crop size       : $PHASE2_CROP_SIZE"
echo "[Phase2] Crop cache dir  : $PHASE2_CROP_CACHE_DIR"
echo "[Phase2] Cache workers   : $PHASE2_CROP_CACHE_WORKERS"
echo "[Phase2] Building crop cache"

"${CMD[@]}"

echo "[Phase2] Crop cache ready"


# PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core PHASE2_CROP_SIZE=112 PHASE2_CROP_CACHE_WORKERS=16 nohup ./scripts/run_phase2_build_crop_cache.sh > outputs_phase2/totalsegmenter_ct_mr_anchor/phase2/core/logs/crop_cache/crop112.log 2>&1 &
# PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core PHASE2_CROP_SIZE=96 PHASE2_CROP_CACHE_WORKERS=16 nohup ./scripts/run_phase2_build_crop_cache.sh > outputs_phase2/totalsegmenter_ct_mr_anchor/phase2/core/logs/crop_cache/crop96.log 2>&1 &





