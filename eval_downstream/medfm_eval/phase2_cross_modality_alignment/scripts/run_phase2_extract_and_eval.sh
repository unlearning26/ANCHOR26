#!/bin/bash
# Phase 2 extraction plus evaluation launcher.
#
# This script runs one checkpoint end to end:
# 1. extract organ-aware embeddings for one feature family into the canonical Phase 2 features tree
# 2. run the Phase 2 primary metrics using the produced NPZ
#
# Usage:
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core ./scripts/run_phase2_extract_and_eval.sh
#
# Examples:
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core \
#   PHASE2_CHECKPOINT=Med3DINO_REL_c96 \
#   ./scripts/run_phase2_extract_and_eval.sh
#
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core \
#   PHASE2_CHECKPOINT=Med3DINO_REL_c96 \
#   PHASE2_SKIP_EXTRACTION=1 \
#   ./scripts/run_phase2_extract_and_eval.sh
#
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core \
#   PHASE2_CHECKPOINT=Med3DINO_REL_c96 \
#   PHASE2_MAX_SAMPLES=16 \
#   PHASE2_MAX_SAMPLES_PER_POOL=8 \
#   PHASE2_MAX_QUERIES_PER_ORGAN=2 \
#   PHASE2_MAX_TARGETS_PER_ORGAN=2 \
#   PHASE2_BOOTSTRAP_RESAMPLES=32 \
#   ./scripts/run_phase2_extract_and_eval.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/phase2_dataset_presets.sh"
phase2_init_namespace

PYTHON_BIN=${PYTHON:-python}
PHASE2_CHECKPOINT=${PHASE2_CHECKPOINT:-Med3DINO_REL_c96}
PHASE2_BATCH_SIZE=${PHASE2_BATCH_SIZE:-8}
PHASE2_BBOX_MARGIN=${PHASE2_BBOX_MARGIN:-5}
PHASE2_MIN_MASK_VOXELS=${PHASE2_MIN_MASK_VOXELS:-100}
PHASE2_DEVICE=${PHASE2_DEVICE:-cuda}
PHASE2_CROP_CACHE_WORKERS=${PHASE2_CROP_CACHE_WORKERS:-4}
PHASE2_BOOTSTRAP_RESAMPLES=${PHASE2_BOOTSTRAP_RESAMPLES:-1000}
PHASE2_SEED=${PHASE2_SEED:-42}
PHASE2_SKIP_EXTRACTION=${PHASE2_SKIP_EXTRACTION:-0}
PHASE2_SKIP_EVALUATION=${PHASE2_SKIP_EVALUATION:-${PHASE2_SKIP_EVALUATION_DEFAULT:-0}}
PHASE2_FEATURE_TYPE="$(phase2_normalize_feature_type "${PHASE2_FEATURE_TYPE:-cls}")"
REQUIRED_MODALITIES_RAW="${PHASE2_REQUIRED_MODALITIES:-${PHASE2_REQUIRED_MODALITIES_DEFAULT:-ct mr}}"
REQUIRED_MODALITIES_RAW="${REQUIRED_MODALITIES_RAW//,/ }"
read -r -a REQUIRED_MODALITIES <<< "$REQUIRED_MODALITIES_RAW"

MANIFEST_VARIANT=$(phase2_manifest_variant_from_manifest "$PHASE2_MANIFEST")
PHASE2_CROP_SIZE=$(phase2_checkpoint_crop_size "$PHASE2_CHECKPOINT")
PHASE2_CROP_CACHE_DIR=${PHASE2_CROP_CACHE_DIR:-$(phase2_resolve_crop_cache_dir "$ROOT_DIR" "$PHASE2_ANALYSIS_NAME" "$MANIFEST_VARIANT" "$PHASE2_CROP_SIZE")}
FEATURES_NPZ=$(phase2_resolve_feature_npz "$ROOT_DIR" "$PHASE2_ANALYSIS_NAME" "$MANIFEST_VARIANT" "$PHASE2_CHECKPOINT" "$PHASE2_FEATURE_TYPE")
mkdir -p "$(dirname "$FEATURES_NPZ")"

EXTRACT_CMD=(
    "$PYTHON_BIN"
    "$ROOT_DIR/phase2_organ_feature_extractor.py"
    -m "$PHASE2_MANIFEST"
    -c "$PHASE2_CHECKPOINT"
    --feature-type "$PHASE2_FEATURE_TYPE"
    -o "$FEATURES_NPZ"
    --batch-size "$PHASE2_BATCH_SIZE"
    --bbox-margin "$PHASE2_BBOX_MARGIN"
    --min-mask-voxels "$PHASE2_MIN_MASK_VOXELS"
    --crop-cache-dir "$PHASE2_CROP_CACHE_DIR"
    --crop-cache-workers "$PHASE2_CROP_CACHE_WORKERS"
    --device "$PHASE2_DEVICE"
)

if [[ -n "${PHASE2_MAX_SAMPLES:-}" ]]; then
    EXTRACT_CMD+=(--max-samples "$PHASE2_MAX_SAMPLES")
fi

EVAL_CMD=(
    "$PYTHON_BIN"
    "$ROOT_DIR/phase2_evaluation_pipeline.py"
    -m "$PHASE2_MANIFEST"
    -a "$PHASE2_ANALYSIS_NAME"
    --required-modalities "${REQUIRED_MODALITIES[@]}"
    --checkpoint-name "$PHASE2_CHECKPOINT"
    --feature-type "$PHASE2_FEATURE_TYPE"
    --embeddings-npz "$FEATURES_NPZ"
    --bootstrap-resamples "$PHASE2_BOOTSTRAP_RESAMPLES"
    --seed "$PHASE2_SEED"
)

if [[ -n "${PHASE2_MAX_SAMPLES_PER_POOL:-}" ]]; then
    EVAL_CMD+=(--max-samples-per-pool "$PHASE2_MAX_SAMPLES_PER_POOL")
fi

if [[ -n "${PHASE2_MAX_QUERIES_PER_ORGAN:-}" ]]; then
    EVAL_CMD+=(--max-queries-per-organ "$PHASE2_MAX_QUERIES_PER_ORGAN")
fi

if [[ -n "${PHASE2_MAX_TARGETS_PER_ORGAN:-}" ]]; then
    EVAL_CMD+=(--max-targets-per-organ "$PHASE2_MAX_TARGETS_PER_ORGAN")
fi

echo "[Phase2] Analysis      : $PHASE2_ANALYSIS_NAME"
echo "[Phase2] Manifest      : $PHASE2_MANIFEST"
echo "[Phase2] Variant       : $MANIFEST_VARIANT"
echo "[Phase2] Checkpoint    : $PHASE2_CHECKPOINT"
echo "[Phase2] Feature type  : $PHASE2_FEATURE_TYPE"
echo "[Phase2] Modalities    : ${REQUIRED_MODALITIES[*]}"
echo "[Phase2] Crop size     : $PHASE2_CROP_SIZE"
echo "[Phase2] Crop cache    : $PHASE2_CROP_CACHE_DIR"
echo "[Phase2] Features NPZ  : $FEATURES_NPZ"

if [[ "$PHASE2_SKIP_EXTRACTION" == "1" ]]; then
    echo "[Phase2] Skipping extraction and reusing existing embeddings"
    if [[ ! -f "$FEATURES_NPZ" ]]; then
        echo "ERROR: PHASE2_SKIP_EXTRACTION=1 but embeddings NPZ does not exist: $FEATURES_NPZ" >&2
        exit 1
    fi
else
    echo "[Phase2] Extracting organ-aware $PHASE2_FEATURE_TYPE embeddings"
    "${EXTRACT_CMD[@]}"
fi

if [[ "$PHASE2_SKIP_EVALUATION" == "1" ]]; then
    echo "[Phase2] Skipping Phase 2 primary metrics for this namespace"
    exit 0
fi

echo "[Phase2] Running primary metrics"
"${EVAL_CMD[@]}"

echo "[Phase2] Completed"