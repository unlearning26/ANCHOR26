#!/bin/bash
# Phase 2 single-organ patient-id-matched alignment launcher.
#
# This script is additive to the canonical Phase 2 workflow. It evaluates a
# single shared organ using patient-id overlap inside an existing feature bundle
# and writes a sidecar JSON artifact without modifying the canonical Phase 2
# metric files.
#
# Usage:
#   PHASE2_ANALYSIS_NAME=chaos_ct_mr \
#   PHASE2_MANIFEST=../data_manifests/phase2_cross_modality_alignment/chaos_ct_mr/core/manifest_sampled.json \
#   PHASE2_CHECKPOINT=3dinov2 \
#   PHASE2_ALIGNMENT_ORGAN=liver \
#   bash ./scripts/run_phase2_single_organ_alignment.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/phase2_dataset_presets.sh"
phase2_init_namespace

PYTHON_BIN=${PYTHON:-python}
PHASE2_CHECKPOINT=${PHASE2_CHECKPOINT:-Med3DINO_REL_c96}
PHASE2_FEATURE_TYPE="$(phase2_normalize_feature_type "${PHASE2_FEATURE_TYPE:-cls}")"
PHASE2_ALIGNMENT_ORGAN=${PHASE2_ALIGNMENT_ORGAN:-liver}
PHASE2_PAIR_ID_FIELD=${PHASE2_PAIR_ID_FIELD:-patient_id}
PHASE2_BATCH_SIZE=${PHASE2_BATCH_SIZE:-8}
PHASE2_BBOX_MARGIN=${PHASE2_BBOX_MARGIN:-5}
PHASE2_MIN_MASK_VOXELS=${PHASE2_MIN_MASK_VOXELS:-100}
PHASE2_DEVICE=${PHASE2_DEVICE:-cuda}
PHASE2_CROP_CACHE_WORKERS=${PHASE2_CROP_CACHE_WORKERS:-4}
PHASE2_SKIP_EXTRACTION=${PHASE2_SKIP_EXTRACTION:-0}
PHASE2_FORCE_EXTRACTION=${PHASE2_FORCE_EXTRACTION:-0}
REQUIRED_MODALITIES_RAW="${PHASE2_REQUIRED_MODALITIES:-${PHASE2_REQUIRED_MODALITIES_DEFAULT:-ct mr}}"
REQUIRED_MODALITIES_RAW="${REQUIRED_MODALITIES_RAW//,/ }"
read -r -a REQUIRED_MODALITIES <<< "$REQUIRED_MODALITIES_RAW"

MANIFEST_VARIANT=$(phase2_manifest_variant_from_manifest "$PHASE2_MANIFEST")
PHASE2_CROP_SIZE=$(phase2_checkpoint_crop_size "$PHASE2_CHECKPOINT")
PHASE2_CROP_CACHE_DIR=${PHASE2_CROP_CACHE_DIR:-$(phase2_resolve_crop_cache_dir "$ROOT_DIR" "$PHASE2_ANALYSIS_NAME" "$MANIFEST_VARIANT" "$PHASE2_CROP_SIZE")}
FEATURES_NPZ=${PHASE2_EMBEDDINGS_NPZ:-$(phase2_resolve_feature_npz "$ROOT_DIR" "$PHASE2_ANALYSIS_NAME" "$MANIFEST_VARIANT" "$PHASE2_CHECKPOINT" "$PHASE2_FEATURE_TYPE")}
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

CMD=(
    "$PYTHON_BIN"
    "$ROOT_DIR/phase2_single_organ_alignment.py"
    -m "$PHASE2_MANIFEST"
    -a "$PHASE2_ANALYSIS_NAME"
    --checkpoint-name "$PHASE2_CHECKPOINT"
    --feature-type "$PHASE2_FEATURE_TYPE"
    --organ "$PHASE2_ALIGNMENT_ORGAN"
    --pair-id-field "$PHASE2_PAIR_ID_FIELD"
    --required-modalities "${REQUIRED_MODALITIES[@]}"
    --embeddings-npz "$FEATURES_NPZ"
)

echo "[Phase2 single-organ] Analysis     : $PHASE2_ANALYSIS_NAME"
echo "[Phase2 single-organ] Manifest     : $PHASE2_MANIFEST"
echo "[Phase2 single-organ] Variant      : $MANIFEST_VARIANT"
echo "[Phase2 single-organ] Checkpoint   : $PHASE2_CHECKPOINT"
echo "[Phase2 single-organ] Feature type : $PHASE2_FEATURE_TYPE"
echo "[Phase2 single-organ] Crop size    : $PHASE2_CROP_SIZE"
echo "[Phase2 single-organ] Crop cache   : $PHASE2_CROP_CACHE_DIR"
echo "[Phase2 single-organ] Organ        : $PHASE2_ALIGNMENT_ORGAN"
echo "[Phase2 single-organ] Pair field   : $PHASE2_PAIR_ID_FIELD"
echo "[Phase2 single-organ] Features NPZ : $FEATURES_NPZ"

if [[ "$PHASE2_FORCE_EXTRACTION" == "1" ]]; then
    echo "[Phase2 single-organ] Force extracting organ-aware embeddings"
    "${EXTRACT_CMD[@]}"
elif [[ ! -f "$FEATURES_NPZ" ]]; then
    if [[ "$PHASE2_SKIP_EXTRACTION" == "1" ]]; then
        echo "ERROR: PHASE2_SKIP_EXTRACTION=1 but embeddings NPZ does not exist: $FEATURES_NPZ" >&2
        exit 1
    fi
    echo "[Phase2 single-organ] Missing embeddings NPZ; extracting organ-aware embeddings first"
    "${EXTRACT_CMD[@]}"
else
    echo "[Phase2 single-organ] Reusing existing embeddings"
fi

"${CMD[@]}"

echo "[Phase2 single-organ] Completed"