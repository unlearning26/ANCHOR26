#!/bin/bash
# Tiny MMWHS context-ablation wrapper.
#
# Runs a small fixed slate of context variants end to end:
# - baseline: bbox margin 5, no outside-organ masking
# - tight: bbox margin 0, no outside-organ masking
# - tight_masked: bbox margin 0, zero outside-organ voxels inside the crop bbox
#
# Each run writes to its own Phase 2 analysis namespace so canonical MMWHS
# outputs are not overwritten.
#
# Usage:
#   bash ./scripts/run_phase2_mmwhs_context_ablation.sh
#
# Examples:
#   PHASE2_CHECKPOINT=3dinov2 \
#   PHASE2_RUNS=baseline,tight_masked \
#   bash ./scripts/run_phase2_mmwhs_context_ablation.sh
#
#   PYTHON=python \
#   PHASE2_SKIP_EXTRACTION=1 \
#   bash ./scripts/run_phase2_mmwhs_context_ablation.sh

print_usage() {
    sed -n '2,22p' "$0"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    print_usage
    exit 0
fi

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/phase2_dataset_presets.sh"

PHASE2_PRESET=${PHASE2_PRESET:-mmwhs_ct_mr_core}
phase2_init_namespace

PYTHON_BIN=${PYTHON:-python}
PHASE2_CHECKPOINT=${PHASE2_CHECKPOINT:-3dinov2}
PHASE2_FEATURE_TYPE="$(phase2_normalize_feature_type "${PHASE2_FEATURE_TYPE:-cls}")"
PHASE2_BATCH_SIZE=${PHASE2_BATCH_SIZE:-8}
PHASE2_MIN_MASK_VOXELS=${PHASE2_MIN_MASK_VOXELS:-100}
PHASE2_CROP_CACHE_WORKERS=${PHASE2_CROP_CACHE_WORKERS:-4}
PHASE2_DEVICE=${PHASE2_DEVICE:-cuda}
PHASE2_BOOTSTRAP_RESAMPLES=${PHASE2_BOOTSTRAP_RESAMPLES:-1000}
PHASE2_SEED=${PHASE2_SEED:-42}
PHASE2_SKIP_EXTRACTION=${PHASE2_SKIP_EXTRACTION:-0}
PHASE2_SKIP_EVALUATION=${PHASE2_SKIP_EVALUATION:-0}
PHASE2_NAMESPACE_PREFIX=${PHASE2_NAMESPACE_PREFIX:-${PHASE2_ANALYSIS_NAME}_ctxablation}
PHASE2_RUNS_RAW=${PHASE2_RUNS:-baseline,tight,tight_masked}
PHASE2_RUNS_RAW=${PHASE2_RUNS_RAW//,/ }
read -r -a PHASE2_RUN_LIST <<< "$PHASE2_RUNS_RAW"

REQUIRED_MODALITIES_RAW="${PHASE2_REQUIRED_MODALITIES:-${PHASE2_REQUIRED_MODALITIES_DEFAULT:-ct mr}}"
REQUIRED_MODALITIES_RAW=${REQUIRED_MODALITIES_RAW//,/ }
read -r -a REQUIRED_MODALITIES <<< "$REQUIRED_MODALITIES_RAW"

MANIFEST_VARIANT=$(phase2_manifest_variant_from_manifest "$PHASE2_MANIFEST")
CROP_SIZE=$(phase2_checkpoint_crop_size "$PHASE2_CHECKPOINT")

resolve_run_spec() {
    local run_name=$1
    case "$run_name" in
        baseline)
            printf '%s\n' "baseline 5 none"
            ;;
        tight)
            printf '%s\n' "tight 0 none"
            ;;
        tight_masked|masked)
            printf '%s\n' "tight_masked 0 zero"
            ;;
        *)
            echo "ERROR: Unknown PHASE2_RUNS entry '$run_name'. Expected baseline, tight, or tight_masked." >&2
            return 1
            ;;
    esac
}

run_one_ablation() {
    local run_label=$1
    local bbox_margin=$2
    local outside_organ_mask=$3
    local run_analysis_name="${PHASE2_NAMESPACE_PREFIX}_${run_label}"
    local output_root
    local features_npz
    local crop_cache_dir
    local metrics_path

    output_root=$(phase2_resolve_output_root "$ROOT_DIR" "$run_analysis_name" "$MANIFEST_VARIANT")
    features_npz=$(phase2_resolve_feature_npz "$ROOT_DIR" "$run_analysis_name" "$MANIFEST_VARIANT" "$PHASE2_CHECKPOINT" "$PHASE2_FEATURE_TYPE")
    crop_cache_dir=$(phase2_resolve_crop_cache_dir "$ROOT_DIR" "$run_analysis_name" "$MANIFEST_VARIANT" "$CROP_SIZE")
    metrics_path=$(phase2_resolve_primary_metrics_path "$ROOT_DIR" "$run_analysis_name" "$MANIFEST_VARIANT" "$PHASE2_CHECKPOINT" "$PHASE2_FEATURE_TYPE")

    mkdir -p "$(dirname "$features_npz")"
    mkdir -p "$(dirname "$metrics_path")"

    echo ""
    echo "=== Phase 2 MMWHS Context Ablation: $run_label ==="
    echo "Analysis namespace : $run_analysis_name"
    echo "Manifest           : $PHASE2_MANIFEST"
    echo "Checkpoint         : $PHASE2_CHECKPOINT"
    echo "Feature type       : $PHASE2_FEATURE_TYPE"
    echo "Crop size          : $CROP_SIZE"
    echo "BBox margin        : $bbox_margin"
    echo "Outside mask       : $outside_organ_mask"
    echo "Crop cache         : $crop_cache_dir"
    echo "Features NPZ       : $features_npz"
    echo "Metrics path       : $metrics_path"

    if [[ "$PHASE2_SKIP_EXTRACTION" == "1" ]]; then
        echo "[ContextAblation] Skipping extraction"
        if [[ ! -f "$features_npz" ]]; then
            echo "ERROR: PHASE2_SKIP_EXTRACTION=1 but embeddings NPZ does not exist: $features_npz" >&2
            exit 1
        fi
    else
        local extract_cmd=(
            "$PYTHON_BIN"
            "$ROOT_DIR/phase2_organ_feature_extractor.py"
            -m "$PHASE2_MANIFEST"
            -c "$PHASE2_CHECKPOINT"
            --feature-type "$PHASE2_FEATURE_TYPE"
            -o "$features_npz"
            --batch-size "$PHASE2_BATCH_SIZE"
            --bbox-margin "$bbox_margin"
            --outside-organ-mask "$outside_organ_mask"
            --min-mask-voxels "$PHASE2_MIN_MASK_VOXELS"
            --crop-cache-dir "$crop_cache_dir"
            --crop-cache-workers "$PHASE2_CROP_CACHE_WORKERS"
            --device "$PHASE2_DEVICE"
        )
        if [[ -n "${PHASE2_MAX_SAMPLES:-}" ]]; then
            extract_cmd+=(--max-samples "$PHASE2_MAX_SAMPLES")
        fi
        echo "[ContextAblation] Extracting embeddings"
        "${extract_cmd[@]}"
    fi

    if [[ "$PHASE2_SKIP_EVALUATION" == "1" ]]; then
        echo "[ContextAblation] Skipping evaluation"
        return 0
    fi

    local eval_cmd=(
        "$PYTHON_BIN"
        "$ROOT_DIR/phase2_evaluation_pipeline.py"
        -m "$PHASE2_MANIFEST"
        -a "$run_analysis_name"
        --required-modalities "${REQUIRED_MODALITIES[@]}"
        --checkpoint-name "$PHASE2_CHECKPOINT"
        --feature-type "$PHASE2_FEATURE_TYPE"
        --embeddings-npz "$features_npz"
        --bootstrap-resamples "$PHASE2_BOOTSTRAP_RESAMPLES"
        --seed "$PHASE2_SEED"
    )
    if [[ -n "${PHASE2_MAX_SAMPLES_PER_POOL:-}" ]]; then
        eval_cmd+=(--max-samples-per-pool "$PHASE2_MAX_SAMPLES_PER_POOL")
    fi
    if [[ -n "${PHASE2_MAX_QUERIES_PER_ORGAN:-}" ]]; then
        eval_cmd+=(--max-queries-per-organ "$PHASE2_MAX_QUERIES_PER_ORGAN")
    fi
    if [[ -n "${PHASE2_MAX_TARGETS_PER_ORGAN:-}" ]]; then
        eval_cmd+=(--max-targets-per-organ "$PHASE2_MAX_TARGETS_PER_ORGAN")
    fi
    if [[ -n "${PHASE2_MAX_LISI_SAMPLES_PER_GROUP:-}" ]]; then
        eval_cmd+=(--max-lisi-samples-per-group "$PHASE2_MAX_LISI_SAMPLES_PER_GROUP")
    fi

    echo "[ContextAblation] Running primary metrics"
    "${eval_cmd[@]}"
}

echo "=== Phase 2 MMWHS Context Ablation Wrapper ==="
echo "Base analysis      : $PHASE2_ANALYSIS_NAME"
echo "Namespace prefix   : $PHASE2_NAMESPACE_PREFIX"
echo "Manifest           : $PHASE2_MANIFEST"
echo "Variant            : $MANIFEST_VARIANT"
echo "Checkpoint         : $PHASE2_CHECKPOINT"
echo "Feature type       : $PHASE2_FEATURE_TYPE"
echo "Planned runs       : ${PHASE2_RUN_LIST[*]}"
echo ""

for requested_run in "${PHASE2_RUN_LIST[@]}"; do
    read -r run_label bbox_margin outside_organ_mask <<< "$(resolve_run_spec "$requested_run")"
    run_one_ablation "$run_label" "$bbox_margin" "$outside_organ_mask"
done

echo ""
echo "=== Completed MMWHS context ablation runs ==="