#!/bin/bash
# Module 3 anatomical generalization batch launcher.

set -euo pipefail

print_usage() {
        cat <<'EOF'
Module 3 anatomical generalization batch launcher

Required inputs:
    Either:
        MODULE3_PRESET=<canonical_preset>
    Or all of:
        MODULE3_ANALYSIS_NAME=<dataset_namespace>
        MODULE3_SOURCE_MANIFEST=<phase2_manifest_path>
        MODULE3_MANIFEST=<module3_manifest_path>

Optional controls:
    MODULE3_FEATURE_TYPES="cls avg_pool multilayer"
    MODULE3_CHECKPOINTS="Med3DINO_REL_c96 Med3DINO_SA_c96"
    MODULE3_MIN_HOLDOUT_ORGANS=2
    MODULE3_MIN_SAMPLES_PER_MODALITY=5
    MODULE3_FEW_SHOT_SUPPORT_PER_MODALITY=2
    MODULE3_FEW_SHOT_QUERY_PER_MODALITY=3
    MODULE3_FEW_SHOT_SEEDS="42 123 456"
    MODULE3_SEED=42
    MODULE3_SKIP_MANIFEST_BUILD=1
    REQUIRE_COMPLETE=1
    MODULE3_SUMMARY_OUTPUT=<summary_json_path>
    MODULE3_CSV_OUTPUT=<summary_csv_path>
    PYTHON=<python_executable>
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    print_usage
    exit 0
fi

cd "$(dirname "$0")"
ROOT_DIR="$(cd .. && pwd)"

source "$ROOT_DIR/scripts/module3_dataset_presets.sh"
module3_init_namespace

PYTHON_BIN="${PYTHON:-python}"
MANIFEST_VARIANT="$(module3_manifest_variant_from_manifest "$MODULE3_SOURCE_MANIFEST")"
OUTPUT_ROOT="$(module3_resolve_output_root "$ROOT_DIR" "$MODULE3_ANALYSIS_NAME" "$MANIFEST_VARIANT")"
REQUIRED_MODALITIES_RAW="${MODULE3_REQUIRED_MODALITIES:-${MODULE3_REQUIRED_MODALITIES_DEFAULT:-ct mr}}"
REQUIRED_MODALITIES_RAW="${REQUIRED_MODALITIES_RAW//,/ }"
read -r -a REQUIRED_MODALITIES <<< "$REQUIRED_MODALITIES_RAW"
LOGDIR="$OUTPUT_ROOT/logs/module3_anatomical_generalization"
mkdir -p "$LOGDIR"

if [[ "${MODULE3_SKIP_MANIFEST_BUILD:-0}" != "1" ]]; then
    bash "$ROOT_DIR/scripts/run_module3_build_manifest.sh"
fi

FEATURE_TYPE_ARRAY=()
if [[ -n "${MODULE3_FEATURE_TYPES:-}" ]]; then
    FEATURE_TYPES_RAW="${MODULE3_FEATURE_TYPES//,/ }"
    read -r -a FEATURE_TYPE_ARRAY <<< "$FEATURE_TYPES_RAW"
else
    mapfile -t FEATURE_TYPE_ARRAY < <(module3_all_feature_types)
fi
if [[ "${#FEATURE_TYPE_ARRAY[@]}" -eq 0 ]]; then
    echo "ERROR: MODULE3_FEATURE_TYPES must contain at least one feature family." >&2
    exit 1
fi

CHECKPOINTS_RAW="${MODULE3_CHECKPOINTS:-}"
CHECKPOINT_ARGS=()
if [[ -n "$CHECKPOINTS_RAW" ]]; then
    CHECKPOINTS_RAW="${CHECKPOINTS_RAW//,/ }"
    read -r -a CHECKPOINT_ARRAY <<< "$CHECKPOINTS_RAW"
    if [[ "${#CHECKPOINT_ARRAY[@]}" -gt 0 ]]; then
        CHECKPOINT_ARGS=(--checkpoints "${CHECKPOINT_ARRAY[@]}")
    fi
fi

REQUIRE_COMPLETE_FLAG=()
if [[ "${REQUIRE_COMPLETE:-0}" == "1" ]]; then
    REQUIRE_COMPLETE_FLAG=(--require-complete)
fi

SUMMARY_OUTPUT="${MODULE3_SUMMARY_OUTPUT:-$OUTPUT_ROOT/results/module3_anatomical_generalization_summary.json}"
CSV_OUTPUT="${MODULE3_CSV_OUTPUT:-$OUTPUT_ROOT/results/module3_anatomical_generalization_summary.csv}"

echo "================================================================="
echo "Module 3 Anatomical Generalization"
echo "================================================================="
echo "Analysis:          $MODULE3_ANALYSIS_NAME"
echo "Source manifest:   $MODULE3_SOURCE_MANIFEST"
echo "Module 3 manifest: $MODULE3_MANIFEST"
echo "Variant:           $MANIFEST_VARIANT"
echo "Modalities:        ${REQUIRED_MODALITIES[*]}"
echo "Feature types:     ${FEATURE_TYPE_ARRAY[*]}"
if [[ "${#CHECKPOINT_ARGS[@]}" -eq 0 ]]; then
    echo "Checkpoints:       all registered checkpoints"
else
    echo "Checkpoints:       ${CHECKPOINT_ARRAY[*]}"
fi
echo "Summary output:    $SUMMARY_OUTPUT"
echo "CSV output:        $CSV_OUTPUT"
echo "Require complete:  ${REQUIRE_COMPLETE:-0}"
echo "Few-shot support:  ${MODULE3_FEW_SHOT_SUPPORT_PER_MODALITY:-2} per organ-modality"
echo "Few-shot query cap:${MODULE3_FEW_SHOT_QUERY_PER_MODALITY:-balanced remainder}"
echo "Few-shot seeds:    ${MODULE3_FEW_SHOT_SEEDS:-42 123 456}"
echo "Log directory:     $LOGDIR"
echo "================================================================="

FEW_SHOT_SEEDS_RAW="${MODULE3_FEW_SHOT_SEEDS:-42 123 456}"
FEW_SHOT_SEEDS_RAW="${FEW_SHOT_SEEDS_RAW//,/ }"
read -r -a FEW_SHOT_SEED_ARRAY <<< "$FEW_SHOT_SEEDS_RAW"

FEW_SHOT_QUERY_ARGS=()
if [[ -n "${MODULE3_FEW_SHOT_QUERY_PER_MODALITY:-}" ]]; then
    FEW_SHOT_QUERY_ARGS=(--few-shot-query-per-modality "${MODULE3_FEW_SHOT_QUERY_PER_MODALITY}")
fi

"$PYTHON_BIN" "$ROOT_DIR/module3_evaluation_pipeline.py" \
    --analysis-name "$MODULE3_ANALYSIS_NAME" \
    --manifest "$MODULE3_MANIFEST" \
    --manifest-variant "$MANIFEST_VARIANT" \
    --required-modalities "${REQUIRED_MODALITIES[@]}" \
    --feature-types "${FEATURE_TYPE_ARRAY[@]}" \
    --summary-output "$SUMMARY_OUTPUT" \
    --csv-output "$CSV_OUTPUT" \
    --min-holdout-organs "${MODULE3_MIN_HOLDOUT_ORGANS:-2}" \
    --min-samples-per-modality "${MODULE3_MIN_SAMPLES_PER_MODALITY:-5}" \
    --few-shot-support-per-modality "${MODULE3_FEW_SHOT_SUPPORT_PER_MODALITY:-2}" \
    "${FEW_SHOT_QUERY_ARGS[@]}" \
    --few-shot-seeds "${FEW_SHOT_SEED_ARRAY[@]}" \
    --seed "${MODULE3_SEED:-42}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${REQUIRE_COMPLETE_FLAG[@]}"

echo "================================================================="
echo "Module 3 anatomical generalization completed at $(date)"
echo "================================================================="
