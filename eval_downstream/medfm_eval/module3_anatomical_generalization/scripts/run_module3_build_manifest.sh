#!/bin/bash
# Module 3 manifest builder.

set -euo pipefail

print_usage() {
        cat <<'EOF'
Module 3 manifest builder

Required inputs:
    Either:
        MODULE3_PRESET=<canonical_preset>
    Or all of:
        MODULE3_ANALYSIS_NAME=<dataset_namespace>
        MODULE3_SOURCE_MANIFEST=<upstream_organ_manifest_path>
        MODULE3_MANIFEST=<module3_manifest_path>

Optional controls:
    MODULE3_REQUIRED_MODALITIES="ct mr"
    MODULE3_MIN_HOLDOUT_ORGANS=2
    MODULE3_MIN_SAMPLES_PER_MODALITY=5
    MODULE3_OUTPUT_MANIFEST_DIR=<explicit_output_dir>
    PYTHON=<python_executable>

Examples:
    MODULE3_PRESET=totalsegmenter_ct_mr_anchor_core ./run_module3_build_manifest.sh
    MODULE3_PRESET=mmwhs_ct_mr_core MODULE3_MIN_SAMPLES_PER_MODALITY=10 ./run_module3_build_manifest.sh
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
REQUIRED_MODALITIES_RAW="${MODULE3_REQUIRED_MODALITIES:-${MODULE3_REQUIRED_MODALITIES_DEFAULT:-ct mr}}"
REQUIRED_MODALITIES_RAW="${REQUIRED_MODALITIES_RAW//,/ }"
read -r -a REQUIRED_MODALITIES <<< "$REQUIRED_MODALITIES_RAW"

OUTPUT_DIR_ARG=()
if [[ -n "${MODULE3_OUTPUT_MANIFEST_DIR:-}" ]]; then
    OUTPUT_DIR_ARG=(--output-dir "$MODULE3_OUTPUT_MANIFEST_DIR")
fi

echo "================================================================="
echo "Module 3 Manifest Builder"
echo "================================================================="
echo "Analysis:         $MODULE3_ANALYSIS_NAME"
echo "Source manifest:  $MODULE3_SOURCE_MANIFEST"
echo "Output manifest:  $MODULE3_MANIFEST"
echo "Variant:          $MANIFEST_VARIANT"
echo "Modalities:       ${REQUIRED_MODALITIES[*]}"
echo "Min hold-outs:    ${MODULE3_MIN_HOLDOUT_ORGANS:-2}"
echo "Min samples/mod:  ${MODULE3_MIN_SAMPLES_PER_MODALITY:-5}"
echo "================================================================="

"$PYTHON_BIN" "$ROOT_DIR/build_module3_manifest.py" \
    --source-manifest "$MODULE3_SOURCE_MANIFEST" \
    --analysis-name "$MODULE3_ANALYSIS_NAME" \
    --manifest-variant "$MANIFEST_VARIANT" \
    --required-modalities "${REQUIRED_MODALITIES[@]}" \
    --min-holdout-organs "${MODULE3_MIN_HOLDOUT_ORGANS:-2}" \
    --min-samples-per-modality "${MODULE3_MIN_SAMPLES_PER_MODALITY:-5}" \
    "${OUTPUT_DIR_ARG[@]}"

echo "================================================================="
echo "Module 3 manifest build completed at $(date)"
echo "================================================================="
