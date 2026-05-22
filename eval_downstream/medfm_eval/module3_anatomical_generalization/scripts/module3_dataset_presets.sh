#!/bin/bash
# Module 3 preset and path helper library.

module3_print_preset_help() {
    cat <<'EOF'
Module 3 preset resolver

Canonical presets:
    totalsegmenter_ct_mr_anchor_core
    mmwhs_ct_mr_core
    amos_ct_mr_core
    abdomenatlas_core
    abdomenatlas_subset_1k

Manual namespace overrides:
  export MODULE3_ANALYSIS_NAME=<dataset_namespace>
  export MODULE3_SOURCE_MANIFEST=<phase2_manifest_path>
  export MODULE3_MANIFEST=<module3_manifest_path>

Examples:
    MODULE3_PRESET=totalsegmenter_ct_mr_anchor_core ./run_module3_build_manifest.sh
    MODULE3_PRESET=mmwhs_ct_mr_core ./run_module3_anatomical_generalization.sh
    MODULE3_PRESET=abdomenatlas_core ./run_module3_anatomical_generalization.sh
EOF
}

_module3_set_preset() {
    local analysis_name=$1
    local manifest_variant=$2
    local required_modalities=${3:-"ct mr"}
    MODULE3_ANALYSIS_NAME=$analysis_name
    MODULE3_SOURCE_MANIFEST="$ROOT_DIR/../data_manifests/phase2_cross_modality_alignment/${analysis_name}/${manifest_variant}/manifest_sampled.json"
    MODULE3_MANIFEST="$ROOT_DIR/../data_manifests/module3_anatomical_generalization/${analysis_name}/${manifest_variant}/manifest_sampled.json"
    MODULE3_REQUIRED_MODALITIES_DEFAULT=$required_modalities
}

resolve_module3_preset() {
    local preset=$1
    case "$preset" in
        totalsegmenter_ct_mr_anchor_core)
            _module3_set_preset "totalsegmenter_ct_mr_anchor" "core" "ct mr"
            ;;
        mmwhs_ct_mr_core)
            _module3_set_preset "mmwhs_ct_mr" "core" "ct mr"
            ;;
        amos_ct_mr_core)
            _module3_set_preset "amos_ct_mr" "core" "ct mr"
            ;;
        abdomenatlas_core)
            _module3_set_preset "abdomenatlas" "core" "ct"
            ;;
        abdomenatlas_subset_1k)
            _module3_set_preset "abdomenatlas" "subset_1k" "ct"
            ;;
        *)
            echo "ERROR: Unknown Module 3 preset '$preset'" >&2
            return 1
            ;;
    esac
}

module3_normalize_manifest_variant() {
    local manifest_variant=$1
    case "$manifest_variant" in
        core|default|shared_organs)
            printf '%s\n' "core"
            ;;
        paired|paired_core)
            printf '%s\n' "paired"
            ;;
        *)
            printf '%s\n' "$manifest_variant"
            ;;
    esac
}

module3_manifest_variant_from_manifest() {
    local manifest_path=$1
    module3_normalize_manifest_variant "$(basename "$(dirname "$manifest_path")")"
}

module3_normalize_feature_type() {
    local feature_type=${1:-cls}
    case "${feature_type,,}" in
        cls)
            printf '%s\n' "cls"
            ;;
        avg_pool|avgpool|avg-pool)
            printf '%s\n' "avg_pool"
            ;;
        multilayer|multi_layer|multi-layer)
            printf '%s\n' "multilayer"
            ;;
        *)
            echo "ERROR: Unknown Module 3 feature type '$feature_type'" >&2
            return 1
            ;;
    esac
}

module3_all_feature_types() {
    printf '%s\n' \
        cls \
        avg_pool \
        multilayer
}

module3_all_checkpoints() {
    printf '%s\n' \
    Med3DINO_REL_c96 \
    Med3DINO_REL_c112 \
    Med3DINO_ISO_c96 \
    Med3DINO_SA_c96 \
    Med3DINO_SA_c112 \
    Med3DINO_ISO_c112 \
    Med3DINO_Base_c96 \
    Med3DINO_Base_c112 \
        3dinov2
}

module3_require_manual_namespace() {
    if [[ -z "${MODULE3_ANALYSIS_NAME:-}" ]]; then
        echo "ERROR: MODULE3_ANALYSIS_NAME must be set" >&2
        return 1
    fi
    if [[ -z "${MODULE3_SOURCE_MANIFEST:-}" ]]; then
        echo "ERROR: MODULE3_SOURCE_MANIFEST must be set" >&2
        return 1
    fi
    if [[ -z "${MODULE3_MANIFEST:-}" ]]; then
        echo "ERROR: MODULE3_MANIFEST must be set" >&2
        return 1
    fi
}

module3_init_namespace() {
    if [[ -n "${MODULE3_PRESET:-}" ]]; then
        resolve_module3_preset "$MODULE3_PRESET"
        return 0
    fi
    module3_require_manual_namespace
}

module3_resolve_output_root() {
    local root_dir=$1
    local analysis_name=$2
    local manifest_variant=$3
    manifest_variant="$(module3_normalize_manifest_variant "$manifest_variant")"
    printf '%s\n' "$root_dir/outputs_module3/${analysis_name}/module3/${manifest_variant}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    module3_print_preset_help
fi
