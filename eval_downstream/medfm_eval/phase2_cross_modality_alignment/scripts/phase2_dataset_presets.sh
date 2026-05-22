#!/bin/bash
# Phase 2 preset and path helper library.
#
# This script resolves canonical Phase 2 presets, manifest variants, output
# roots, feature bundle paths, and checkpoint slates for the shell launchers.
#
# Usage:
#   source ./scripts/phase2_dataset_presets.sh
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core
#   phase2_init_namespace
#
# Examples:
#   source ./scripts/phase2_dataset_presets.sh
#   PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core
#   phase2_init_namespace
#   phase2_resolve_output_root "$PWD" "$PHASE2_ANALYSIS_NAME" "core"


phase2_print_preset_help() {
    cat <<'EOF'
Phase 2 preset resolver

Canonical preset:
    totalsegmenter_ct_mr_anchor_core
    mmwhs_ct_mr_core
    amos_ct_mr_core
    chaos_ct_mr_core
    abdomenatlas_core
    abdomenatlas_subset_1k

Validation surfaces:
    start with manual namespace overrides until the manifest namespace is stable
    enough to deserve a canonical preset.

Use manual namespace overrides:
    export PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core

Or set manual namespace overrides:
  export PHASE2_ANALYSIS_NAME=<dataset_namespace>
  export PHASE2_MANIFEST=<manifest_path>

Optional:
  export PHASE2_EMBEDDINGS_NPZ=<npz_path>

Example:
    PHASE2_PRESET=totalsegmenter_ct_mr_anchor_core ./run_phase2_manifest_audit.sh

    or

    PHASE2_ANALYSIS_NAME=mmwhs_ct_mr \
    PHASE2_MANIFEST=../data_manifests/phase2_cross_modality_alignment/mmwhs_ct_mr/core/manifest_sampled.json \
    ./run_phase2_manifest_audit.sh
EOF
}

_phase2_set_preset() {
        local analysis_name=$1
        local manifest_variant=$2
    local required_modalities=${3:-"ct mr"}
    local skip_evaluation=${4:-0}
        PHASE2_ANALYSIS_NAME=$analysis_name
        PHASE2_MANIFEST="$ROOT_DIR/../data_manifests/phase2_cross_modality_alignment/${analysis_name}/${manifest_variant}/manifest_sampled.json"
    PHASE2_REQUIRED_MODALITIES_DEFAULT=$required_modalities
    PHASE2_SKIP_EVALUATION_DEFAULT=$skip_evaluation
}

resolve_phase2_preset() {
        local preset=$1
        case "$preset" in
                totalsegmenter_ct_mr_anchor_core)
                        _phase2_set_preset "totalsegmenter_ct_mr_anchor" "core" "ct mr" 0
                        ;;
                mmwhs_ct_mr_core)
                    _phase2_set_preset "mmwhs_ct_mr" "core" "ct mr" 0
                    ;;
                amos_ct_mr_core)
                    _phase2_set_preset "amos_ct_mr" "core" "ct mr" 0
                    ;;
                chaos_ct_mr_core)
                    _phase2_set_preset "chaos_ct_mr" "core" "ct mr" 0
                    ;;
                abdomenatlas_core)
                    _phase2_set_preset "abdomenatlas" "core" "ct" 1
                    ;;
                abdomenatlas_subset_1k)
                    _phase2_set_preset "abdomenatlas" "subset_1k" "ct" 1
                    ;;
                *)
                        echo "ERROR: Unknown Phase 2 preset '$preset'" >&2
                        return 1
                        ;;
        esac
}

phase2_normalize_manifest_variant() {
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

phase2_manifest_variant_from_manifest() {
    local manifest_path=$1
    phase2_normalize_manifest_variant "$(basename "$(dirname "$manifest_path")")"
}

phase2_normalize_feature_type() {
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
            echo "ERROR: Unknown Phase 2 feature type '$feature_type'" >&2
            return 1
            ;;
    esac
}

phase2_all_feature_types() {
    printf '%s\n' \
        cls \
        avg_pool \
        multilayer
}

phase2_normalize_checkpoint_name() {
    local checkpoint_name=$1
    case "$checkpoint_name" in
        c96_rel|Med3DINO_REL_c96)
            printf '%s\n' "Med3DINO_REL_c96"
            ;;
        c112_rel|Med3DINO_REL_c112)
            printf '%s\n' "Med3DINO_REL_c112"
            ;;
        c96_sa|Med3DINO_SA_c96)
            printf '%s\n' "Med3DINO_SA_c96"
            ;;
        c112_sa|Med3DINO_SA_c112)
            printf '%s\n' "Med3DINO_SA_c112"
            ;;
        c96_iso|Med3DINO_ISO_c96)
            printf '%s\n' "Med3DINO_ISO_c96"
            ;;
        c112_iso|Med3DINO_ISO_c112)
            printf '%s\n' "Med3DINO_ISO_c112"
            ;;
        c96_base|Med3DINO_Base_c96)
            printf '%s\n' "Med3DINO_Base_c96"
            ;;
        c112_base|Med3DINO_Base_c112)
            printf '%s\n' "Med3DINO_Base_c112"
            ;;
        3dinov2)
            printf '%s\n' "3dinov2"
            ;;
        *)
            echo "ERROR: Unknown checkpoint '$checkpoint_name'" >&2
            return 1
            ;;
    esac
}

phase2_feature_embedding_filename() {
    local feature_type
    feature_type="$(phase2_normalize_feature_type "${1:-cls}")"
    case "$feature_type" in
        cls)
            printf '%s\n' "phase2_organ_cls_embeddings.npz"
            ;;
        avg_pool)
            printf '%s\n' "phase2_organ_avg_pool_embeddings.npz"
            ;;
        multilayer)
            printf '%s\n' "phase2_organ_multilayer_embeddings.npz"
            ;;
    esac
}

phase2_feature_comparison_json_name() {
    local feature_type
    feature_type="$(phase2_normalize_feature_type "${1:-cls}")"
    case "$feature_type" in
        cls)
            printf '%s\n' "phase2_checkpoint_comparison.json"
            ;;
        avg_pool)
            printf '%s\n' "phase2_checkpoint_comparison_avg_pool.json"
            ;;
        multilayer)
            printf '%s\n' "phase2_checkpoint_comparison_multilayer.json"
            ;;
    esac
}

phase2_feature_comparison_csv_name() {
    local feature_type
    feature_type="$(phase2_normalize_feature_type "${1:-cls}")"
    case "$feature_type" in
        cls)
            printf '%s\n' "phase2_checkpoint_comparison.csv"
            ;;
        avg_pool)
            printf '%s\n' "phase2_checkpoint_comparison_avg_pool.csv"
            ;;
        multilayer)
            printf '%s\n' "phase2_checkpoint_comparison_multilayer.csv"
            ;;
    esac
}

phase2_require_manual_namespace() {
    if [[ -z "${PHASE2_ANALYSIS_NAME:-}" ]]; then
        echo "ERROR: PHASE2_ANALYSIS_NAME must be set" >&2
        return 1
    fi
    if [[ -z "${PHASE2_MANIFEST:-}" ]]; then
        echo "ERROR: PHASE2_MANIFEST must be set" >&2
        return 1
    fi
}

phase2_normalize_manifest_path() {
    local manifest_path=$1
    if [[ "$manifest_path" == /* ]]; then
        printf '%s\n' "$manifest_path"
    else
        printf '%s\n' "$ROOT_DIR/${manifest_path#./}"
    fi
}

phase2_init_namespace() {
    if [[ -n "${PHASE2_PRESET:-}" ]]; then
        resolve_phase2_preset "$PHASE2_PRESET"
        return 0
    fi
    phase2_require_manual_namespace
    PHASE2_MANIFEST="$(phase2_normalize_manifest_path "$PHASE2_MANIFEST")"
}

phase2_resolve_cache_root() {
    local root_dir=$1
    local cache_namespace=$2
    local manifest_variant=$3
    manifest_variant="$(phase2_normalize_manifest_variant "$manifest_variant")"
    printf '%s\n' "$root_dir/../caches/${cache_namespace}/phase2/${manifest_variant}"
}

phase2_resolve_crop_cache_dir() {
    local root_dir=$1
    local cache_namespace=$2
    local manifest_variant=$3
    local crop_size=$4
    manifest_variant="$(phase2_normalize_manifest_variant "$manifest_variant")"
    printf '%s\n' "$root_dir/../caches/${cache_namespace}/phase2/${manifest_variant}/crop${crop_size}/organ_crop_cache"
}

phase2_resolve_feature_npz() {
    local root_dir=$1
    local analysis_name=$2
    local manifest_variant=$3
    local checkpoint_name=$4
    local feature_type=${5:-cls}
    manifest_variant="$(phase2_normalize_manifest_variant "$manifest_variant")"
    checkpoint_name="$(phase2_normalize_checkpoint_name "$checkpoint_name")"
    feature_type="$(phase2_normalize_feature_type "$feature_type")"
    printf '%s\n' "$root_dir/outputs_phase2/${analysis_name}/phase2/${manifest_variant}/features/${checkpoint_name}/${feature_type}/$(phase2_feature_embedding_filename "$feature_type")"
}

phase2_resolve_primary_metrics_path() {
    local root_dir=$1
    local analysis_name=$2
    local manifest_variant=$3
    local checkpoint_name=$4
    local feature_type=${5:-cls}
    manifest_variant="$(phase2_normalize_manifest_variant "$manifest_variant")"
    checkpoint_name="$(phase2_normalize_checkpoint_name "$checkpoint_name")"
    feature_type="$(phase2_normalize_feature_type "$feature_type")"
    if [[ "$feature_type" == "cls" ]]; then
        printf '%s\n' "$root_dir/outputs_phase2/${analysis_name}/phase2/${manifest_variant}/results/${checkpoint_name}/phase2_primary_metrics.json"
    else
        printf '%s\n' "$root_dir/outputs_phase2/${analysis_name}/phase2/${manifest_variant}/results/${checkpoint_name}/${feature_type}/phase2_primary_metrics.json"
    fi
}

phase2_resolve_output_root() {
    local root_dir=$1
    local analysis_name=$2
    local manifest_variant=$3
    manifest_variant="$(phase2_normalize_manifest_variant "$manifest_variant")"
    printf '%s\n' "$root_dir/outputs_phase2/${analysis_name}/phase2/${manifest_variant}"
}

phase2_all_checkpoints() {
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

phase2_checkpoint_crop_size() {
    local checkpoint_name
    checkpoint_name="$(phase2_normalize_checkpoint_name "$1")"
    case "$checkpoint_name" in
        Med3DINO_REL_c96|Med3DINO_SA_c96|Med3DINO_Base_c96|Med3DINO_ISO_c96)
            printf '%s\n' 96
            ;;
        Med3DINO_Base_c112|Med3DINO_REL_c112|Med3DINO_SA_c112|Med3DINO_ISO_c112|3dinov2)
            printf '%s\n' 112
            ;;
        *)
            echo "ERROR: Unknown checkpoint '$checkpoint_name'" >&2
            return 1
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    phase2_print_preset_help
fi