#!/bin/bash
# Phase 1 preset resolver.
#
# Canonical naming:
#   <dataset>_core   -> original_bins manifest
#   <dataset>_coarse -> coarse_bins manifest when available
#
# Direct usage:
#   source ./phase1_dataset_presets.sh
#   resolve_phase1_preset abdomenatlas_core
#   printf '%s\n' "$PHASE1_PRESET_MANIFEST"

phase1_print_preset_help() {
    cat <<'EOF'
Phase 1 preset matrix

Setting A and Setting B:
  abdomenatlas_core
  abdomenct1k_core
  cirrmri600_core
  duke_liver_core
  jhu_stroke_core
    kits23_core
  pansegdata_core
  totalsegmentermri_core
  abdomenatlas_coarse
  abdomenct1k_coarse

Setting B only:
        hecktor25_core
        imagecas_core
    totalsegmenter_ct_core

Examples:
  source ./phase1_dataset_presets.sh && resolve_phase1_preset abdomenatlas_core
    source ./phase1_dataset_presets.sh && resolve_phase1_preset totalsegmenter_ct_core
EOF
}

phase1_known_presets() {
    printf '%s\n' \
        abdomenatlas_core \
        abdomenct1k_core \
        cirrmri600_core \
        duke_liver_core \
        jhu_stroke_core \
        kits23_core \
        pansegdata_core \
        totalsegmentermri_core \
        hecktor25_core \
        imagecas_core \
        totalsegmenter_ct_core \
        abdomenatlas_coarse \
        abdomenct1k_coarse
}

_phase1_set_preset() {
    local analysis_name=$1
    local manifest_variant=$2
    local supported_tasks=$3

    PHASE1_PRESET_ANALYSIS_NAME="$analysis_name"
    PHASE1_PRESET_MANIFEST="../../data_manifests/phase1_anisotropy_robustness/${analysis_name}/${manifest_variant}/manifest_sampled.json"
    PHASE1_PRESET_SUPPORTED_TASKS="$supported_tasks"
}

resolve_phase1_preset() {
    local preset=$1

    case "$preset" in
        abdomenatlas_core)
            _phase1_set_preset "abdomenatlas" "original_bins" "full_phase1 perturbation_only"
            ;;
        abdomenct1k_core)
            _phase1_set_preset "abdomenct1k" "original_bins" "full_phase1 perturbation_only"
            ;;
        abdomenatlas_coarse)
            _phase1_set_preset "abdomenatlas" "coarse_bins" "full_phase1 perturbation_only"
            ;;
        abdomenct1k_coarse)
            _phase1_set_preset "abdomenct1k" "coarse_bins" "full_phase1 perturbation_only"
            ;;
        cirrmri600_core)
            _phase1_set_preset "cirrmri600" "original_bins" "full_phase1 perturbation_only"
            ;;
        duke_liver_core)
            _phase1_set_preset "duke_liver" "original_bins" "full_phase1 perturbation_only"
            ;;
        jhu_stroke_core)
            _phase1_set_preset "jhu_stroke" "original_bins" "full_phase1 perturbation_only"
            ;;
        kits23_core)
            _phase1_set_preset "kits23" "original_bins" "full_phase1 perturbation_only"
            ;;
        pansegdata_core)
            _phase1_set_preset "pansegdata" "original_bins" "full_phase1 perturbation_only"
            ;;
        totalsegmentermri_core)
            _phase1_set_preset "totalsegmentermri" "original_bins" "full_phase1 perturbation_only"
            ;;
        hecktor25_core)
            _phase1_set_preset "hecktor25" "original_bins" "perturbation_only"
            ;;
        imagecas_core)
            _phase1_set_preset "imagecas" "original_bins" "perturbation_only"
            ;;
        totalsegmenter_ct_core)
            _phase1_set_preset "totalsegmenter_ct" "original_bins" "perturbation_only"
            ;;
        "")
            echo "ERROR: PHASE1_PRESET cannot be empty" >&2
            return 1
            ;;
        *)
            echo "ERROR: Unknown PHASE1_PRESET '$preset'" >&2
            echo "Known presets:" >&2
            phase1_known_presets | sed 's/^/  - /' >&2
            return 1
            ;;
    esac
}

phase1_preset_supports() {
    local task_name=$1
    case " ${PHASE1_PRESET_SUPPORTED_TASKS:-} " in
        *" ${task_name} "*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

phase1_normalize_manifest_variant() {
    local manifest_variant=$1
    case "$manifest_variant" in
        original|original_bin|original_bins)
            printf '%s\n' "original_bins"
            ;;
        coarse|coarse_bin|coarse_bins|coarse_ratio_thickness|coarse_ratio_thickness_bins)
            printf '%s\n' "coarse_bins"
            ;;
        *)
            printf '%s\n' "$manifest_variant"
            ;;
    esac
}

phase1_manifest_variant_from_manifest() {
    local manifest_path=$1
    phase1_normalize_manifest_variant "$(basename "$(dirname "$manifest_path")")"
}

phase1_resolve_cache_root() {
    local root_dir=$1
    local cache_namespace=$2
    local manifest_variant=$3
    manifest_variant="$(phase1_normalize_manifest_variant "$manifest_variant")"
    printf '%s\n' "$root_dir/../caches/${cache_namespace}/phase1/${manifest_variant}"
}

phase1_resolve_cache_signature() {
    local manifest_path=$1
    local phase1_root
    local python_bin="${PYTHON:-python}"
    phase1_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    "$python_bin" - "$phase1_root" "$manifest_path" <<'PY'
import json
import sys
from pathlib import Path

phase1_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(phase1_root))

from config import get_controlled_perturbation_config_from_manifest

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(get_controlled_perturbation_config_from_manifest(manifest).cache_signature())
PY
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    phase1_print_preset_help
fi