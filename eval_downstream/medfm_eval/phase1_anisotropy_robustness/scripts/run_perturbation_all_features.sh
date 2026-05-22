#!/bin/bash
# Controlled perturbation evaluation - All feature families
# Usage:
#   PHASE1_PRESET=abdomenatlas_core ./run_perturbation_all_features.sh
#   PHASE1_PRESET=totalsegmentermri_core ./run_perturbation_all_features.sh
#   PHASE1_PRESET=totalsegmenter_ct_core ./run_perturbation_all_features.sh
#   GPU_IDS="0 1 2" PHASE1_PRESET=abdomenatlas_core ./run_perturbation_all_features.sh

print_usage() {
    sed -n '2,5p' "$0"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    print_usage
    exit 0
fi

set -e
cd "$(dirname "$0")"

FEATURE_TYPES="${FEATURE_TYPES:-cls avg_pool multilayer}"
if [ -n "${GPU_DIS:-}" ] && [ -z "${GPU_IDS:-}" ]; then
    echo "WARNING: GPU_DIS is deprecated; using it as an alias for GPU_IDS." >&2
    GPU_IDS="$GPU_DIS"
elif [ -n "${GPU_DIS:-}" ] && [ -n "${GPU_IDS:-}" ]; then
    echo "WARNING: Both GPU_DIS and GPU_IDS are set; using GPU_IDS and ignoring GPU_DIS." >&2
fi
GPU_IDS="${GPU_IDS:-0 1 2 3}"

echo "==============================================================="
echo "Controlled perturbation evaluation - all feature families"
echo "Preset: ${PHASE1_PRESET:-manual-or-default}"
echo "Feature types: $FEATURE_TYPES"
echo "GPU IDs: $GPU_IDS"
echo "Started at: $(date)"
echo "==============================================================="

for feature_type in $FEATURE_TYPES; do
    case "$feature_type" in
        cls)
            ./run_perturbation_parallel_cls.sh
            ;;
        avg_pool)
            ./run_perturbation_parallel_avg_pool.sh
            ;;
        multilayer)
            ./run_perturbation_parallel_multilayer.sh
            ;;
        *)
            echo "ERROR: Unknown feature type '$feature_type'" >&2
            echo "Known feature types: cls, avg_pool, multilayer" >&2
            exit 1
            ;;
    esac
done

echo "==============================================================="
echo "All controlled perturbation feature families completed at $(date)"
echo "==============================================================="