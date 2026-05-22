#!/bin/bash
# Controlled perturbation evaluation - Parallel - AVG_POOL features
# Runs all checkpoints across multiple GPUs in parallel
#
# PREREQUISITE: Run the cache generation first.
#   python ../build_perturbation_cache.py -m ../data_manifests/phase1_anisotropy_robustness/abdomenatlas/original_bins/manifest_sampled.json --crop-size 96 112
#
# With cached volumes, this runs in ~10 min total (vs ~4 hours without cache)
# Usage:
#   PHASE1_PRESET=abdomenatlas_core ./run_perturbation_parallel_avg_pool.sh
#   PHASE1_PRESET=totalsegmentermri_core ./run_perturbation_parallel_avg_pool.sh
#   PHASE1_PRESET=totalsegmenter_ct_core ./run_perturbation_parallel_avg_pool.sh

print_usage() {
    sed -n '2,10p' "$0"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    print_usage
    exit 0
fi

set -e
cd "$(dirname "$0")"
ROOT_DIR="$(cd .. && pwd)"
PYTHON_BIN="${PYTHON:-python}"
source ./phase1_dataset_presets.sh

# ========== CONFIGURATION ==========
if [ -n "${ANALYSIS_NAME:-}" ] || [ -n "${MANIFEST:-}" ]; then
    if [ -z "${ANALYSIS_NAME:-}" ] || [ -z "${MANIFEST:-}" ]; then
        echo "ERROR: Set both ANALYSIS_NAME and MANIFEST, or neither." >&2
        exit 1
    fi
    PRESET_LABEL="manual"
else
    PHASE1_PRESET="${PHASE1_PRESET:-abdomenatlas_core}"
    resolve_phase1_preset "$PHASE1_PRESET"
    if ! phase1_preset_supports "perturbation_only"; then
        echo "ERROR: Preset '$PHASE1_PRESET' is not valid for controlled perturbation runs." >&2
        exit 1
    fi
    ANALYSIS_NAME="$PHASE1_PRESET_ANALYSIS_NAME"
    MANIFEST="$PHASE1_PRESET_MANIFEST"
    PRESET_LABEL="$PHASE1_PRESET"
fi
FEATURE_TYPE="avg_pool"
MANIFEST_VARIANT="$(phase1_manifest_variant_from_manifest "$MANIFEST")"
OUTPUT_ROOT="$ROOT_DIR/outputs_phase1/${ANALYSIS_NAME}/phase1/${MANIFEST_VARIANT}"
CACHE_NAMESPACE="$ANALYSIS_NAME"
CACHE_ROOT="$(phase1_resolve_cache_root "$ROOT_DIR" "$CACHE_NAMESPACE" "$MANIFEST_VARIANT")"
if [ -n "${GPU_DIS:-}" ] && [ -z "${GPU_IDS:-}" ]; then
    echo "WARNING: GPU_DIS is deprecated; using it as an alias for GPU_IDS." >&2
    GPU_IDS="$GPU_DIS"
elif [ -n "${GPU_DIS:-}" ] && [ -n "${GPU_IDS:-}" ]; then
    echo "WARNING: Both GPU_DIS and GPU_IDS are set; using GPU_IDS and ignoring GPU_DIS." >&2
fi
GPU_IDS_RAW="${GPU_IDS:-0 1 2 3}"
GPU_IDS_RAW="${GPU_IDS_RAW//,/ }"
read -r -a GPU_IDS <<< "$GPU_IDS_RAW"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "ERROR: GPU_IDS must contain at least one GPU id." >&2
    exit 1
fi
# ===================================

LOGDIR="$OUTPUT_ROOT/logs/perturbation_${FEATURE_TYPE}"
mkdir -p "$LOGDIR"

CACHE_SIGNATURE="${CACHE_SIGNATURE:-$(phase1_resolve_cache_signature "$MANIFEST")}"

# Check if cache exists (crop96)
CACHE_DIR_96="$CACHE_ROOT/crop96/${CACHE_SIGNATURE}"
CACHE_COUNT_96=$(find "$CACHE_DIR_96" -type f -name '*.pt' 2>/dev/null | wc -l)
if [ "$CACHE_COUNT_96" -eq 0 ]; then
    echo "ERROR: Cache not found at $CACHE_DIR_96"
    echo "Please pre-generate cache with the matching manifest and analysis name. See readme.md for preset-specific commands."
    exit 1
fi

echo "Found $CACHE_COUNT_96 cached volumes (crop96)"

# Check if cache exists (crop112)
CACHE_DIR_112="$CACHE_ROOT/crop112/${CACHE_SIGNATURE}"
if [ -d "$CACHE_DIR_112" ]; then
    CACHE_COUNT_112=$(find "$CACHE_DIR_112" -type f -name '*.pt' 2>/dev/null | wc -l)
fi
if [ "${CACHE_COUNT_112:-0}" -gt 0 ]; then
    echo "Found $CACHE_COUNT_112 cached volumes (crop112)"
else
    echo "WARNING: crop112 cache not found - skipping c112 checkpoints"
fi

run_controlled_perturbation() {
    local gpu=$1
    shift
    for ckpt in "$@"; do
        echo "[$(date '+%H:%M:%S')] GPU $gpu: Starting controlled perturbation ($FEATURE_TYPE) for $ckpt"
        CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" "$ROOT_DIR/phase1_evaluation_pipeline.py" \
            --setting-b-only \
            -m "$MANIFEST" -a "$ANALYSIS_NAME" \
            -c "$ckpt" \
            -f "$FEATURE_TYPE" \
            > "$LOGDIR/${ckpt}.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] GPU $gpu: Completed $ckpt"
        else
            echo "[$(date '+%H:%M:%S')] GPU $gpu: FAILED $ckpt"
        fi
    done
}

echo "=== Controlled Perturbation Parallel Evaluation ($FEATURE_TYPE) ==="
echo "Preset: $PRESET_LABEL"
echo "Analysis: $ANALYSIS_NAME"
echo "Cache namespace: $CACHE_NAMESPACE"
echo "Manifest: $MANIFEST"
echo "Cache signature: $CACHE_SIGNATURE"
echo "GPU IDs: ${GPU_IDS[*]}"
echo "Logs: $LOGDIR"
echo "Starting at $(date)"
echo ""

pids=()
checkpoints=(Med3DINO_REL_c96 Med3DINO_SA_c96 Med3DINO_Base_c96 Med3DINO_ISO_c96)
if [ "${CACHE_COUNT_112:-0}" -gt 0 ]; then
    checkpoints+=(Med3DINO_Base_c112 Med3DINO_REL_c112 Med3DINO_SA_c112 Med3DINO_ISO_c112 3dinov2)
else
    echo "Skipping crop112 checkpoints: Med3DINO_Base_c112 Med3DINO_REL_c112 Med3DINO_SA_c112 Med3DINO_ISO_c112 3dinov2"
fi

declare -a gpu_assignments
for i in "${!GPU_IDS[@]}"; do
    gpu_assignments[$i]=""
done

for i in "${!checkpoints[@]}"; do
    slot=$(( i % ${#GPU_IDS[@]} ))
    gpu_assignments[$slot]="${gpu_assignments[$slot]} ${checkpoints[$i]}"
done

for i in "${!GPU_IDS[@]}"; do
    assignment="${gpu_assignments[$i]}"
    if [ -z "${assignment// }" ]; then
        continue
    fi
    read -r -a ckpt_group <<< "$assignment"
    echo "GPU ${GPU_IDS[$i]} checkpoints: ${ckpt_group[*]}"
    run_controlled_perturbation "${GPU_IDS[$i]}" "${ckpt_group[@]}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "=== One or more controlled perturbation checkpoints failed. See logs in $LOGDIR ===" >&2
    exit 1
fi

echo ""
echo "=== All controlled perturbation evaluations ($FEATURE_TYPE) completed at $(date) ==="
